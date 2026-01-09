use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder};
use candle_transformers::generation::LogitsProcessor;
use hf_hub::{Repo, RepoType, api::sync::Api};
use std::path::PathBuf;
use tokenizers::Tokenizer;

mod punc_norm;
mod s3gen_flow;
mod voice_encoder;

use candle_nn::{ConvTranspose1d, ConvTranspose1dConfig};
use punc_norm::punc_norm;
use pycandle_core::gpt2::{Config as GPTConfig, GPTModel, KVCache};
use voice_encoder::{Config as VEConfig, VoiceEncoder};

// Helper to load WeightNorm-parametrized weights (g * v / ||v||)
// Helper to load WeightNorm-parametrized weights (g * v / ||v||)
fn load_weight_norm(
    vb: VarBuilder,
    v_shape: (usize, usize, usize),
    g_shape: (usize, usize, usize),
) -> Result<Linear> {
    // 1. Load Direction Vector (v) -> "original1"
    let v = vb
        .pp("parametrizations.weight")
        .get(v_shape, "original1")
        .map_err(anyhow::Error::msg)?;

    // 2. Load Magnitude (g) -> "original0"
    let g = vb
        .pp("parametrizations.weight")
        .get(g_shape, "original0")
        .map_err(anyhow::Error::msg)?;

    // 3. Compute Norm of v along all dimensions except 0 (output channels)
    // v shape: [C_out, ...]
    let v_dims = v.dims();
    // Flatten all dims except the first (output channels)
    let v_flat = v.flatten_from(1).map_err(anyhow::Error::msg)?;
    let v_norm = v_flat
        .sqr()
        .map_err(anyhow::Error::msg)?
        .sum_keepdim(1)
        .map_err(anyhow::Error::msg)?
        .sqrt()
        .map_err(anyhow::Error::msg)?;

    // 4. Calculate Effective Weight: w = g * (v / ||v||)
    let mut broadcast_shape = vec![v_dims[0]];
    for _ in 1..v_dims.len() {
        broadcast_shape.push(1);
    }

    let g_reshaped = g
        .reshape(&broadcast_shape[..])
        .map_err(anyhow::Error::msg)?;
    let norm_reshaped = v_norm
        .reshape(&broadcast_shape[..])
        .map_err(anyhow::Error::msg)?;

    let w = v
        .broadcast_div(&norm_reshaped)
        .map_err(anyhow::Error::msg)?
        .broadcast_mul(&g_reshaped)
        .map_err(anyhow::Error::msg)?;

    // 5. Load Bias
    let bias = vb.get(v_dims[0], "bias").ok();

    Ok(Linear::new(w, bias))
}

const HF_CACHE: &str = "D:/huggingface";
const REPO_ID: &str = "ResembleAI/chatterbox-turbo";

pub struct LearnedPositionEmbeddings {
    pub emb: Embedding,
}

impl LearnedPositionEmbeddings {
    pub fn load(vb: VarBuilder, vocab_size: usize, dim: usize) -> Result<Self> {
        let emb = candle_nn::embedding(vocab_size, dim, vb.pp("emb"))?;
        Ok(Self { emb })
    }

    pub fn forward(&self, t: usize, device: &Device) -> Result<Tensor> {
        let pos = Tensor::arange(0u32, t as u32, device)?.unsqueeze(0)?;
        Ok(self.emb.forward(&pos)?)
    }

    pub fn get_at(&self, idx: usize, device: &Device) -> Result<Tensor> {
        let pos = Tensor::new(&[idx as u32], device)?.unsqueeze(0)?;
        Ok(self.emb.forward(&pos)?)
    }
}

pub struct T3CondEnc {
    pub spkr_enc: Linear,
    pub emotion_adv_fc: Option<Linear>,
}

impl T3CondEnc {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let spkr_enc = candle_nn::linear(256, 1024, vb.pp("spkr_enc"))?;
        let emotion_adv_fc = vb
            .pp("emotion_adv_fc")
            .get((1024, 1), "weight")
            .ok()
            .map(|w| Linear::new(w, None));
        Ok(Self {
            spkr_enc,
            emotion_adv_fc,
        })
    }

    pub fn forward(&self, speaker_emb: &Tensor, device: &Device) -> Result<Tensor> {
        let b = speaker_emb.dim(0).map_err(anyhow::Error::msg)?;
        let emb = if speaker_emb.rank() == 3 {
            speaker_emb.mean(1).map_err(anyhow::Error::msg)?
        } else {
            speaker_emb.clone()
        };
        let cond_spkr = self
            .spkr_enc
            .forward(&emb.reshape((b, 256)).map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?;
        let mut res = cond_spkr
            .reshape((b, 1, 1024))
            .map_err(anyhow::Error::msg)?;

        if let Some(ref fc) = self.emotion_adv_fc {
            // Default emotion_adv is 0.5
            let val = Tensor::new(&[0.5f32], device)
                .map_err(anyhow::Error::msg)?
                .reshape((b, 1, 1))
                .map_err(anyhow::Error::msg)?;
            let emotion_emb = fc.forward(&val).map_err(anyhow::Error::msg)?;
            res = (res + emotion_emb).map_err(anyhow::Error::msg)?;
        }
        Ok(res)
    }
}

pub struct T3Model {
    pub tfmr: GPTModel,
    pub speech_head: Linear,
    pub cond_enc: T3CondEnc,
}

impl T3Model {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let config = GPTConfig {
            vocab_size: 50276,
            context_length: 8196,
            emb_dim: 1024,
            n_heads: 16,
            n_layers: 24,
            ..Default::default()
        };

        let tfmr = GPTModel::new(config, vb.pp("tfmr"))?;
        let speech_head = candle_nn::linear(1024, 6563, vb.pp("speech_head"))?;
        let cond_enc = T3CondEnc::load(vb.pp("cond_enc"))?;

        Ok(Self {
            tfmr,
            speech_head,
            cond_enc,
        })
    }

    pub fn forward_manual(
        &self,
        x: &Tensor, // (B, T, D)
        mut kv_cache: Option<&mut Vec<Option<KVCache>>>,
        offset: usize,
    ) -> Result<Tensor> {
        let (_b, t, _d) = x.dims3()?;
        let device = x.device();

        // GPT-2 backbone positional encoding (wpe)
        let pos = Tensor::arange(offset as u32, (offset + t) as u32, device)?;
        let pos_emb = self.tfmr.wpe.forward(&pos)?;
        let h = x.broadcast_add(&pos_emb.unsqueeze(0)?)?;

        let mut h = h.clone();
        for (i, block) in self.tfmr.h.iter().enumerate() {
            let layer_cache = if let Some(ref mut cache) = kv_cache {
                if cache.len() <= i {
                    cache.resize_with(i + 1, || None);
                }
                cache.get_mut(i)
            } else {
                None
            };
            h = block.forward(&h, self.tfmr.mask.as_ref(), layer_cache)?;
        }

        Ok(self.tfmr.ln_f.forward(&h)?)
    }

    pub fn generate_tokens(
        &self,
        text_tokens: &Tensor, // [1, N]
        speaker_emb: &Tensor,
        device: &Device,
    ) -> Result<Vec<u32>> {
        // Wrap text with BOT (255) and EOT (0)
        let mut wrapped_text = vec![255u32];
        wrapped_text.extend(text_tokens.flatten_all()?.to_vec1::<u32>()?);
        wrapped_text.push(0u32);
        let text_tokens = Tensor::new(wrapped_text.as_slice(), device)?.unsqueeze(0)?;

        let cond_emb = self.cond_enc.forward(speaker_emb, device)?;
        let text_embed = self.tfmr.wte.forward(&text_tokens)?;

        let inputs_embeds = Tensor::cat(&[&cond_emb, &text_embed], 1)?;
        let mut generated_tokens = Vec::new();
        let mut kv_cache = Vec::new();

        // SOS token is 6561
        let mut current_token = Tensor::new(&[6561u32], device)?.unsqueeze(0)?;
        let mut logits_processor = LogitsProcessor::new(1337, Some(0.8), Some(0.95));

        println!("  (Sampling Loop Start)");

        for i in 0..1000 {
            let current_emb = if i == 0 {
                let bos_emb = self.tfmr.wte.forward(&current_token)?;
                Tensor::cat(&[&inputs_embeds, &bos_emb], 1)?
            } else {
                self.tfmr.wte.forward(&current_token)?
            };

            let offset = if i == 0 { 0 } else { inputs_embeds.dim(1)? + i };
            let h = self.forward_manual(&current_emb, Some(&mut kv_cache), offset)?;

            let last_h = h.i((.., h.dim(1)? - 1, ..))?;
            let logits = self.speech_head.forward(&last_h)?;

            let next_token = logits_processor.sample(&logits.i(0)?)?;
            if next_token == 6562 {
                // EOS
                break;
            }

            generated_tokens.push(next_token);
            current_token = Tensor::new(&[next_token], device)?.unsqueeze(0)?;

            if i % 100 == 0 && i > 0 {
                println!("    Sampled {} tokens...", i);
            }
        }

        Ok(generated_tokens)
    }
}

// ============================================================================
// S3Gen / HiFTGenerator Components
// ============================================================================

pub struct F0Predictor {
    pub condnet: Vec<Linear>,
    pub classifier: candle_nn::Linear,
}

fn cumsum_2d(x: &Tensor) -> Result<Tensor> {
    let (b, c, t) = x.dims3().map_err(anyhow::Error::msg)?;
    let mut data = x
        .to_device(&Device::Cpu)
        .map_err(anyhow::Error::msg)?
        .to_vec3::<f32>()
        .map_err(anyhow::Error::msg)?;
    for bi in 0..b {
        for ci in 0..c {
            for ti in 1..t {
                data[bi][ci][ti] += data[bi][ci][ti - 1];
            }
        }
    }
    let flattened: Vec<f32> = data
        .into_iter()
        .flat_map(|bc| bc.into_iter().flat_map(|t| t.into_iter()))
        .collect();
    Tensor::from_vec(flattened, (b, c, t), &Device::Cpu)?
        .to_device(x.device())
        .map_err(anyhow::Error::msg)
}

pub struct SineGen {
    pub sampling_rate: f64,
    pub harmonic_num: usize,
    pub sine_amp: f64,
    pub noise_std: f64,
    pub voiced_threshold: f64,
}

impl SineGen {
    pub fn new(sr: f64, harm: usize, amp: f64, noise: f64, thresh: f64) -> Self {
        Self {
            sampling_rate: sr,
            harmonic_num: harm,
            sine_amp: amp,
            noise_std: noise,
            voiced_threshold: thresh,
        }
    }

    pub fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        let (_b, _c, _t) = f0.dims3().map_err(anyhow::Error::msg)?;
        let _device = f0.device();

        // F_mat = f0 * (i + 1) / sr
        let mut f_mats = Vec::new();
        for i in 0..=self.harmonic_num {
            let f_i = f0
                .affine((i + 1) as f64 / self.sampling_rate, 0.0)
                .map_err(anyhow::Error::msg)?;
            f_mats.push(f_i);
        }
        let f_mat = Tensor::cat(&f_mats, 1).map_err(anyhow::Error::msg)?; // [B, harm+1, T]

        // theta_mat = 2 * PI * cumsum(f_mat)
        let theta_mat = cumsum_2d(&f_mat)?
            .affine(2.0 * std::f32::consts::PI as f64, 0.0)
            .map_err(anyhow::Error::msg)?;

        let sine_waves = theta_mat
            .sin()
            .map_err(anyhow::Error::msg)?
            .affine(self.sine_amp, 0.0)
            .map_err(anyhow::Error::msg)?;

        // U/V
        let uv = f0
            .affine(1.0, -self.voiced_threshold as f64)
            .map_err(anyhow::Error::msg)?
            .gt(&f0.zeros_like().map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?
            .to_dtype(DType::F32)
            .map_err(anyhow::Error::msg)?;

        // Noise
        let voiced_noise = uv.affine(self.noise_std, 0.0).map_err(anyhow::Error::msg)?;
        let unvoiced_noise = uv
            .affine(-1.0, 1.0)
            .map_err(anyhow::Error::msg)?
            .affine(self.sine_amp / 3.0, 0.0)
            .map_err(anyhow::Error::msg)?;
        let noise_amp = (voiced_noise + unvoiced_noise).map_err(anyhow::Error::msg)?;
        let randn = sine_waves
            .randn_like(0.0, 1.0)
            .map_err(anyhow::Error::msg)?;
        let noise = randn
            .broadcast_mul(&noise_amp)
            .map_err(anyhow::Error::msg)?;

        // first: set the unvoiced part to 0 by uv
        // then: additive noise
        let voiced_part = sine_waves.broadcast_mul(&uv).map_err(anyhow::Error::msg)?;
        let res = (voiced_part + noise).map_err(anyhow::Error::msg)?;
        Ok(res)
    }
}

pub struct SourceModule {
    pub sin_gen: SineGen,
    pub linear: candle_nn::Linear,
}

impl SourceModule {
    pub fn load(vb: VarBuilder, sr: f64, harm: usize) -> Result<Self> {
        let sin_gen = SineGen::new(sr, harm, 0.1, 0.003, 0.0);
        let linear_w = vb.get((1, harm + 1), "l_linear.weight")?;
        let linear_b = vb.get((1,), "l_linear.bias")?;
        let linear = candle_nn::Linear::new(linear_w, Some(linear_b));
        Ok(Self { sin_gen, linear })
    }

    pub fn forward(&self, f0: &Tensor) -> Result<Tensor> {
        // f0 is [B, 1, T] at 24kHz
        let sine_wavs = self.sin_gen.forward(f0)?; // [B, harm+1, T]
        let sine_wavs = sine_wavs.transpose(1, 2).map_err(anyhow::Error::msg)?; // [B, T, harm+1]
        let merged = self
            .linear
            .forward(&sine_wavs)
            .map_err(anyhow::Error::msg)?; // [B, T, 1]
        let res = merged
            .transpose(1, 2)
            .map_err(anyhow::Error::msg)?
            .tanh()
            .map_err(anyhow::Error::msg)?;
        Ok(res)
    }
}

impl F0Predictor {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let mut condnet = Vec::new();
        let channels = [80, 512, 512, 512, 512, 512];
        for i in 0..5 {
            let vb_i = vb.pp(&format!("condnet.{}", i * 2)); // ELU is at 1, 3, 5...
            condnet.push(load_weight_norm(vb_i, (512, channels[i], 3), (512, 1, 1))?);
        }
        let classifier_w = vb
            .get((1, 512), "classifier.weight")
            .map_err(anyhow::Error::msg)?;
        let classifier_b = vb
            .get((1,), "classifier.bias")
            .map_err(anyhow::Error::msg)?;
        let classifier = candle_nn::Linear::new(classifier_w, Some(classifier_b));
        Ok(Self {
            condnet,
            classifier,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for conv in &self.condnet {
            x = x
                .conv1d(&conv.weight(), 1, 1, 1, 1) // padding 1, stride 1
                .map_err(anyhow::Error::msg)?;
            if let Some(bias) = conv.bias() {
                x = x
                    .broadcast_add(&bias.reshape((1, (), 1)).map_err(anyhow::Error::msg)?)
                    .map_err(anyhow::Error::msg)?;
            }
            x = x.elu(1.0).map_err(anyhow::Error::msg)?;
        }
        // x: (B, 512, T) -> (B, T, 512)
        x = x.transpose(1, 2).map_err(anyhow::Error::msg)?;
        let x = self.classifier.forward(&x).map_err(anyhow::Error::msg)?;
        // abs(squeeze(-1))
        x.squeeze(2)
            .map_err(anyhow::Error::msg)?
            .abs()
            .map_err(anyhow::Error::msg)
    }
}

pub struct ResBlock {
    pub convs1: Vec<Linear>, // Use Linear for WeightNorm original1
    pub convs2: Vec<Linear>,
    pub activations1: Vec<pycandle_core::layers::Snake>,
    pub activations2: Vec<pycandle_core::layers::Snake>,
    pub kernel_size: usize,
}

impl ResBlock {
    pub fn load(
        vb: VarBuilder,
        channels: usize,
        kernel_size: usize,
        dilations: &[usize],
    ) -> Result<Self> {
        let mut convs1 = Vec::new();
        let mut convs2 = Vec::new();
        let mut activations1 = Vec::new();
        let mut activations2 = Vec::new();

        for (i, &_dilation) in dilations.iter().enumerate() {
            // WeightNorm in Python means weight is in parametrizations.weight.original1
            convs1.push(load_weight_norm(
                vb.pp(format!("convs1.{}", i)),
                (channels, channels, kernel_size),
                (channels, 1, 1),
            )?);
            convs2.push(load_weight_norm(
                vb.pp(format!("convs2.{}", i)),
                (channels, channels, kernel_size),
                (channels, 1, 1),
            )?);

            activations1.push(
                pycandle_core::layers::Snake::load(vb.pp(&format!("activations1.{}", i)), channels)
                    .map_err(anyhow::Error::msg)?,
            );
            activations2.push(
                pycandle_core::layers::Snake::load(vb.pp(&format!("activations2.{}", i)), channels)
                    .map_err(anyhow::Error::msg)?,
            );
        }

        Ok(Self {
            convs1,
            convs2,
            activations1,
            activations2,
            kernel_size,
        })
    }

    pub fn forward(&self, x: &Tensor, dilations: &[usize]) -> Result<Tensor> {
        let mut x = x.clone();
        for i in 0..self.convs1.len() {
            let dilation = dilations[i];
            let padding = (self.kernel_size * dilation - dilation) / 2;

            let mut xt = self.activations1[i]
                .forward(&x)
                .map_err(anyhow::Error::msg)?;
            xt = xt
                .conv1d(&self.convs1[i].weight(), padding, 1, dilation, 1)
                .map_err(anyhow::Error::msg)?;
            if let Some(bias) = self.convs1[i].bias() {
                xt = xt
                    .broadcast_add(&bias.reshape((1, (), 1)).map_err(anyhow::Error::msg)?)
                    .map_err(anyhow::Error::msg)?;
            }

            xt = self.activations2[i]
                .forward(&xt)
                .map_err(anyhow::Error::msg)?;
            let p2 = (self.kernel_size - 1) / 2;
            xt = xt
                .conv1d(&self.convs2[i].weight(), p2, 1, 1, 1)
                .map_err(anyhow::Error::msg)?;
            if let Some(bias) = self.convs2[i].bias() {
                xt = xt
                    .broadcast_add(&bias.reshape((1, (), 1)).map_err(anyhow::Error::msg)?)
                    .map_err(anyhow::Error::msg)?;
            }
            x = (x + xt).map_err(anyhow::Error::msg)?;
        }
        Ok(x)
    }
}

pub struct HiFTGenerator {
    pub f0_predictor: F0Predictor,
    pub m_source: SourceModule,
    pub conv_pre: Linear,
    pub ups: Vec<candle_nn::ConvTranspose1d>,
    pub source_downs: Vec<Linear>,
    pub source_resblocks: Vec<ResBlock>,
    pub resblocks: Vec<ResBlock>,
    pub conv_post: Linear,
}

impl HiFTGenerator {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let f0_predictor = F0Predictor::load(vb.pp("mel2wav.f0_predictor"))?;
        let m_source = SourceModule::load(vb.pp("mel2wav.m_source"), 24000.0, 8)?;

        let conv_pre = load_weight_norm(vb.pp("mel2wav.conv_pre"), (512, 80, 7), (512, 1, 1))?;

        let mut ups = Vec::new();
        let mut source_downs = Vec::new();
        let mut source_resblocks = Vec::new();

        let rates = [8, 5, 3];
        let kernels = [16, 11, 7];
        let channels = [512, 256, 128, 64];
        let sd_res_kernels = [7, 7, 11];
        let sd_kernels = [30, 6, 1];

        for i in 0..3 {
            // UPS: Manually load WeightNorm weights for ConvTranspose1d
            let vb_u = vb.pp(format!("mel2wav.ups.{}", i));
            let weight = load_weight_norm(
                vb_u.clone(),
                (channels[i], channels[i + 1], kernels[i]),
                (channels[i], 1, 1),
            )?;
            let bias = vb_u.get((channels[i + 1],), "bias").ok();
            let config = candle_nn::ConvTranspose1dConfig {
                stride: rates[i],
                padding: (kernels[i] - rates[i]) / 2,
                ..Default::default()
            };
            ups.push(candle_nn::ConvTranspose1d::new(
                weight.weight().clone(),
                bias,
                config,
            ));

            // SOURCE DOWNS: Standard Linear/Conv1d (No Weight Norm)
            let vb_sd = vb.pp(format!("mel2wav.source_downs.{}", i));
            let w_sd = vb_sd
                .get((channels[i + 1], 18, sd_kernels[i]), "weight")
                .map_err(anyhow::Error::msg)?;
            let b_sd = vb_sd
                .get((channels[i + 1],), "bias")
                .map_err(anyhow::Error::msg)?;
            source_downs.push(Linear::new(w_sd, Some(b_sd)));

            source_resblocks.push(ResBlock::load(
                vb.pp(&format!("mel2wav.source_resblocks.{}", i)),
                channels[i + 1],
                sd_res_kernels[i],
                &[1, 3, 5],
            )?);
        }

        let mut resblocks = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                resblocks.push(ResBlock::load(
                    vb.pp(&format!("mel2wav.resblocks.{}", i * 3 + j)),
                    channels[i + 1],
                    [3, 7, 11][j],
                    &[1, 3, 5],
                )?);
            }
        }

        let conv_post = load_weight_norm(vb.pp("mel2wav.conv_post"), (18, 64, 7), (18, 1, 1))?;

        Ok(Self {
            f0_predictor,
            m_source,
            conv_pre,
            ups,
            source_downs,
            source_resblocks,
            resblocks,
            conv_post,
        })
    }

    pub fn forward(&self, mel: &Tensor) -> Result<Tensor> {
        let b = mel.dim(0)?;
        let t = mel.dim(2)?;

        // F0 branch
        let f0 = self.f0_predictor.forward(mel)?; // [B, T]
        let f0 = f0.unsqueeze(1)?; // [B, 1, T]

        // Upsample F0 to 24kHz (50Hz -> 24kHz: factor 480)
        let f0_up = f0
            .unsqueeze(2)?
            .upsample_nearest2d(1, t * 480)?
            .squeeze(2)?;

        let s = self.m_source.forward(&f0_up)?; // [B, 1, 120T]

        // s_stft: STFT of s with hop 4
        use pycandle_audio::{PadMode, StftConfig, stft};
        let stft_cfg = StftConfig {
            n_fft: 16,
            hop_length: Some(4),
            win_length: Some(16),
            center: true,
            pad_mode: PadMode::Reflect,
            ..Default::default()
        };
        // s is [B, 1, T_24k], stft expects [B, T_24k]
        let s_stft_complex = stft(&s.squeeze(1).map_err(anyhow::Error::msg)?, &stft_cfg, None)
            .map_err(anyhow::Error::msg)?; // [B, 9, T_voc, 2]
        let s_stft = s_stft_complex
            .reshape((b, 18, ()))
            .map_err(anyhow::Error::msg)?;

        let mut x = mel
            .conv1d(&self.conv_pre.weight(), 3, 1, 1, 1) // padding 3, stride 1
            .map_err(anyhow::Error::msg)?;
        if let Some(bias) = self.conv_pre.bias() {
            x = x
                .broadcast_add(&bias.reshape((1, (), 1)).map_err(anyhow::Error::msg)?)
                .map_err(anyhow::Error::msg)?;
        }

        let sd_strides = [15, 3, 1];
        let sd_paddings = [7, 1, 0];

        for i in 0..3 {
            x = candle_nn::ops::leaky_relu(&x, 0.1).map_err(anyhow::Error::msg)?;
            x = self.ups[i].forward(&x).map_err(anyhow::Error::msg)?;

            if i == 2 {
                // Reflection pad (1, 0)
                let zeros = Tensor::zeros((b, x.dim(1)?, 1), x.dtype(), x.device())?;
                x = Tensor::cat(&[&zeros, &x], 2)?;
            }

            // Fusion
            let mut si = s_stft
                .conv1d(
                    &self.source_downs[i].weight(),
                    sd_paddings[i],
                    sd_strides[i],
                    1,
                    1,
                )
                .map_err(anyhow::Error::msg)?;
            if let Some(bias) = self.source_downs[i].bias() {
                si = si
                    .broadcast_add(&bias.reshape((1, (), 1)).map_err(anyhow::Error::msg)?)
                    .map_err(anyhow::Error::msg)?;
            }
            si = self.source_resblocks[i].forward(&si, &[1, 3, 5])?;

            // si and x should now match exactly thanks to reflection_pad
            x = (x + si).map_err(anyhow::Error::msg)?;

            let mut res_sum: Option<Tensor> = None;
            for j in 0..3 {
                let r = self.resblocks[i * 3 + j].forward(&x, &[1, 3, 5])?;
                res_sum = match res_sum {
                    Some(s) => Some((s + r).map_err(anyhow::Error::msg)?),
                    None => Some(r),
                };
            }
            x = (res_sum.unwrap() / 3.0).map_err(anyhow::Error::msg)?;
        }

        x = x.relu().map_err(anyhow::Error::msg)?;
        x = x
            .conv1d(&self.conv_post.weight(), 3, 1, 1, 1) // padding 3, stride 1
            .map_err(anyhow::Error::msg)?;
        if let Some(bias) = self.conv_post.bias() {
            x = x
                .broadcast_add(&bias.reshape((1, (), 1)).map_err(anyhow::Error::msg)?)
                .map_err(anyhow::Error::msg)?;
        }

        // ISTFT: 18 channels -> 9 Mag, 9 Phase
        let mag = x
            .narrow(1, 0, 9)
            .map_err(anyhow::Error::msg)?
            .exp()
            .map_err(anyhow::Error::msg)?
            .clamp(0.0f32, 100.0f32)
            .map_err(anyhow::Error::msg)?;
        let phase = x.narrow(1, 9, 9).map_err(anyhow::Error::msg)?;

        let real = mag
            .broadcast_mul(&phase.cos().map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?;
        let imag = mag
            .broadcast_mul(&phase.sin().map_err(anyhow::Error::msg)?)
            .map_err(anyhow::Error::msg)?;

        // Zero out imaginary part for DC and Nyquist bins (0 and 8 for n_fft=16)
        // Hermitian symmetry: DC and Nyquist bins must be real (imag part must be exactly zero)
        let device = x.device();
        let t = imag.dim(2)?;
        let zero_row = Tensor::zeros((1, 1, t), DType::F32, device).map_err(anyhow::Error::msg)?;
        let mid = imag.narrow(1, 1, 7).map_err(anyhow::Error::msg)?;
        let imag = Tensor::cat(&[&zero_row, &mid, &zero_row], 1).map_err(anyhow::Error::msg)?;

        let spec = Tensor::cat(
            &[
                real.unsqueeze(3).map_err(anyhow::Error::msg)?,
                imag.unsqueeze(3).map_err(anyhow::Error::msg)?,
            ],
            3,
        )
        .map_err(anyhow::Error::msg)?;

        spec.squeeze(0).map_err(anyhow::Error::msg)
    }
}

pub struct Orchestrator {
    pub device: Device,
    pub tokenizer: Tokenizer,
    pub repo: Repo,
}

impl Orchestrator {
    pub fn load(device: &Device) -> Result<Self> {
        unsafe {
            std::env::set_var("HF_HOME", HF_CACHE);
        }
        let api = Api::new()?;
        let repo = Repo::new(REPO_ID.to_string(), RepoType::Model);
        let repo_handle = api.repo(repo.clone());

        println!("📦 Loading Tokenizer...");
        let vocab_path = repo_handle.get("vocab.json")?;
        let merges_path = repo_handle.get("merges.txt")?;

        use tokenizers::models::bpe::BPE;
        let bpe = BPE::from_file(vocab_path.to_str().unwrap(), merges_path.to_str().unwrap())
            .build()
            .map_err(anyhow::Error::msg)?;
        let tokenizer = Tokenizer::new(bpe);

        Ok(Self {
            device: device.clone(),
            tokenizer,
            repo,
        })
    }

    pub fn generate_sequential(
        &self,
        text: &str,
        _ref_audio_path: Option<PathBuf>,
    ) -> Result<Vec<f32>> {
        let text = punc_norm(text);
        println!("📝 Normalized text: \"{}\"", text);
        let tokens = self
            .tokenizer
            .encode(text, true)
            .map_err(anyhow::Error::msg)?;
        let text_tokens = Tensor::new(tokens.get_ids(), &self.device)?.unsqueeze(0)?;

        let api = Api::new()?;
        let repo_handle = api.repo(self.repo.clone());

        // Stage 1: Voice Encoding
        println!("🎙️ Stage 1: Voice Encoding...");
        let speaker_emb = {
            let ve_path = repo_handle.get("ve.safetensors")?;
            let vb_ve = unsafe {
                VarBuilder::from_mmaped_safetensors(&[ve_path], DType::F32, &self.device)?
            };
            let ve = VoiceEncoder::load(VEConfig {}, vb_ve, None).map_err(anyhow::Error::msg)?;
            let ref_path = if let Some(p) = _ref_audio_path {
                p
            } else {
                PathBuf::from("reference.wav")
            };
            println!("📖 Loading reference audio from {:?}...", ref_path);
            let mut reader = hound::WavReader::open(ref_path).map_err(anyhow::Error::msg)?;
            let spec = reader.spec();
            let samples: Vec<f32> = reader
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect();
            let audio_tensor = Tensor::from_vec(samples, (1, reader.len() as usize), &self.device)
                .map_err(anyhow::Error::msg)?;

            // Compute Mel Spectrogram for VoiceEncoder (40 pins, 16k sr)
            use pycandle_audio::{
                MelNorm, MelScale, MelSpectrogramConfig, PadMode, StftConfig, mel_spectrogram,
            };
            let melspec_cfg = MelSpectrogramConfig {
                stft_config: StftConfig {
                    n_fft: 400,
                    hop_length: Some(160),
                    win_length: Some(400),
                    center: true,
                    pad_mode: PadMode::Reflect,
                    ..Default::default()
                },
                sample_rate: spec.sample_rate as usize,
                n_mels: 40,
                f_min: 0.0,
                f_max: Some(8000.0),
                mel_scale: MelScale::Htk,
                norm: MelNorm::None,
            };

            let mel =
                mel_spectrogram(&audio_tensor, &melspec_cfg, None).map_err(anyhow::Error::msg)?;
            // mel is (B, 40, T), VoiceEncoder expects (B, T, 40)
            let mel = mel.transpose(1, 2).map_err(anyhow::Error::msg)?;

            ve.forward(&mel).map_err(anyhow::Error::msg)?
        };
        // VoiceEncoder is dropped here

        // Stage 2: T3 Token Generation
        println!("🧠 Stage 2: T3 Token Generation...");
        let speech_tokens = {
            let t3_path = repo_handle.get("t3_turbo_v1.safetensors")?;
            let vb_t3 = unsafe {
                VarBuilder::from_mmaped_safetensors(&[t3_path], DType::F32, &self.device)?
            };
            let t3 = T3Model::load(vb_t3).map_err(anyhow::Error::msg)?;
            t3.generate_tokens(&text_tokens, &speaker_emb, &self.device)
                .map_err(anyhow::Error::msg)?
        };
        println!("✅ Generated {} speech tokens.", speech_tokens.len());
        // T3Model is dropped here

        // Stage 3: S3Gen Waveform
        println!("🔊 Stage 3: S3Gen Waveform Generation...");
        let audio_data = {
            let s3_path = repo_handle.get("s3gen_meanflow.safetensors")?;
            let vb_s3 = unsafe {
                VarBuilder::from_mmaped_safetensors(&[s3_path], DType::F32, &self.device)?
            };

            // Link Flow model to generate latents from T3 tokens
            use s3gen_flow::{Config as FlowConfig, S3GenFlow};
            let flow_cfg = FlowConfig {
                hidden_dim: 512,
                vocab_size: 6561,
            };
            let flow =
                S3GenFlow::load(flow_cfg, vb_s3.pp("flow"), None).map_err(anyhow::Error::msg)?;

            // tokens to tensor
            let speech_tokens_tensor = Tensor::new(speech_tokens, &self.device)?.unsqueeze(0)?;
            let latents = flow
                .mean_flow(&speech_tokens_tensor)
                .map_err(anyhow::Error::msg)?;
            println!("  (Latents Check: {:?})", latents.dims());

            let hift = HiFTGenerator::load(vb_s3).map_err(anyhow::Error::msg)?;
            let spec = hift.forward(&latents).map_err(anyhow::Error::msg)?;

            // ISTFT
            use pycandle_audio::{PadMode, StftConfig, istft};
            let istft_cfg = StftConfig {
                n_fft: 16,
                hop_length: Some(4),
                win_length: Some(16),
                center: true,
                pad_mode: PadMode::Reflect,
                ..Default::default()
            };
            let audio_tensor = istft(&spec, &istft_cfg, None).map_err(anyhow::Error::msg)?;
            audio_tensor.to_vec1::<f32>().map_err(anyhow::Error::msg)?
        };

        Ok(audio_data)
    }
}

fn main() -> Result<()> {
    let device = Device::Cpu;
    let orchestrator = Orchestrator::load(&device)?;

    let text = "I am a powerful agentic AI coding assistant designed by the Google Deepmind team.";
    let audio = orchestrator.generate_sequential(text, None)?;

    println!("💾 Saving output.wav ({} samples)...", audio.len());
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 24000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create("output.wav", spec)?;
    for &sample in audio.iter() {
        let s = (sample.max(-1.0).min(1.0) * 32767.0) as i16;
        writer.write_sample(s)?;
    }
    writer.finalize()?;
    println!("✨ Done! Audio saved to output.wav");

    Ok(())
}
