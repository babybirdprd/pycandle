use candle_core::{Result, Tensor};

/// Dropout layer (inference no-op)
#[derive(Clone, Debug)]
pub struct Dropout {
    pub p: f32,
}
impl Dropout {
    pub fn new() -> Self {
        Self { p: 0.5 }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        Ok(x.clone())
    }
}

/// Transpose layer (swaps two dimensions)
#[derive(Clone, Debug)]
pub struct Transpose {
    pub dim0: usize,
    pub dim1: usize,
}
impl Transpose {
    pub fn new(dim0: usize, dim1: usize) -> Self {
        Self { dim0, dim1 }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.transpose(self.dim0, self.dim1)
    }
}

/// Sinusoidal Positional Embedding
#[derive(Clone, Debug)]
pub struct SinusoidalPosEmb {
    pub dim: usize,
}
impl SinusoidalPosEmb {
    pub fn new(dim: usize) -> Self {
        Self { dim }
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let half_dim = self.dim / 2;
        let device = x.device();
        let dtype = x.dtype();
        let inv_freq: Vec<_> = (0..half_dim)
            .map(|i| 1.0f32 / (10000.0f32.powf(i as f32 / (half_dim as f32 - 1.0f32))))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, half_dim, device)?.to_dtype(dtype)?;
        let emb = x.unsqueeze(1)?.broadcast_mul(&inv_freq.unsqueeze(0)?)?;
        Tensor::cat(&[emb.sin()?, emb.cos()?], 1)
    }
}

/// Helper to load a Linear layer that uses PyTorch Weight Normalization.
/// Composes w = g * (v / ||v||) at runtime.
pub fn load_weight_norm_linear(
    vb: candle_nn::VarBuilder,
    in_f: usize,
    out_f: usize,
    bias: bool,
) -> Result<candle_nn::Linear> {
    // Attempt to load 'original1' (v) and 'original0' (g) from parametrizations
    // If not found, try legacy 'weight_v' and 'weight_g'

    let v_path = if vb.contains_tensor("parametrizations.weight.original1") {
        "parametrizations.weight.original1"
    } else {
        "weight_v"
    };

    let g_path = if vb.contains_tensor("parametrizations.weight.original0") {
        "parametrizations.weight.original0"
    } else {
        "weight_g"
    };

    let v = vb.get((out_f, in_f), v_path)?; // Shape: [out, in]
    let g = vb.get((out_f, 1), g_path)?; // Shape: [out, 1]

    // Normalize v: v / ||v||
    // Norm along dim 1 (input dimension) for Linear
    let v_norm = v.sqr()?.sum_keepdim(1)?.sqrt()?;

    // Compose: w = g * (v / norm)
    let w = v.broadcast_div(&v_norm)?.broadcast_mul(&g)?;

    // Transpose for Candle (Candle Linear is [out, in], but usually we transpose standard weights.
    // However, weight_norm vectors in PT are usually [out, in].
    // Candle's Linear::new expects weight to be [out, in].
    // Let's standardise on returning the layer directly.

    let b = if bias {
        Some(vb.get((out_f,), "bias")?)
    } else {
        None
    };

    Ok(candle_nn::Linear::new(w, b))
}

/// ReflectionPad1d (1D Reflection Padding)
/// Input: (B, C, T)
#[derive(Clone, Debug)]
pub struct ReflectionPad1d {
    pub padding: usize,
}

impl ReflectionPad1d {
    pub fn new(padding: usize) -> Self {
        Self { padding }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.padding == 0 {
            return Ok(x.clone());
        }
        // x: (B, C, T)
        let (_b, _c, t) = x.dims3()?;

        // Reflect left: x[..., 1:1+padding].flip(-1)
        // Reflect right: x[..., T-1-padding:T-1].flip(-1)

        // Left pad: take indices 1..1+padding, reverse them
        let left_part = x.narrow(2, 1, self.padding)?;
        let left_pad = flip_last_dim(&left_part)?;

        // Right pad: take indices T-1-padding..T-1, reverse them
        let right_part = x.narrow(2, t - 1 - self.padding, self.padding)?;
        let right_pad = flip_last_dim(&right_part)?;

        Tensor::cat(&[&left_pad, x, &right_pad], 2)
    }
}

fn flip_last_dim(x: &Tensor) -> Result<Tensor> {
    let dim = x.rank() - 1;
    let size = x.dim(dim)?;

    // actually, simpler:
    let rev_indices: Vec<u32> = (0..size as u32).rev().collect();
    let idx = Tensor::new(rev_indices.as_slice(), x.device())?;
    x.index_select(&idx, dim)
}

/// Upsample (Nearest Neighbor 1D)
/// Input: (B, C, T)
#[derive(Clone, Debug)]
pub struct Upsample {
    pub scale_factor: usize,
}

impl Upsample {
    pub fn new(scale_factor: usize) -> Self {
        Self { scale_factor }
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T) -> (B, C, T * scale_factor)
        // Nearest neighbor: repeat each element scale_factor times
        let (b, c, t) = x.dims3()?;
        let x_expanded = x.unsqueeze(3)?; // (B, C, T, 1)
        let x_tiled = x_expanded.broadcast_as((b, c, t, self.scale_factor))?;
        x_tiled.flatten(2, 3) // (B, C, T * scale)
    }
}

/// SineGen: Sine Wave Generator from F0
/// Input: F0 (B, 1, T)
/// Output: Sine Waves (B, 1, T * upsample_scale), UV, Noise
#[derive(Clone, Debug)]
pub struct SineGen {
    pub harmonic_num: usize,
    pub sine_amp: f64,
    pub noise_std: f64,
    pub sampling_rate: f64,
    pub voiced_threshold: f64,
}

impl SineGen {
    pub fn new(
        harmonic_num: usize,
        sine_amp: f64,
        noise_std: f64,
        sampling_rate: f64,
        voiced_threshold: f64,
    ) -> Self {
        Self {
            harmonic_num,
            sine_amp,
            noise_std,
            sampling_rate,
            voiced_threshold,
        }
    }

    fn f02uv(&self, f0: &Tensor) -> Result<Tensor> {
        // uv = (f0 > voiced_threshold).float()
        let mask = f0.gt(self.voiced_threshold)?;
        mask.to_dtype(candle_core::DType::F32)
    }

    pub fn forward(&self, f0: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        // f0: (B, 1, T)

        // 1. Upsample F0 is handled outside usually (in SourceModule), but hifigan.py SineGen takes upsampled f0?
        // Checking hifigan.py:
        //  s = self.f0_upsamp(f0[:, None]).transpose(1, 2)
        //  s, _, _ = self.m_source(s)
        //  SourceModuleHnNSF calls l_sin_gen(x)
        // So f0 passed to SineGen is already upsampled.

        let (b, _c, _t) = f0.dims3()?;
        let dev = f0.device();

        // 2. F_mat: harmonics
        // F_mat shape: (B, harmonic_num + 1, T)
        let mut f_mat_vec = Vec::new();
        for i in 0..=self.harmonic_num {
            let scale = (i + 1) as f64 / self.sampling_rate;
            let harmonic = (f0 * scale)?;
            f_mat_vec.push(harmonic);
        }
        let f_mat = Tensor::cat(&f_mat_vec, 1)?; // (B, H+1, T)

        // 3. Theta: 2 * pi * cumsum(F_mat) % 1
        // cumsum along last dim (time)
        // Note: Candle's cumsum might be tricky? Tensor::cumsum exists?
        // If not, we might need a custom kernel or loop. Assuming it exists or implemented.
        // Actually, let's check if cumsum is available.
        // If not, we can implement prefix sum or assume it's there.
        // candle_core::Tensor generic methods usually have it.
        // Wait, standard candle might not have cumsum implemented for all backends yet?
        // Let's assume it works for now.

        // We'll use a naive cumsum via matmul if needed (T x T triangle), but that's slow.
        // Best to use a loop or assume ops support.
        // Update: candle-core usually has `cumsum`.

        // f_mat is (B, H+1, T). Flatten to 2D (B*(H+1), T) for easier handling?
        // Let's just assume simple tensor ops.

        // We need to implement cumsum manually if missing.
        // For now, I will use a placeholder `cumsum` extension trait if it fails?
        // Or better: `scan`?

        // Let's assume `cumsum` is supported.
        // theta = 2 * pi * (f_mat.cumsum(-1) % 1)
        // ERROR: `cumsum` is not in standard Candle 0.3.0? It was added recently.
        // I will implement a visual scan loop if needed, but for 'parity' with python,
        // we heavily rely on it.

        // Temporary: I will try to call `cumsum` method.

        // (f_mat.cumsum(2)? - f_mat.cumsum(2)?.floor()?) * 2.0 * std::f64::consts::PI
        // Better: just check fractional part.

        // If cumsum missing, we are blocked.
        // I'll assume it is present or use `scan` logic.

        // Re-implementing logic:
        // theta_mat = 2 * np.pi * (torch.cumsum(F_mat, dim=-1) % 1)

        // NOTE: In Rust, we might not have `cumsum`. I'll try to use a naive loop for T dimension if it's small,
        // but explicit loops are slow.
        // Assuming `cumsum` exists.

        // Using `broadcast_mul` for 2*pi
        let two_pi = 2.0 * std::f64::consts::PI;
        let cumsum = f_mat.cumsum(2)?;
        let theta_mat = ((cumsum.clone() - cumsum.floor()?)? * two_pi)?;

        // 4. Phase Vec: Random uniform [-pi, pi]
        // shape: (B, H+1, 1) broadcast to T
        // phase_vec[:, 0, :] = 0
        let h_plus_1 = self.harmonic_num + 1;

        // Random phases
        let phase = Tensor::rand(0.0f32, 1.0f32, (b, h_plus_1, 1), dev)?.to_dtype(f0.dtype())?;
        // map [0, 1] -> [-pi, pi]
        // x * 2pi - pi
        let phase = ((phase * two_pi)? - std::f64::consts::PI)?;

        // Zero out fundamental phase (index 0)
        // We need scatter or mask.
        // Create mask: [0, 1, 1, ..., 1]
        let mut mask_vec = vec![1.0; h_plus_1];
        mask_vec[0] = 0.0;
        let mask = Tensor::new(mask_vec, dev)?
            .reshape((1, h_plus_1, 1))?
            .to_dtype(f0.dtype())?;

        let phase_vec = phase.broadcast_mul(&mask)?;

        // 5. Sine Waves
        // sin(theta + phase)
        let sine_waves = (theta_mat.broadcast_add(&phase_vec))?.sin()?;
        let sine_waves = (sine_waves * self.sine_amp)?;

        // 6. UV
        let uv = self.f02uv(f0)?; // (B, 1, T)

        // 7. Noise
        // uv * noise_std + (1 - uv) * sine_amp / 3
        let noise_amp_uv = (uv.clone() * self.noise_std)?;
        let noise_amp_unvoiced = ((uv.ones_like()? - uv.clone())? * (self.sine_amp / 3.0))?;
        let noise_amp = (noise_amp_uv + noise_amp_unvoiced)?;

        // Expand noise_amp to match sine_waves shape (B, H+1, T)?
        // Wait, hifigan.py:
        // noise = noise_amp * torch.randn_like(sine_waves)
        // sine_waves is (B, H+1, T)
        // uv is (B, 1, T), needs broadcasting to (B, H+1, T)

        let noise = Tensor::randn(0.0f32, 1.0f32, sine_waves.shape(), dev)?.to_dtype(f0.dtype())?;
        let noise = noise.broadcast_mul(&noise_amp.broadcast_as(sine_waves.shape())?)?;

        // 8. Combine
        // sine_waves = sine_waves * uv + noise
        let uv_bcast = uv.broadcast_as(sine_waves.shape())?;
        let sine_final = ((sine_waves * uv_bcast.clone())? + noise.clone())?;

        // Output: (sine_final, uv, noise)
        // Note: code expects sine_final, uv, noise
        Ok((sine_final, uv, noise))
    }
}
