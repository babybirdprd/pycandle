//! PyCandle Audio - Audio operations with PyTorch parity
//!
//! This crate provides STFT, iSTFT, and padding operations that match
//! PyTorch's behavior for audio model porting.

use candle_core::{Result, Tensor};

/// Padding modes for audio operations
#[derive(Debug, Clone, Copy)]
pub enum PadMode {
    /// Reflect padding: [1,2,3] -> [3,2,1,2,3,2,1]
    Reflect,
    /// Replicate edge values
    Replicate,
    /// Pad with constant value
    Constant(f64),
}

/// Mel scale types
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MelScale {
    /// HTK scale: 2595 * log10(1 + f / 700)
    Htk,
    /// Slaney scale: linear below 1kHz, log above
    Slaney,
}

/// Normalization modes for Mel banks
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MelNorm {
    /// No normalization
    None,
    /// Slaney-style area normalization
    Slaney,
}

/// MelSpectrogram configuration matching torchaudio.transforms.MelSpectrogram
#[derive(Debug, Clone)]
pub struct MelSpectrogramConfig {
    pub stft_config: StftConfig,
    pub sample_rate: usize,
    pub n_mels: usize,
    pub f_min: f64,
    pub f_max: Option<f64>,
    pub mel_scale: MelScale,
    pub norm: MelNorm,
}

impl Default for MelSpectrogramConfig {
    fn default() -> Self {
        Self {
            stft_config: StftConfig::default(),
            sample_rate: 16000,
            n_mels: 128,
            f_min: 0.0,
            f_max: None,
            mel_scale: MelScale::Htk,
            norm: MelNorm::None,
        }
    }
}

/// STFT configuration matching PyTorch's torch.stft
#[derive(Debug, Clone)]
pub struct StftConfig {
    pub n_fft: usize,
    pub hop_length: Option<usize>,
    pub win_length: Option<usize>,
    pub center: bool,
    pub pad_mode: PadMode,
    pub normalized: bool,
    pub onesided: bool,
    pub return_complex: bool,
}

impl Default for StftConfig {
    fn default() -> Self {
        Self {
            n_fft: 400,
            hop_length: None,
            win_length: None,
            center: true,
            pad_mode: PadMode::Reflect,
            normalized: false,
            onesided: true,
            return_complex: true,
        }
    }
}

/// Apply 1D padding to a tensor
///
/// # Arguments
/// * `input` - Tensor of shape (B, T) or (B, C, T)
/// * `pad_left` - Amount of padding on the left
/// * `pad_right` - Amount of padding on the right
/// * `mode` - Padding mode
pub fn pad_1d(input: &Tensor, pad_left: usize, pad_right: usize, mode: PadMode) -> Result<Tensor> {
    let ndim = input.dims().len();
    let time_dim = ndim - 1;
    let time_len = input.dim(time_dim)?;

    match mode {
        PadMode::Reflect => {
            // Reflect padding - build indices and use index_select
            if pad_left > time_len - 1 || pad_right > time_len - 1 {
                return Err(candle_core::Error::Msg(
                    "Reflect padding size exceeds input size".to_string(),
                ));
            }

            // Build reflection indices
            let mut indices = Vec::with_capacity(pad_left + time_len + pad_right);

            // Left reflection: [pad_left, pad_left-1, ..., 1]
            for i in (1..=pad_left).rev() {
                indices.push(i as u32);
            }

            // Original: [0, 1, ..., time_len-1]
            for i in 0..time_len {
                indices.push(i as u32);
            }

            // Right reflection: [time_len-2, time_len-3, ..., time_len-1-pad_right]
            for i in 0..pad_right {
                indices.push((time_len - 2 - i) as u32);
            }

            let idx_tensor =
                Tensor::from_vec(indices, (pad_left + time_len + pad_right,), input.device())?;
            input.index_select(&idx_tensor, time_dim)
        }
        PadMode::Replicate => {
            // Edge/replicate padding using constant padding with edge values
            // For simplicity, use constant padding with zero and then copy edge values
            let total_len = pad_left + time_len + pad_right;
            let mut indices = Vec::with_capacity(total_len);

            for _ in 0..pad_left {
                indices.push(0u32);
            }
            for i in 0..time_len {
                indices.push(i as u32);
            }
            for _ in 0..pad_right {
                indices.push((time_len - 1) as u32);
            }

            let idx_tensor = Tensor::from_vec(indices, (total_len,), input.device())?;
            input.index_select(&idx_tensor, time_dim)
        }
        PadMode::Constant(val) => {
            let mut left_shape: Vec<usize> = input.dims().to_vec();
            left_shape[time_dim] = pad_left;
            let mut right_shape: Vec<usize> = input.dims().to_vec();
            right_shape[time_dim] = pad_right;

            let left =
                Tensor::full(val as f32, left_shape, input.device())?.to_dtype(input.dtype())?;
            let right =
                Tensor::full(val as f32, right_shape, input.device())?.to_dtype(input.dtype())?;
            Tensor::cat(&[&left, input, &right], time_dim)
        }
    }
}

/// Create a Hann window
pub fn hann_window(length: usize, device: &candle_core::Device) -> Result<Tensor> {
    let mut values = vec![0f32; length];
    for i in 0..length {
        values[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * i as f32 / length as f32).cos());
    }
    Tensor::from_vec(values, (length,), device)
}

/// Convert Hz to Mel frequency
pub fn hz_to_mel(freq: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 2595.0 * (1.0 + freq / 700.0).log10(),
        MelScale::Slaney => {
            let min_log_hz = 1000.0;
            let min_log_mel = 15.0;
            let log_step = (6.4f64).ln() / 27.0;
            if freq >= min_log_hz {
                min_log_mel + (freq / min_log_hz).ln() / log_step
            } else {
                3.0 * freq / 200.0
            }
        }
    }
}

/// Convert Mel frequency to Hz
pub fn mel_to_hz(mel: f64, scale: MelScale) -> f64 {
    match scale {
        MelScale::Htk => 700.0 * (10f64.powf(mel / 2595.0) - 1.0),
        MelScale::Slaney => {
            let min_log_hz = 1000.0;
            let min_log_mel = 15.0;
            let log_step = (6.4f64).ln() / 27.0;
            if mel >= min_log_mel {
                min_log_hz * (log_step * (mel - min_log_mel)).exp()
            } else {
                200.0 * mel / 3.0
            }
        }
    }
}

/// Create a Mel filterbank matrix of shape (n_mels, n_fft / 2 + 1)
pub fn get_mel_banks(
    n_mels: usize,
    n_fft: usize,
    sample_rate: usize,
    f_min: f64,
    f_max: f64,
    scale: MelScale,
    norm: MelNorm,
) -> Result<Tensor> {
    let n_bins = n_fft / 2 + 1;
    let mel_min = hz_to_mel(f_min, scale);
    let mel_max = hz_to_mel(f_max, scale);

    let mut mel_points = vec![0.0f64; n_mels + 2];
    for i in 0..n_mels + 2 {
        mel_points[i] = mel_min + i as f64 * (mel_max - mel_min) / (n_mels + 1) as f64;
    }

    let mut hz_points = vec![0.0f64; n_mels + 2];
    for i in 0..n_mels + 2 {
        hz_points[i] = mel_to_hz(mel_points[i], scale);
    }

    let mut fft_freqs = vec![0.0f64; n_bins];
    for i in 0..n_bins {
        fft_freqs[i] = i as f64 * sample_rate as f64 / n_fft as f64;
    }

    let mut filterbank = vec![0.0f32; n_mels * n_bins];

    for i in 0..n_mels {
        let left = hz_points[i];
        let center = hz_points[i + 1];
        let right = hz_points[i + 2];

        for j in 0..n_bins {
            let f = fft_freqs[j];
            if f > left && f < right {
                let val = if f <= center {
                    (f - left) / (center - left)
                } else {
                    (right - f) / (right - center)
                };

                // Area normalization (Slaney)
                let val = if norm == MelNorm::Slaney {
                    val * 2.0 / (right - left)
                } else {
                    val
                };

                filterbank[i * n_bins + j] = val as f32;
            }
        }
    }

    Tensor::from_vec(filterbank, (n_mels, n_bins), &candle_core::Device::Cpu)
}

/// MelSpectrogram transformation
pub fn mel_spectrogram(
    input: &Tensor,
    config: &MelSpectrogramConfig,
    window: Option<&Tensor>,
) -> Result<Tensor> {
    let n_fft = config.stft_config.n_fft;
    let f_max = config.f_max.unwrap_or(config.sample_rate as f64 / 2.0);

    // 1. STFT
    let spec = stft(input, &config.stft_config, window)?;

    // spec is (B, n_bins, n_frames, 2) or (n_bins, n_frames, 2)
    // 2. Power Spectrogram (magnitude squared)
    let power = spec
        .narrow(spec.dims().len() - 1, 0, 1)?
        .sqr()?
        .add(&spec.narrow(spec.dims().len() - 1, 1, 1)?.sqr()?)?;
    let power = power.squeeze(spec.dims().len() - 1)?;

    // 3. Mel Filterbank
    let mel_banks = get_mel_banks(
        config.n_mels,
        n_fft,
        config.sample_rate,
        config.f_min,
        f_max,
        config.mel_scale,
        config.norm,
    )?
    .to_device(input.device())?
    .to_dtype(input.dtype())?;

    // Apply filterbank: (n_mels, n_bins) * (..., n_bins, n_frames)
    // We need to reorder power to (..., n_frames, n_bins) to use matmul or just use a custom dot product
    // Or just use matmul if we transpose.
    // power is (B, n_bins, n_frames)
    let power = power.transpose(power.dims().len() - 2, power.dims().len() - 1)?;
    // power is (B, n_frames, n_bins)

    let mel_spec = power.matmul(&mel_banks.transpose(0, 1)?)?;
    // mel_spec is (B, n_frames, n_mels)

    mel_spec.transpose(mel_spec.dims().len() - 2, mel_spec.dims().len() - 1)
}

/// Short-time Fourier Transform (STFT)
pub fn stft(input: &Tensor, config: &StftConfig, window: Option<&Tensor>) -> Result<Tensor> {
    let device = input.device();
    let dtype = input.dtype();

    // 1. Padding
    let n_fft = config.n_fft;
    let hop_length = config.hop_length.unwrap_or(n_fft / 4);
    let win_length = config.win_length.unwrap_or(n_fft);

    let mut x = input.clone();
    if config.center {
        let pad = n_fft / 2;
        x = pad_1d(&x, pad, pad, config.pad_mode)?;
    }

    // 2. Framing & Windowing
    // x is (B, T) or (T,)
    let dims = x.dims();
    let (batch_size, time_len) = if dims.len() == 1 {
        (1, dims[0])
    } else if dims.len() == 2 {
        (dims[0], dims[1])
    } else {
        return Err(candle_core::Error::Msg(format!(
            "STFT input must be (B, T) or (T,), got {:?}",
            dims
        )));
    };

    let n_frames = (time_len - win_length) / hop_length + 1;

    // Move to CPU for FFT if on GPU
    let x_cpu = x.to_device(&candle_core::Device::Cpu)?;
    let x_vec = x_cpu.flatten_all()?.to_vec1::<f32>()?;

    let window_vec = if let Some(w) = window {
        w.to_device(&candle_core::Device::Cpu)?.to_vec1::<f32>()?
    } else {
        vec![1.0; win_length]
    };

    // 3. FFT Setup
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let r2c = planner.plan_fft_forward(n_fft);

    let n_bins = n_fft / 2 + 1;
    let mut output_vec = vec![0.0f32; batch_size * n_frames * n_bins * 2];

    for b in 0..batch_size {
        let batch_offset = b * time_len;
        let out_batch_offset = b * n_bins * n_frames * 2;
        for f in 0..n_frames {
            let start = batch_offset + f * hop_length;
            let mut frame = vec![0.0f32; n_fft];

            // Apply window and copy into frame (handling win_length < n_fft with zero padding)
            for i in 0..win_length {
                frame[i] = x_vec[start + i] * window_vec[i];
            }

            let mut spectrum = r2c.make_output_vec();
            r2c.process(&mut frame, &mut spectrum)
                .map_err(|e| candle_core::Error::Msg(format!("FFT error: {:?}", e)))?;

            for (i, c) in spectrum.iter().enumerate() {
                let out_idx = out_batch_offset + (i * n_frames + f) * 2;
                output_vec[out_idx] = c.re;
                output_vec[out_idx + 1] = c.im;
            }
        }
    }

    // 4. Result Formatting
    let shape = if batch_size == 1 && dims.len() == 1 {
        candle_core::Shape::from((n_bins, n_frames, 2usize))
    } else {
        candle_core::Shape::from((batch_size, n_bins, n_frames, 2usize))
    };

    let result = Tensor::from_vec(output_vec, shape, &candle_core::Device::Cpu)?;

    // Move back to original device and dtype
    result.to_device(device)?.to_dtype(dtype)
}

/// Inverse Short-time Fourier Transform (iSTFT)
pub fn istft(input: &Tensor, config: &StftConfig, window: Option<&Tensor>) -> Result<Tensor> {
    let device = input.device();
    let dtype = input.dtype();

    let n_fft = config.n_fft;
    let hop_length = config.hop_length.unwrap_or(n_fft / 4);
    let win_length = config.win_length.unwrap_or(n_fft);

    // input is (B, n_bins, n_frames, 2) or (n_bins, n_frames, 2)
    let dims = input.dims();
    let (batch_size, n_bins, n_frames) = if dims.len() == 3 {
        (1, dims[0], dims[1])
    } else if dims.len() == 4 {
        (dims[0], dims[1], dims[2])
    } else {
        return Err(candle_core::Error::Msg(format!(
            "iSTFT input must be (B, n_bins, n_frames, 2) or (n_bins, n_frames, 2), got {:?}",
            dims
        )));
    };

    if n_bins != n_fft / 2 + 1 {
        return Err(candle_core::Error::Msg(format!(
            "Expected {} bins for n_fft={}, got {}",
            n_fft / 2 + 1,
            n_fft,
            n_bins
        )));
    }

    // Move to CPU
    let input_cpu = input.to_device(&candle_core::Device::Cpu)?;
    let input_vec = input_cpu.flatten_all()?.to_vec1::<f32>()?;

    let window_vec = if let Some(w) = window {
        w.to_device(&candle_core::Device::Cpu)?.to_vec1::<f32>()?
    } else {
        vec![1.0; win_length]
    };

    // 2. Inverse FFT Setup
    let mut planner = realfft::RealFftPlanner::<f32>::new();
    let c2r = planner.plan_fft_inverse(n_fft);

    let mut output_audio = Vec::with_capacity(batch_size * (n_frames * hop_length + n_fft));

    for b in 0..batch_size {
        let batch_offset = b * n_bins * n_frames * 2;
        let expected_len = n_frames * hop_length + n_fft;
        let mut reconstructed = vec![0.0f32; expected_len];
        let mut window_sum = vec![0.0f32; expected_len];

        for f in 0..n_frames {
            let start = f * hop_length;
            let mut spectrum = Vec::with_capacity(n_bins);
            for i in 0..n_bins {
                let idx = batch_offset + (i * n_frames + f) * 2;
                spectrum.push(num_complex::Complex::new(
                    input_vec[idx],
                    input_vec[idx + 1],
                ));
            }

            let mut frame = c2r.make_output_vec();
            c2r.process(&mut spectrum, &mut frame)
                .map_err(|e| candle_core::Error::Msg(format!("iFFT error: {:?}", e)))?;

            // Normalize iFFT (realfft doesn't normalize by default)
            let norm = 1.0 / n_fft as f32;
            for i in 0..win_length {
                reconstructed[start + i] += frame[i] * norm * window_vec[i];
                window_sum[start + i] += window_vec[i] * window_vec[i];
            }
        }

        // Apply OLA normalization (Over-Lap Add)
        for i in 0..reconstructed.len() {
            if window_sum[i] > 1e-10 {
                reconstructed[i] /= window_sum[i];
            }
        }

        output_audio.extend_from_slice(&reconstructed);
    }

    // 3. Finalize and Crop
    let mut result = Tensor::from_vec(
        output_audio,
        (batch_size, n_frames * hop_length + n_fft),
        &candle_core::Device::Cpu,
    )?;

    if config.center {
        let pad = n_fft / 2;
        let total_len = result.dim(1)?;
        result = result.narrow(1, pad, total_len - 2 * pad)?;
    }

    if batch_size == 1 && dims.len() == 3 {
        result = result.squeeze(0)?;
    }

    result.to_device(device)?.to_dtype(dtype)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hann_window() {
        let device = candle_core::Device::Cpu;
        let window = hann_window(4, &device).unwrap();
        let data: Vec<f32> = window.to_vec1().unwrap();
        // Hann window for n=4: [0, 0.5, 1, 0.5]
        assert!((data[0] - 0.0).abs() < 1e-5);
        assert!((data[1] - 0.5).abs() < 1e-5);
        assert!((data[2] - 1.0).abs() < 1e-5);
        assert!((data[3] - 0.5).abs() < 1e-5);
    }

    #[test]
    fn test_stft_istft_roundtrip() {
        let device = candle_core::Device::Cpu;
        let config = StftConfig {
            n_fft: 16,
            hop_length: Some(4),
            win_length: Some(16),
            center: true,
            pad_mode: PadMode::Reflect,
            ..Default::default()
        };

        // Create a simple signal: sum of sines
        let mut signal = vec![0.0f32; 64];
        for i in 0..64 {
            signal[i] = (2.0 * std::f32::consts::PI * 440.0 * i as f32 / 16000.0).sin();
        }
        let x = Tensor::from_vec(signal, (64,), &device).unwrap();
        let window = hann_window(config.win_length.unwrap(), &device).unwrap();

        let spec = stft(&x, &config, Some(&window)).unwrap();
        let x_hat = istft(&spec, &config, Some(&window)).unwrap();

        let x_vec = x.to_vec1::<f32>().unwrap();
        let x_hat_vec = x_hat.to_vec1::<f32>().unwrap();

        // Roundtrip should be reasonably close
        // Note: OLA with Hann window and hop=4 (n_fft/4) meets COLA
        for i in 4..60 {
            // Avoid edges due to OLA ramp-up/down if not perfectly handled by padding
            assert!(
                (x_vec[i] - x_hat_vec[i]).abs() < 1e-3,
                "At index {}: {} != {}",
                i,
                x_vec[i],
                x_hat_vec[i]
            );
        }
    }

    #[test]
    fn test_mel_scales() {
        // HTK parity
        let hz = 1000.0;
        let mel = hz_to_mel(hz, MelScale::Htk);
        assert!((mel - 1000.0).abs() < 0.1);
        let hz_back = mel_to_hz(mel, MelScale::Htk);
        assert!((hz_back - hz).abs() < 1e-5);

        // Slaney parity
        let hz_s = 2000.0;
        let mel_s = hz_to_mel(hz_s, MelScale::Slaney);
        // Slaney: 15 + log(2000/1000) / log_step
        assert!((mel_s - 25.0).abs() < 0.1);
        let hz_back_s = mel_to_hz(mel_s, MelScale::Slaney);
        assert!((hz_back_s - hz_s).abs() < 1e-5);
    }

    #[test]
    fn test_get_mel_banks() {
        let n_mels = 40;
        let n_fft = 1024;
        let sample_rate = 16000;
        let f_min = 0.0;
        let f_max = 8000.0;

        let banks = get_mel_banks(
            n_mels,
            n_fft,
            sample_rate,
            f_min,
            f_max,
            MelScale::Htk,
            MelNorm::None,
        )
        .unwrap();

        assert_eq!(banks.dims(), &[40, 513]);
    }
}
