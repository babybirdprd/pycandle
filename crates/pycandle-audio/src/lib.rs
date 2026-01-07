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

/// Short-time Fourier Transform (STFT)
///
/// NOTE: This is a placeholder. Full implementation requires:
/// - Frame extraction with proper striding
/// - Windowing
/// - Real FFT (requires candle FFT support or external crate)
///
/// For now, audio models should use pre-computed spectrograms or
/// integrate with a Python preprocessing step.
pub fn stft(_input: &Tensor, _config: &StftConfig) -> Result<Tensor> {
    // TODO: Implement STFT when Candle adds FFT support
    // See roadmap in README
    Err(candle_core::Error::Msg(
        "STFT not yet implemented - see README roadmap".to_string(),
    ))
}

/// Inverse Short-time Fourier Transform (iSTFT)
///
/// NOTE: This is a placeholder. Requires STFT implementation first.
pub fn istft(_input: &Tensor, _config: &StftConfig) -> Result<Tensor> {
    // TODO: Implement iSTFT when Candle adds FFT support
    // See roadmap in README
    Err(candle_core::Error::Msg(
        "iSTFT not yet implemented - see README roadmap".to_string(),
    ))
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
}
