//! PyCandle Core Library
//!
//! Core functionality for PyTorch → Candle porting with parity verification.
//!
//! # Features
//! - `PyChecker` for layer-wise verification against golden tensors
//! - `py_check!` macro for embedded verification in generated code
//! - Layer implementations: BatchNorm, LSTM, activations
//! - Code generation from manifests

mod checker;
pub mod codegen;
pub mod gpt2;
mod layers;
pub mod samplers;

pub use checker::{ComparisonResult, LayerMeta, PyChecker};
pub use layers::*;
pub use samplers::*;

/// Verify tensor against golden record, panics on mismatch
#[macro_export]
macro_rules! py_check {
    ($checker:expr, $name:expr, $tensor:expr) => {
        if let Some(ref c) = $checker {
            c.verify($name, $tensor).expect("Parity Check Failed");
        }
    };
}
