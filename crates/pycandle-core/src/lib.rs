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
mod layers;

pub use checker::{ComparisonResult, LayerMeta, PyChecker};
pub use layers::*;

/// Verify tensor against golden record, panics on mismatch
#[macro_export]
macro_rules! py_check {
    ($checker:expr, $name:expr, $tensor:expr) => {
        if let Some(ref c) = $checker {
            c.verify($name, $tensor).expect("Parity Check Failed");
        }
    };
}
