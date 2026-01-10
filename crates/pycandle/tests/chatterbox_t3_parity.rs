#[cfg(test)]
mod tests {
    use super::*;
    use anyhow::Context;
    use candle_core::{DType, Device, Tensor};
    use pycandle_core::{PyChecker, VerificationMode};
    use std::collections::HashMap;

    mod model {
        #![allow(dead_code)]
        #![allow(unused_imports)]
        #![allow(non_snake_case)]
        #![allow(unused_variables)]
        include!("../../../.pycandle/generated_chatterbox_t3.rs"); // Point to the generated model file
    }
    use model::{ChatterboxT3, Config};

    #[test]
    fn test_parity() -> anyhow::Result<()> {
        // 1. Setup Device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("Running on device: {:?}", device);

        // 2. Load Checker and Golden Trace
        let checker = PyChecker::load("chatterbox_t3", "chatterbox-repo/traces", &device)?
            .with_mode(VerificationMode::Strict);
        println!("Loaded checker with trace: {}", checker.name);

        // 3. Load Model
        // NOTE: Using zero weights for structural verification as weights file is missing
        let vb = candle_nn::VarBuilder::zeros(DType::F32, &device);

        // 4. Load Inputs from Trace
        let trace_path = "chatterbox-repo/traces/chatterbox_t3_trace.safetensors";
        let tensors = candle_core::safetensors::load(trace_path, &device)?;

        let config = Config {
            n_head: 16,
            vocab_size: 50276,
            context_length: 8196,
            n_layers: 24,
            hidden_dim: 1024,
            ..Default::default()
        };
        let mut model = ChatterboxT3::load(config, vb, Some(checker.clone()))?;

        let x0 = tensors
            .get("model_input.0")
            .context("Missing model_input.0")?
            .clone();

        // 5. Run Forward Pass & Verify
        let output = model.forward(&x0)?;
        checker.verify("last_hidden_state", &output)?;
        println!("✅ Parity test passed for ChatterboxT3!");

        Ok(())
    }
}
