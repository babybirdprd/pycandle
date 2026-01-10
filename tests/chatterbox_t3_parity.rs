#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor, DType};
    use pycandle_core::{PyChecker, VerificationMode};
    use anyhow::Context;
    use std::collections::HashMap;

    mod model {
        #![allow(dead_code)]
        #![allow(unused_imports)]
        #![allow(non_snake_case)]
        #![allow(unused_variables)]
        include!("../.pycandle/generated_ChatterboxT3.rs"); // Point to the generated model file
    }
    use model::{Config, ChatterboxT3};

    #[test]
    fn test_parity() -> anyhow::Result<()> {
        // 1. Setup Device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("Running on device: {:?}", device);

        // 2. Load Checker and Golden Trace
        let checker = PyChecker::load("chatterbox_t3", "chatterbox_t3/traces", &device)?
            .with_mode(VerificationMode::Strict);
        println!("Loaded checker with trace: {}", checker.name);

        // 3. Load Model
        let weights_path = "chatterbox_t3/src/chatterbox_t3.safetensors";
        println!("Loading weights from: {}", weights_path);
        let weight_map = candle_core::safetensors::load(weights_path, &device)?;
        let vb = candle_nn::VarBuilder::from_tensors(weight_map, DType::F32, &device);
        
        // 4. Load Inputs from Trace
        let trace_path = "chatterbox_t3/traces/chatterbox_t3_trace.safetensors";
        let tensors = candle_core::safetensors::load(trace_path, &device)?;
        
        let config = Config {
            context_length: 8196,
            hidden_dim: 1024,
            n_head: 16,
            vocab_size: 50276,
            n_layers: 24,
            ..Default::default()
        };
        let mut model = ChatterboxT3::load(config, vb, Some(checker.clone()))?;

        let x0 = tensors.get("model_input.0").context("Missing model_input.0")?.clone();


        // 5. Run Forward Pass & Verify
        let output = model.forward(&x0)?;
        checker.verify("last_hidden_state", &output)?;
        println!("✅ Parity test passed for ChatterboxT3!");

        Ok(())
    }
}
