#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{Device, Tensor};
    use pycandle_core::{PyChecker, VerificationMode};
    use my_project::SimpleOnnxModel;
    use anyhow::Result;

    #[test]
    fn test_parity() -> Result<()> {
        // 1. Setup Device
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
        println!("Running on device: {:?}", device);

        // 2. Load Checker and Golden Trace
        // Assumes the trace directory is in the current project root
        let checker = PyChecker::load("simple_onnx", "pycandle_trace", &device)?
            .with_mode(VerificationMode::Strict);
        println!("Loaded checker with trace: {}", checker.name);

        // 3. Load Model
        // We use zeros VB as a placeholder; in a real parity test, 
        // you might want to load weights using pycandle weight tools.
        let vb = candle_nn::VarBuilder::zeros(candle_core::DType::F32, &device);
        let model = SimpleOnnxModel::load(vb, Some(checker.clone()))?;

        // 4. Load Inputs from Trace
        let trace_path = format!("pycandle_trace/simple_onnx_trace.safetensors", "simple_onnx");
        let tensors = candle_core::safetensors::load(&trace_path, &device)?;
        
        let x0 = tensors.get("model_input.0").context("Missing model_input.0")?.clone();


        // 5. Run Forward Pass & Verify
        let output = model.forward(&x0)?;
        checker.verify("node_relu", &output)?;
        println!("✅ Parity test passed for SimpleOnnxModel!");

        Ok(())
    }
}
