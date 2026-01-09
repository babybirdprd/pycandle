use candle_core::{Device, Tensor};
use pycandle_core::checker::LayerMeta; // Needed for mock manifest
use pycandle_core::{PyChecker, VerificationMode};
use std::collections::HashMap;

#[test]
fn test_quantization_drift() -> anyhow::Result<()> {
    let device = Device::Cpu;

    // 1. Create a dummy manifest
    let manifest_json = r#"{
        "layer_perfect": {
            "name": "layer_perfect", "module_type": "Linear", 
            "input_shapes": [], "output_shapes": [], "parameters": [], "is_leaf": true, "config": {} 
        },
        "layer_drift_small": {
            "name": "layer_drift_small", "module_type": "Linear", 
            "input_shapes": [], "output_shapes": [], "parameters": [], "is_leaf": true, "config": {} 
        },
        "layer_drift_large": {
            "name": "layer_drift_large", "module_type": "Linear", 
            "input_shapes": [], "output_shapes": [], "parameters": [], "is_leaf": true, "config": {} 
        }
    }"#;

    // Write dummy files for PyChecker::load
    std::fs::create_dir_all("test_trace_drift")?;
    std::fs::write("test_trace_drift/drift_run_manifest.json", manifest_json)?;

    // Create dummy golden tensors
    let t_perfect = Tensor::new(&[1.0f32, 2.0, 3.0], &device)?;
    let t_drift_small = Tensor::new(&[10.0f32, 20.0, 30.0], &device)?;
    let t_drift_large = Tensor::new(&[100.0f32, 200.0, 300.0], &device)?;

    let tensors_map: HashMap<String, Tensor> = HashMap::from([
        ("layer_perfect.out.0".to_string(), t_perfect.clone()),
        ("layer_drift_small.out.0".to_string(), t_drift_small.clone()),
        ("layer_drift_large.out.0".to_string(), t_drift_large.clone()),
    ]);
    candle_core::safetensors::save(&tensors_map, "test_trace_drift/drift_run_trace.safetensors")?;

    // 2. Load Checker in DriftTracking mode
    let checker = PyChecker::load("drift_run", "test_trace_drift", &device)?
        .with_mode(VerificationMode::DriftTracking);

    println!("Checking in mode: {:?}", checker.mode);

    // 3. Verify Layers

    // A) Perfect Match
    let res1 = checker.verify("layer_perfect", &t_perfect)?;
    assert!(res1.mse < 1e-6);

    // B) Small Drift (MSE = 1e-6, barely passing strict if atol=1e-4, but let's make it small enough)
    // Let's add noise: 1e-3. 1e-3^2 = 1e-6.
    let noise_small = Tensor::new(&[0.001f32, 0.001, 0.001], &device)?;
    let t_small_noisy = (&t_drift_small + noise_small)?;
    let res2 = checker.verify("layer_drift_small", &t_small_noisy)?;
    println!("Layer Small Drift MSE: {}", res2.mse);

    // C) Large Drift (MSE ~ 0.01) - Should fail in Strict, but pass here
    // Noise 0.1. 0.1^2 = 0.01. > 1e-4.
    let noise_large = Tensor::new(&[0.1f32, 0.1, 0.1], &device)?;
    let t_large_noisy = (&t_drift_large + noise_large)?;

    // This should NOT panic or return Err, unlike Strict mode
    let res3 = checker.verify("layer_drift_large", &t_large_noisy)?;
    println!("Layer Large Drift MSE: {}", res3.mse);

    assert!(
        res3.mse > checker.atol,
        "Expected large drift to exceed atol"
    );

    // 4. Print Report (Manual verify on console output)
    checker.print_drift_report();

    // Cleanup
    std::fs::remove_dir_all("test_trace_drift")?;

    Ok(())
}
