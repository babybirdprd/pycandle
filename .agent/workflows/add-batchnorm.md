---
description: Add BatchNorm1d/2d support to PyCandle codegen
---

# Add BatchNorm Support to PyCandle

BatchNorm is frequently used in CNNs and speaker encoders (CAMPPlus uses ~100 BatchNorm layers).

## Steps

### 1. Update `src/codegen/mod.rs` to handle BatchNorm

Add BatchNorm to the `map_type` function:

```rust
// In src/codegen/mod.rs, add to map_type():
"BatchNorm1d" => "BatchNorm1d".to_string(),
"BatchNorm2d" => "BatchNorm2d".to_string(),
```

### 2. Create BatchNorm struct in lib.rs (Candle doesn't have built-in BatchNorm)

```rust
pub struct BatchNorm1d {
    weight: Tensor,  // gamma
    bias: Tensor,    // beta
    running_mean: Tensor,
    running_var: Tensor,
    eps: f64,
}

impl BatchNorm1d {
    pub fn load(vb: VarBuilder, num_features: usize) -> Result<Self> {
        let weight = vb.get((num_features,), "weight")?;
        let bias = vb.get((num_features,), "bias")?;
        let running_mean = vb.get((num_features,), "running_mean")?;
        let running_var = vb.get((num_features,), "running_var")?;
        Ok(Self { weight, bias, running_mean, running_var, eps: 1e-5 })
    }

    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T) for 1d
        // Normalize: (x - mean) / sqrt(var + eps) * weight + bias
        let mean = self.running_mean.unsqueeze(0)?.unsqueeze(2)?;
        let var = self.running_var.unsqueeze(0)?.unsqueeze(2)?;
        let weight = self.weight.unsqueeze(0)?.unsqueeze(2)?;
        let bias = self.bias.unsqueeze(0)?.unsqueeze(2)?;
        
        let normalized = x.broadcast_sub(&mean)?.broadcast_div(&(var + self.eps)?.sqrt()?)?;
        normalized.broadcast_mul(&weight)?.broadcast_add(&bias)
    }
}
```

### 3. Update `generate_init` in src/codegen/mod.rs

```rust
"BatchNorm1d" => {
    let num_features = meta.config.get("num_features")
        .and_then(|v| v.as_u64())
        .unwrap_or(0) as usize;
    format!("BatchNorm1d::load(vb.pp(\"{}\"), {})?", layer_name, num_features)
}
```

### 4. Extract num_features in spy.py

In `_get_module_config`, add:

```python
elif isinstance(m, nn.BatchNorm1d):
    config['num_features'] = m.num_features
    config['eps'] = m.eps
    config['momentum'] = m.momentum
```

## Testing

// turbo
```bash
cd chatterbox-repo
.\.venv\Scripts\python record_s3gen.py
cargo run -- codegen --manifest py_trace/s3gen_components_manifest.json --out test_bn.rs --model Test
```

Verify BatchNorm layers show proper initialization instead of `todo!()`.
