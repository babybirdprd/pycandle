---
description: Improve codegen for custom activation functions
---

# Improve Custom Activation Support

Currently, custom activations like Snake, ELU, and Swish generate `todo!()` markers.

## Steps

### 1. Add activation mappings in src/codegen/mod.rs

Update `map_type` to handle common activations:

```rust
fn map_type(&self, py_type: &str) -> String {
    match py_type {
        // Existing...
        "ReLU" => "ReLU".to_string(),
        "GELU" => "GELU".to_string(),
        "Sigmoid" => "Sigmoid".to_string(),
        "Tanh" => "Tanh".to_string(),
        "ELU" => "ELU".to_string(),
        "LeakyReLU" => "LeakyReLU".to_string(),
        "Snake" => "Snake".to_string(),  // Custom
        _ => format!("() /* TODO: {} */", py_type),
    }
}
```

### 2. Create activation structs in lib.rs

```rust
// Simple activations (stateless)
pub struct ReLU;
impl ReLU {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        x.relu()
    }
}

pub struct Sigmoid;
impl Sigmoid {
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        candle_nn::ops::sigmoid(x)
    }
}

// ELU with alpha parameter
pub struct ELU {
    alpha: f64,
}
impl ELU {
    pub fn new(alpha: f64) -> Self { Self { alpha } }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // ELU(x) = x if x > 0, else alpha * (exp(x) - 1)
        let mask = x.ge(&Tensor::zeros_like(x)?)?;
        let positive = x.clone();
        let negative = ((x.exp()? - 1.0)? * self.alpha)?;
        mask.where_cond(&positive, &negative)
    }
}

// Snake activation: x + sin²(αx)/α
pub struct Snake {
    alpha: Tensor,
}
impl Snake {
    pub fn load(vb: VarBuilder, in_features: usize) -> Result<Self> {
        let alpha = vb.get((in_features,), "alpha")?;
        Ok(Self { alpha })
    }
    pub fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, C, T), alpha: (C,)
        let alpha = self.alpha.unsqueeze(0)?.unsqueeze(2)?;
        let sin_term = (x.broadcast_mul(&alpha)?)?.sin()?;
        let sin_sq = sin_term.sqr()?;
        x.broadcast_add(&sin_sq.broadcast_div(&alpha)?)
    }
}
```

### 3. Update generate_init for activations

```rust
fn generate_init(&self, layer_name: &str, meta: &LayerMeta) -> String {
    match meta.module_type.as_str() {
        "ReLU" => "ReLU".to_string(),
        "Sigmoid" => "Sigmoid".to_string(),
        "GELU" => "GELU".to_string(),
        "ELU" => {
            let alpha = meta.config.get("alpha").and_then(|v| v.as_f64()).unwrap_or(1.0);
            format!("ELU::new({})", alpha)
        }
        "Snake" => {
            let in_features = meta.config.get("in_features").and_then(|v| v.as_u64()).unwrap_or(0);
            format!("Snake::load(vb.pp(\"{}\"), {})?", layer_name, in_features)
        }
        // ... other cases
    }
}
```

### 4. Update spy.py to extract activation configs

```python
def _get_module_config(self, m: nn.Module) -> Dict[str, Any]:
    config = {}
    # ... existing code ...
    
    elif isinstance(m, nn.ELU):
        config['alpha'] = m.alpha
    elif isinstance(m, nn.LeakyReLU):
        config['negative_slope'] = m.negative_slope
    # Snake is custom, check for alpha attribute
    elif hasattr(m, 'alpha') and hasattr(m, 'in_features'):
        config['in_features'] = m.in_features
        # alpha is a parameter, will be saved in weights
    
    return config
```

## Testing

```bash
cd chatterbox-repo
.\.venv\Scripts\python record_vocoder.py
cargo run -- codegen --manifest py_trace/vocoder_components_manifest.json --out test_act.rs --model Test
```

Verify Snake and ELU show proper initialization instead of `todo!()`.
