---
description: Add LSTM support to PyCandle codegen
---

# Add LSTM Support to PyCandle

LSTM is used in VoiceEncoder and other sequence models.

## Steps

### 1. Create LSTM struct in lib.rs

Candle doesn't have a built-in LSTM, so we need to implement one:

```rust
pub struct LSTM {
    weight_ih: Vec<Tensor>,  // One per layer
    weight_hh: Vec<Tensor>,
    bias_ih: Vec<Tensor>,
    bias_hh: Vec<Tensor>,
    num_layers: usize,
    hidden_size: usize,
    batch_first: bool,
}

impl LSTM {
    pub fn load(vb: VarBuilder, input_size: usize, hidden_size: usize, num_layers: usize) -> Result<Self> {
        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();
        
        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { hidden_size };
            weight_ih.push(vb.get((4 * hidden_size, in_size), &format!("weight_ih_l{}", layer))?);
            weight_hh.push(vb.get((4 * hidden_size, hidden_size), &format!("weight_hh_l{}", layer))?);
            bias_ih.push(vb.get((4 * hidden_size,), &format!("bias_ih_l{}", layer))?);
            bias_hh.push(vb.get((4 * hidden_size,), &format!("bias_hh_l{}", layer))?);
        }
        
        Ok(Self { weight_ih, weight_hh, bias_ih, bias_hh, num_layers, hidden_size, batch_first: true })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, (Tensor, Tensor))> {
        // x: (B, T, input_size) if batch_first
        let (batch, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();
        
        let mut h = Tensor::zeros((self.num_layers, batch, self.hidden_size), dtype, device)?;
        let mut c = Tensor::zeros((self.num_layers, batch, self.hidden_size), dtype, device)?;
        
        let mut output = x.clone();
        
        for layer in 0..self.num_layers {
            let mut h_t = h.i(layer)?;
            let mut c_t = c.i(layer)?;
            let mut outputs = Vec::new();
            
            for t in 0..seq_len {
                let x_t = output.i((.., t, ..))?;
                
                // gates = x @ W_ih.T + h @ W_hh.T + b_ih + b_hh
                let gates = x_t.matmul(&self.weight_ih[layer].t()?)?
                    .broadcast_add(&h_t.matmul(&self.weight_hh[layer].t()?)?)?
                    .broadcast_add(&self.bias_ih[layer])?
                    .broadcast_add(&self.bias_hh[layer])?;
                
                // Split into i, f, g, o
                let chunks = gates.chunk(4, 1)?;
                let i_gate = candle_nn::ops::sigmoid(&chunks[0])?;
                let f_gate = candle_nn::ops::sigmoid(&chunks[1])?;
                let g_gate = chunks[2].tanh()?;
                let o_gate = candle_nn::ops::sigmoid(&chunks[3])?;
                
                c_t = (f_gate * &c_t)? + (i_gate * g_gate)?;
                h_t = o_gate * c_t.tanh()?;
                
                outputs.push(h_t.unsqueeze(1)?);
            }
            
            output = Tensor::cat(&outputs, 1)?;
        }
        
        Ok((output, (h, c)))
    }
}
```

### 2. Update codegen.rs map_type

```rust
"LSTM" => "LSTM".to_string(),
```

### 3. Update generate_init

```rust
"LSTM" => {
    let input_size = meta.config.get("input_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let hidden_size = meta.config.get("hidden_size").and_then(|v| v.as_u64()).unwrap_or(0) as usize;
    let num_layers = meta.config.get("num_layers").and_then(|v| v.as_u64()).unwrap_or(1) as usize;
    format!("LSTM::load(vb.pp(\"{}\"), {}, {}, {})?", layer_name, input_size, hidden_size, num_layers)
}
```

### 4. Extract config in spy.py

```python
elif isinstance(m, nn.LSTM):
    config['input_size'] = m.input_size
    config['hidden_size'] = m.hidden_size
    config['num_layers'] = m.num_layers
    config['batch_first'] = m.batch_first
    config['bidirectional'] = m.bidirectional
```

## Testing

```bash
cd chatterbox-repo
.\.venv\Scripts\python record_voice_encoder.py
cargo run -- codegen --manifest py_trace/voice_encoder_manifest.json --out test_lstm.rs --model Test
```

Verify LSTM shows proper initialization.
