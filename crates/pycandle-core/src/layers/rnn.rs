use candle_core::{IndexOp, Result, Tensor};

/// LSTM layer (multi-layer, unidirectional)
/// Input: (B, T, input_size) if batch_first=true
/// Output: (output, (h_n, c_n))
#[derive(Clone, Debug)]
pub struct LSTM {
    pub weight_ih: Vec<Tensor>, // One per layer: (4*hidden, input_size or hidden_size)
    pub weight_hh: Vec<Tensor>, // One per layer: (4*hidden, hidden_size)
    pub bias_ih: Vec<Tensor>,   // One per layer: (4*hidden,)
    pub bias_hh: Vec<Tensor>,   // One per layer: (4*hidden,)
    pub num_layers: usize,
    pub hidden_size: usize,
}

impl LSTM {
    pub fn load(
        vb: candle_nn::VarBuilder,
        input_size: usize,
        hidden_size: usize,
        num_layers: usize,
    ) -> Result<Self> {
        let mut weight_ih = Vec::new();
        let mut weight_hh = Vec::new();
        let mut bias_ih = Vec::new();
        let mut bias_hh = Vec::new();

        for layer in 0..num_layers {
            let in_size = if layer == 0 { input_size } else { hidden_size };
            weight_ih.push(vb.get((4 * hidden_size, in_size), &format!("weight_ih_l{}", layer))?);
            weight_hh.push(vb.get(
                (4 * hidden_size, hidden_size),
                &format!("weight_hh_l{}", layer),
            )?);
            bias_ih.push(vb.get((4 * hidden_size,), &format!("bias_ih_l{}", layer))?);
            bias_hh.push(vb.get((4 * hidden_size,), &format!("bias_hh_l{}", layer))?);
        }

        Ok(Self {
            weight_ih,
            weight_hh,
            bias_ih,
            bias_hh,
            num_layers,
            hidden_size,
        })
    }

    pub fn forward(&self, x: &Tensor) -> Result<(Tensor, (Tensor, Tensor))> {
        // x: (B, T, input_size) assuming batch_first
        let (batch, seq_len, _) = x.dims3()?;
        let device = x.device();
        let dtype = x.dtype();

        let mut h_layers = Vec::new();
        let mut c_layers = Vec::new();
        let mut current_input = x.clone();

        for layer in 0..self.num_layers {
            let mut h_t = Tensor::zeros((batch, self.hidden_size), dtype, device)?;
            let mut c_t = Tensor::zeros((batch, self.hidden_size), dtype, device)?;
            let mut layer_outputs = Vec::new();

            for t in 0..seq_len {
                let x_t = current_input.i((.., t, ..))?;

                // gates = x @ W_ih.T + b_ih + h @ W_hh.T + b_hh
                let gates = x_t
                    .matmul(&self.weight_ih[layer].t()?)?
                    .broadcast_add(&self.bias_ih[layer])?
                    .broadcast_add(&h_t.matmul(&self.weight_hh[layer].t()?)?)?
                    .broadcast_add(&self.bias_hh[layer])?;

                // Split into i, f, g, o (each of size hidden_size)
                let chunks = gates.chunk(4, 1)?;
                let i_gate = candle_nn::ops::sigmoid(&chunks[0])?;
                let f_gate = candle_nn::ops::sigmoid(&chunks[1])?;
                let g_gate = chunks[2].tanh()?;
                let o_gate = candle_nn::ops::sigmoid(&chunks[3])?;

                let next_c_t = f_gate
                    .broadcast_mul(&c_t)?
                    .broadcast_add(&i_gate.broadcast_mul(&g_gate)?)?;
                let next_h_t = o_gate.broadcast_mul(&next_c_t.tanh()?)?;

                c_t = next_c_t;
                h_t = next_h_t;

                layer_outputs.push(h_t.unsqueeze(1)?);
            }

            current_input = Tensor::cat(&layer_outputs, 1)?;
            h_layers.push(h_t.unsqueeze(0)?);
            c_layers.push(c_t.unsqueeze(0)?);
        }

        let h_n = Tensor::cat(&h_layers, 0)?;
        let c_n = Tensor::cat(&c_layers, 0)?;
        Ok((current_input, (h_n, c_n)))
    }
}
