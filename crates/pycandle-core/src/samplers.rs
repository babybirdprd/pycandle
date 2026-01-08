use candle_core::{Result, Tensor};

pub trait LogitsProcessor {
    fn apply(&self, logits: &Tensor) -> Result<Tensor>;
}

/// Repetition Penalty
/// Reference: https://arxiv.org/pdf/1909.05858.pdf
pub struct RepetitionPenalty {
    pub penalty: f64,
    pub context: Vec<u32>,
}

impl RepetitionPenalty {
    pub fn new(penalty: f64, context: Vec<u32>) -> Self {
        Self { penalty, context }
    }
}

impl LogitsProcessor for RepetitionPenalty {
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        if self.penalty <= 1.0 || self.context.is_empty() {
            return Ok(logits.clone());
        }

        let mut logits_vec = logits.squeeze(0)?.to_vec1::<f32>()?;

        for &token in &self.context {
            if (token as usize) < logits_vec.len() {
                let logit = logits_vec[token as usize];
                if logit < 0.0 {
                    logits_vec[token as usize] = logit * self.penalty as f32;
                } else {
                    logits_vec[token as usize] = logit / self.penalty as f32;
                }
            }
        }

        Tensor::from_vec(logits_vec, (1, logits.dim(1)?), logits.device())
    }
}

/// Temperature scaling
pub struct Temperature {
    pub temperature: f64,
}

impl LogitsProcessor for Temperature {
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        if self.temperature <= 0.0 {
            // Greedy sampling handled by caller usually, but here we can just return
            return Ok(logits.clone());
        }
        logits / self.temperature
    }
}

/// Top-P (Nucleus) Sampling
pub struct TopP {
    pub p: f64,
}

impl LogitsProcessor for TopP {
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        if self.p >= 1.0 {
            return Ok(logits.clone());
        }

        let probs = candle_nn::ops::softmax(logits, 1)?;
        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;

        // Sort descending, keep indices
        let mut indices: Vec<usize> = (0..probs_vec.len()).collect();
        indices.sort_by(|&a, &b| probs_vec[b].partial_cmp(&probs_vec[a]).unwrap());

        let mut cum_sum = 0.0;
        let mut cutoff_index = indices.len() - 1;

        for (i, &idx) in indices.iter().enumerate() {
            cum_sum += probs_vec[idx];
            if cum_sum > self.p as f32 {
                cutoff_index = i;
                break;
            }
        }

        // Everything after cutoff gets -inf
        let mut new_logits = logits.squeeze(0)?.to_vec1::<f32>()?;
        let neg_inf = f32::NEG_INFINITY;

        // Mask out the tail
        for &idx in &indices[cutoff_index + 1..] {
            new_logits[idx] = neg_inf;
        }

        Tensor::from_vec(new_logits, (1, logits.dim(1)?), logits.device())
    }
}

/// Min-P Sampling (Alternative to Top-P)
/// Reference: https://github.com/huggingface/transformers/issues/27670
pub struct MinP {
    pub p: f64,
}

impl LogitsProcessor for MinP {
    fn apply(&self, logits: &Tensor) -> Result<Tensor> {
        if self.p <= 0.0 {
            return Ok(logits.clone());
        }

        let probs = candle_nn::ops::softmax(logits, 1)?;
        let max_prob = probs.max_keepdim(1)?.squeeze(0)?.to_vec1::<f32>()?[0];
        let scaled_min = max_prob * self.p as f32;

        let probs_vec = probs.squeeze(0)?.to_vec1::<f32>()?;
        let mut new_logits = logits.squeeze(0)?.to_vec1::<f32>()?;
        let neg_inf = f32::NEG_INFINITY;

        for (i, &prob) in probs_vec.iter().enumerate() {
            if prob < scaled_min {
                new_logits[i] = neg_inf;
            }
        }

        Tensor::from_vec(new_logits, (1, logits.dim(1)?), logits.device())
    }
}
