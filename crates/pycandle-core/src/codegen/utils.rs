use super::gpt2;

impl super::Codegen {
    pub fn sanitize_name(&self, name: &str) -> String {
        // Handle ONNX paths like /layers/0/Gather_1_output -> gather_1
        if name.contains('/') {
            let clean = name.trim_start_matches('/');
            let parts: Vec<&str> = clean.split('/').collect();
            let last = parts.last().unwrap_or(&name);
            let mut sanitized = last.to_lowercase().replace("_output", "");

            if sanitized.starts_with("node_") {
                sanitized = sanitized.replace("node_", "x_");
            }

            // Ensure valid identifier
            if sanitized
                .chars()
                .next()
                .map(|c| !c.is_alphabetic())
                .unwrap_or(true)
            {
                return format!("x_{}", sanitized);
            }
            return sanitized;
        }

        // Standard PyTorch names: encoder.layers.0 -> encoder_layers_0
        let mut sanitized = name.replace(".", "_").replace("-", "_");
        if sanitized.starts_with("node_") {
            sanitized = sanitized.replace("node_", "x_");
        }

        // Ensure valid identifier
        if sanitized
            .chars()
            .next()
            .map(|c| !c.is_alphabetic() && c != '_')
            .unwrap_or(true)
        {
            return format!("x_{}", sanitized);
        }

        sanitized
    }

    /// Check if a module type is supported by the codegen
    pub fn is_supported(&self, module_type: &str) -> bool {
        // Check GPT2 helper first
        if gpt2::map_type(module_type).is_some() {
            return true;
        }

        matches!(
            module_type,
            "Linear"
                | "Conv1d"
                | "LayerNorm"
                | "Embedding"
                | "ReLU"
                | "GELU"
                | "Sigmoid"
                | "Tanh"
                | "ELU"
                | "LeakyReLU"
                | "Snake"
                | "BatchNorm1d"
                | "BatchNorm2d"
                | "LSTM"
                | "Mish"
                | "SiLU"
                | "CausalConv1d"
                | "Transpose"
                | "Conv1D"
                | "Dropout"
                | "NewGELUActivation"
                | "LoRACompatibleLinear"
                | "SinusoidalPosEmb"
        )
    }

    /// Get implementation suggestion for an unsupported module type
    pub fn get_suggestion(&self, module_type: &str) -> String {
        match module_type {
            "LSTM" => "Use /add-lstm workflow".to_string(),
            "BatchNorm1d" | "BatchNorm2d" => "Use /add-batchnorm workflow".to_string(),
            "Snake" | "ELU" => "Use /add-activations workflow".to_string(),
            _ => format!("Implement {} manually", module_type),
        }
    }

    pub fn map_type(&self, py_type: &str) -> String {
        // Check GPT2 helper first
        if let Some(t) = gpt2::map_type(py_type) {
            return t;
        }

        // Core types
        match py_type {
            "Linear" => "candle_nn::Linear".to_string(),
            "ParametrizedLinear" => "candle_nn::Linear".to_string(),
            "Conv1d" => "candle_nn::Conv1d".to_string(),
            "ParametrizedConv1d" => "candle_nn::Conv1d".to_string(),
            "LayerNorm" => "candle_nn::LayerNorm".to_string(),
            "Embedding" => "candle_nn::Embedding".to_string(),
            // Activations
            "ReLU" => "ReLU".to_string(),
            "GELU" => "GELU".to_string(),
            "Sigmoid" => "Sigmoid".to_string(),
            "Tanh" => "Tanh".to_string(),
            "ELU" => "ELU".to_string(),
            "LeakyReLU" => "LeakyReLU".to_string(),
            "Snake" => "Snake".to_string(),
            "BatchNorm1d" => "BatchNorm1d".to_string(),
            "BatchNorm2d" => "BatchNorm2d".to_string(),
            "LSTM" => "LSTM".to_string(),
            "Mish" => "Mish".to_string(),
            "SiLU" => "SiLU".to_string(),
            "CausalConv1d" => "CausalConv1d".to_string(),
            "Transpose" => "Transpose".to_string(),
            "Conv1D" => "candle_nn::Linear".to_string(),
            "Dropout" => "Dropout".to_string(),
            "NewGELUActivation" => "candle_nn::Activation".to_string(),
            "LoRACompatibleLinear" => "candle_nn::Linear".to_string(),
            "LlamaRMSNorm" => "LlamaRMSNorm".to_string(),
            "SinusoidalPosEmb" => "SinusoidalPosEmb".to_string(),
            _ => format!("() /* TODO: {} */", py_type),
        }
    }

    pub fn has_gpt2_types(&self) -> bool {
        self.manifest
            .values()
            .any(|meta| gpt2::is_gpt2_type(&meta.module_type))
    }
}
