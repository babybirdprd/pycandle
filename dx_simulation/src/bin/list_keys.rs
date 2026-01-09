use candle_core::safetensors::load;
use candle_core::{Device, Result};
use std::path::PathBuf;

fn main() -> anyhow::Result<()> {
    let api = hf_hub::api::sync::ApiBuilder::new()
        .with_cache_dir(std::path::PathBuf::from("D:/huggingface"))
        .build()?;
    let repo = api.model("ResembleAI/chatterbox-turbo".to_string());
    let path = repo.get("s3gen_meanflow.safetensors")?;

    let tensors = candle_core::safetensors::load(path, &candle_core::Device::Cpu)?;

    let mut keys: Vec<_> = tensors.keys().collect();
    keys.sort();

    println!("Keys in s3gen_meanflow.safetensors:");
    for key in keys {
        let tensor = tensors.get(key).unwrap();
        println!("{}: {:?}", key, tensor.shape());
    }

    Ok(())
}
