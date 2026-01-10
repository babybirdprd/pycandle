#[test]
fn verify_s3gen_codegen() {
    use pycandle_core::LayerMeta;
    use pycandle_core::codegen::Codegen;
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::Read;
    use std::path::PathBuf;

    let manifest_path =
        PathBuf::from("d:/pycandle/chatterbox-turbo-port/.pycandle/traces/s3gen_manifest.json");
    if !manifest_path.exists() {
        eprintln!("Manifest not found at {:?}", manifest_path);
        return;
    }

    let mut file = File::open(manifest_path).unwrap();
    let mut content = String::new();
    file.read_to_string(&mut content).unwrap();

    let manifest: HashMap<String, LayerMeta> = serde_json::from_str(&content).unwrap();

    let codegen = Codegen::new(manifest, None);
    let code = codegen.generate_model_rs("S3Gen");

    println!("Generated Code Preview (first 500 chars):");
    println!("{}", &code[..500]);

    // Save to temp file for inspection
    std::fs::write("d:/pycandle/s3gen_generated.rs", code).unwrap();
}
