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

    let graph_nodes = vec![
        pycandle_core::codegen::GraphNode {
            name: "x_1".to_string(),
            op: "call_module".to_string(),
            target: "flow.encoder.embed".to_string(),
            args: vec![serde_json::to_value("xs").unwrap()],
            kwargs: HashMap::new(),
            module_type: Some("Embedding".to_string()),
        },
        pycandle_core::codegen::GraphNode {
            name: "x_2".to_string(),
            op: "call_module".to_string(),
            target: "flow.decoder.blocks.0.attn1".to_string(),
            args: vec![serde_json::to_value("x_1").unwrap()],
            kwargs: HashMap::new(),
            module_type: Some("Attention".to_string()),
        },
        pycandle_core::codegen::GraphNode {
            name: "x_3".to_string(),
            op: "call_module".to_string(),
            target: "flow.encoder.up_encoders.0".to_string(),
            args: vec![
                serde_json::to_value("x_1").unwrap(),
                serde_json::to_value("x_2").unwrap(),
            ],
            kwargs: HashMap::new(),
            module_type: Some("ResnetBlock".to_string()),
        },
    ];

    let codegen = Codegen::new(manifest, None).with_graph(graph_nodes);
    let code = codegen.generate_model_rs("S3Gen");

    println!("Generated Code Preview (first 500 chars):");
    println!("{}", &code[..500]);

    // Save to temp file for inspection
    std::fs::write("d:/pycandle/s3gen_generated.rs", code).unwrap();
}
