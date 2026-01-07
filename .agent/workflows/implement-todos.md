---
description: Extract TODOs from generated code and implement them incrementally
---

# TODO Extraction and Implementation Workflow

Parse generated Rust code to extract all `todo!()` markers, implement them one-by-one, and merge back.

## CLI Interface

```bash
# Extract TODOs to JSON
pycandle todos --file generated_model.rs --json

# Output:
# {"todos":[{"line":13,"type":"LSTM","field":"lstm","context":"let lstm = todo!(...)"},...]"}

# After implementing, validate
pycandle todos --file generated_model.rs --check
```

## Steps

### 1. Add Todos command

```rust
#[derive(Subcommand)]
enum Commands {
    // ... existing ...
    
    /// Extract and manage TODO markers in generated code
    Todos {
        /// Path to generated Rust file
        #[arg(short, long)]
        file: PathBuf,
        
        /// Just check if TODOs remain (exit code 1 if any)
        #[arg(long)]
        check: bool,
        
        /// Output as JSON
        #[arg(long)]
        json: bool,
    },
}
```

### 2. Create todos.rs module

```rust
// src/todos.rs
use regex::Regex;
use serde::Serialize;

#[derive(Serialize, Debug)]
pub struct TodoItem {
    pub line: usize,
    pub module_type: String,
    pub field_name: String,
    pub context: String,
    pub suggestion: String,
}

pub fn extract_todos(content: &str) -> Vec<TodoItem> {
    let mut todos = Vec::new();
    
    // Match: let field_name = todo!("Implement initialization for ModuleType");
    let re = Regex::new(r#"let\s+(\w+)\s*=\s*todo!\("Implement initialization for (\w+)"\)"#).unwrap();
    
    for (line_num, line) in content.lines().enumerate() {
        if let Some(caps) = re.captures(line) {
            let field_name = caps.get(1).unwrap().as_str().to_string();
            let module_type = caps.get(2).unwrap().as_str().to_string();
            
            todos.push(TodoItem {
                line: line_num + 1,
                field_name,
                module_type: module_type.clone(),
                context: line.trim().to_string(),
                suggestion: get_implementation_hint(&module_type),
            });
        }
    }
    
    todos
}

fn get_implementation_hint(module_type: &str) -> String {
    match module_type {
        "LSTM" => r#"LSTM::load(vb.pp("field"), input_size, hidden_size, num_layers)?"#.to_string(),
        "BatchNorm1d" => r#"BatchNorm1d::load(vb.pp("field"), num_features)?"#.to_string(),
        "BatchNorm2d" => r#"BatchNorm2d::load(vb.pp("field"), num_features)?"#.to_string(),
        "Snake" => r#"Snake::load(vb.pp("field"), in_features)?"#.to_string(),
        "ReLU" => "ReLU".to_string(),
        "Sigmoid" => "Sigmoid".to_string(),
        "ELU" => "ELU::new(1.0)".to_string(),
        "_WeightNorm" => "// weight_norm is handled by loading underlying conv weights".to_string(),
        "Conv2d" => r#"candle_nn::conv2d(in_c, out_c, kernel, cfg, vb.pp("field"))?"#.to_string(),
        _ => format!("// Manual implementation needed for {}", module_type),
    }
}

#[derive(Serialize)]
pub struct TodoReport {
    pub file: String,
    pub total: usize,
    pub by_type: std::collections::HashMap<String, usize>,
    pub todos: Vec<TodoItem>,
}

pub fn generate_report(file_path: &str, todos: Vec<TodoItem>) -> TodoReport {
    let mut by_type = std::collections::HashMap::new();
    for todo in &todos {
        *by_type.entry(todo.module_type.clone()).or_default() += 1;
    }
    
    TodoReport {
        file: file_path.to_string(),
        total: todos.len(),
        by_type,
        todos,
    }
}
```

### 3. Handle Todos command

```rust
Commands::Todos { file, check, json } => {
    let content = std::fs::read_to_string(&file)?;
    let todos = todos::extract_todos(&content);
    let report = todos::generate_report(file.to_str().unwrap(), todos);
    
    if json {
        println!("{}", serde_json::to_string_pretty(&report)?);
    } else {
        println!("📋 TODOs in {:?}:", file);
        println!("   Total: {}", report.total);
        for (t, c) in &report.by_type {
            println!("   - {}: {}", t, c);
        }
        println!("\nDetails:");
        for todo in &report.todos {
            println!("  L{}: {} ({}) -> {}", 
                todo.line, todo.field_name, todo.module_type, todo.suggestion);
        }
    }
    
    if check && report.total > 0 {
        std::process::exit(1);
    }
}
```

## Agent Workflow

```bash
# 1. Generate code
pycandle codegen --manifest m.json --out model.rs --model MyModel

# 2. Extract TODOs as JSON
TODOS=$(pycandle todos --file model.rs --json)

# 3. Agent processes each TODO type
for type in $(echo $TODOS | jq -r '.by_type | keys[]'); do
    echo "Implementing $type..."
    # Agent implements the type in lib.rs
    # Then updates src/codegen/mod.rs to use it
done

# 4. Re-generate and verify no TODOs remain
pycandle codegen --manifest m.json --out model.rs --model MyModel
pycandle todos --file model.rs --check && echo "All TODOs resolved!"
```

## Testing

```bash
cargo run -- todos --file generated_s3gen_components.rs --json | jq '.by_type'
# Expected: {"BatchNorm1d": 50, "ReLU": 100, ...}
```
