// TODO extraction and management for generated code
use regex::Regex;
use serde::Serialize;
use std::collections::HashMap;

#[derive(Serialize, Debug)]
pub struct TodoItem {
    pub line: usize,
    pub module_type: String,
    pub field_name: String,
    pub context: String,
    pub suggestion: String,
}

#[derive(Serialize, Debug)]
pub struct TodoReport {
    pub file: String,
    pub total: usize,
    pub by_type: HashMap<String, usize>,
    pub todos: Vec<TodoItem>,
}

/// Extract TODO markers from generated Rust code
pub fn extract_todos(content: &str) -> Vec<TodoItem> {
    let mut todos = Vec::new();

    // Match: let field_name = todo!("Implement initialization for ModuleType");
    let re = Regex::new(r#"let\s+(\w+)\s*=\s*todo!\("Implement initialization for (\w+)"\)"#)
        .expect("Invalid regex");

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

/// Get implementation hint for a module type
pub fn get_implementation_hint(module_type: &str) -> String {
    match module_type {
        "LSTM" => r#"LSTM::load(vb.pp("field"), input_size, hidden_size, num_layers)?"#.to_string(),
        "BatchNorm1d" => r#"BatchNorm1d::load(vb.pp("field"), num_features)?"#.to_string(),
        "BatchNorm2d" => r#"BatchNorm2d::load(vb.pp("field"), num_features)?"#.to_string(),
        "Snake" => r#"Snake::load(vb.pp("field"), in_features)?"#.to_string(),
        "ReLU" => "ReLU".to_string(),
        "Sigmoid" => "Sigmoid".to_string(),
        "ELU" => "ELU::new(1.0)".to_string(),
        "Dropout" => "// Dropout is a no-op at inference time".to_string(),
        "Conv1D" => r#"// HuggingFace Conv1D: implement as Linear with transpose"#.to_string(),
        "NewGELUActivation" => "GELU // or x.gelu_erf()".to_string(),
        _ => format!("// Manual implementation needed for {}", module_type),
    }
}

/// Generate a report from extracted TODOs
pub fn generate_report(file_path: &str, todos: Vec<TodoItem>) -> TodoReport {
    let mut by_type: HashMap<String, usize> = HashMap::new();
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
