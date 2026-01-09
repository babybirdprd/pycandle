use regex::Regex;

pub fn punc_norm(text: &str) -> String {
    if text.is_empty() {
        return "You need to add some text for me to talk.".to_string();
    }

    let mut text = text.trim().to_string();

    // Capitalise first letter
    if let Some(first_char) = text.chars().next() {
        if first_char.is_lowercase() {
            let mut chars = text.chars();
            text = format!("{}{}", chars.next().unwrap().to_uppercase(), chars.as_str());
        }
    }

    // Remove multiple space chars
    let re_spaces = Regex::new(r"\s+").unwrap();
    text = re_spaces.replace_all(&text, " ").to_string();

    // Replace uncommon/llm punc
    let punc_to_replace = [
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ];
    for (old, new) in punc_to_replace {
        text = text.replace(old, new);
    }

    // Add full stop if no ending punc
    let sentence_enders = ['.', '!', '?', '-', ','];
    if !text.is_empty() && !sentence_enders.contains(&text.chars().last().unwrap()) {
        text.push('.');
    }

    text
}
