use super::types::ReturnType;

/// Compiles a torch.einsum equation into a sequence of Candle operations.
///
/// # Arguments
/// * `equation` - The einsum equation string (e.g., "bhld,hdm->bhlm")
/// * `args` - The variable names of the input tensors
///
/// # Returns
/// * A String containing the Rust code to execute the operation.
/// * A ReturnType indicating the result type (usually Tensor).
pub fn compile_einsum(equation: &str, args: &[String]) -> (String, ReturnType) {
    // 1. Parse equation
    let parts: Vec<&str> = equation.split("->").collect();
    if parts.len() != 2 {
        return (
            format!("todo!(/* Invalid einsum equation: {} */)", equation),
            ReturnType::Tensor,
        );
    }

    let inputs_str = parts[0];
    let output_str = parts[1].trim();

    let input_patterns: Vec<&str> = inputs_str.split(',').map(|s| s.trim()).collect();

    if input_patterns.len() != args.len() {
        return (
            format!(
                "todo!(/* Einsum arg mismatch: expected {}, got {} */)",
                input_patterns.len(),
                args.len()
            ),
            ReturnType::Tensor,
        );
    }

    // CASE 1: 2 Arguments (Matmul / Batched Matmul)
    if args.len() == 2 {
        return compile_binary_einsum(
            input_patterns[0],
            input_patterns[1],
            output_str,
            &args[0],
            &args[1],
        );
    }

    // Fallback
    (
        format!(
            "todo!(/* Complex einsum not yet supported: {} */)",
            equation
        ),
        ReturnType::Tensor,
    )
}

fn compile_binary_einsum(
    p1: &str,
    p2: &str,
    out: &str,
    arg1: &str,
    arg2: &str,
) -> (String, ReturnType) {
    let p1_chars: Vec<char> = p1.chars().collect();
    let p2_chars: Vec<char> = p2.chars().collect();
    let out_chars: Vec<char> = out.chars().collect();

    // 1. Identify Contracting Dimensions (present in p1 AND p2, BUT NOT in out)
    let mut contracting: Vec<char> = Vec::new();
    for c in &p1_chars {
        if p2_chars.contains(c) && !out_chars.contains(c) {
            contracting.push(*c);
        }
    }

    // 2. Identify Batch Dimensions (present in out)
    // Note: In Candle's matmul, 'Batch' dims are all dims before the last 2, which MUST broadcast.

    // Simplest heuristic:
    // If there is exactly 1 contracting dimension.
    if contracting.len() == 1 {
        let k = contracting[0];

        // We want lhs to be (..., k) and rhs to be (..., k, m)
        // Then lhs.matmul(&rhs) -> (..., m)

        // Check LHS
        let lhs_code = permute_for_matmul(arg1, &p1_chars, k, true);

        // Check RHS
        let rhs_code = permute_for_matmul(arg2, &p2_chars, k, false);

        return (
            format!("{}.matmul(&{})?", lhs_code, rhs_code),
            ReturnType::Tensor,
        );
    }

    // If 0 contracting dimensions -> Outer product or Elementwise?
    if contracting.is_empty() {
        // If all indices match -> Elementwise Mul
        if p1 == p2 && p2 == out {
            return (format!("({} * {})", arg1, arg2), ReturnType::Tensor);
        }
        // Outer product not easily done with single matmul without reshaping.
        // todo
    }

    (
        format!(
            "todo!(/* Binary einsum with {} contracting dims: {}, {}, {} */)",
            contracting.len(),
            p1,
            p2,
            out
        ),
        ReturnType::Tensor,
    )
}

// Ensure the contracting dim is in the correct position for matmul.
// is_lhs = true  => target shape (..., K)     [Last dim is K]
// is_lhs = false => target shape (..., K, N)  [Second to last dim is K]
fn permute_for_matmul(arg: &str, pattern: &[char], k: char, is_lhs: bool) -> String {
    let k_pos = pattern.iter().position(|&c| c == k).unwrap(); // must exist
    let rank = pattern.len();

    if is_lhs {
        // Wanted: K is at index (rank - 1)
        if k_pos == rank - 1 {
            return arg.to_string();
        }
        // Permute to move K to end
        let mut new_perm: Vec<usize> = (0..rank).filter(|&i| i != k_pos).collect();
        new_perm.push(k_pos);
        let perm_str: Vec<String> = new_perm.iter().map(|i| i.to_string()).collect();
        return format!("{}.permute(({}))?", arg, perm_str.join(", "));
    } else {
        // Wanted: K is at index (rank - 2)
        if k_pos == rank - 2 {
            return arg.to_string();
        }
        // Permute to move K to rank-2
        if k_pos == rank - 1 {
            // K is last, just transpose last two? Or generic permute
            return format!("{}.t()?", arg);
        }

        // Generic permute: Move K to second-to-last, preserve others order?
        // Actually, we usually want the OTHER dim at the very end.
        // Assuming there is exactly one other non-batch dim?
        // For general batched mm, we need (B..., K, N).
        // So we move K to -2.
        // Let's just move K to the end, then transpose.
        // "Make K the last dimension"
        let mut simple_perm: Vec<usize> = (0..rank).filter(|&i| i != k_pos).collect();
        simple_perm.push(k_pos);
        let perm_str: Vec<String> = simple_perm.iter().map(|i| i.to_string()).collect();

        // Now it's (..., K). We want (..., K, N).
        // So we transpose the result.
        return format!("{}.permute(({}))?.t()?", arg, perm_str.join(", "));
    }
}
