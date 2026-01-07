import os
import json
import torch
import torch.nn as nn
from safetensors.torch import save_file
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

@dataclass
class LayerMeta:
    name: str
    module_type: str
    input_shapes: List[List[int]]
    output_shapes: List[List[int]]
    parameters: List[str]
    is_leaf: bool
    config: Dict[str, Any]

class GoldenRecorder:
    def __init__(self, output_dir: str = "pycandle_trace"):
        self.output_dir = output_dir
        self.records: Dict[str, torch.Tensor] = {}
        self.manifest: Dict[str, LayerMeta] = {}
        self.call_counts = defaultdict(int)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _get_module_config(self, module: nn.Module) -> Dict[str, Any]:
        cfg = {}
        if isinstance(module, nn.Linear):
            cfg = {"in_features": module.in_features, "out_features": module.out_features, "bias": module.bias is not None}
        elif isinstance(module, nn.Conv1d):
            cfg = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size[0],
                "stride": module.stride[0],
                "padding": module.padding[0],
                "bias": module.bias is not None
            }
        elif isinstance(module, nn.LayerNorm):
            cfg = {"normalized_shape": list(module.normalized_shape), "eps": module.eps}
        elif isinstance(module, nn.Embedding):
            cfg = {"num_embeddings": module.num_embeddings, "embedding_dim": module.embedding_dim}
        # Activation functions
        elif isinstance(module, nn.ELU):
            cfg = {"alpha": module.alpha}
        elif isinstance(module, nn.LeakyReLU):
            cfg = {"negative_slope": module.negative_slope}
        elif isinstance(module, nn.BatchNorm1d):
            cfg = {"num_features": module.num_features, "eps": module.eps}
        elif isinstance(module, nn.BatchNorm2d):
            cfg = {"num_features": module.num_features, "eps": module.eps}
        elif isinstance(module, nn.LSTM):
            cfg = {
                "input_size": module.input_size,
                "hidden_size": module.hidden_size,
                "num_layers": module.num_layers,
                "batch_first": module.batch_first,
                "bidirectional": module.bidirectional
            }
        # Snake activation (custom) - check for alpha parameter and in_features
        elif hasattr(module, 'alpha') and isinstance(getattr(module, 'alpha', None), torch.nn.Parameter):
            # Snake: alpha is a learnable parameter, extract in_features from its shape
            alpha = module.alpha
            if alpha.dim() >= 2:
                cfg = {"in_features": alpha.shape[1]}
            elif alpha.dim() == 1:
                cfg = {"in_features": alpha.shape[0]}
        
        # GPT2 from HuggingFace transformers
        if hasattr(module, 'config') and hasattr(module.config, 'n_embd'):
            cfg['vocab_size'] = module.config.vocab_size
            cfg['n_positions'] = module.config.n_positions  # context_length
            cfg['n_embd'] = module.config.n_embd  # emb_dim
            cfg['n_head'] = module.config.n_head  # n_heads
            cfg['n_layer'] = module.config.n_layer  # n_layers
            cfg['resid_pdrop'] = module.config.resid_pdrop  # drop_rate
        
        return cfg

    def _tensor_to_cpu(self, t: Any) -> Optional[torch.Tensor]:
        if isinstance(t, torch.Tensor):
            return t.detach().clone().contiguous().float().cpu()
        return None

    def _extract_shapes(self, data: Any) -> List[List[int]]:
        shapes = []
        if isinstance(data, torch.Tensor):
            shapes.append(list(data.shape))
        elif isinstance(data, (tuple, list)):
            for item in data:
                if isinstance(item, torch.Tensor):
                    shapes.append(list(item.shape))
        return shapes

    def hook_factory(self, name: str):
        def hook(m, inp, out):
            call_idx = self.call_counts[name]
            self.call_counts[name] += 1
            
            trace_key = f"{name}.{call_idx}" if call_idx > 0 else name
            
            # Record Inputs
            if isinstance(inp, tuple):
                for i, x in enumerate(inp):
                    cpu_x = self._tensor_to_cpu(x)
                    if cpu_x is not None:
                        self.records[f"{trace_key}.in.{i}"] = cpu_x
            
            # Record Outputs
            if isinstance(out, torch.Tensor):
                self.records[f"{trace_key}.out.0"] = self._tensor_to_cpu(out)
            elif isinstance(out, (tuple, list)):
                for i, x in enumerate(out):
                    cpu_x = self._tensor_to_cpu(x)
                    if cpu_x is not None:
                        self.records[f"{trace_key}.out.{i}"] = cpu_x

            # Record Metadata
            self.manifest[trace_key] = LayerMeta(
                name=name,
                module_type=type(m).__name__,
                input_shapes=self._extract_shapes(inp),
                output_shapes=self._extract_shapes(out),
                parameters=[n for n, _ in m.named_parameters(recurse=False)],
                is_leaf=len(list(m.children())) == 0,
                config=self._get_module_config(m)
            )
        return hook

    def record(self, model: nn.Module, *args, **kwargs):
        model.eval()
        hooks = []
        for name, module in model.named_modules():
            if name == "": continue 
            hooks.append(module.register_forward_hook(self.hook_factory(name)))
        
        try:
            with torch.no_grad():
                output = model(*args, **kwargs)
        finally:
            for h in hooks:
                h.remove()
        
        return output

    def save(self, project_name: str):
        tensor_path = os.path.join(self.output_dir, f"{project_name}_trace.safetensors")
        save_file(self.records, tensor_path)
        
        manifest_path = os.path.join(self.output_dir, f"{project_name}_manifest.json")
        with open(manifest_path, "w") as f:
            json.dump({k: asdict(v) for k, v in self.manifest.items()}, f, indent=4)
        print(f"✅ Trace and Manifest saved for {project_name}")
