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
    buffers: List[str]
    is_leaf: bool
    config: Dict[str, Any]

class GoldenRecorder:
    def __init__(self, output_dir: str = "pycandle_trace", keep_dtype: bool = False):
        self.output_dir = output_dir
        self.keep_dtype = keep_dtype
        self.records: Dict[str, torch.Tensor] = {}
        self.manifest: Dict[str, LayerMeta] = {}
        self.call_counts = defaultdict(int)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

    def _get_module_config(self, module: nn.Module) -> Dict[str, Any]:
        cfg = {}
        
        # --- NEW BLOCK: Detect Weight Normalization ---
        is_weight_norm = False
        # Check PyTorch 2.0+ parametrizations
        if hasattr(module, 'parametrizations') and 'weight' in module.parametrizations:
             # Check if it's actually WeightNorm
             for p in module.parametrizations.weight:
                 if type(p).__name__ == "WeightNorm":
                     is_weight_norm = True
        # Check Legacy PyTorch weight_norm
        elif hasattr(module, 'weight_g') and hasattr(module, 'weight_v'):
            is_weight_norm = True
        
        if is_weight_norm:
            cfg['weight_norm'] = True
        # ---------------------------------------------

        if isinstance(module, nn.Linear):
            cfg = {
                "in_features": module.in_features,
                "out_features": module.out_features,
                "bias": module.bias is not None,
                # Record actual weight shape for transpose detection
                "weight_shape": list(module.weight.shape),  # [out, in] in PyTorch
            }
        elif isinstance(module, nn.Conv1d) or type(module).__name__ == "Conv1d":
            cfg = {
                "in_channels": module.in_channels,
                "out_channels": module.out_channels,
                "kernel_size": module.kernel_size[0] if isinstance(module.kernel_size, (list, tuple)) else module.kernel_size,
                "stride": module.stride[0] if isinstance(module.stride, (list, tuple)) else module.stride,
                "padding": module.padding[0] if isinstance(module.padding, (list, tuple)) else module.padding,
                "bias": module.bias is not None
            }
        # HF GPT2 often uses Conv1D
        elif type(module).__name__ == "Conv1D":
            cfg = {
                "in_features": module.weight.shape[0],
                "out_features": module.weight.shape[1],
                "bias": module.bias is not None,
                "weight_shape": list(module.weight.shape)
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
        # Custom Transpose
        elif type(module).__name__ == "Transpose":
             cfg = {
                "dim0": getattr(module, "dim0", 1),
                "dim1": getattr(module, "dim1", 2),
            }
        
        # GPT2 from HuggingFace transformers
        if hasattr(module, 'config') and hasattr(module.config, 'n_embd'):
            cfg['vocab_size'] = module.config.vocab_size
            cfg['n_positions'] = module.config.n_positions  # context_length
            cfg['n_embd'] = module.config.n_embd  # emb_dim
            cfg['n_head'] = module.config.n_head  # n_heads
            cfg['n_layer'] = module.config.n_layer  # n_layers
            cfg['resid_pdrop'] = module.config.resid_pdrop  # drop_rate
        
        # --- Universal Fallback ---
        # Capture all scalar/list attributes to support generic loading
        # This allows us to handle layers we haven't explicitly mapped yet
        base_attrs = dir(torch.nn.Module())
        for k in dir(module):
            if k.startswith('_') or k in base_attrs: continue
            try:
                v = getattr(module, k)
                if isinstance(v, (int, float, bool, str)):
                    if k not in cfg:
                        cfg[k] = v
                elif isinstance(v, (tuple, list)):
                    if all(isinstance(x, (int, float, bool, str)) for x in v):
                         if k not in cfg:
                            cfg[k] = v
            except:
                pass
        
        return cfg

    def _tensor_to_cpu(self, t: Any) -> Optional[torch.Tensor]:
        if isinstance(t, torch.Tensor):
            if t.device.type == 'meta':
                # Return placeholder or None
                return None
            t = t.detach().clone().contiguous().cpu()
            if not self.keep_dtype:
                t = t.float()
            return t
        return None

    def _extract_shapes(self, data: Any) -> List[List[int]]:
        shapes = []
        if isinstance(data, torch.Tensor):
            shapes.append(list(data.shape))
        elif isinstance(data, (tuple, list)):
            for item in data:
                shapes.extend(self._extract_shapes(item))
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
            module_type = type(m).__name__
            supported_ops = {
                "Linear", "Conv1d", "Conv2d", "LayerNorm", "Embedding", "ReLU", "GELU", 
                "Sigmoid", "Tanh", "ELU", "LeakyReLU", "Snake", "BatchNorm1d", "BatchNorm2d",
                "LSTM", "CausalConv1d", "Mish", "SiLU", "NewGELUActivation", "Dropout",
                "Transpose", "SinusoidalPosEmb", "LlamaRMSNorm", "ModuleList", "Sequential"
            }
            if module_type not in supported_ops:
                print(f"⚠️  WARNING: Custom or potentially unsupported Op detected: {module_type} ({name})")
                print(f"    PyCandle might not support this layer automatically. You may need to implement a custom kernel.")

            self.manifest[trace_key] = LayerMeta(
                name=name,
                module_type=module_type,
                input_shapes=self._extract_shapes(inp),
                output_shapes=self._extract_shapes(out),
                parameters=[n for n, _ in m.named_parameters(recurse=False)],
                buffers=[n for n, _ in m.named_buffers(recurse=False)],
                is_leaf=len(list(m.children())) == 0,
                config=self._get_module_config(m)
            )
        return hook

    def record(self, model: nn.Module, *args, trace_fx: bool = False, fx_concrete_args: Optional[Dict[str, Any]] = None, **kwargs):
        model.eval()
        
        # Record model inputs
        for i, arg in enumerate(args):
            cpu_arg = self._tensor_to_cpu(arg)
            if cpu_arg is not None:
                self.records[f"model_input.{i}"] = cpu_arg

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
        
        if trace_fx:
            self.trace_fx(model, concrete_args=fx_concrete_args, *args, **kwargs)
            
        return output

    def trace_fx(self, model: nn.Module, *example_inputs, concrete_args=None, **kwargs):
        """Use torch.fx to capture the computation graph."""
        import torch.fx as fx
        
        # Try to use transformers.utils.fx if available for HF models
        # This handles control flow and other HF-specific quirks better than vanilla fx
        try:
            from transformers.utils.fx import symbolic_trace as hf_symbolic_trace
            # Check if it's a transformers model (heuristic)
            is_hf = any(c.__module__.startswith("transformers") for c in model.__class__.__mro__)
            if is_hf:
                print("🕵️ Using transformers.utils.fx.symbolic_trace for robustness...")
                # HF tracer requires input_names usually, but let's try basic first
                # Note: HF tracer might expect 'input_names' kwarg if we want to be safe, 
                # but let's see if it works with just concrete_args.
                traced = hf_symbolic_trace(model, input_names=["input_ids"], disable_check=True)
            else:
                traced = fx.symbolic_trace(model, concrete_args=concrete_args)
        except ImportError:
            traced = fx.symbolic_trace(model, concrete_args=concrete_args)
        except Exception as e:
            print(f"⚠️ transformers trace failed ({e}), falling back to torch.fx...")
            traced = fx.symbolic_trace(model, concrete_args=concrete_args)
        
        def serialize_arg(arg):
            if isinstance(arg, (list, tuple)):
                return [serialize_arg(a) for a in arg]
            if isinstance(arg, dict):
                return {k: serialize_arg(v) for k, v in arg.items()}
            if hasattr(arg, "name"):
                return arg.name
            return str(arg)

        def serialize_target(target):
            if isinstance(target, str):
                return target
            if hasattr(target, "__module__") and hasattr(target, "__name__"):
                 # e.g. torch.arange, operator.getitem
                 return f"{target.__module__}.{target.__name__}"
            if hasattr(target, "__name__"):
                return target.__name__
            return str(target)

        graph_nodes = []
        for node in traced.graph.nodes:
            # We want to map these nodes to the modules or functions
            node_info = {
                "name": node.name,
                "op": node.op,
                "target": serialize_target(node.target),
                "args": [serialize_arg(arg) for arg in node.args],
                "kwargs": {k: serialize_arg(v) for k, v in node.kwargs.items()},
            }
            
            if node.op == "call_module":
                try:
                    submod = traced.get_submodule(node.target)
                    node_info["module_type"] = type(submod).__name__
                except:
                    pass
            
            graph_nodes.append(node_info)
            
        self._fx_graph = {
            "graph_nodes": graph_nodes,
            "graph_code": traced.code
        }
        return self._fx_graph

    def save(self, project_name: str, use_fx: bool = False, hints: Optional[Dict[str, int]] = None):
        tensor_path = os.path.join(self.output_dir, f"{project_name}_trace.safetensors")
        # Filter out None values (meta placeholders)
        real_records = {k: v for k, v in self.records.items() if v is not None}
        if real_records:
            save_file(real_records, tensor_path)
        else:
            print("⚠️ No real tensors recorded (Meta-only mode)")
        
        manifest_path = os.path.join(self.output_dir, f"{project_name}_manifest.json")
        manifest_data = {k: asdict(v) for k, v in self.manifest.items()}
        
        if use_fx and hasattr(self, '_fx_graph'):
            manifest_data["_graph_nodes"] = self._fx_graph["graph_nodes"]
            manifest_data["_graph_code"] = self._fx_graph["graph_code"]
            
            # Auto-detect stateful
            for node in self._fx_graph["graph_nodes"]:
                if node["op"] == "placeholder" and (
                    "past" in node["name"] or "cache" in node["name"]
                ):
                    manifest_data["_auto_stateful"] = True


        if hints:
            manifest_data["_symbolic_hints"] = hints

        with open(manifest_path, "w") as f:
            json.dump(manifest_data, f, indent=4)
        print(f"✅ Trace and Manifest saved for {project_name}")
