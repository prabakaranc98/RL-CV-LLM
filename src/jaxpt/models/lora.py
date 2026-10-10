"""LoRA on Tunix's Qwen3 (via qwix), and merging it back into dense weights.

qwix wraps each matched projection: y = x W + (x A B) * (alpha / rank), with A, B stored next to the frozen weight as
nnx.LoRAParam (`w_lora_a`, `w_lora_b` for attention's per-head `w`; `kernel_lora_a`, `kernel_lora_b` for the MLP).
Tunix then trains only the LoRAParams and syncs only them to the sampler each step.

merge_lora() turns a LoRA model's parameters into the dense equivalent, W + (alpha / rank) * A @ B, with shapes
A [..in.., r] and B [r, ..out..] contracting over r. It is the one place this is done, used by:
  - the sampler copy (generation runs on dense weights: no LoRA overhead, and our last-token prefill model),
  - the sampler's per-step weight sync (base weights kept once, the new A, B merged in),
  - the export to a Hugging Face checkpoint for vLLM evaluation.
"""
import jax
import jax.numpy as jnp
from flax import nnx

# all attention and MLP projections of Tunix's Qwen3
MODULES = ".*q_proj|.*k_proj|.*v_proj|.*o_proj|.*gate_proj|.*up_proj|.*down_proj"


def apply_lora(model, rank, alpha, module_path=MODULES):
    import qwix
    provider = qwix.LoraProvider(module_path=module_path, rank=rank, alpha=float(alpha))
    return qwix.apply_lora_to_model(model, provider, **model.get_model_input())


def has_lora(model):
    return any(isinstance(v, nnx.LoRAParam) for _, v in nnx.iter_graph(model))


def flatten(tree):
    return {tuple(getattr(k, "key", k) for k in path): v for path, v in jax.tree_util.tree_flatten_with_path(tree)[0]}


def unflatten(flat):
    out = {}
    for path, v in flat.items():
        d = out
        for k in path[:-1]:
            d = d.setdefault(k, {})
        d[path[-1]] = v
    return out


def is_lora_key(name):
    return isinstance(name, str) and (name.endswith("_lora_a") or name.endswith("_lora_b"))


def merge_lora(params, scale, dtype=None):
    """params: nested dict of a LoRA model's parameters (base weights and LoRA A, B). Returns the nested dict of dense
    weights W + scale * A @ B, without LoRA entries, optionally cast to dtype."""
    flat = flatten(params)
    dense = {}
    for path, w in flat.items():
        if is_lora_key(path[-1]):
            continue
        a = flat.get(path[:-1] + (f"{path[-1]}_lora_a",))
        if a is not None:
            b = flat[path[:-1] + (f"{path[-1]}_lora_b",)]
            delta = jnp.tensordot(a.astype(jnp.float32), b.astype(jnp.float32), axes=1)
            w = (w.astype(jnp.float32) + scale * delta).astype(w.dtype)
        dense[path] = w.astype(dtype) if dtype is not None else w
    return unflatten(dense)


def base_params(model, dtype=None):
    """Nested dict of the model's non-LoRA parameters (the frozen base under LoRA), optionally cast."""
    flat = flatten(nnx.to_pure_dict(nnx.state(model, nnx.Param)))
    return unflatten({p: (v.astype(dtype) if dtype is not None else v) for p, v in flat.items() if not is_lora_key(p[-1])})


def dense_params(model, scale, dtype=None):
    """Nested dict of the model's dense-equivalent parameters (identity for a model without LoRA)."""
    return merge_lora(nnx.to_pure_dict(nnx.state(model, nnx.Param)), scale, dtype)
