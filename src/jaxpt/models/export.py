"""Export a Tunix Qwen3 (dense) to a Hugging Face checkpoint directory, so vLLM can evaluate trained policies
with the same harness as the base model (E0).

The mapping is the inverse of tunix.models.qwen3.params._get_key_and_transform_mapping:
attention projections are stored by Tunix per head ([D, H, E] and [H, E, D]), MLP kernels as [in, out].
"""
import os
import re
import shutil

import jax
import numpy as np
from flax import nnx
from safetensors.numpy import save_file

from .load import fetch

# Tunix path -> (HF name, how to turn the Tunix array into the HF [out, in] layout)
_RULES = [
    (r"embedder\.input_embedding", "model.embed_tokens.weight", lambda w: w),
    (r"final_norm\.w", "model.norm.weight", lambda w: w),
    (r"lm_head\.w", "lm_head.weight", lambda w: w.T),
    (r"layers\.(\d+)\.attn\.(q|k|v)_proj\.w", r"model.layers.\1.self_attn.\2_proj.weight",
     lambda w: w.reshape(w.shape[0], -1).T),                       # [D, H, E] -> [H*E, D]
    (r"layers\.(\d+)\.attn\.o_proj\.w", r"model.layers.\1.self_attn.o_proj.weight",
     lambda w: w.reshape(-1, w.shape[-1]).T),                      # [H, E, D] -> [D, H*E]
    (r"layers\.(\d+)\.attn\.(q|k)_norm\.w", r"model.layers.\1.self_attn.\2_norm.weight", lambda w: w),
    (r"layers\.(\d+)\.mlp\.(gate|up|down)_proj\.kernel", r"model.layers.\1.mlp.\2_proj.weight", lambda w: w.T),
    (r"layers\.(\d+)\.input_layernorm\.w", r"model.layers.\1.input_layernorm.weight", lambda w: w),
    (r"layers\.(\d+)\.post_attention_layernorm\.w", r"model.layers.\1.post_attention_layernorm.weight", lambda w: w),
]
_META = ["config.json", "generation_config.json", "tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt"]


def hf_tensors(model, dtype=np.float32, lora_scale=None):
    """{HF name: numpy array} from a Tunix Qwen3 module (a LoRA model is exported as its dense equivalent)."""
    import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
    from .lora import dense_params, has_lora
    pure = dense_params(model, lora_scale) if has_lora(model) else nnx.to_pure_dict(nnx.state(model, nnx.Param))
    out = {}
    for path, w in jax.tree.flatten_with_path(pure)[0]:
        key = ".".join(str(p.key) for p in path)
        for pat, name, f in _RULES:
            if re.fullmatch(pat, key):
                out[re.sub(pat, name, key)] = np.ascontiguousarray(f(np.asarray(w, np.float32)).astype(dtype))
                break
        else:
            raise KeyError(f"no export rule for Tunix param {key}")
    return out


def export_hf(model, out_dir, base_repo="Qwen/Qwen3-0.6B-Base", dtype="bfloat16", lora_scale=None):
    """Write model.safetensors (bf16 by default) plus the base model's config and tokenizer files."""
    import ml_dtypes
    os.makedirs(out_dir, exist_ok=True)
    np_dtype = ml_dtypes.bfloat16 if dtype == "bfloat16" else np.float32
    save_file(hf_tensors(model, np_dtype, lora_scale), os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    base = fetch(base_repo)
    for f in _META:
        if os.path.exists(os.path.join(base, f)):
            shutil.copy(os.path.join(base, f), out_dir)
    return out_dir
