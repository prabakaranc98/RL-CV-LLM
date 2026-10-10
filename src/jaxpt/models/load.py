"""Load a Hugging Face Qwen2 / Qwen3 checkpoint into the params pytree used by qwen3.py.

Linear weights are stored by Hugging Face as [out, in]; here they become [in, out] so that x @ w works.
Per-layer tensors are stacked along a leading L axis for lax.scan.
"""
import glob
import json
import os

import jax.numpy as jnp
import numpy as np
from huggingface_hub import snapshot_download
from safetensors.numpy import load_file

from .qwen3 import Config

_PATTERNS = ["*.json", "*.safetensors", "*.txt", "*.model", "tokenizer*", "merges.txt", "vocab.json"]


def fetch(repo_id, revision=None, cache_dir=None):
    """Download (or reuse) the checkpoint; returns the local directory."""
    return snapshot_download(repo_id, revision=revision, cache_dir=cache_dir, allow_patterns=_PATTERNS)


def load_config(repo_or_dir, revision=None):
    path = repo_or_dir if os.path.isdir(repo_or_dir) else fetch(repo_or_dir, revision)
    return Config.from_hf(json.load(open(os.path.join(path, "config.json"))))


def load(repo_or_dir, dtype=jnp.float32, revision=None):
    """Returns (cfg, params). dtype is the storage dtype of the params (float32 master weights by default)."""
    path = repo_or_dir if os.path.isdir(repo_or_dir) else fetch(repo_or_dir, revision)
    cfg = load_config(path)
    raw = {}
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        raw.update(load_file(f))

    def t(name):                       # a linear weight, transposed to [in, out]
        return raw[name].T

    def stack(fn):
        return jnp.asarray(np.stack([fn(i) for i in range(cfg.layers)]), dtype)

    p = "model.layers.{}."
    layers = {
        "ln1": stack(lambda i: raw[p.format(i) + "input_layernorm.weight"]),
        "ln2": stack(lambda i: raw[p.format(i) + "post_attention_layernorm.weight"]),
        "wq": stack(lambda i: t(p.format(i) + "self_attn.q_proj.weight")),
        "wk": stack(lambda i: t(p.format(i) + "self_attn.k_proj.weight")),
        "wv": stack(lambda i: t(p.format(i) + "self_attn.v_proj.weight")),
        "wo": stack(lambda i: t(p.format(i) + "self_attn.o_proj.weight")),
        "w_gate": stack(lambda i: t(p.format(i) + "mlp.gate_proj.weight")),
        "w_up": stack(lambda i: t(p.format(i) + "mlp.up_proj.weight")),
        "w_down": stack(lambda i: t(p.format(i) + "mlp.down_proj.weight")),
    }
    if cfg.qkv_bias:
        for n in ("q", "k", "v"):
            layers[f"b{n}"] = stack(lambda i, n=n: raw[p.format(i) + f"self_attn.{n}_proj.bias"])
    if cfg.qk_norm:
        layers["q_norm"] = stack(lambda i: raw[p.format(i) + "self_attn.q_norm.weight"])
        layers["k_norm"] = stack(lambda i: raw[p.format(i) + "self_attn.k_norm.weight"])
    params = {"embed": jnp.asarray(raw["model.embed_tokens.weight"], dtype), "layers": layers,
              "norm": jnp.asarray(raw["model.norm.weight"], dtype)}
    if not cfg.tied:
        params["lm_head"] = jnp.asarray(raw["lm_head.weight"].T, dtype)
    return cfg, params


def tokenizer(repo_or_dir):
    from transformers import AutoTokenizer
    path = repo_or_dir if os.path.isdir(repo_or_dir) else fetch(repo_or_dir)
    return AutoTokenizer.from_pretrained(path)


def count_params(params):
    import jax
    return sum(x.size for x in jax.tree.leaves(params))
