"""Qwen2 / Qwen3 dense decoder as pure JAX functions over a params pytree.

Shapes: B batch, T or S sequence, M cache slots, D width, H query heads, K key/value heads,
E head dim, F feed-forward width, V vocab, L layers.

Two entry points share one layer:
  forward(...)         a full sequence, no cache: training and scoring
  forward_cached(...)  S new tokens written into a fixed-size KV cache: prefill (S = prompt) and decode (S = 1)
Layers are stacked along a leading L axis and run with lax.scan, so the model compiles once.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp

NEG = -1e30   # additive mask value; finite, so a fully masked row gives a uniform (not NaN) softmax


@dataclass(frozen=True)
class Config:
    vocab: int
    d: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    ffn: int
    rope_theta: float
    eps: float
    tied: bool
    qkv_bias: bool      # Qwen2: biases on q, k, v
    qk_norm: bool       # Qwen3: RMSNorm on each head of q and k

    @staticmethod
    def from_hf(c):
        mt = c["model_type"]
        if mt not in ("qwen2", "qwen3"):
            raise ValueError(f"unsupported model_type {mt!r} (dense Qwen2 / Qwen3 only)")
        return Config(vocab=c["vocab_size"], d=c["hidden_size"], layers=c["num_hidden_layers"],
                      heads=c["num_attention_heads"], kv_heads=c["num_key_value_heads"],
                      head_dim=c.get("head_dim") or c["hidden_size"] // c["num_attention_heads"],
                      ffn=c["intermediate_size"], rope_theta=float(c["rope_theta"]), eps=c["rms_norm_eps"],
                      tied=bool(c.get("tie_word_embeddings", False)),
                      qkv_bias=(mt == "qwen2"), qk_norm=(mt == "qwen3"))


def rms_norm(x, w, eps):
    """As in Hugging Face: normalize in fp32, cast back, then scale."""
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(x32 * x32, axis=-1, keepdims=True) + eps)
    return w.astype(x.dtype) * y.astype(x.dtype)


def rope(x, positions, theta):
    """Rotary embedding, half-split layout (rotate_half). x: [B, S, N, E]; positions: [B, S]."""
    E = x.shape[-1]
    half = E // 2
    inv_freq = 1.0 / (theta ** (jnp.arange(0, E, 2, dtype=jnp.float32) / E))      # [E/2]
    ang = positions[..., None].astype(jnp.float32) * inv_freq                        # [B, S, E/2]
    cos = jnp.cos(ang)[:, :, None, :].astype(x.dtype)
    sin = jnp.sin(ang)[:, :, None, :].astype(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return jnp.concatenate([x1 * cos - x2 * sin, x2 * cos + x1 * sin], axis=-1)


def _qkv(lp, cfg, h, positions, dtype):
    B, S, _ = h.shape
    q = h @ lp["wq"].astype(dtype)
    k = h @ lp["wk"].astype(dtype)
    v = h @ lp["wv"].astype(dtype)
    if cfg.qkv_bias:
        q, k, v = q + lp["bq"].astype(dtype), k + lp["bk"].astype(dtype), v + lp["bv"].astype(dtype)
    q = q.reshape(B, S, cfg.heads, cfg.head_dim)
    k = k.reshape(B, S, cfg.kv_heads, cfg.head_dim)
    v = v.reshape(B, S, cfg.kv_heads, cfg.head_dim)
    if cfg.qk_norm:
        q, k = rms_norm(q, lp["q_norm"], cfg.eps), rms_norm(k, lp["k_norm"], cfg.eps)
    return rope(q, positions, cfg.rope_theta), rope(k, positions, cfg.rope_theta), v


def _attend(cfg, q, k, v, mask, dtype):
    """q: [B, S, H, E]; k, v: [B, M, K, E]; mask: [B, S, M] bool (True = may attend)."""
    B, S, H, E = q.shape
    g = H // cfg.kv_heads
    qg = q.reshape(B, S, cfg.kv_heads, g, E)
    scores = jnp.einsum("bskge,bmke->bkgsm", qg, k).astype(jnp.float32) / jnp.sqrt(jnp.float32(E))
    scores = jnp.where(mask[:, None, None, :, :], scores, NEG)
    probs = jax.nn.softmax(scores, axis=-1).astype(dtype)
    out = jnp.einsum("bkgsm,bmke->bskge", probs, v)
    return out.reshape(B, S, H * E)


def _mlp(lp, h, dtype):
    gate = h @ lp["w_gate"].astype(dtype)
    up = h @ lp["w_up"].astype(dtype)
    return (jax.nn.silu(gate) * up) @ lp["w_down"].astype(dtype)


def _logits(params, cfg, x, dtype):
    x = rms_norm(x, params["norm"], cfg.eps)
    w = params["embed"].T if cfg.tied else params["lm_head"]
    return x @ w.astype(dtype)


def forward(params, cfg, tokens, positions, valid, dtype=jnp.bfloat16, remat=False):
    """Full-sequence forward. tokens, positions: [B, T] int; valid: [B, T] bool. Returns logits [B, T, V]."""
    T = tokens.shape[1]
    causal = jnp.tril(jnp.ones((T, T), bool))
    mask = causal[None] & valid[:, None, :]                                   # [B, T, T]
    mask = mask | jnp.eye(T, dtype=bool)[None]                                # a row is never empty

    def layer(x, lp):
        h = rms_norm(x, lp["ln1"], cfg.eps)
        q, k, v = _qkv(lp, cfg, h, positions, dtype)
        x = x + _attend(cfg, q, k, v, mask, dtype) @ lp["wo"].astype(dtype)
        x = x + _mlp(lp, rms_norm(x, lp["ln2"], cfg.eps), dtype)
        return x, None

    x = params["embed"][tokens].astype(dtype)
    x, _ = jax.lax.scan(jax.checkpoint(layer) if remat else layer, x, params["layers"])
    return _logits(params, cfg, x, dtype)


def flops_per_token(cfg, T):
    """Matmul FLOPs of one forward pass, per token, at sequence length T (the PaLM-style count):
    2 x (weight-matrix params, LM head included) + 4 x layers x (heads x head_dim) x T for QK^T and PV.
    Training (forward + backward) is 3x this."""
    q, kv = cfg.heads * cfg.head_dim, cfg.kv_heads * cfg.head_dim
    per_layer = cfg.d * (q + 2 * kv) + q * cfg.d + 3 * cfg.d * cfg.ffn
    return 2 * (cfg.layers * per_layer + cfg.d * cfg.vocab) + 4 * cfg.layers * q * T


def init_cache(cfg, batch, slots, dtype=jnp.bfloat16):
    shape = (cfg.layers, batch, slots, cfg.kv_heads, cfg.head_dim)
    return {"k": jnp.zeros(shape, dtype), "v": jnp.zeros(shape, dtype), "valid": jnp.zeros((batch, slots), bool)}


def forward_cached(params, cfg, tokens, positions, valid, cache, write_idx, dtype=jnp.bfloat16):
    """Write S new tokens at cache slots [write_idx, write_idx + S) and attend over the cache.
    tokens, positions: [B, S]; valid: [B, S] bool. Returns (logits [B, S, V], new cache)."""
    B, S = tokens.shape
    M = cache["valid"].shape[1]
    kv_valid = jax.lax.dynamic_update_slice(cache["valid"], valid, (0, write_idx))
    slot = jnp.arange(M)[None, :]                                              # [1, M]
    q_slot = write_idx + jnp.arange(S)[:, None]                                # [S, 1]
    mask = (slot <= q_slot)[None] & kv_valid[:, None, :]                        # [B, S, M]
    mask = mask | (slot == q_slot)[None]

    def layer(x, inputs):
        lp, kc, vc = inputs
        h = rms_norm(x, lp["ln1"], cfg.eps)
        q, k, v = _qkv(lp, cfg, h, positions, dtype)
        kc = jax.lax.dynamic_update_slice(kc, k.astype(kc.dtype), (0, write_idx, 0, 0))
        vc = jax.lax.dynamic_update_slice(vc, v.astype(vc.dtype), (0, write_idx, 0, 0))
        x = x + _attend(cfg, q, kc.astype(dtype), vc.astype(dtype), mask, dtype) @ lp["wo"].astype(dtype)
        x = x + _mlp(lp, rms_norm(x, lp["ln2"], cfg.eps), dtype)
        return x, (kc, vc)

    x = params["embed"][tokens].astype(dtype)
    x, (k_all, v_all) = jax.lax.scan(layer, x, (params["layers"], cache["k"], cache["v"]))
    return _logits(params, cfg, x, dtype), {"k": k_all, "v": v_all, "valid": kv_valid}
