"""LoRA merge is exact: the dense copy used for generation and export computes what the LoRA policy computes.

A 2-layer Qwen3 (fp32 compute) gets qwix LoRA on all projections, with B set to random values so the update is not
zero. Then:
  1. logits of the LoRA model == logits of sampler_copy(model) (our dense Qwen3 subclass), max |diff| < 1e-4;
  2. the sync path (frozen base + synced A, B -> merge_lora) gives the same dense weights as merging the model;
  3. the export has no LoRA tensors, and every tensor has its Hugging Face shape;
  4. the frozen base is untouched by merging.

Run on a pod (CPU is enough):  JAX_PLATFORMS=cpu pytest -q -s -p no:warnings tests/unit/test_lora_merge.py
"""
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tunix.models.qwen3 import model as tmodel

from jaxpt.models import lora
from jaxpt.models.export import hf_tensors
from jaxpt.rollout.tunix_qwen3 import sampler_copy

RANK, ALPHA = 4, 8.0
SCALE = ALPHA / RANK


def lora_model():
    cfg = dataclasses.replace(tmodel.ModelConfig.qwen3_0p6b_base(), num_layers=2, dtype=jnp.float32,
                              param_dtype=jnp.float32)
    m = lora.apply_lora(tmodel.Qwen3(cfg, rngs=nnx.Rngs(0)), RANK, ALPHA)
    st = nnx.state(m, nnx.LoRAParam)
    keys = iter(jax.random.split(jax.random.key(1), len(jax.tree.leaves(st))))
    nnx.update(m, jax.tree.map(lambda x: 0.02 * jax.random.normal(next(keys), x.shape, x.dtype), st))
    return m


def logits(model, T=16):
    tokens = jax.random.randint(jax.random.key(2), (2, T), 0, 151_000)
    positions = jnp.broadcast_to(jnp.arange(T), (2, T))
    out, _ = model(tokens, positions, None, jnp.tril(jnp.ones((1, T, T), bool)))
    return np.asarray(out)


def test_dense_copy_matches_lora_model():
    m = lora_model()
    assert lora.has_lora(m)
    dense = sampler_copy(m, dtype=jnp.float32, lora_scale=SCALE)
    assert not lora.has_lora(dense)
    a, b = logits(m), logits(dense)
    print(f"\n  max |logits LoRA - dense| {np.abs(a - b).max():.2e} (logits scale {np.abs(a).max():.1f})")
    assert np.abs(a - b).max() < 1e-4


def test_sync_path_equals_model_merge():
    m = lora_model()
    base = lora.base_params(m)
    synced = lora.flatten(nnx.to_pure_dict(nnx.state(m, nnx.LoRAParam)))
    full = lora.flatten(base)
    full.update({p: v for p, v in synced.items() if lora.is_lora_key(p[-1])})
    via_sync = lora.flatten(lora.merge_lora(lora.unflatten(full), SCALE))
    direct = lora.flatten(lora.dense_params(m, SCALE))
    assert via_sync.keys() == direct.keys()
    assert all(np.array_equal(via_sync[k], direct[k]) for k in direct)
    changed = [k for k in direct if not np.array_equal(direct[k], lora.flatten(base)[k])]
    print(f"\n  {len(direct)} dense tensors, {len(changed)} changed by the LoRA update")
    assert len(changed) == 2 * 7                                     # 7 projections x 2 layers


def test_export_is_dense_with_hf_shapes():
    m = lora_model()
    t = hf_tensors(m, lora_scale=SCALE)
    assert not any("lora" in k for k in t)
    assert t["model.layers.0.self_attn.q_proj.weight"].shape == (2048, 1024)
    assert t["model.layers.0.self_attn.o_proj.weight"].shape == (1024, 2048)
    assert t["model.layers.1.mlp.down_proj.weight"].shape == (1024, 3072)
    print(f"\n  exported {len(t)} dense tensors")
