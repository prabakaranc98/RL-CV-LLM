"""Walkthrough step 1: the token embedding and RMSNorm match Hugging Face.

Run on the pod:  pytest -s tests/parity/test_01_embedding_rmsnorm.py
"""
import jax
import jax.numpy as jnp
import numpy as np

from jaxpt.models.load import load
from jaxpt.models.qwen3 import rms_norm

from hf_reference import REPO, capture, compare

# fp32 matmuls on Ampere default to TF32 (about 3 significant digits); parity needs full fp32.
jax.config.update("jax_default_matmul_precision", "highest")


def test_embedding_and_rmsnorm():
    rec = capture()
    cfg, params = load(REPO, dtype=jnp.float32)
    ids = rec["ids"]
    print(f"\n  model {REPO}: {cfg.layers} layers, width {cfg.d}, vocab {cfg.vocab}; {ids.shape[1]} tokens")

    # 1. Embedding: a table lookup, so the result must be bit-identical.
    emb = params["embed"][ids]
    ok = compare("embedding (table lookup)", emb, rec["embed.out"], tol=0.0)

    # 2. RMSNorm: x / sqrt(mean(x^2) + eps) * weight, before attention and before the MLP of layer 0,
    #    and the final norm after the last layer.
    for name, w in (("ln1", params["layers"]["ln1"][0]), ("ln2", params["layers"]["ln2"][0]), ("norm", params["norm"])):
        y = rms_norm(jnp.asarray(rec[f"{name}.in"]), w, cfg.eps)
        ok &= compare(f"RMSNorm {name}", y, rec[f"{name}.out"], tol=1e-5)

    # 3. Qwen3's QK-norm: the same RMSNorm, applied to each head of q and k (weights of size head_dim).
    for name, w in (("q_norm", params["layers"]["q_norm"][0]), ("k_norm", params["layers"]["k_norm"][0])):
        y = rms_norm(jnp.asarray(rec[f"{name}.in"]), w, cfg.eps)
        ok &= compare(f"QK-norm {name} (per head)", y, rec[f"{name}.out"], tol=1e-5)

    assert ok
