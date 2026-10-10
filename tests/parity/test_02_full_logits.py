"""Block 1: full-model logits of three implementations agree on the same text.

  ours    jaxpt.models.qwen3   pure functions, 28 layers stacked and run with lax.scan
  Tunix   tunix.models.qwen3   Flax NNX modules, layers in a Python loop
  HF      transformers         the reference (fp32, CPU, eager attention)

Run on the pod:  XLA_PYTHON_CLIENT_PREALLOCATE=false pytest -s tests/parity/test_02_full_logits.py
"""
import time

import jax
import jax.numpy as jnp
import numpy as np

from jaxpt.models import load as jload
from jaxpt.models import qwen3 as ours

from hf_reference import REPO, capture, compare

jax.config.update("jax_default_matmul_precision", "highest")    # full fp32 on Ampere, for parity


def tunix_model(path):
    from tunix.models.qwen3 import model as tmodel, params as tparams
    cfg = tmodel.ModelConfig.qwen3_0p6b_base()
    return tparams.create_model_from_safe_tensors(path, cfg, mesh=None, dtype=jnp.float32)


def test_full_logits_three_ways():
    rec = capture()
    ids = jnp.asarray(rec["ids"])
    T = ids.shape[1]
    positions = jnp.arange(T)[None, :]
    print(f"\n  {REPO}: {T} tokens, fp32, highest matmul precision")

    # ours
    cfg, params = jload.load(REPO, dtype=jnp.float32)
    t0 = time.time()
    logits_ours = ours.forward(params, cfg, ids, positions, jnp.ones_like(ids, bool), dtype=jnp.float32)
    logits_ours.block_until_ready()
    print(f"  ours:  first call (compile + run) {time.time() - t0:.1f}s")

    # Tunix
    model = tunix_model(jload.fetch(REPO))
    mask = jnp.tril(jnp.ones((1, T, T), bool))
    t0 = time.time()
    logits_tunix, _ = model(ids, positions, None, mask)
    logits_tunix.block_until_ready()
    print(f"  Tunix: first call (trace + run, eager NNX) {time.time() - t0:.1f}s")

    ok = compare("ours  vs HF   (logits)", logits_ours, rec["logits"], tol=2e-3)
    ok &= compare("Tunix vs HF   (logits)", logits_tunix, rec["logits"], tol=2e-3)
    ok &= compare("ours  vs Tunix (logits)", logits_ours, logits_tunix, tol=2e-3)
    same_next = np.mean(np.argmax(np.asarray(logits_ours), -1) == np.argmax(rec["logits"], -1))
    print(f"  next-token argmax agreement with HF: ours {same_next:.1%}, "
          f"Tunix {np.mean(np.argmax(np.asarray(logits_tunix), -1) == np.argmax(rec['logits'], -1)):.1%}")
    assert ok
