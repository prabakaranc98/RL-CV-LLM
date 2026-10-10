"""Block 1b: forward throughput, our Qwen3 (lax.scan over layers) vs Tunix's (Flax NNX, Python loop over layers).

Same setting for both:
  - fp32 master weights, bf16 compute, one jitted function;
  - output = log-prob of each next token, which is what GRPO scoring needs,
    so both pay the same fp32 log-softmax over the 152K vocabulary;
  - random tokens, no padding (padding does not change the work).
Reports compile time, step time, tokens/s, MFU (analytic FLOPs / bf16 peak) and device memory.

Run on a pod:
  XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/x1_medcalc_grpo/bench_forward.py --shapes 1x2048 4x2048 8x2048
"""
import argparse
import dataclasses
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from jaxpt.models import load as jload
from jaxpt.models import qwen3 as ours
from jaxpt.tracking.perf import measure

REPO = "Qwen/Qwen3-0.6B-Base"
OUT = Path(__file__).parent / "results"


def next_token_logp(logits, tokens):
    """[B, T, V] logits, [B, T] tokens -> [B, T-1] log p(token t+1 | tokens <= t), in fp32."""
    logp = jax.nn.log_softmax(logits.astype(jnp.float32), axis=-1)
    return jnp.take_along_axis(logp[:, :-1], tokens[:, 1:, None], axis=-1)[..., 0]


def ours_scorer():
    cfg, params = jload.load(REPO, dtype=jnp.float32)

    @jax.jit
    def score(params, tokens, positions, valid):
        return next_token_logp(ours.forward(params, cfg, tokens, positions, valid, dtype=jnp.bfloat16), tokens)

    return score, params


def tunix_scorer():
    from flax import nnx
    from tunix.models.qwen3 import model as tmodel, params as tparams
    cfg = dataclasses.replace(tmodel.ModelConfig.qwen3_0p6b_base(), dtype=jnp.bfloat16, param_dtype=jnp.float32)
    model = tparams.create_model_from_safe_tensors(jload.fetch(REPO), cfg, mesh=None, dtype=jnp.float32)
    graphdef, state = nnx.split(model)      # NNX module -> (static structure, params pytree), so plain jax.jit works

    @jax.jit
    def score(state, tokens, positions, valid):
        T = tokens.shape[1]
        mask = jnp.tril(jnp.ones((T, T), bool))[None] & valid[:, None, :]
        logits, _ = nnx.merge(graphdef, state)(tokens, positions, None, mask)
        return next_token_logp(logits, tokens)

    return score, state


def inputs(B, T, seed=0):
    tokens = jax.random.randint(jax.random.key(seed), (B, T), 0, 151_643)    # below the special-token ids
    positions = jnp.broadcast_to(jnp.arange(T), (B, T))
    return tokens, positions, jnp.ones((B, T), bool)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["1x2048", "4x2048", "8x2048"], help="BxT")
    ap.add_argument("--iters", type=int, default=10)
    args = ap.parse_args()
    shapes = [tuple(map(int, s.split("x"))) for s in args.shapes]
    gpu = jax.devices()[0].device_kind
    print(f"{gpu}; {REPO}; fp32 weights, bf16 compute; next-token log-probs")

    impls = {"ours (scan)": ours_scorer(), "Tunix (NNX loop)": tunix_scorer()}
    cfg = jload.load_config(REPO)

    # Same inputs -> same log-probs (bf16 compute, so close, not identical).
    x = inputs(1, 256)
    a, b = (fn(p, *x) for fn, p in impls.values())
    print(f"agreement on 1x256: max |d logp| {float(jnp.max(jnp.abs(a - b))):.3f}, "
          f"mean {float(jnp.mean(jnp.abs(a - b))):.4f}")

    rows = []
    print(f"\n{'impl':18} {'BxT':>7} {'compile s':>9} {'step ms':>8} {'tok/s':>9} {'TFLOP/s':>8} {'MFU':>6} {'mem GB':>7}")
    for name, (fn, p) in impls.items():
        for B, T in shapes:
            r = measure(fn, p, *inputs(B, T), flops=B * T * ours.flops_per_token(cfg, T), iters=args.iters)
            r.update(impl=name, B=B, T=T, tokens_per_s=B * T / r["step_s"])
            rows.append(r)
            mfu = f"{r['mfu']:.1%}" if r["mfu"] is not None else "-"
            print(f"{name:18} {f'{B}x{T}':>7} {r['compile_s']:9.1f} {r['step_s'] * 1e3:8.1f} "
                  f"{r['tokens_per_s']:9,.0f} {r['tflops_per_s']:8.1f} {mfu:>6} {r['memory_gb']:7.1f}")

    OUT.mkdir(exist_ok=True)
    path = OUT / f"bench_forward_{gpu.replace(' ', '-')}.json"
    path.write_text(json.dumps({"gpu": gpu, "model": REPO, "rows": rows}, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
