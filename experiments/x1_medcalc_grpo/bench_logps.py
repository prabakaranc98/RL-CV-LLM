"""Tunix's per-token log-probs vs jaxpt.train.chunked_logps, at the training shape. Equal answers, less memory?

Two jitted functions per implementation, on Qwen3-0.6B-Base (fp32 weights, bf16 compute, block remat, as the actor):
  train:  value_and_grad of  -mean(log p) - 0.01 mean(entropy)  over `--rows` rows (the shape of one micro-step)
  score:  log-probs without gradient over 2 x rows (the reference / teacher pass)
Reports, for each: compile time, step time, device memory of the executable; and between the two implementations:
max |d log p|, max |d entropy|, and the relative difference of the gradients.

  XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/x1_medcalc_grpo/bench_logps.py --rows 8
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx
from tunix.rl import algo_core

from jaxpt.tracking.perf import measure
from jaxpt.train import chunked_logps, tunix_rl

REPO = "Qwen/Qwen3-0.6B-Base"
PAD = EOS = 151643


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=8)
    ap.add_argument("--prompt", type=int, default=1024)
    ap.add_argument("--completion", type=int, default=1024)
    ap.add_argument("--chunk", type=int, default=128)
    args = ap.parse_args()
    model = tunix_rl.load_tunix_qwen3(REPO, param_dtype=jnp.float32, remat=True)
    graphdef, state = nnx.split(model)
    key = jax.random.key(0)
    make = lambda rows, n, k: jax.random.randint(jax.random.fold_in(key, k), (rows, n), 0, PAD)
    train_in = (make(args.rows, args.prompt, 1), make(args.rows, args.completion, 2))
    score_in = (make(2 * args.rows, args.prompt, 3), make(2 * args.rows, args.completion, 4))

    def build():
        logps = chunked_logps.common.compute_per_token_logps         # whichever is installed at trace time

        def loss(st, prompt, completion):
            lp, x = logps(graphdef, st, prompt, completion, PAD, EOS, stop_gradient=False, return_logits=True)
            ent = algo_core.compute_entropy_from_logits(x)
            return -lp.mean() - 0.01 * ent.mean(), (lp, ent)

        train = jax.jit(jax.value_and_grad(loss, has_aux=True))
        score = jax.jit(lambda st, prompt, completion: logps(graphdef, st, prompt, completion, PAD, EOS))
        return train, score

    results, outputs = {}, {}
    for name, swap in (("tunix", chunked_logps.uninstall), ("chunked", chunked_logps.install)):
        swap()
        train, score = build()
        results[name] = {"train": measure(train, state, *train_in, iters=3), "score": measure(score, state, *score_in, iters=3)}
        (_, (lp, ent)), grads = train(state, *train_in)
        outputs[name] = (np.asarray(lp), np.asarray(ent), grads)
        jax.clear_caches()
    chunked_logps.uninstall()

    (lp_a, ent_a, g_a), (lp_b, ent_b, g_b) = outputs["tunix"], outputs["chunked"]
    gnorm = lambda t: float(jnp.sqrt(sum(jnp.sum(jnp.square(x.astype(jnp.float32))) for x in jax.tree.leaves(t))))
    agree = {"max_abs_dlogp": float(np.abs(lp_a - lp_b).max()), "mean_abs_dlogp": float(np.abs(lp_a - lp_b).mean()),
             "max_abs_dentropy": float(np.abs(ent_a - ent_b).max()),
             "grad_rel_diff": gnorm(jax.tree.map(lambda a, b: a - b, g_a, g_b)) / gnorm(g_a)}

    print(f"\nQwen3-0.6B-Base, {args.rows} rows x ({args.prompt} prompt + {args.completion} completion), "
          f"chunk {args.chunk}, {jax.devices()[0].device_kind}")
    print(f"  {'':10} {'train GB':>9} {'train s':>8} {'score GB':>9} {'score s':>8} {'compile s':>9}")
    for name, r in results.items():
        print(f"  {name:10} {r['train']['memory_gb']:9.1f} {r['train']['step_s']:8.2f} "
              f"{r['score']['memory_gb']:9.1f} {r['score']['step_s']:8.2f} {r['train']['compile_s']:9.1f}")
    print("  agreement: " + ", ".join(f"{k} {v:.2e}" for k, v in agree.items()))
    out = Path(__file__).parent / "results" / f"bench_logps_{jax.devices()[0].device_kind.replace(' ', '-')}.json"
    out.write_text(json.dumps({"args": vars(args), "results": results, "agreement": agree}, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
