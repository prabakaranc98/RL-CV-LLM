"""Device memory of one update micro-step, GRPO loss vs OPD loss, at the training shape (diagnosis of X2's OOM).

Compiles jax.value_and_grad of each loss (Tunix's grpo_loss_fn, our opd_loss_fn) over the actor's state, on one
8-row micro-batch of 1,024 prompt + 1,024 completion tokens, and prints XLA's memory analysis (arguments, outputs,
temporaries). Same model setup as the runs: Qwen3-0.6B-Base, fp32 weights, bf16 compute, block remat.

  python experiments/x2_medcalc_opd/bench_loss_memory.py [--no-chunked]
"""
import argparse

import jax
import jax.numpy as jnp
from flax import nnx

from jaxpt.algorithms import opd
from jaxpt.train import chunked_logps, tunix_rl
from tunix.rl import algo_core
from tunix.rl.grpo.grpo_learner import GRPOConfig, TrainExample

REPO = "Qwen/Qwen3-0.6B-Base"
PAD = EOS = 151643


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chunked", action="store_true")
    ap.add_argument("--rows", type=int, default=8)
    args = ap.parse_args()
    if not args.no_chunked:
        chunked_logps.install()
    model = tunix_rl.load_tunix_qwen3(REPO, param_dtype=jnp.float32, remat=True)
    graphdef, state = nnx.split(model)
    k = jax.random.key(0)
    B, P, C = args.rows, 1024, 1024
    tokens = lambda n, i: jax.random.randint(jax.random.fold_in(k, i), (B, n), 0, PAD)
    ex = TrainExample(prompt_ids=tokens(P, 1), prompt_mask=jnp.ones((B, P), bool), completion_ids=tokens(C, 2),
                      completion_mask=jnp.ones((B, C), jnp.int32), advantages=jnp.linspace(-1, 1, B),
                      ref_per_token_logps=-jnp.ones((B, C)), old_per_token_logps=None)
    cfgs = {"grpo": (algo_core.grpo_loss_fn, GRPOConfig(num_generations=8, beta=0.04, kl_loss_mode="low_var_kl",
                                                         loss_agg_mode="sequence-mean-token-mean")),
            "opd": (opd.opd_loss_fn, GRPOConfig(num_generations=8, beta=1.0, policy_loss_fn="opd",
                                                 loss_agg_mode="sequence-mean-token-mean"))}
    print(f"{B} rows x ({P} + {C}), chunked log-probs: {not args.no_chunked}, {jax.devices()[0].device_kind}")
    for name, (loss_fn, cfg) in cfgs.items():
        cfg.temperature = 1.0
        f = jax.jit(jax.value_and_grad(lambda st: loss_fn(nnx.merge(graphdef, st), ex, cfg, pad_id=PAD, eos_id=EOS)[0]))
        m = f.lower(state).compile().memory_analysis()
        gib = lambda x: x / 2**30
        print(f"  {name:4}  args {gib(m.argument_size_in_bytes):5.1f}  out {gib(m.output_size_in_bytes):5.1f}  "
              f"temp {gib(m.temp_size_in_bytes):5.1f} GiB")


if __name__ == "__main__":
    main()
