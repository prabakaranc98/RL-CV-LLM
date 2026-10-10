"""Block 2: Tunix's sampler (plus stop strings) on MedCalc prompts, checked against vLLM's E0 answers.

1. Prompt lengths of the X1 variant (train and test). They set max_prompt_length and the KV-cache size, since the
   Tunix sampler left-pads every prompt to max_prompt_length (static shapes, one compile).
2. Greedy on n test problems: does Tunix give the answers vLLM gave in E0? Both run bf16, so long greedy texts
   may drift apart; the extracted answer and the reward are what must agree.
3. Throughput at GRPO shape: P prompts x G samples at temperature 1, against vLLM's sampled tokens/s in E0.

Run on a pod (needs E0's answers.jsonl for the same variant in results/):
  XLA_PYTHON_CLIENT_PREALLOCATE=false python experiments/x1_medcalc_grpo/b2_rollout.py --shots per-calculator
"""
import argparse
import dataclasses
import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from transformers import AutoTokenizer

from jaxpt.envs import medcalc
from jaxpt.models import load as jload
from jaxpt.rollout.stop_strings import StopStringRollout

REPO = "Qwen/Qwen3-0.6B-Base"
RESULTS = Path(__file__).parent / "results"


def length_stats(name, lengths):
    q = np.percentile(lengths, [50, 90, 99, 100]).astype(int)
    fits = {c: float(np.mean(np.asarray(lengths) <= c)) for c in (1024, 1536, 2048, 3072)}
    print(f"  {name:5} n={len(lengths):6}  p50 {q[0]}  p90 {q[1]}  p99 {q[2]}  max {q[3]}  "
          + "  ".join(f"<={c}: {v:.1%}" for c, v in fits.items()))
    return {"p50": int(q[0]), "p90": int(q[1]), "p99": int(q[2]), "max": int(q[3]), "fits": fits}


def tunix_rollout(max_prompt, max_new):
    from tunix.models.qwen3 import model as tmodel, params as tparams
    from tunix.rl.rollout.base_rollout import RolloutConfig
    cfg = dataclasses.replace(tmodel.ModelConfig.qwen3_0p6b_base(), dtype=jnp.bfloat16, param_dtype=jnp.bfloat16)
    model = tparams.create_model_from_safe_tensors(jload.fetch(REPO), cfg, mesh=None, dtype=jnp.bfloat16)
    base = RolloutConfig(max_prompt_length=max_prompt, max_tokens_to_generate=max_new,
                         kv_cache_size=max_prompt + max_new, eos_tokens=None)
    rollout = StopStringRollout(model, AutoTokenizer.from_pretrained(REPO), rollout_config=base, stop=medcalc.STOP)
    return rollout, base


def generate(rollout, cfg, prompts, batch):
    """Fixed-size batches (the last one padded with repeats), so every call reuses one compiled shape."""
    texts, stats, t0 = [], [], time.time()
    for i in range(0, len(prompts), batch):
        chunk = prompts[i:i + batch]
        out = rollout.generate(chunk + [chunk[-1]] * (batch - len(chunk)), cfg)
        texts += out.text[:len(chunk)]
        stats.append(rollout.last)
    return texts, stats, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shots", default="per-calculator", choices=["generic", "per-calculator"])
    ap.add_argument("--input", default="note", choices=["note", "values"])
    ap.add_argument("--n", type=int, default=64, help="test problems for the greedy check")
    ap.add_argument("--batch", type=int, default=16, help="rows per sampler call (greedy and sampling)")
    ap.add_argument("--prompts", type=int, default=8, help="P, prompts per GRPO step")
    ap.add_argument("--group", type=int, default=8, help="G, samples per prompt")
    ap.add_argument("--max-prompt", type=int, default=2048)
    ap.add_argument("--max-new", type=int, default=1024)
    args = ap.parse_args()
    tag = f"{args.shots}_{args.input}"
    prompt_of = medcalc.prompter(args.shots, args.input)
    tok = AutoTokenizer.from_pretrained(REPO)
    report = {"variant": tag, "args": vars(args)}

    # 1. prompt lengths
    print(f"1. prompt tokens, {tag}")
    train, test = medcalc.load("train"), medcalc.load("test")
    lens = {split: [len(ids) for ids in tok([prompt_of(e) for e in exs])["input_ids"]]
            for split, exs in (("train", train), ("test", test))}
    report["prompt_tokens"] = {s: length_stats(s, l) for s, l in lens.items()}

    # 2. greedy agreement with vLLM (E0) on the first n test problems that fit
    e0 = {r["id"]: r for r in map(json.loads, open(RESULTS / f"e0_Qwen3-0.6B-Base_verified_{tag}" / "answers.jsonl"))}
    picked = [e for e, n in zip(test, lens["test"]) if n <= args.max_prompt][:args.n]
    rollout, base = tunix_rollout(args.max_prompt, args.max_new)
    greedy_cfg = dataclasses.replace(base, temperature=0.0, top_p=None)
    texts, stats, wall = generate(rollout, greedy_cfg, [prompt_of(e) for e in picked], args.batch)
    rows = []
    for e, t in zip(picked, texts):
        r, info = medcalc.reward(t, e)
        v = e0[e.id]
        rows.append({"id": e.id, "tunix_reward": r, "vllm_reward": v["reward"], "same_text": t == v["text"],
                     "same_answer": info["parsed"] == v["parsed"], "tunix_parsed": info["parsed"],
                     "vllm_parsed": v["parsed"]})
    agree = {k: float(np.mean([r[k] for r in rows])) for k in ("same_text", "same_answer")}
    acc = {k: float(np.mean([r[k] for r in rows])) for k in ("tunix_reward", "vllm_reward")}
    print(f"\n2. greedy on {len(rows)} test problems (batch {args.batch}, includes compile): {wall:.0f}s")
    print(f"  same text {agree['same_text']:.1%}, same extracted answer {agree['same_answer']:.1%}; "
          f"accuracy Tunix {acc['tunix_reward']:.1%} vs vLLM {acc['vllm_reward']:.1%}")
    print(f"  stop-string rate {np.mean([s['stop_string_rate'] for s in stats]):.1%}, "
          f"hit max tokens {np.mean([s['hit_max_tokens_rate'] for s in stats]):.1%}")
    report["greedy"] = {"n": len(rows), "wall_s": wall, **agree, **acc, "rows": rows}

    # 3. GRPO-shaped sampling: P prompts x G samples at temperature 1, in micro-batches of --batch rows
    #    (Tunix's rollout_micro_batch_size); the first pass compiles, the second is timed.
    sample_cfg = dataclasses.replace(base, temperature=1.0, top_p=1.0, seed=0)
    group = [prompt_of(e) for e in picked[:args.prompts] for _ in range(args.group)]
    B = len(group)
    _, _, first = generate(rollout, sample_cfg, group, args.batch)
    _, stats, wall = generate(rollout, sample_cfg, group, args.batch)
    s = {k: (sum if k.endswith("tokens") else np.mean)([x[k] for x in stats]) for k in stats[0]}
    vllm = json.load(open(RESULTS / f"e0_Qwen3-0.6B-Base_verified_{tag}" / "summary.json"))["systems"]
    print(f"\n3. sampling {args.prompts} prompts x {args.group} = {B} rows in batches of {args.batch}, max_new {args.max_new}")
    print(f"  first call (compile + run) {first:.0f}s; timed call {wall:.1f}s")
    print(f"  kept tokens {s['kept_tokens']:,} -> {s['kept_tokens'] / wall:,.0f} tok/s "
          f"(vLLM E0 sampled: {vllm['sampled_gen_tok_per_s']:,.0f} tok/s)")
    print(f"  decoded tokens {s['decoded_tokens']:,}; stop-string rate {s['stop_string_rate']:.1%}; "
          f"hit max tokens {s['hit_max_tokens_rate']:.1%}")
    report["sampling"] = {"B": B, "batch": args.batch, "first_call_s": first, "wall_s": wall, **s,
                          "kept_tok_per_s": s["kept_tokens"] / wall, "vllm_sampled_tok_per_s": vllm["sampled_gen_tok_per_s"]}

    path = RESULTS / f"b2_rollout_{tag}.json"
    path.write_text(json.dumps(report, indent=1, default=str))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
