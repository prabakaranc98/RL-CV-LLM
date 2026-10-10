"""E0: the base model's MedCalc baseline, before any training (vLLM inference).

What it measures
  * greedy accuracy on all 1,100 test cases, overall and by category / calculator / answer type / note type;
  * format-valid rate, truncation rate, response length;
  * pass@k (k = 1, 2, 4, 8; 8 samples at temperature 1) on a 5-per-calculator subset, and how many of those
    prompts sit in the "learnable band" (0 < correct < 8) where GRPO gets a gradient;
  * a constant-guess baseline per calculator (the typical training answer), to expose reward hacking later;
  * systems: load time, generation time, prompt and generated tokens/s, GPU memory, dollar cost.

Run on the pod (vLLM venv):
  python experiments/x1_medcalc_grpo/e0_vllm.py --data verified --shots per-calculator --wandb
Variants: --input note|values (read the note, or get the gold extracted values), --data verified|v1.2 (label version),
--shots generic|per-calculator (one fixed MAP example for every
question, or the benchmark's own worked example of the same calculator, which also shows that calculator's formula).
Outputs: experiments/x1_medcalc_grpo/results/e0_<model>_<data>_<shots>_<input>/{answers.jsonl, passk.jsonl, summary.json}
"""
import argparse
import collections
import json
import math
import os
import statistics as st
import time
from jaxpt.envs import medcalc
from jaxpt.tracking.text import is_degenerate

GPU_PRICE = {"NVIDIA A40": 0.59, "NVIDIA A100-SXM4-80GB": 1.79}   # $/h, RunPod Secure (2026-10-09)


def pass_at_k(n, c, k):
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    return 1.0 if n - c < k else 1.0 - math.comb(n - c, k) / math.comb(n, k)


def constant_guess(train, test):
    """Accuracy of always answering the typical training value per calculator (median; mode for integers)."""
    by_calc = collections.defaultdict(list)
    for e in train:
        by_calc[e.calculator].append(e)
    hits = 0
    for e in test:
        pool = by_calc[e.calculator]
        if e.output_type == "integer":
            guess = collections.Counter(x.answer for x in pool).most_common(1)[0][0]
        elif e.output_type == "decimal":
            guess = str(st.median(float(x.answer) for x in pool))
        else:
            guess = collections.Counter(x.answer for x in pool).most_common(1)[0][0]
        hits += medcalc.reward(f"\\boxed{{{guess}}}", e)[0]
    return hits / len(test)


def outcome(reward, info, truncated, text, ex):
    """One label per answer, so failures can be counted by cause, not just as 'wrong'."""
    if reward == 1.0:
        return "correct"
    if info["valid"]:
        try:
            gold = float(ex.answer)
            rel = abs(float(info["parsed"]) - gold) / max(abs(gold), 1e-9)
            return "wrong value (<=10% off)" if rel <= 0.10 else "wrong value (<=50% off)" if rel <= 0.5 else "wrong value (>50% off)"
        except (TypeError, ValueError):
            return "wrong value (date / weeks)"
    if info["parsed"] is not None:
        return "bad format in box"
    if truncated:
        return "no answer: degenerate loop" if is_degenerate(text) else "no answer: truncated (no loop)"
    return "no answer: stopped without box"


def grouped(records, key):
    acc = collections.defaultdict(list)
    for r in records:
        acc[r[key]].append(r["reward"])
    return {k: {"n": len(v), "accuracy": sum(v) / len(v)} for k, v in sorted(acc.items())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base", help="Hub id or a local HF checkpoint directory")
    ap.add_argument("--name", default=None, help="label for outputs; default: the model id's last part")
    ap.add_argument("--data", default="verified", choices=["verified", "v1.2"])
    ap.add_argument("--shots", default="generic", choices=["generic", "per-calculator"])
    ap.add_argument("--input", default="note", choices=["note", "values"],
                    help="note: extract the values from the patient note; values: given the gold extracted values")
    ap.add_argument("--max-new", type=int, default=1024,
                    help="1,024: worked solutions for scoring calculators are long; 512 truncated 36% of one condition")
    ap.add_argument("--max-model-len", type=int, default=8192,
                    help="prompt + generation limit; same-calculator examples reach 4,567 prompt tokens (model window 32K)")
    ap.add_argument("--passk-per-calc", type=int, default=5, help="0: greedy only (checkpoint curves)")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--mean-k", type=int, default=0,
                    help=">0: also sample the full test this many times at the --sample-* setting; report mean accuracy")
    ap.add_argument("--sample-temperature", type=float, default=0.7, help="Qwen3 card, non-thinking: T 0.7")
    ap.add_argument("--sample-top-p", type=float, default=0.8, help="Qwen3 card, non-thinking: top-p 0.8")
    ap.add_argument("--sample-top-k", type=int, default=20, help="Qwen3 card, non-thinking: top-k 20")
    ap.add_argument("--wandb", action="store_true")
    args = ap.parse_args()
    tag = f"{args.name or args.model.rstrip('/').split('/')[-1]}_{args.data}_{args.shots}_{args.input}"
    out = os.path.join(os.path.dirname(__file__), "results", f"e0_{tag}")
    os.makedirs(out, exist_ok=True)

    import torch
    from vllm import LLM, SamplingParams, __version__ as vllm_version

    train, test = medcalc.load("train", args.data), medcalc.load("test", args.data)
    if args.shots == "per-calculator":
        shots = medcalc.per_calculator_shots(medcalc.load("one_shot", "verified"))
        shot_of = lambda e: shots[e.calculator]
    else:
        generic = medcalc.default_shot(train)
        shot_of = lambda e: generic
    prompts = [medcalc.build_prompt(e, shot_of(e), args.input) for e in test]

    t0 = time.time()
    llm = LLM(model=args.model, dtype="bfloat16", max_model_len=args.max_model_len, gpu_memory_utilization=0.85, seed=0)
    load_s = time.time() - t0
    free, total = torch.cuda.mem_get_info()

    # ---- greedy pass over the full test set
    greedy = SamplingParams(temperature=0.0, max_tokens=args.max_new, stop=medcalc.STOP)
    t0 = time.time()
    outs = llm.generate(prompts, greedy)
    greedy_s = time.time() - t0
    records = []
    for e, o in zip(test, outs):
        c = o.outputs[0]
        r, info = medcalc.reward(c.text, e)
        records.append({"id": e.id, "calculator": e.calculator, "category": e.category, "output_type": e.output_type,
                        "note_type": e.note_type, "gold": e.answer, "lower": e.lower, "upper": e.upper,
                        "parsed": info["parsed"], "reward": r, "valid": info["valid"], "reason": info["reason"],
                        "truncated": c.finish_reason == "length",
                        "outcome": outcome(r, info, c.finish_reason == "length", c.text, e),
                        "prompt_tokens": len(o.prompt_token_ids),
                        "gen_tokens": len(c.token_ids), "text": c.text})
    with open(os.path.join(out, "answers.jsonl"), "w") as f:
        for r in records:
            f.write(json.dumps(r, default=str) + "\n")

    # ---- pass@k on a balanced subset (the first few test cases of each calculator)
    per_calc = collections.defaultdict(list)
    for e in test:
        if len(per_calc[e.calculator]) < args.passk_per_calc:
            per_calc[e.calculator].append(e)
    subset = [e for v in per_calc.values() for e in v]
    sampled = SamplingParams(n=args.k, temperature=1.0, top_p=1.0, max_tokens=args.max_new, stop=medcalc.STOP, seed=0)
    t0 = time.time()
    souts = llm.generate([medcalc.build_prompt(e, shot_of(e), args.input) for e in subset], sampled) if subset else []
    sampled_s = time.time() - t0
    correct = []
    with open(os.path.join(out, "passk.jsonl"), "w") as f:
        for e, o in zip(subset, souts):
            rs = [medcalc.reward(c.text, e)[0] for c in o.outputs]
            correct.append(int(sum(rs)))
            f.write(json.dumps({"id": e.id, "calculator": e.calculator, "correct": int(sum(rs)), "n": args.k}) + "\n")

    # ---- sampled decoding on the full test (the model card's recommended setting): mean accuracy over mean_k samples
    sampled_full = None
    if args.mean_k > 0:
        mk = SamplingParams(n=args.mean_k, temperature=args.sample_temperature, top_p=args.sample_top_p,
                            top_k=args.sample_top_k, max_tokens=args.max_new, stop=medcalc.STOP, seed=0)
        mouts = llm.generate(prompts, mk)
        per_sample = [[medcalc.reward(c.text, e)[0] for c in o.outputs] for e, o in zip(test, mouts)]
        accs = [sum(rs[j] for rs in per_sample) / len(per_sample) for j in range(args.mean_k)]
        sampled_full = {"n": args.mean_k, "temperature": args.sample_temperature, "top_p": args.sample_top_p,
                        "top_k": args.sample_top_k, "mean_accuracy": sum(accs) / len(accs),
                        "per_sample_accuracy": accs,
                        "any_correct": sum(any(rs) for rs in per_sample) / len(per_sample)}

    # ---- summary
    gen_tok = sum(r["gen_tokens"] for r in records)
    prm_tok = sum(r["prompt_tokens"] for r in records)
    s_gen_tok = sum(len(c.token_ids) for o in souts for c in o.outputs)
    gpu = torch.cuda.get_device_name(0)
    wall_h = (load_s + greedy_s + sampled_s) / 3600
    summary = {
        "model": args.model, "data": args.data, "shots": args.shots, "input": args.input, "n_test": len(records),
        "accuracy": sum(r["reward"] for r in records) / len(records),
        "format_valid_rate": sum(r["valid"] for r in records) / len(records),
        "truncation_rate": sum(r["truncated"] for r in records) / len(records),
        "outcomes": dict(collections.Counter(r["outcome"] for r in records).most_common()),
        "degenerate_rate": sum(r["outcome"] == "no answer: degenerate loop" for r in records) / len(records),
        "max_new": args.max_new,
        "gen_tokens_mean": gen_tok / len(records), "gen_tokens_median": st.median(r["gen_tokens"] for r in records),
        "constant_guess_accuracy": constant_guess(train, test),
        "by_category": grouped(records, "category"), "by_output_type": grouped(records, "output_type"),
        "by_note_type": grouped(records, "note_type"), "by_calculator": grouped(records, "calculator"),
        "passk_subset": {"n_prompts": len(subset), "samples_per_prompt": args.k,
                         **{f"pass@{k}": sum(pass_at_k(args.k, c, k) for c in correct) / len(correct) for k in (1, 2, 4, 8)},
                         "all_wrong": sum(c == 0 for c in correct), "all_right": sum(c == args.k for c in correct),
                         "learnable_band": sum(0 < c < args.k for c in correct)} if correct else None,
        "sampled_full": sampled_full,
        "systems": {"gpu": gpu, "vllm": vllm_version, "load_s": load_s, "greedy_s": greedy_s, "sampled_s": sampled_s,
                    "greedy_prompt_tok_per_s": prm_tok / greedy_s, "greedy_gen_tok_per_s": gen_tok / greedy_s,
                    "sampled_gen_tok_per_s": s_gen_tok / sampled_s if subset else None,
                    "gpu_mem_used_gb_after_load": (total - free) / 1e9, "gpu_mem_total_gb": total / 1e9,
                    "cost_usd": wall_h * GPU_PRICE.get(gpu, float("nan"))},
    }
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps({k: v for k, v in summary.items() if not k.startswith("by_")}, indent=1))

    if args.wandb:
        import wandb
        run = wandb.init(project="rl-cv-llm", name=f"e0-{tag}", tags=["E0", "medcalc", args.data, args.shots, args.input], config=vars(args))
        flat = {k: v for k, v in summary.items() if isinstance(v, (int, float))}
        flat.update({f"passk/{k}": v for k, v in summary["passk_subset"].items()})
        flat.update({f"systems/{k}": v for k, v in summary["systems"].items() if isinstance(v, (int, float))})
        run.summary.update(flat)
        rows = [[k, v["n"], v["accuracy"]] for k, v in summary["by_calculator"].items()]
        run.log({"accuracy_by_calculator": wandb.Table(columns=["calculator", "n", "accuracy"], data=rows)})
        run.finish()


if __name__ == "__main__":
    main()
