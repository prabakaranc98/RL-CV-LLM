"""Warm-up demonstrations for X3: the oracle's episodes (make_tasks.plan) on the training patients' tasks.

Each demo is the list of agent actions the oracle takes: the task's lookup (with _count=5000, as the grader queries),
the POSTs it requires, and FINISH with the reference answer. train_agentic_grpo.py --sft-demos replays them through
the environment, so the replies are the real FHIR responses, and keeps a demo only if it passes the official grader
and the lookup gate. Sources: the 208 official tasks of the 66 training patients, plus --per-type generated tasks on
demographics-only patients for the four types that need no lab data (task1 / 2 / 3 / 8).

    python build_demos.py --mab /root/MedAgentBench --out /root/mab_train/demos.jsonl
"""
import argparse
import json
import random
import sys

sys.path.insert(0, "/workspace/RL-CV-LLM/src")
sys.path.insert(0, "/workspace/RL-CV-LLM/experiments/x3_mab_agentic")
from jaxpt.envs.medagentbench import Benchmark  # noqa: E402
import make_tasks  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mab", default="/root/MedAgentBench")
    ap.add_argument("--train", default="/root/mab_train/train_official.json")
    ap.add_argument("--generated", default="/root/mab_train/train_tasks.json")
    ap.add_argument("--per-type", type=int, default=30)
    ap.add_argument("--out", default="/root/mab_train/demos.jsonl")
    args = ap.parse_args()
    bench = Benchmark(args.mab)
    patients = {p["mrn"]: p for p in make_tasks.all_patients(bench.get, bench.base)}
    cases = json.load(open(args.train))
    gen = json.load(open(args.generated))
    rng = random.Random(0)
    for t in ("task1", "task2", "task3", "task8"):
        pool = [x for x in gen if x["id"].split("_")[0] == t]
        cases += rng.sample(pool, min(args.per_type, len(pool)))
    n = 0
    with open(args.out, "w") as f:
        for case in cases:
            gets, posts, answer, _ = make_tasks.plan(case, bench.base, bench.get, patients)
            actions = [f"GET {u.replace('&_format=json', '')}" for u in gets]
            actions += [f"POST {u}\n{json.dumps(p)}" for u, p in posts]
            actions.append(f"FINISH({json.dumps(answer)})")
            f.write(json.dumps({"case": case, "actions": actions}) + "\n")
            n += 1
    print(f"wrote {n} demos ({len(json.load(open(args.train)))} official training tasks + generated) to {args.out}")


if __name__ == "__main__":
    main()
