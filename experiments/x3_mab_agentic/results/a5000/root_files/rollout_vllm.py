"""Run a policy served by vLLM through the MedAgentBench environment: evaluation and RL-signal measurement.

The environment is jaxpt.envs.medagentbench (replay_check.py: identical to the official harness on 1,500 recorded
episodes); the request matches the official HTTPAgent (chat roles user / assistant, thinking off). Each task runs
--samples times at --temperature; per episode we record the official grade, the lookup-gated training reward, and the
actions taken.

Signal for GRPO: a task's group of G samples carries a gradient only if the rewards in it are not all equal, so the
share of tasks whose G samples include at least one reward-1 (pass@G of the training reward) bounds what RL can learn.

    python rollout_vllm.py --tasks /root/mab_train/train_official.json --samples 8 --temperature 1.0 --name signal_t1
    python rollout_vllm.py --tasks /root/mab_train/heldout_official.json --samples 1 --temperature 0 --name base_greedy
"""
import argparse
import collections
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import requests

sys.path.insert(0, "/root/jaxpt_src")
from jaxpt.envs.medagentbench import Benchmark, Episode  # noqa: E402


def chat(url, model, history, temperature, max_tokens, top_p, top_k, seed):
    messages = [{"role": "user" if t.role == "user" else "assistant", "content": t.content} for t in history]
    body = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": False}, "seed": seed}
    if top_p is not None:
        body["top_p"] = top_p
    if top_k is not None:
        body["top_k"] = top_k
    for attempt in range(3):
        try:
            r = requests.post(url, json=body, timeout=600)
            if r.status_code == 400 and "context" in r.text.lower() or r.status_code == 400 and "max" in r.text.lower():
                return None                      # the conversation outgrew the model: AgentBench's AGENT_CONTEXT_LIMIT
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"] or ""
        except Exception:
            if attempt == 2:
                raise
            time.sleep(2)


def run_episode(bench, case, args, seed):
    ep = Episode(bench, case, max_rounds=8, max_reply_chars=args.max_reply_chars)
    while not ep.done:
        text = chat(args.url, args.model, ep.history, args.temperature, args.max_tokens, args.top_p, args.top_k, seed)
        if text is None:
            ep.status = "agent context limit"                 # graded as a failure, as in the official harness
            break
        ep.step(text)
    agent = [t.content.strip() for t in ep.history[1::2]]
    return {"id": case["id"], "seed": seed, "status": ep.status, "official": bench.official(case, ep),
            "reward": bench.reward(case, ep), "looked_up": bench.looked_up(case, ep),
            "gets": sum(a.startswith("GET") for a in agent), "posts": sum(a.startswith("POST") for a in agent),
            "get_ok": sum(r.content.startswith("Here is the response") for r in ep.history[2::2]),
            "answer": ep.result, "turns": [t.content[:400] for t in ep.history[1:]]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mab", default="/root/MedAgentBench")
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--out", default="/root/mab_runs")
    ap.add_argument("--url", default="http://localhost:8011/v1/chat/completions")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--samples", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=None)
    ap.add_argument("--top-k", type=int, default=None)
    ap.add_argument("--max-tokens", type=int, default=2048, help="official HTTPAgent: 2048")
    ap.add_argument("--max-reply-chars", type=int, default=None, help="None = official (no truncation)")
    ap.add_argument("--workers", type=int, default=32)
    args = ap.parse_args()

    bench = Benchmark(args.mab)
    cases = json.load(open(args.tasks))
    jobs = [(c, s) for c in cases for s in range(args.samples)]
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as pool:
        rows = list(pool.map(lambda j: run_episode(bench, j[0], args, j[1]), jobs))
    os.makedirs(args.out, exist_ok=True)
    with open(f"{args.out}/{args.name}.jsonl", "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    by_task = collections.defaultdict(list)
    for r in rows:
        by_task[r["id"]].append(r)
    by_type = collections.defaultdict(lambda: collections.Counter())
    for tid, rs in by_task.items():
        c = by_type[tid.split("_")[0]]
        c["tasks"] += 1
        c["official_mean"] += sum(r["official"] for r in rs) / len(rs)
        c["reward_mean"] += sum(r["reward"] for r in rs) / len(rs)
        c["reward_any"] += any(r["reward"] for r in rs)                       # group carries signal
        c["get_ok_any"] += any(r["get_ok"] for r in rs)
    n = len(by_task)
    tot = lambda k: sum(c[k] for c in by_type.values())
    print(f"{args.name}: {n} tasks x {args.samples} samples at T={args.temperature} in {time.time() - t0:.0f}s | "
          f"official {100 * tot('official_mean') / n:.1f}% | gated reward {100 * tot('reward_mean') / n:.1f}% | "
          f"tasks with >=1 reward-1 sample {tot('reward_any')}/{n} | with >=1 successful GET {tot('get_ok_any')}/{n}")
    print(f"statuses {dict(collections.Counter(r['status'] for r in rows))}")
    for t, c in sorted(by_type.items(), key=lambda kv: int(kv[0][4:])):
        print(f"  {t:7} tasks {c['tasks']:3}  official {100 * c['official_mean'] / c['tasks']:5.1f}%  "
              f"gated {100 * c['reward_mean'] / c['tasks']:5.1f}%  signal groups {c['reward_any']:3}  "
              f"successful-GET groups {c['get_ok_any']:3}")
    json.dump({"args": vars(args), "by_type": {t: dict(c) for t, c in by_type.items()}},
              open(f"{args.out}/{args.name}.summary.json", "w"), indent=1)


if __name__ == "__main__":
    main()
