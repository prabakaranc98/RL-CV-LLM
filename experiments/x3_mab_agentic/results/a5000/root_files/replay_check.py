"""Gate for the RL environment: replay recorded official MedAgentBench episodes through jaxpt.envs.medagentbench.

For each episode the official AgentBench harness recorded (outputs/MedAgentBenchv1/<agent>/medagentbench-std), feed the
agent's turns, in order, to Episode.step and require:
  * every environment reply equals the recorded one, character for character;
  * the episode ends in the recorded status (completed / agent invalid action / task limit reached);
  * Benchmark.official() equals the recorded grade (completedCorrect vs everything else).
Also prints, per agent, how many official passes the lookup-gated training reward would keep.

Run on the MedAgentBench pod:  python replay_check.py --mab /root/MedAgentBench qwen3-0.6b qwen3-1.7b qwen3-4b do-nothing
"""
import argparse
import collections
import json
import sys

sys.path.insert(0, "/root/jaxpt_src")
from jaxpt.envs.medagentbench import Benchmark, Episode  # noqa: E402

HEAD, TAIL = "Here is the response from the GET request:\n", ". Please call FINISH if"


def same_reply(a, b):
    """Equal, ignoring what HAPI regenerates on every search: the Bundle's id, meta and paging links."""
    if a == b:
        return True
    if not (a and b and a.startswith(HEAD) and b.startswith(HEAD)):
        return False
    try:
        ja, jb = (json.loads(x[len(HEAD):x.rindex(TAIL)]) for x in (a, b))
    except ValueError:
        return False
    for j in (ja, jb):
        for k in ("id", "meta", "link"):
            j.pop(k, None)
    return ja == jb and a[a.rindex(TAIL):] == b[b.rindex(TAIL):]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mab", default="/root/MedAgentBench")
    ap.add_argument("agents", nargs="+")
    args = ap.parse_args()
    bench = Benchmark(args.mab)
    cases = bench.tasks()
    ok_all = True
    for agent in args.agents:
        out = f"{args.mab}/outputs/MedAgentBenchv1/{agent}/medagentbench-std"
        status = {r["index"]: r["status"] for r in json.load(open(f"{out}/overall.json"))["custom"]["raw_results"]}
        n, reply_mismatch, status_mismatch, grade_mismatch, gated = 0, 0, 0, 0, collections.Counter()
        example = None
        for line in open(f"{out}/runs.jsonl"):
            rec = json.loads(line)
            i, hist = rec["index"], rec["output"]["history"]
            case, ep = cases[i], Episode(bench, cases[i])
            n += 1
            if ep.history[0].content != hist[0]["content"]:
                reply_mismatch += 1
                example = example or (i, "prompt")
            for t, turn in enumerate(hist[1:], start=1):
                if turn["role"] != "agent":
                    continue
                ep.step(turn["content"])
                mine = ep.history[-1].content if ep.history[-1].role == "user" else None   # None: ended, no reply
                recorded = hist[t + 1]["content"] if t + 1 < len(hist) else None
                if not same_reply(mine, recorded):
                    reply_mismatch += 1
                    example = example or (i, t, (mine or "")[:120], (recorded or "")[:120])
                    break
                if ep.done:
                    break
            want = status[i]
            want_status = want.replace("Correct", "").replace("Incorrect", "")
            if ep.status != want_status:
                status_mismatch += 1
                example = example or (i, "status", ep.status, want)
            official = bench.official(case, ep)
            if official != (want == "completedCorrect"):
                grade_mismatch += 1
                example = example or (i, "grade", official, want)
            if official:
                gated[case["id"].split("_")[0]] += 1
                gated["kept"] += bench.reward(case, ep) == 1.0
        ok = reply_mismatch == status_mismatch == grade_mismatch == 0
        ok_all &= ok
        passes = sum(v for k, v in gated.items() if k != "kept")
        print(f"{agent:14} episodes {n}  reply mismatches {reply_mismatch}  status mismatches {status_mismatch}  "
              f"grade mismatches {grade_mismatch}  -> {'PASS' if ok else 'FAIL'} | official passes {passes}, "
              f"kept by the lookup gate {gated['kept']}" + (f" | first problem: {example}" if example else ""))
    print("ALL PASS" if ok_all else "SOME FAILED")


if __name__ == "__main__":
    main()
