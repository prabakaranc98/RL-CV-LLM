"""Per-task-type MedAgentBench result from an AgentBench output dir: correct / incorrect / limit per task type."""
import collections, json, sys
run = sys.argv[1]
o = json.load(open(f"{run}/overall.json"))
data = json.load(open("data/medagentbench/test_data_v2.json"))
by = collections.defaultdict(collections.Counter)
for r in o["custom"]["raw_results"]:
    by[data[r["index"]]["id"].split("_")[0]][r["status"]] += 1
print(f"success rate {100 * o['custom']['success rate']:.1f}%  ({run.split('/')[-2]})")
print("task    correct  incorrect  other   instruction (first task)")
first = {d["id"].split("_")[0]: d["instruction"] for d in data if d["id"].endswith("_1")}
for t in sorted(by, key=lambda x: int(x[4:])):
    c = by[t]
    other = sum(v for k, v in c.items() if k not in ("completedCorrect", "completedIncorrect"))
    print(f"{t:7} {c['completedCorrect']:7}  {c['completedIncorrect']:9}  {other:5}   {first.get(t, '')[:70]}")
print(dict(collections.Counter(r["status"] for r in o["custom"]["raw_results"])))
