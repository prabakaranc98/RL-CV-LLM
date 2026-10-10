"""Base vs GRPO vs OPD, read with one ruler: the E0 vLLM harness on Verified test, all four conditions.

For each policy (an E0 result name, e.g. Qwen3-0.6B-Base, grpo-c, opd-c) this reports:
  1. the 2 x 2 grid: greedy accuracy, pass@1 / pass@8 (275-problem subset), learnable band, degenerate rate;
  2. the gaps: knowledge (formula shown vs not) and extraction (gold values vs the note), before and after;
  3. how failures moved: the outcome taxonomy per policy (did "> 50% off" become "<= 10% off", or correct?);
  4. per-category and per-calculator accuracy, with the largest gains and losses vs the base;
  5. flips on the same problems vs the base: wrong -> right and right -> wrong (net gain hides both);
  6. one problem the policy newly solves, base answer vs policy answer.
Writes results/compare_<names>.md and prints it.

  python experiments/x1_medcalc_grpo/compare.py Qwen3-0.6B-Base grpo-c opd-c
"""
import json
import sys
from collections import Counter
from pathlib import Path

RESULTS = Path(__file__).parent / "results"
CONDITIONS = [("C", "per-calculator", "values"), ("B", "per-calculator", "note"),
              ("D", "generic", "values"), ("A", "generic", "note")]


def load(name, shots, inp):
    d = RESULTS / f"e0_{name}_verified_{shots}_{inp}"
    if not (d / "summary.json").exists():
        return None
    summary = json.load(open(d / "summary.json"))
    answers = {r["id"]: r for r in map(json.loads, open(d / "answers.jsonl"))}
    return summary, answers


def pct(x):
    return "—" if x is None else f"{100 * x:.1f}"


def main(names):
    base = names[0]
    runs = {(n, c): load(n, s, i) for n in names for c, s, i in CONDITIONS}
    md = [f"# MedCalc-Bench Verified test: {' vs '.join(names)}", ""]

    md += ["## 1. The 2 × 2 grid (greedy accuracy %, pass@1 / pass@8 % on 275 problems, learnable band /275)", "",
           "| policy | " + " | ".join(f"{c}: {s}, {i}" for c, s, i in CONDITIONS) + " |",
           "|---|" + "---|" * len(CONDITIONS)]
    for n in names:
        cells = []
        for c, _, _ in CONDITIONS:
            r = runs[(n, c)]
            if r is None:
                cells.append("—")
                continue
            s, pk = r[0], r[0]["passk_subset"]
            cells.append(f"**{pct(s['accuracy'])}** · {pct(pk['pass@1'])} / {pct(pk['pass@8'])} · "
                         f"band {pk['learnable_band']} · loops {pct(s['degenerate_rate'])}")
        md.append(f"| {n} | " + " | ".join(cells) + " |")

    md += ["", "## 2. Gaps (accuracy points)", "", "| policy | knowledge B−A | knowledge C−D | extraction D−A | extraction C−B |",
           "|---|---|---|---|---|"]
    for n in names:
        acc = {c: (runs[(n, c)][0]["accuracy"] if runs[(n, c)] else None) for c, _, _ in CONDITIONS}
        gap = lambda a, b: "—" if acc[a] is None or acc[b] is None else f"{100 * (acc[a] - acc[b]):+.1f}"
        md.append(f"| {n} | {gap('B', 'A')} | {gap('C', 'D')} | {gap('D', 'A')} | {gap('C', 'B')} |")

    for c, s, i in CONDITIONS[:2]:                          # the training condition and the transfer condition
        if runs[(base, c)] is None:
            continue
        md += ["", f"## 3–6. Condition {c} ({s}, {i})", "", "**Outcome taxonomy (counts of 1,100)**", "",
               "| outcome | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        outcomes = {n: Counter(a["outcome"] for a in runs[(n, c)][1].values()) for n in names if runs[(n, c)]}
        for o in sorted({o for v in outcomes.values() for o in v}, key=lambda o: -outcomes[base][o]):
            md.append(f"| {o} | " + " | ".join(str(outcomes.get(n, {}).get(o, "—")) for n in names) + " |")

        md += ["", "**Accuracy by category (%)**", "", "| category | n | " + " | ".join(names) + " |",
               "|---|---|" + "---|" * len(names)]
        cats = Counter(a["category"] for a in runs[(base, c)][1].values())
        for cat, k in cats.most_common():
            row = [pct(sum(a["reward"] for a in runs[(n, c)][1].values() if a["category"] == cat) / k)
                   if runs[(n, c)] else "—" for n in names]
            md.append(f"| {cat} | {k} | " + " | ".join(row) + " |")

        for n in names[1:]:
            if runs[(n, c)] is None:
                continue
            b, p = runs[(base, c)][1], runs[(n, c)][1]
            up = [k for k in b if b[k]["reward"] < 1 <= p[k]["reward"]]
            down = [k for k in b if b[k]["reward"] >= 1 > p[k]["reward"]]
            md += ["", f"**{n} vs {base}: {len(up)} problems wrong → right, {len(down)} right → wrong "
                   f"(net {len(up) - len(down):+d})**", ""]
            calc = Counter()
            for k in b:
                calc[b[k]["calculator"]] += p[k]["reward"] - b[k]["reward"]
            best, worst = calc.most_common(5), sorted(calc.items(), key=lambda kv: kv[1])[:5]
            md.append("largest gains (problems): " + "; ".join(f"{k} {v:+.0f}" for k, v in best if v > 0))
            md.append("")
            md.append("largest losses (problems): " + "; ".join(f"{k} {v:+.0f}" for k, v in worst if v < 0))
            if up:
                k = min(up, key=lambda k: len(p[k]["text"]))
                md += ["", f"Newly solved, `{b[k]['calculator']}` (gold {b[k]['gold']}):", "",
                       f"- base answered `{b[k]['parsed']}` ({b[k]['outcome']}):", "", "```",
                       b[k]["text"][:1200].strip(), "```", "", f"- {n} answered `{p[k]['parsed']}`:", "", "```",
                       p[k]["text"][:1200].strip(), "```"]

    text = "\n".join(md) + "\n"
    out = RESULTS / f"compare_{'_vs_'.join(names)}.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main(sys.argv[1:] or ["Qwen3-0.6B-Base"])
