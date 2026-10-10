"""Hill-climbing figure for one or more runs (GRPO, OPD) on shared axes, from local files only:
  <run>/completions.jsonl   every completion (train and held-out eval), written by the training run
  <run>/history.csv         the W&B metric history (scripts/pull_wandb.py), optional

Top row, the climb:
  (a) held-out pass@1 and pass@8 vs step (64 fixed test problems x 8 samples, T = 1; step 0 = the base model)
  (b) train reward (5-step moving average) and the share of groups with zero advantage, split into
      all-right and all-wrong groups (no learning signal either way)
  (c) format-valid share and the typical-value (constant-guess) rate
Bottom row, what is happening inside (from history.csv):
  (d) KL to the reference (GRPO: to the base model; OPD: reverse KL to the teacher)
  (e) policy entropy (nats / token)
  (f) answer length (tokens)

  python experiments/x1_medcalc_grpo/curves.py <run dir> [<run dir> ...] --out <png>
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def held_out_and_train(run):
    by = defaultdict(lambda: defaultdict(list))
    for line in open(Path(run) / "completions.jsonl"):
        r = json.loads(line)
        by[(r["mode"], r["step"])][r["idx"]].append(r["reward"])
    evals, train = {}, {}
    for (mode, step), groups in by.items():
        g = list(groups.values())
        if mode == "eval":
            if sum(map(len, g)) < 8 * len(g):            # an eval still being written
                continue
            evals[step] = (np.mean([np.mean(x) for x in g]), np.mean([max(x) > 0 for x in g]))
        else:
            train[step] = (np.mean([np.mean(x) for x in g]), np.mean([min(x) > 0 for x in g]),
                           np.mean([max(x) == 0 for x in g]))
    return dict(sorted(evals.items())), dict(sorted(train.items()))


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--labels", nargs="*")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = args.labels or [Path(r).name for r in args.runs]
    colors = ["#b5446e", "#5b7f3a", "#3a6ea5", "#c98a2b"]
    fig, ax = plt.subplots(2, 3, figsize=(16, 8.5))
    (a, b, f), (c, d, e) = ax                                    # f = format panel (top right), c-e = inside
    for run, label, col in zip(args.runs, labels, colors):
        evals, train = held_out_and_train(run)
        es = list(evals)
        a.plot(es, [evals[s][0] for s in es], "o-", color=col, label=f"{label} accuracy (pass@1)")
        a.plot(es, [evals[s][1] for s in es], "s--", color=col, alpha=0.6, label=f"{label} solved by any of 8 (pass@8)")
        ts = list(train)
        b.plot(ts, smooth([train[s][0] for s in ts]), color=col, label=f"{label} train accuracy")
        b.plot(ts, smooth([train[s][1] for s in ts]), "--", color=col, alpha=0.7, label=f"{label} groups all right")
        b.plot(ts, smooth([train[s][2] for s in ts]), ":", color=col, label=f"{label} groups all wrong")
        print(f"{label}: held-out " + ", ".join(f"step {s}: {p1:.3f} / {p8:.3f}" for s, (p1, p8) in evals.items()))
        hist = Path(run) / "history.csv"
        if hist.exists():
            import pandas as pd
            h = pd.read_csv(hist)
            def plot(axis, col_name, style="-", name=None):
                if col_name in h:
                    s = h[["_step", col_name]].dropna()
                    axis.plot(s["_step"], s[col_name], style, color=col, label=f"{label} {name or ''}".strip())
            plot(c, "actor/train/kl")
            plot(d, "actor/train/entropy")
            plot(e, "completions/train/mean_length")
            plot(f, "medcalc/train/format_valid", "-", "format valid")
            plot(f, "medcalc/train/typical_value_rate", ":", "typical-value rate")
    titles = ["(a) held-out accuracy: 64 test problems x 8 samples, T = 1", "(b) train accuracy (reward) and no-signal groups (5-step avg)",
              "(c) format valid / constant-guess monitor", "(d) KL to reference", "(e) entropy (nats/token)",
              "(f) answer length (tokens)"]
    for axis, t in zip([a, b, f, c, d, e], titles):
        axis.set_title(t, fontsize=11)
        axis.set_xlabel("step")
        axis.grid(alpha=0.3)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
    fig.tight_layout()
    out = Path(args.out or Path(args.runs[0]) / "curves.png")
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
