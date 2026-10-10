"""Figures for the experiments part, drawn from the result files only (no number typed by hand).

Reads experiments/*/results (E0 summaries, completions.jsonl, history.csv) and writes figures/exp-*.svg in the deck's
style: fern for the model being trained, greys for everything else.

  .venv/bin/python talks/01-post-training-llms-with-rl/code/exp_figures.py [--runs grpo-c=GRPO ...]
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.ticker
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
X1 = ROOT / "experiments/x1_medcalc_grpo/results"
X2 = ROOT / "experiments/x2_medcalc_opd/results"
X3 = ROOT / "experiments/x3_mab_agentic/results"
OUT = Path(__file__).resolve().parents[1] / "figures"
FERN, FERN_DARK, TINT = "#4F7942", "#2F4A28", "#EAF1E6"
INK, G700, G500, G400, G200 = "#111111", "#4A4F55", "#6B7178", "#8A9097", "#D5D9DD"
RUN_COLORS = [FERN, "#B5835A", "#3E6E9E", G500]          # one accent per run, fern first

plt.rcParams.update({
    "font.family": ["Avenir Next", "Avenir", "DejaVu Sans"], "font.size": 13, "axes.edgecolor": G400,
    "axes.labelcolor": G700, "xtick.color": G700, "ytick.color": G700, "axes.spines.top": False,
    "axes.spines.right": False, "axes.titleweight": "semibold", "axes.titlesize": 14, "axes.titlecolor": INK,
    "legend.frameon": False, "svg.fonttype": "none", "axes.grid": True, "grid.color": "#EEF0F2", "grid.linewidth": 0.8,
})


def summary(name, shots, inp):
    p = X1 / f"e0_{name}_verified_{shots}_{inp}" / "summary.json"
    return json.load(open(p)) if p.exists() else None


def save(fig, name):
    OUT.mkdir(exist_ok=True)
    fig.savefig(OUT / f"{name}.svg", bbox_inches="tight")
    fig.savefig(OUT.parent / "dist" / f"{name}.png", dpi=110, bbox_inches="tight") if (OUT.parent / "dist").exists() else None
    plt.close(fig)
    print(f"figure: figures/{name}.svg")


def fig_e0_grid():
    """Left: the 2 x 2 at 0.6B with the constant-guess line. Right: condition B across sizes, next to published
    one-shot results (hatched; their label versions differ, see the footer)."""
    conds = [("A", "generic", "note"), ("D", "generic", "values"), ("B", "per-calculator", "note"),
             ("C", "per-calculator", "values")]
    labels = ["A\nno formula\nreads note", "D\nno formula\ngold values", "B\nformula shown\nreads note",
              "C\nformula shown\ngold values"]
    acc = [100 * summary("Qwen3-0.6B-Base", s, i)["accuracy"] for _, s, i in conds]
    fig, (a, b) = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1, 1.45]})
    bars = a.bar(range(4), acc, color=[G200, G200, TINT, TINT], edgecolor=[G400, G400, FERN, FERN], linewidth=1.5)
    for r, v in zip(bars, acc):
        a.text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v:.1f}%", ha="center", color=INK, fontsize=13)
    guess = 100 * summary("Qwen3-0.6B-Base", "generic", "note")["constant_guess_accuracy"]
    a.axhline(guess, color=G500, ls="--", lw=1.2)
    a.text(-0.4, guess + 0.9, f"constant guess {guess:.1f}%", ha="left", color=G500, fontsize=11)
    a.set_xticks(range(4), labels, fontsize=11)
    a.set_ylabel("greedy accuracy (%), Verified test 1,100")
    a.set_ylim(0, 34)
    a.set_title("Qwen3-0.6B-Base: the formula matters, the note barely does")
    ours = [("Qwen3-0.6B-Base", "0.6B\nbase"), ("Qwen3-1.7B-Base", "1.7B\nbase"), ("Qwen3-4B-Base", "4B\nbase"),
            ("Qwen3-4B", "4B post-\ntrained\n(teacher)")]
    vals = [100 * summary(n, "per-calculator", "note")["accuracy"] for n, _ in ours]
    published = [("Llama 3 8B\n(v1.0)", 27.1), ("GPT-4\n(v1.0)", 50.9), ("Qwen3-4B\nMedRaC", 56.0)]
    approx = {2: "≈56%"}          # MedRaC reports rule / calc separately; weighted by our category mix
    xs_ours, xs_pub = np.arange(4), np.arange(len(published)) + 4.8
    bars = b.bar(xs_ours, vals, color=[TINT, G200, G200, G200], edgecolor=[FERN, G400, G400, G400], linewidth=1.5)
    pub = b.bar(xs_pub, [v for _, v in published], color="white", edgecolor=G400, hatch="///", linewidth=1.2)
    for r, v in zip(bars, vals):
        b.text(r.get_x() + r.get_width() / 2, v + 1, f"{v:.1f}%", ha="center", color=INK, fontsize=12)
    for j, (r, (_, v)) in enumerate(zip(pub, published)):
        b.text(r.get_x() + r.get_width() / 2, v + 1, approx.get(j, f"{v:.1f}%"), ha="center", color=INK, fontsize=12)
    b.set_xticks(list(xs_ours) + list(xs_pub), [l for _, l in ours] + [l for l, _ in published], fontsize=10.5)
    b.text(1.5, 63, "ours (Verified labels)", ha="center", color=FERN_DARK, fontsize=11)
    b.text(xs_pub.mean(), 63, "published one-shot", ha="center", color=G500, fontsize=11)
    b.set_ylim(0, 68)
    b.set_title("Condition B (the benchmark's one-shot protocol) across sizes")
    fig.tight_layout()
    save(fig, "exp-e0-grid")


def held_out_and_train(run_dir):
    by = defaultdict(lambda: defaultdict(list))
    for line in open(run_dir / "completions.jsonl"):
        r = json.loads(line)
        by[(r["mode"], r["step"])][r["idx"]].append(r["reward"])
    evals, train = {}, {}
    for (mode, step), groups in by.items():
        g = list(groups.values())
        if mode == "eval":
            if sum(map(len, g)) >= 8 * len(g):
                evals[step] = (np.mean([np.mean(x) for x in g]), np.mean([max(x) > 0 for x in g]))
        else:
            train[step] = (np.mean([np.mean(x) for x in g]), np.mean([max(x) == 0 for x in g]),
                           np.mean([min(x) > 0 for x in g]))
    return dict(sorted(evals.items())), dict(sorted(train.items()))


def smooth(y, k=5):
    y = np.asarray(y, float)
    return np.array([y[max(0, i - k + 1):i + 1].mean() for i in range(len(y))])


def fig_climb(runs, name):
    """Held-out accuracy and pass@8 vs step; train accuracy and no-signal groups."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 5.0))
    for j, ((run_dir, label), col) in enumerate(zip(runs, RUN_COLORS)):
        evals, train = held_out_and_train(run_dir)
        es = list(evals)
        a.plot(es, [100 * evals[s][0] for s in es], "o-", color=col, lw=2.2, label=f"{label}: pass@1")
        a.plot(es, [100 * evals[s][1] for s in es], "s--", color=col, lw=1.6, alpha=0.75, label=f"{label}: pass@8")
        dy = 9 if j == 0 else -16                         # first run's labels above its points, the second's below
        for s, dx in ((es[0], -12 if j == 0 else 12), (es[-1], 0)):
            a.annotate(f"{100 * evals[s][0]:.1f}", (s, 100 * evals[s][0]), textcoords="offset points", xytext=(dx, dy),
                       ha="center", color=col, fontsize=11)
            a.annotate(f"{100 * evals[s][1]:.1f}", (s, 100 * evals[s][1]), textcoords="offset points", xytext=(dx, dy),
                       ha="center", color=col, fontsize=11, alpha=0.8)
        ts = list(train)
        b.plot(ts, smooth([100 * train[s][0] for s in ts]), color=col, lw=2.2, label=f"{label}: train accuracy")
        b.plot(ts, smooth([100 * train[s][1] for s in ts]), ":", color=col, lw=1.6, label=f"{label}: groups all wrong")
        b.plot(ts, smooth([100 * train[s][2] for s in ts]), "--", color=col, lw=1.2, alpha=0.7,
               label=f"{label}: groups all right")
    a.set_title("Held-out: 64 test problems × 8 samples (T = 1)")
    a.set_xlabel("step (16 prompts × 8 samples each)")
    a.set_ylabel("%")
    a.set_ylim(0, 75)
    a.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    b.set_title("Training: accuracy and groups with no signal (5-step mean)")
    b.set_xlabel("step")
    b.set_ylim(0, 60)
    b.legend(fontsize=9.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.tight_layout()
    save(fig, name)


def fig_inside(runs, name, kl_title="KL to the reference"):
    """KL (GRPO: to the base model; OPD: reverse KL to the teacher), entropy, answer length, format-valid share,
    from history.csv."""
    import pandas as pd
    fig, ax = plt.subplots(1, 4, figsize=(15, 3.6))
    panels = [("actor/train/kl", kl_title, 1), ("actor/train/entropy", "entropy (nats / token)", 1),
              ("completions/train/mean_length", "answer length (tokens)", 1),
              ("medcalc/train/format_valid", "format valid (%)", 100)]
    for (run_dir, label), col in zip(runs, RUN_COLORS):
        h = pd.read_csv(run_dir / "history.csv")
        for axis, (key, title, k) in zip(ax, panels):
            if key in h:
                s = h[["_step", key]].dropna()
                axis.plot(s["_step"], k * s[key], color=col, lw=1.8, label=label)
    for axis, (_, title, _) in zip(ax, panels):
        axis.set_title(title, fontsize=13)
        axis.set_xlabel("step")
    ax[0].legend(fontsize=10)
    fig.tight_layout()
    save(fig, name)


def fig_before_after(policies, name, cond=("per-calculator", "values")):
    """Full test (1,100), one condition: greedy accuracy, pass@1, pass@8 per policy."""
    metrics = [("accuracy", "greedy accuracy"), ("pass@1", "pass@1 (T = 1)"), ("pass@8", "pass@8")]
    fig, a = plt.subplots(figsize=(10, 4.4))
    w = 0.8 / len(policies)
    for j, ((pname, label), col) in enumerate(zip(policies, [G400] + RUN_COLORS)):
        s = summary(pname, *cond)
        if s is None:
            continue
        vals = [100 * (s["accuracy"] if k == "accuracy" else s["passk_subset"][k]) for k, _ in metrics]
        xs = np.arange(len(metrics)) + (j - (len(policies) - 1) / 2) * w
        bars = a.bar(xs, vals, w * 0.92, color=col if j else G200, edgecolor=col, linewidth=1.2, label=label)
        for r, v in zip(bars, vals):
            a.text(r.get_x() + r.get_width() / 2, v + 0.8, f"{v:.1f}", ha="center", fontsize=11, color=INK)
    a.set_xticks(range(len(metrics)), [m for _, m in metrics])
    a.set_ylabel("% of Verified test")
    a.legend(fontsize=11)
    fig.tight_layout()
    save(fig, name)


def fig_fulltest_curve(runs, name, refs=()):
    """Greedy accuracy on the full test (1,100, condition C) at every exported checkpoint, one line per run, the
    final policy plotted at its own last step; horizontal references (base, other policies) from their E0 summaries."""
    fig, a = plt.subplots(figsize=(10, 4.6))
    for (run, label, final_step), col in zip(runs, RUN_COLORS):
        pts = []
        for d in sorted(X1.glob(f"e0_{run}-step*_verified_per-calculator_values")):
            step = int(d.name.split("-step")[1].split("_")[0])
            pts.append((step, 100 * json.load(open(d / "summary.json"))["accuracy"]))
        final = summary(run, "per-calculator", "values")
        if final and final_step not in [p[0] for p in pts]:
            pts.append((final_step, 100 * final["accuracy"]))
        pts.sort()
        if pts:
            a.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=col, lw=2.2, label=label)
            a.annotate(f"{pts[-1][1]:.1f}", pts[-1], textcoords="offset points", xytext=(6, 4), color=col, fontsize=11)
    xmax = a.get_xlim()[1]
    for (pname, label), ls in zip(refs, ["--", ":", "-."]):
        s_ = summary(pname, "per-calculator", "values")
        if s_:
            a.axhline(100 * s_["accuracy"], color=G500, ls=ls, lw=1.2)
            a.text(xmax, 100 * s_["accuracy"] + 0.4, f"{label} {100 * s_['accuracy']:.1f}", color=G500, fontsize=10,
                   ha="right")
    a.set_xlabel("step (16 prompts × 8 samples each)")
    a.set_ylabel("greedy accuracy (%), full test 1,100")
    a.set_title("Condition C, full test, every checkpoint (vLLM, greedy)")
    a.legend(fontsize=11, loc="upper left")
    fig.tight_layout()
    save(fig, name)


def fig_x3(runs, name):
    """Agentic GRPO on MedAgentBench, from each run's metrics.jsonl. Left: training rollouts per step (5-step mean):
    the right lookup made, episodes that end with FINISH, lookup-gated passes. Right: the held-out patients' official
    tasks at each evaluation, lookup-gated (solid) and official grader (dashed)."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(13, 4.8))
    for (run_dir, label), col in zip(runs, [G400, "#B5835A", FERN]):   # cold starts muted, the warm start the accent
        rows = [json.loads(l) for l in open(run_dir / "metrics.jsonl")]
        train = [r for r in rows if "step" in r]
        evals = [r for r in rows if "eval_step" in r]
        st = [r["step"] for r in train]
        n = lambda r: sum(r["statuses"].values())
        a.plot(st, [100 * r["lookup_rate"] for r in train], "-", color=col, lw=2.2, label=f"{label}: right lookup")
        a.plot(st, [100 * r["statuses"].get("completed", 0) / n(r) for r in train], "--", color=col, lw=1.6,
               label=f"{label}: ends with FINISH")
        a.plot(st, [100 * r["gated_rate"] for r in train], ":", color=col, lw=2.0, label=f"{label}: gated pass")
        es = [r["eval_step"] for r in evals]
        b.plot(es, [100 * r["heldout_gated"] for r in evals], "o-", color=col, lw=2.2, label=f"{label}: lookup-gated")
        b.plot(es, [100 * r["heldout_official"] for r in evals], "s--", color=col, lw=1.4, alpha=0.8,
               label=f"{label}: official")
    a.set_title("Training rollouts (64 episodes per step)")
    a.set_xlabel("step")
    a.set_ylabel("% of episodes")
    a.set_ylim(0, 100)
    a.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    b.set_title("Held-out: 92 tasks of 32 unseen patients, greedy")
    b.set_xlabel("step")
    b.set_ylabel("% of tasks")
    b.set_ylim(0, 60)
    b.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    b.legend(fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    fig.tight_layout()
    save(fig, name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    jobs = {
        "e0": fig_e0_grid,
        "grpo-climb": lambda: fig_climb([(X1 / "grpo-c", "GRPO (full)")], "exp-grpo-climb"),
        "grpo-inside": lambda: fig_inside([(X1 / "grpo-c", "GRPO (full)")], "exp-grpo-inside"),
        "grpo-before-after": lambda: fig_before_after([("Qwen3-0.6B-Base", "base"), ("grpo-c", "GRPO (full FT)")],
                                                      "exp-grpo-before-after"),
        "lora-pair": lambda: fig_climb([(X1 / "grpo-c-lora32", "GRPO (LoRA)"), (X2 / "opd-c-lora32", "OPD (LoRA)")],
                                       "exp-lora-climb"),
        "lora-inside": lambda: fig_inside([(X1 / "grpo-c-lora32", "GRPO (LoRA)"), (X2 / "opd-c-lora32", "OPD (LoRA)")],
                                          "exp-lora-inside", kl_title="KL: GRPO to base, OPD to teacher"),
        "fulltest-curve": lambda: fig_fulltest_curve(
            [("grpo-c-lora32", "GRPO (LoRA)", 124), ("opd-c-lora32", "OPD (LoRA)", 161)], "exp-fulltest-curve",
            refs=[("Qwen3-0.6B-Base", "base"), ("grpo-c", "GRPO full FT, 58 steps")]),
        "x3": lambda: fig_x3([(X3 / "a5000/x3_runs/grpo-shaped-lora32-t4k", "run 1"),
                              (X3 / "a5000/x3_runs/grpo-shaped-finish-lora32", "run 2"),
                              (X3 / "a5000/x3_runs/sft-then-grpo-lora32", "run 3 (warm start)")],
                             "exp-x3-agentic"),
    }
    for k, f in jobs.items():
        if not args.only or k in args.only:
            f()


if __name__ == "__main__":
    main()
