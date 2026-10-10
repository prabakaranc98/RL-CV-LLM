"""Save a W&B run's full metric history next to its results, so figures and tables are built from local files.

  python scripts/pull_wandb.py x1-grpo-grpo-c experiments/x1_medcalc_grpo/results/grpo-c/history.csv
Reads the API key from WANDB_API_KEY (or WAND_API_KEY, as named in .env).
"""
import os
import sys

import wandb


def main(name, out):
    os.environ.setdefault("WANDB_API_KEY", os.environ.get("WAND_API_KEY", ""))
    runs = [r for r in wandb.Api().runs("karan98/rl-cv-llm", order="-created_at") if r.name == name]
    if not runs:
        sys.exit(f"no W&B run named {name}")
    history = runs[0].history(samples=10_000)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    history.to_csv(out, index=False)
    print(f"{name} ({runs[0].state}): {len(history)} rows, {len(history.columns)} metrics -> {out}")


if __name__ == "__main__":
    main(*sys.argv[1:3])
