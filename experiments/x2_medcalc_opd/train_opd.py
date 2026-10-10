"""X2: vanilla on-policy distillation (OPD) on MedCalc-Bench Verified, variant C, Qwen3-4B -> Qwen3-0.6B-Base, on Tunix.

The same task, prompts, rollout, aggregation, optimizer and held-out curve as X1 (GRPO); only the per-token signal
changes: the teacher's log-prob of each token the student sampled, minus the student's (jaxpt.algorithms.opd).
No reward enters the loss; the MedCalc reward is still logged, so the two runs are read on the same axes.
Teacher: Qwen3-4B (the post-trained hybrid model), scoring the same completion-style prompt, no chat template.

Run (one GPU with room for the 4B teacher):
  python experiments/x2_medcalc_opd/train_opd.py --run opd-c --wandb --hours 2 --eval-every 15
Smoke test:
  python experiments/x2_medcalc_opd/train_opd.py --run smoke --max-steps 3 --prompts-per-step 2 \\
      --rollout-micro 2 --logps-micro 1 --max-new 256 --eval-prompts 4 --eval-every 2
"""
from pathlib import Path

from jaxpt.train import medcalc_run

if __name__ == "__main__":
    medcalc_run.main("opd", Path(__file__).parent, tag="x2-opd")
