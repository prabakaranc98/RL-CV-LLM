"""X2b: GRPO + on-policy distillation in one update, on MedCalc-Bench Verified, variant C, Qwen3-4B -> Qwen3-0.6B-Base.

Each token's advantage = its answer's GRPO group advantage (checker reward) + lambda x (teacher - student log-prob):
the verifier says whether the answer is right, the teacher says which tokens are good (jaxpt.algorithms.opd,
"grpo_opd"; GKD / KDRL-style weighted combination). Everything else is shared with X1 / X2.

  python experiments/x2_medcalc_opd/train_hybrid.py --run hybrid-c-lora32 --wandb --lora-rank 32 --lr 3e-5 ...
"""
from pathlib import Path

from jaxpt.train import medcalc_run

if __name__ == "__main__":
    medcalc_run.main("hybrid", Path(__file__).parent, tag="x2b-hybrid")
