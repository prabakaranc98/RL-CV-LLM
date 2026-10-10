"""X1: vanilla GRPO (DeepSeekMath definition) on MedCalc-Bench Verified, variant C, from Qwen3-0.6B-Base, on Tunix.

Variant C: the benchmark's worked example for the same calculator, and the patient's gold values (not the note).
E0 base on C: 26.2% greedy, pass@8 52.7%, 53% of prompts in the learnable band.

The algorithm, unchanged from the paper (arXiv:2402.03300):
  G samples per prompt at temperature 1; advantage (r - mean_group) / (std_group + 1e-6);
  PPO-clipped surrogate (eps = 0.2), per-sequence token mean, then mean over sequences;
  KL to the base model with the k3 estimator, beta = 0.04; one on-policy update per batch.
Systems: Tunix GRPOLearner, our StopStringRollout (Tunix sampler + stop strings + last-token prefill),
fp32 master weights, bf16 compute, one GPU.

Logged live to W&B (karan98/rl-cv-llm) and to results/<run>/: the config, every completion (completions.jsonl),
and the final policy as a Hugging Face checkpoint (hf/) for the E0 vLLM evaluation.

The run itself lives in jaxpt.train.medcalc_run, shared with X2 (OPD).

Run (A40, 2 hours):
  python experiments/x1_medcalc_grpo/train_grpo.py --run grpo-c-a40 --wandb --hours 2 --eval-every 15
Smoke test:
  python experiments/x1_medcalc_grpo/train_grpo.py --run smoke --max-steps 3 --prompts-per-step 2 \
      --rollout-micro 2 --logps-micro 1 --max-new 256 --eval-prompts 4 --eval-every 2
"""
from pathlib import Path

from jaxpt.train import medcalc_run

if __name__ == "__main__":
    medcalc_run.main("grpo", Path(__file__).parent, tag="x1-grpo")
