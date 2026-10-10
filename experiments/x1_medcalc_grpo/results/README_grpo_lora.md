---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B-Base
library_name: transformers
pipeline_tag: text-generation
tags:
  - experimental
  - work-in-progress
  - medical
  - medcalc-bench
  - reinforcement-learning
  - grpo
  - rlvr
  - lora
  - jax
---

# ex1-medbench-rl · MedCalc-Bench: GRPO with a verifier (Qwen3-0.6B-Base, LoRA) — EXPERIMENTAL / WIP

> **Experimental, work in progress.** A research checkpoint from an ongoing study; it has not been validated beyond the evaluations reported here.
> **Not for clinical use.** Accuracy on clinical calculations is below 50%; outputs must not inform care.

Qwen3-0.6B-Base trained with **GRPO** (group-relative policy optimization) against MedCalc-Bench's answer checker: for each prompt the model samples 8 answers, each answer's reward is whether its value falls inside the reference range, and the advantage is the reward standardized within the group. LoRA (r32, α32, all attention and MLP projections) merged into the dense weights. The companion model trained by on-policy distillation is in the same collection.

## Results

MedCalc-Bench Verified, **full test (1,100 problems)**, greedy, vLLM. Each cell: greedy accuracy / pass@1 / pass@8 (%); pass@k on a 275-problem subset (5 per calculator), 8 samples at T = 1.

| Prompt condition (trained on C only) | Qwen3-0.6B-Base | **this model (124 steps)** |
|---|---|---|
| **C** per-calculator worked example, gold extracted values | 26.2 / 15.5 / 52.7 | **32.4 / 25.0 / 59.3** |
| **B** per-calculator worked example, reads the patient note | 20.6 / 10.2 / 38.9 | 22.5 / 15.8 / 43.6 |
| **D** generic example, gold values | 4.9 / 3.1 / 17.8 | 6.8 / 3.8 / 16.0 |
| **A** generic example, reads the note | 4.9 / 2.8 / 17.8 | 5.5 / 3.0 / 14.9 |

- Sampled decoding (T 0.7, top-p 0.8, top-k 20; mean of 4) on C: **31.6%**.
- At matched steps LoRA matches GRPO with full fine-tuning (29.7% on C at step 50; full fine-tuning: 29.7% at 58 steps).
- Under the same budget, on-policy distillation from Qwen3-4B reaches 41.2% on C (companion model). GRPO sharpens what the base model already samples; it does not supply missing formulas (conditions A and D stay below 7%).

## Performance plots

![Full-test accuracy at every checkpoint](figures/fulltest-curve.png)

*Greedy accuracy on the full MedCalc-Bench Verified test (1,100 problems), condition C, at every exported checkpoint (vLLM). GRPO (this model) and on-policy distillation with the same LoRA configuration and step budget; horizontal lines: the base model and GRPO with full fine-tuning (58 steps).*

![Held-out accuracy and training signal](figures/heldout-climb.png)

*Left: held-out accuracy during training (64 test problems × 8 samples at T = 1): pass@1 (solid) and pass@8 (dashed). Right: training accuracy and the share of prompt groups that carry no learning signal (all 8 samples wrong, or all right), 5-step mean.*

![Training dynamics](figures/training-dynamics.png)

*KL divergence (GRPO: to the base model, k3 estimator; OPD: reverse KL to the teacher on the student's samples), entropy, answer length and the share of well-formed answers, per training step.*

## Training

- Data: MedCalc-Bench Verified train split (10,538 problems; prompts ≤ 1,024 tokens: 9,798). No test problems used in training.
- 124 steps × 16 prompts × 8 samples, T = 1, lr 3e-5, PPO clip ε = 0.2, KL penalty β = 0.04 (k3) to the base model, `sequence-mean-token-mean` aggregation; 2.9 GPU-hours on one A100.
- Stack: JAX, Tunix's GRPO learner with chunked log-probs, vLLM for evaluation.

## Limitations

Results are from a single seed. Checkpoint-to-checkpoint accuracy varies by about ±1.5 points (standard error at n = 1,100), and vLLM greedy decoding is not bit-reproducible. The peak was 32.9% at step 100; this is the final checkpoint.
