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
  - on-policy-distillation
  - distillation
  - lora
  - jax
---

# ex1-medbench-rl · MedCalc-Bench: on-policy distillation from Qwen3-4B (Qwen3-0.6B-Base, LoRA) — EXPERIMENTAL / WIP

> **Experimental, work in progress.** A research checkpoint from an ongoing study; it has not been validated beyond the evaluations reported here.
> **Not for clinical use.** Accuracy on clinical calculations is below 50%; outputs must not inform care.

Qwen3-0.6B-Base trained by **on-policy distillation (OPD)** on MedCalc-Bench Verified: the student writes its own answers, and the teacher (Qwen3-4B, post-trained) scores every token it wrote; each token's advantage is `log π_teacher − log π_student` (the per-token reverse KL). LoRA (r32, α32, all attention and MLP projections) merged into the dense weights.

## Results

MedCalc-Bench Verified, **full test (1,100 problems)**, greedy, vLLM. Each cell: greedy accuracy / pass@1 / pass@8 (%); pass@k on a 275-problem subset (5 per calculator), 8 samples at T = 1.

| Prompt condition (trained on C only) | Qwen3-0.6B-Base | **this model (161 steps)** |
|---|---|---|
| **C** per-calculator worked example, gold extracted values | 26.2 / 15.5 / 52.7 | **41.2 / 39.1 / 62.5** |
| **B** per-calculator worked example, reads the patient note | 20.6 / 10.2 / 38.9 | **31.5 / 27.7 / 49.8** |
| **D** generic example, gold values | 4.9 / 3.1 / 17.8 | 8.2 / 6.1 / 22.9 |
| **A** generic example, reads the note | 4.9 / 2.8 / 17.8 | 6.9 / 5.8 / 21.8 |

- Sampled decoding (T 0.7, top-p 0.8, top-k 20; mean of 4) on C: **40.9%**, close to greedy.
- Same setup with GRPO instead of OPD (LoRA, 124 steps): 32.4% on C. The teacher's per-token signal moved the 0.6B model further, in the same budget.
- The formulas themselves are not learned: conditions A and D (no worked example) stay below 9%.

## Performance plots

![Full-test accuracy at every checkpoint](figures/fulltest-curve.png)

*Greedy accuracy on the full MedCalc-Bench Verified test (1,100 problems), condition C, at every exported checkpoint (vLLM). OPD (this model) and GRPO with the same LoRA configuration and step budget; horizontal lines: the base model and GRPO with full fine-tuning (58 steps). Final checkpoints: OPD 41.2% (161 steps), GRPO 32.4% (124 steps).*

![Held-out accuracy and training signal](figures/heldout-climb.png)

*Left: held-out accuracy during training (64 test problems × 8 samples at T = 1): pass@1 (solid) and pass@8 (dashed). Right: training accuracy and the share of prompt groups that carry no learning signal (all 8 samples wrong, or all right), 5-step mean.*

![Training dynamics](figures/training-dynamics.png)

*KL divergence (GRPO: to the base model, k3 estimator; OPD: reverse KL to the teacher on the student's samples), entropy, answer length and the share of well-formed answers, per training step.*

## Training

- Data: MedCalc-Bench Verified train split (10,538 problems; prompts ≤ 1,024 tokens: 9,798). No test problems used in training.
- 161 steps × 16 prompts × 8 samples, T = 1, lr 3e-5, `sequence-mean-token-mean` aggregation; 2.9 GPU-hours on one H100.
- Stack: JAX, Tunix's RL loop with our OPD loss, chunked log-probs, vLLM for evaluation.

## Limitations

Results are from a single seed. Checkpoint-to-checkpoint accuracy varies by about ±1.5 points (standard error at n = 1,100), and vLLM greedy decoding is not bit-reproducible. The peak was 43.3% at step 100; this is the final checkpoint.
