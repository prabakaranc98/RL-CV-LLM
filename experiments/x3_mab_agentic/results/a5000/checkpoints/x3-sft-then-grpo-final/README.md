---
license: apache-2.0
base_model: Qwen/Qwen3-0.6B
library_name: transformers
pipeline_tag: text-generation
tags:
  - experimental
  - work-in-progress
  - medical
  - agent
  - fhir
  - medagentbench
  - reinforcement-learning
  - grpo
  - sft
  - lora
  - jax
---

# ex1-medbench-rl · MedAgentBench agent: SFT warm-up + GRPO (Qwen3-0.6B) — EXPERIMENTAL / WIP

> **Experimental, work in progress.** A research checkpoint from an ongoing study; it has not been validated beyond the evaluations reported here.
> **Not for clinical use.** Do not use it to look up, order, or act on real patient data.
> **Contaminated for the official MedAgentBench test:** it was trained on 208 of the official 300 v2 tasks (see *Data*). Do not report its score on the official 300.

Qwen3-0.6B (post-trained, thinking off) trained to act as an agent in MedAgentBench's simulated health record (FHIR): it issues `GET` / `POST` / `FINISH(...)` actions, reads the server's replies, and answers or places orders. LoRA (r32, α32, all attention and MLP projections) merged into the dense weights.

## Result

92 official tasks of **32 patients held out from training**, official protocol (the benchmark's prompt, default chat template, thinking off, greedy, ≤ 8 rounds, 2,048 tokens per turn, untruncated FHIR replies):

| Model | Official success | Real success (lookup-gated) |
|---|---|---|
| Qwen3-0.6B (base) | 6.5% | 0% |
| **this model** | **46.7%** (43 / 92) | **46.7%** |

*Lookup-gated* = the official grader passes **and** the episode made a successful GET of the task's patient record or lab code (e.g. `Observation?code=GLU&patient=...`). It removes passes that come from guessing (the base model's 6.5% is all `FINISH([-1])` / no-action answers).

Per task type (held-out): find patient 8/9 · age 0/9 · record BP 9/9 · Mg (24 h) 0/9 · Mg + order if low 8/10 · avg glucose 2/10 · latest glucose 2/9 · referral 9/9 · K + order if low 2/9 · HbA1C 3/9.

## Performance plots

![Three agentic RL runs](figures/agentic-runs.png)

*Three runs on MedAgentBench with Qwen3-0.6B and LoRA. Run 1: cold-start GRPO with partial credit and no FINISH requirement (the policy learns the lookup, then stops answering). Run 2: cold start with the FINISH requirement (no successes to reinforce). Run 3 (this model): SFT warm-up on oracle demonstrations, then GRPO; its step 0 is measured after the warm-up. Left: training rollouts (64 episodes per step): the right lookup made, episodes that end with FINISH, lookup-gated passes. Right: held-out success on 92 tasks of 32 unseen patients, greedy, with the training-time evaluation settings (256 tokens per turn, replies capped at 3,000 characters); the final checkpoint scores the same 46.7% under the official protocol.*

## How it was trained

1. **Warm start (SFT):** 235 oracle demonstrations: scripted episodes that follow the official grader's logic (the oracle scores 300 / 300 on the official tasks), replayed through the real FHIR environment and kept only if they pass the grader and the lookup gate. 2 epochs, LoRA, lr 1e-4.
2. **GRPO:** 15 steps, 8 tasks × 8 episodes per step at T = 1, group z-score advantages, on-policy, no KL (β = 0), lr 3e-5. Reward: 1 for a lookup-gated pass; otherwise partial credit (real server URL, right request, successful lookup) **only if the episode ends with FINISH** (without this condition, an earlier run learned to issue lookups until the round limit without answering).
3. **System:** rollouts in vLLM (token in / token out), updates in JAX, one RTX A5000. Training caps environment replies at 3,000 characters and turns at 256 tokens; evaluation does not.

Motivation for the warm start: cold-start GRPO from Qwen3-0.6B got **0 of 1,664** samples rewarded (no learning signal), and with partial credit it exploited the reward (reward hacking). After SFT, held-out real success was 39.1%; GRPO raised it to 46.7%, mostly on the conditional order tasks.

## Data

- MedAgentBench (Stanford; Jiang et al., arXiv:2501.14654), official `test_data_v2.json` (300 tasks, 10 types).
- The benchmark ships no training split, and its FHIR server's other 597 patients have no clinical data, so the 98 test patients were split **by patient**: 66 for training (208 tasks), 32 held out (92 tasks, 9–10 per type).
- Hence the contamination warning above: only the held-out patients' tasks are a fair test.

## Use

Serve with vLLM or transformers using the Qwen3 chat template with `enable_thinking=False`; the system prompt and action protocol are MedAgentBench's (`src/server/tasks/medagentbench/__init__.py`).

## Limitations

The held-out set is small (92 tasks; standard error about 5 points), and results are from a single seed. Value-reporting tasks regressed during GRPO (Mg 24 h, HbA1C); age (date arithmetic) is never solved. The SFT-only weights were not saved.
