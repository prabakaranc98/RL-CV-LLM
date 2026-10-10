
---
<!-- _footer: "Wang et al., arXiv:2512.19691 (2025): ≥ 27% of v1.0 test labels were wrong. Numbers on this slide are from our runs." -->

<!-- _class: dense small -->

## Before training: check the labels, the checker and the cheapest shortcut

| Check | What we found | What we did |
|---|---|---|
| **Labels** | v1.2 and Verified differ on 10.6% of test rows; one v1.2 label is a GFR of 4.7 × 10⁻⁷ | train and test on **Verified** labels |
| **The checker** | ground-truth answers must all score 1 | **23,336 / 23,336** pass; 7 reward-hacking cases rejected (ranges, two different boxes, …) |
| **The cheapest shortcut** | always answering the calculator's typical training value | scores **12.7%**: a reward to watch, logged during training |
| **The answer budget** | a 512-token cap truncated 36% of answers in one condition | cap raised to **1,024 tokens** |
| **Degenerate loops** | a compression test missed them | a repeated-sentence / repeated-phrase rule, calibrated on real outputs |

<!--
The reward is only as good as the labels and the checker. Both were tested like code before any model was trained.
-->

---
<!-- _footer: "Our runs. Every failure and its fix is in the run log; the crashed runs are kept." -->

<!-- _class: dense small -->

## What broke, and what each failure taught

| Failure | Cause | Fix and lesson |
|---|---|---|
| reward on wrong labels | v1.2 labels (e.g. GFR 4.7 × 10⁻⁷) | Verified labels; *the reward is only as good as the labels* |
| 36% of answers cut off | 512-token cap | 1,024 tokens; *read the truncated answers before setting a cap* |
| sampler ran out of memory | prefill built full-vocabulary logits for every prompt token | last-token prefill (Tunix's own hook) |
| answers ran on into a new problem | base models do not emit EOS; Tunix has no stop strings | cut at the stop string, keeping junk out of reward and gradient |
| `ShardingTypeError`, then `TraceContextError` | Explicit mesh axes; remat inside the decode loop | Auto axes; no remat in the sampler |
| GRPO run 1 crashed after step 0 | eval loss over 512 answers at once (297 GiB) | chunked eval |
| OPD crashed twice in its first update | JAX's 75% memory cap, then fragmentation | 95% cap; LoRA for OPD |
| LoRA + OPD crashed at start | Tunix assumes reference = the actor's base and **would overwrite the teacher with the student** | keep the teacher separate. *Framework defaults encode the algorithm they were written for* |
| nonsense MFU (0.3–18%) | XLA's cost analysis counts a `scan` body once | analytic FLOP count |

<!--
The most important one is the reference sharing. With a LoRA actor, Tunix treats the reference model as the actor's base and overwrites it. For GRPO that is correct; for distillation it silently replaces the teacher with the student. It surfaced only because the shapes differ.
-->
