---
marp: true
theme: fern
paginate: true
size: 16:9
header: "Prabakaran Chandran | pracha.me"
math: mathjax
title: "Post-training LLMs with RL and On-Policy Distillation (TPU day)"
description: "From policy gradients to PPO, GRPO and on-policy distillation, with JAX"
---

<!-- _class: title -->
<!-- _header: "" -->
<!-- _paginate: false -->

# Post-training LLMs with RL and On-Policy Distillation

<p class="subtitle">A beginner-friendly tour, and two experiments on a 0.6B model · TPU day</p>

**Prabakaran Chandran**

Engineer / Applied Researcher · pracha.me

<!--
No RL background assumed. First twenty slides: the ideas, in plain words. Last ten: what happened when we tried them.
-->

---

<!-- _class: dense -->

## What you will take away

- **What post-training is**, and its main tools: imitation (SFT), RL from human feedback, RL with a checker (GRPO), and learning from a teacher (distillation).
- **How RL changes a language model:** sample answers, score them, and nudge the probability of each token up or down.
- **When to use which:** RL sharpens what a model can already sometimes do; demos or a teacher bring in what it cannot.
- **What goes wrong:** shortcuts and reward hacking, and why you check the ruler before training.
- **How it runs:** a fast generator plus a trainer, in JAX, on one GPU.
- **Two worked examples on a 0.6B model:** clinical calculations, and an agent working in a health record.

---

<!-- _footer: "OLMo 3: Team Olmo, arXiv:2512.13961 (2025). Qwen3: Yang et al., arXiv:2505.09388 (2025), §4.5. Lambert, RLHF book (rlhfbook.com), ch. 3." -->

<!-- _class: figure -->

## Post-training turns a next-token predictor into a useful model

![w:1140](figures/post-training-stages.svg)

<!--
A pretrained model only continues text. Post-training teaches it what to do with what it knows. Four signals do most of the work: examples, comparisons, checkers and a stronger model. This talk covers the RL-based stages and on-policy distillation. Stress the footnote: real recipes mix and repeat stages.
-->

---

<!-- _footer: "Ouyang et al., InstructGPT (2022); Rafailov et al., DPO (2023); Shao et al., DeepSeekMath (2024); DeepSeek-AI, R1 (2025); Agarwal et al., GKD (ICLR 2024); Qwen3 Technical Report (2025), §4.5; Jiang et al., MedAgentBench (2025)." -->

<!-- _class: dense small -->

## The post-training toolbox at a glance

| Tool | What the model learns from | Where the signal comes from | Known for |
|---|---|---|---|
| **SFT** (imitation) | copying good answers, token by token | answers written by people or a stronger model | following instructions, formats |
| **RLHF** (PPO) | a reward model trained on people's rankings | people comparing two answers | helpful chat assistants (InstructGPT) |
| **DPO** | the ranked pairs directly, no reward model | people comparing two answers | cheaper preference tuning |
| **RL with a checker** (GRPO) | right or wrong, decided by a program | a verifier: math checker, unit tests, a grader | reasoning models (DeepSeek-R1) |
| **Distillation**, off-policy | the teacher's own answers | a bigger model | small models (R1-Distill) |
| **On-policy distillation** | the teacher's score for every token the *student* writes | a bigger model with the same tokenizer | Qwen3's small models |
| **Agentic RL** | a reward after many actions in an environment | the world: a tool, a database, a grader | agents (code, search, health records) |

<!--
Read the table by its middle column: every method is a different answer to "where does the learning signal come from?". The rest of the talk builds this up one row at a time.
-->

---

<!-- _footer: "Schulman et al., PPO (2017); Ouyang et al. (2022); Rafailov et al. (2023); Agarwal et al. (2023 / ICLR 2024); Shao et al. (2024); DeepSeek-AI (2025); Qwen3 (2025); Thinking Machines, On-Policy Distillation (2025); Jiang et al. (2025); ICML 2026 RLxF workshop." -->

<!-- _class: dense small -->

## How we got here

| Year | What changed | Why it mattered |
|---|---|---|
| 2017 | **PPO**: a stable way to update a policy a little at a time | the RL algorithm LLM post-training started from |
| 2022 | **InstructGPT**: SFT, then a reward model, then PPO | the RLHF recipe for assistants that follow instructions |
| 2023 | **DPO**: the same goal with a simple loss on ranked pairs | preference tuning without RL machinery |
| 2023 | **GKD**: on-policy distillation for language models | the student learns from the teacher on its *own* answers |
| 2024 | **GRPO** (DeepSeekMath): RL without a critic network | cheaper RL with a checker |
| 2025 | **DeepSeek-R1**: RL with checkers grows long reasoning | reasoning models; distilled into small ones |
| 2025 | **Qwen3**: small models built by distillation, off- then on-policy | a teacher beats RL for small models, at a tenth of the compute |
| 2025–26 | **agents and world feedback**: MedAgentBench; ICML's RLxF workshop | rewards from environments and outcomes, not only people |

---

<!-- _class: divider -->

###### Part 1

# The ideas, in plain words

Reinforcement learning, a language model as a policy, and four ways to train it.

---

<!-- _footer: "Sutton & Barto, Reinforcement Learning: An Introduction, 2nd ed. (2018), ch. 3; OpenAI Spinning Up, Part 1: Key Concepts in RL." -->

<!-- _class: figtext -->

## An agent learns from the consequences of its own actions

![w:1120](figures/rl-loop.svg)

- **Policy** $\pi_\theta(a \mid s)$: the probability of taking action $a$ in state $s$; $\theta$ are the parameters we train.
- **Trajectory** $\tau = (s_0, a_0, r_0, s_1, \dots)$ and **return** $R(\tau) = \sum_t r_t$: the total reward of one episode.
- **Goal**: choose $\theta$ to maximize the expected return $J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[R(\tau)]$.

<!--
The only difference between the two panels is what an action is. For an LLM the environment is trivial while writing (it appends the token) and gives a reward only when the response is finished.
-->

---

<!-- _footer: "Qwen3-0.6B config.json and model card (huggingface.co/Qwen/Qwen3-0.6B): vocabulary 151,936, 28 layers." -->

<!-- _class: figure -->

## The model maps a context to a distribution over the next token

![w:1150](figures/llm-policy.svg)

> Generation repeats one step: score every token in the vocabulary, turn the scores into probabilities with a softmax, sample one, append it. That probability, $\pi_\theta(y_t \mid x, y_{<t})$, is the policy.

<!--
Everything RL needs from the model is this distribution and the log-probability of the token that was actually sampled.
-->

---

<!-- _footer: "Shao et al., DeepSeekMath, arXiv:2402.03300 (2024), §4.1.2: with an outcome reward, every token of a response gets the same advantage." -->

<!-- _class: figtext -->

## Generating a response is a sequence of decisions

![w:1150](figures/generation-as-decisions.svg)

| RL | Language model |
|---|---|
| state $s_t$, action $a_t$ | the context $(x, y_{<t})$, the next token $y_t$ |
| reward | usually one number for the finished response, from a reward model or a checker |

<!--
The key consequence: the reward arrives once, at the end, so the method has to decide how much credit each token gets. PPO asks the critic; GRPO gives every token the same credit; on-policy distillation gives each token its own score.
-->

---

<!-- _footer: "Williams (1992), REINFORCE; Sutton et al. (2000), policy gradient theorem; OpenAI Spinning Up, Part 3: Intro to Policy Optimization." -->

<!-- _class: figtext -->

## The policy gradient: make good actions more likely

![w:1140](figures/policy-gradient.svg)

- **Where it comes from**: $\nabla_\theta p_\theta(\tau) = p_\theta(\tau)\,\nabla_\theta \log p_\theta(\tau)$, and the environment's probabilities do not depend on $\theta$, so only the policy's terms remain.
- **How to read it**: a weighted log-likelihood. With every weight equal to 1 and tokens from a dataset, this is exactly the supervised fine-tuning gradient.

<!--
This one line is the root of everything in the talk. PPO, GRPO and on-policy distillation all keep the grad-log-pi term and differ only in how they compute the weight.
-->

---

<!-- _footer: "Ouyang et al., Training language models to follow instructions with human feedback (InstructGPT), arXiv:2203.02155 (2022), Figure 2 and §3. Christiano et al., arXiv:1706.03741 (2017)." -->

<!-- _class: figure -->

## RLHF: learn a reward from people, then optimize it with PPO

![w:1120](figures/rlhf-pipeline.svg)

> People are better at comparing two answers than at writing a score. Step 2 turns comparisons into a reward model; step 3 optimizes the policy against it.

<!--
InstructGPT asked labellers to rank between 4 and 9 model outputs per prompt. The SFT model is used twice more: it initializes the policy, and it is the reference for the KL penalty.
-->

---

<!-- _footer: "Gao, Schulman & Hilton, arXiv:2210.10760 (2022), §1 and Fig. 1 (curves here are illustrative); Singhal et al., arXiv:2310.03716 (2023), §3.1; Sharma et al., arXiv:2310.13548 (2023), §4; Weng, Reward Hacking in Reinforcement Learning (Lil'Log, 2024)." -->

<!-- _class: figtext -->

## Optimized too hard, the policy exploits the reward model

![w:1060](figures/overoptimization.svg)

- **Measured**: Gao et al. optimize a policy against a proxy reward model and score it with a larger "gold" one. The gold score rises, then falls, as $\sqrt{\mathrm{KL}}$ grows.
- **What it looks like**: longer answers (in two of three RLHF setups, length explains 70–90% of the reward gain; Singhal et al.), and sycophancy, agreeing with the user over being correct (Sharma et al.).

<!--
Reward hacking is not a bug in PPO: any optimizer pushed hard against an imperfect score finds the score's blind spots. Lilian Weng's post is the best long read.
-->

---

<!-- _footer: "Lambert et al., Tülu 3, arXiv:2411.15124 (2024), §6, Eqs. 7–8; DeepSeek-AI, DeepSeek-R1, arXiv:2501.12948, v1 (2025), §2.2.2; Hugging Face, Math-Verify (github.com/huggingface/Math-Verify); Team Olmo, OLMo 3 (2025), §4.4.1." -->

<!-- _class: figtext -->

## RL with verifiable rewards: a program checks the answer

![w:1120](figures/verifiers.svg)

- **RLVR** (the name is from Tülu 3) swaps the learned reward model for a checker: Tülu 3 gives a reward of 10 when the answer verifies and 0 otherwise; DeepSeek-R1-Zero adds a format reward for the think tags.
- **Math checkers** extract the final answer and compare it with the reference, symbolically where needed (Math-Verify; OLMo 3 uses SymPy). Code checkers run tests.

<!--
Checkers exist only where answers can be checked: math, code, constrained instructions, some puzzles. Open-ended writing still needs a learned or LLM judge.
-->

---

<!-- _footer: "Shao et al. (2024), §4.1.2 (outcome supervision). Worked example." -->

<!-- _class: figure -->

## GRPO on one prompt

![w:1150](figures/grpo-group.svg)

> Correct answers move up and wrong ones move down, by an amount set by how the response compares with its siblings. Every token of a response gets that response's advantage.

<!--
If all four had been correct, every advantage would be zero and the prompt would teach nothing. That is why later recipes filter or resample such prompts.
-->

---

<!-- _footer: "Shao et al. (2024), §4.1.2. Code tested with JAX 0.11.2. Liu et al., Dr. GRPO, arXiv:2503.20783 (2025), proposes dropping the division by the standard deviation." -->

<!-- _class: figtext -->

## The GRPO advantage is the reward, standardized within its group

![h:285](figures/grpo-advantage.svg)

```python
def grpo_advantages(rewards, eps=1e-6):
    """rewards: [num_prompts, G], one row per prompt's group of G responses."""
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + eps)      # [[1, 0, 0, 1]] -> [[1, -1, -1, 1]]
```

<!--
It is a z-score. The division by the standard deviation is debated: Dr. GRPO argues it gives very easy and very hard prompts too much weight.
-->

---

<!-- _footer: "Agarwal et al., GKD, arXiv:2306.13649 (2023); Lu (Thinking Machines Lab), On-Policy Distillation (2025); Zhao et al., OPSD, arXiv:2601.18734 (2026)." -->

<!-- _class: figure -->

## Three ways to learn from a teacher

![w:1150](figures/teacher-modes.svg)

> **Off-policy**: the student imitates states the teacher visited. **On-policy**: the student learns on its own responses, so it is corrected where it actually makes mistakes.

<!--
The off-policy problem: at test time the student's own early mistakes lead it into states the teacher's data never covered. Training on its own samples fixes exactly that.
-->

---

<!-- _footer: "Lu (2025), Thinking Machines blog: per-token reverse KL, discount 0. Qwen3 report, arXiv:2505.09388, §4.5 and Table 21." -->

<!-- _class: figtext -->

## On-policy distillation: the teacher scores every token

![w:1060](figures/opd-advantage.svg)

- **Same machinery as RL**: plug $A_t$ into the policy-gradient loss. Averaged over the student's samples, $-A_t$ is the reverse KL from student to teacher.
- **Cost**: Qwen3-8B reached 74.4 on AIME'24 with 1,800 GPU-hours of on-policy distillation, against 67.6 with 17,920 GPU-hours of RL. In this basic form, teacher and student must share a tokenizer.

<!--
The teacher only runs a forward pass on the student's tokens; it never generates. That is why it is cheap.
-->

---

<!-- _footer: "Summary of the slides above." -->

<!-- _class: dense small -->

## Four methods, one loop

| | PPO (RLHF) | GRPO | On-policy distillation | On-policy self-distillation |
|---|---|---|---|---|
| Who writes responses | the policy | the policy, $G$ per prompt | the student | the student, 1 per prompt |
| Signal | reward model | reward model or checker | teacher log-probs | frozen self, shown the solution |
| Signal density | per response | per response | per token | per token |
| Advantage | critic (GAE) | group z-score | $\log\pi_T - \log\pi_\theta$ | full-distribution KL |
| Models in memory | policy, critic, reward, reference | policy, reference (optional) | student, teacher | student, frozen copy |

> All four sample from the model being trained and push its token log-probabilities; they differ in where the weight on each token comes from.

<!--
This is the slide to leave up during questions.
-->

---

<!--
This is the picture behind every systems choice that follows: two jobs with different needs, sharing one set of weights. Continuous batching, which keeps the inference GPUs full, comes from Orca; vLLM adds paged KV-cache memory.
-->

<!-- _class: figure -->

## Generation dominates the time: a rollout engine and a trainer

![w:1150](figures/jax-system.svg)

> OpenRLHF measured generation at 80% of training time. In OLMo 3, a 1,000-second RL step spent 125 seconds training, and the learner waited 75% of the time. Inference engines such as vLLM (paged KV cache) exist for exactly this.

<!--
This is where the experiments part picks up: what we run locally on CPU, what needs a GPU or TPU, and what we measure.
-->

---

<!-- _footer: "JAX documentation: Quickstart: How to think in JAX (docs.jax.dev)." -->

<!-- _class: figure -->

## JAX: composable transformations of pure functions

![w:1150](figures/jax-stack.svg)

> JAX post-training libraries: **Tunix** (Google), **EasyDeL**, and **MaxText** for large-scale training; Part 6 lists what each supports.

<!--
For RL the useful transformations are grad for the loss, jit for the training step, vmap across the group of samples, and scan for the token-by-token decoding loop.
-->

---

<!-- _footer: "Code tested on CPU with JAX 0.11.2; Optax would replace the plain SGD line." -->

<!-- _class: dense -->

## A GRPO update in JAX

```python
def grpo_loss(params, batch, apply_fn, beta=0.0, eps=0.2):
    logits = apply_fn(params, batch["ids"])[:, :-1]          # scores for the next token
    logp = token_logprobs(logits, batch["ids"][:, 1:])        # log pi_theta(y_t | x, y_<t)
    adv = batch["adv"][:, None]                               # one advantage per response
    loss = clipped_pg_loss(logp, batch["logp_old"], adv, batch["mask"], eps)
    if beta > 0:                                              # optional KL penalty to pi_ref
        loss += beta * (kl_k3(logp, batch["logp_ref"]) * batch["mask"]).sum() / batch["mask"].sum()
    return loss

lr = 1e-2

@jax.jit
def train_step(params, batch):
    loss, grads = jax.value_and_grad(grpo_loss)(params, batch, apply_fn)
    params = jax.tree.map(lambda p, g: p - lr * g, params, grads)   # params are a pytree
    return params, loss
```

> `logp_old`, `logp_ref` and `adv` are plain arrays computed before the step; only `params` is differentiated. In practice the update uses an optimizer such as Adam (Optax).

<!--
Point out what is not there: no separate copy of the old policy, no critic. The reference model is needed only to compute logp_ref once per batch.
-->

---

<!-- _class: dense -->

## A recipe: the steps, and the decisions at each one

1. **Write the checker first:** a program, a grader or a teacher that says what "good" is.
2. **Check the ruler:** what does doing nothing score? What does a blind guess score? (Doing nothing passes **23%** of MedAgentBench.)
3. **Measure the signal:** when the model samples freely, does it *ever* succeed? (Our 0.6B agent: **0 of 1,664** tries.)
4. **Choose the method from that answer:** never succeeds → bring knowledge in with demos (SFT) or a teacher (distillation); sometimes succeeds → RL sharpens it.
5. **Train cheaply:** LoRA adapters, one GPU, a generator (vLLM) and a trainer (JAX).
6. **Watch for hacking:** read the transcripts, and track one number the reward cannot fake.
7. **Judge on held-out data,** with the official ruler.

<!--
Each step here is something we did tonight, and each has a slide later that shows what happened when we did it, or when we skipped it.
-->

---

<!-- _class: divider -->

###### Part 2

# Two experiments on a 0.6B model

Clinical calculation (one checked answer) and an agent in a health record (many turns).

---

<!-- _class: dense small -->

## The experiments: two benchmarks, GRPO and OPD, everything measured

| | |
|---|---|
| **Problem** | a 0.6B model is poor at clinical calculation (**26.2%** with the formula shown, **4.9%** without) and, as an agent in the health record, solves **0%** of tasks for real |
| **Question** | how much can post-training lift it, and *which* skill moves: knowledge (formulas), reading the note, or computation? |
| **Runs** | MedCalc-Bench: base · **GRPO** full fine-tuning (58 steps) and LoRA (124) · **on-policy distillation** from Qwen3-4B, LoRA (161); 16 prompts × 8 samples per step · MedAgentBench: **multi-turn GRPO**, LoRA, 8 tasks × 8 episodes per step |
| **Data** | MedCalc-Bench Verified: train 10,538, test the **full 1,100**, four prompt conditions · MedAgentBench (Stanford, official 300): train on 66 patients' tasks, test on **32 unseen patients' 92 tasks** |
| **System** | **JAX** (our Qwen3, parity with Hugging Face to 10⁻⁴) · Tunix for the RL loop, with our fixes · vLLM for evaluation · agentic: vLLM rollouts + JAX update · 1 GPU per run |
| **Measured** | accuracy, pass@k, the failure type of every answer, KL, entropy, length, the shortcut rate, memory and time per phase |
| **Budget** | **$29** of rented GPU time in total (RunPod billing, both benchmarks, failures included) |

<!--
This slide is the map for the rest of the part: first the baseline and its diagnosis, then GRPO, then LoRA and distillation, then systems and failures.
-->

---

<!-- _footer: "Khandekar et al., MedCalc-Bench, arXiv:2406.12036 (NeurIPS 2024); Wang et al., MedCalc-Bench Verified, arXiv:2512.19691 (2025)." -->

<!-- _class: dense -->

## The task: clinical calculators, with answers a program can check

- **Input:** a patient note (or the values extracted from it) and a question, e.g. *"What is the patient's CHA₂DS₂-VASc score?"* **Output:** a number, a date or a gestational age, in `\boxed{}`.
- **55 calculators** in seven families: lab, physical, risk, severity, diagnosis, date, dosage.
- **Data:** MedCalc-Bench **Verified** (physician-reviewed labels): 10,538 train, **1,100 test**, plus one worked example per calculator.
- **Reward:** 1 if the answer falls inside the gold range, else 0. A verifiable reward, so RL needs no reward model.
- **Why this task:** a real use (clinicians use these calculators daily), a checkable answer, and a 0.6B model that starts far from the ceiling.

<!--
The worked example per calculator is the benchmark's own one-shot protocol. It shows the formula, which lets us separate "does not know the formula" from "cannot compute".
-->

---

<!-- _footer: "Our implementation: JAX (model, sampler fixes, losses, LoRA merge, chunked log-probs) with Tunix's RL loop; vLLM for evaluation." -->

<!-- _class: figure -->

## One loop for both methods: only the per-token signal differs

![w:1180](figures/exp-loop.svg)

> GRPO gives every token of an answer the same advantage, from a checker; OPD gives each token its own, from the teacher. The sampler, the loss, the LoRA update and the evaluation are shared, so the comparison has one variable.

<!--
The green path is what trains: the gradient reaches only the LoRA matrices A and B; after each step the sampler receives the dense equivalent W + (alpha / r) A B, so generation runs at full speed with no adapter overhead. Every 25 steps a checkpoint is exported and scored by vLLM on the full test.
-->

---

<!-- _footer: "Our runs: Qwen3-0.6B-Base, LoRA r32 / α32 on all projections, lr 3e-5, 16 prompts × 8 samples per step, 2.9 h each (A100 GRPO, H100 OPD). OPD teacher: Qwen3-4B." -->

<!-- _class: figure -->

## With LoRA on both, OPD keeps climbing where GRPO flattens

![w:1180](figures/exp-lora-climb.svg)

> Held-out pass@1 (64 problems × 8 samples, T = 1): GRPO **16.6 → 33.4** by step 100, OPD **17.0 → 43.4** by step 150. Same prompts, same rollouts, same LoRA. The only difference is the per-token signal: a checker's group advantage versus the teacher's log-probs.

<!--
The dotted lines are the prompts where all 8 samples are wrong. GRPO gets exactly zero signal there; OPD still gets a teacher signal on every token. That is the mechanism behind the gap.
-->

---

<!-- _footer: "Our runs: vLLM, greedy, Verified test (1,100), condition C (the training condition), every exported checkpoint. Final policies: GRPO-LoRA at step 124, OPD-LoRA at step 161; we report finals, not the best checkpoint." -->

<!-- _class: figure -->

## On the full test, OPD gains about 15 points and GRPO 6

![w:1000](figures/exp-fulltest-curve.svg)

> Final policies, greedy, all 1,100 problems: GRPO-LoRA **26.3 → 32.4** (+6.1, 124 steps), OPD-LoRA **26.5 → 41.2** (+14.7, 161 steps; peak 43.3). At matched steps LoRA GRPO equals the full fine-tune (29.7 at step 50 and at 58).

<!--
The two step-0 points are the same weights, 0.2 points apart: vLLM's greedy is not bit-reproducible.
Standard error at n = 1,100 is about 1.5 points, so both gains are well outside noise. Step 25 (25.7) is within noise of step 0. OPD's reverse KL to the teacher had already fallen from 0.32 to 0.07 by step 25, so the student matched the teacher's token distribution first; the accuracy gain came between steps 25 and 50.
-->

---

<!-- _footer: "Our runs: vLLM, Verified test (1,100); greedy on all; pass@k on a 275-problem subset (5 per calculator), 8 samples at T = 1. Final policies. Sampled: T 0.7, top-p 0.8, top-k 20 (Qwen3 card), mean of 4." -->

<!-- _class: dense small -->

## All four conditions: OPD transfers to reading the note; nobody learns the formulas

| Condition (all trained on C) | Base | GRPO full (58) | GRPO LoRA (124) | OPD LoRA (161) |
|---|---|---|---|---|
| **C** formula shown, gold values | 26.2 / 15.5 / 52.7 | 29.7 / 24.6 / 58.2 | 32.4 / 25.0 / 59.3 | **41.2 / 39.1 / 62.5** |
| **B** formula shown, reads the note | 20.6 / 10.2 / 38.9 | 22.9 / 17.2 / 48.0 | 22.5 / 15.8 / 43.6 | **31.5 / 27.7 / 49.8** |
| **D** no formula, gold values | 4.9 / 3.1 / 17.8 | 6.5 / 5.0 / 24.4 | 6.8 / 3.8 / 16.0 | 8.2 / 6.1 / 22.9 |
| **A** no formula, reads the note | 4.9 / 2.8 / 17.8 | 5.8 / 3.8 / 19.3 | 5.5 / 3.0 / 14.9 | 6.9 / 5.8 / 21.8 |

<span class="small">Each cell: greedy accuracy / pass@1 / pass@8 (%).</span>

- **OPD transfers:** B rises **+10.9** (20.6 → 31.5) without being trained on it; GRPO-LoRA moves B by +1.9.
- **The formulas stay missing:** A and D stay below 9% for every method. A teacher on formula-shown prompts does not install the formulas.
- **Greedy is the honest number here:** sampled at the model card's setting, mean of 4: base **25.9**, GRPO full **29.0**, GRPO-LoRA **31.6**, OPD-LoRA **40.9**, each within 1.3 points of greedy.

<!--
The pass@8 column says whether a method widens what the model can reach or only sharpens it. OPD lifts pass@1 by 23.6 points but pass@8 by 9.8: mostly sharpening, some widening. GRPO-LoRA's pass@8 on A and D falls below the base (14.9 vs 17.8, 16.0 vs 17.8): sharpening at the cost of breadth off the training condition.
-->

---

<!-- _class: dense small -->

## Words for the agent experiments

| Word | Plain meaning | Example |
|---|---|---|
| **episode** | one full attempt at one task, question to final answer, at most 8 turns | `GET …` → record → `FINISH([140.0])` |
| **lookup** | the agent asking the health record for data | `GET …/Observation?code=GLU&patient=S6534835` |
| **official success** | the benchmark's grader accepts the final answer | 140.0 is the latest glucose → pass |
| **real (gated) success** | official success **and** the right lookup was made | a blind `FINISH([-1])` can pass officially, but is a guess |
| **shortcut** | passing without the skill | doing nothing passes 23% of the tasks |
| **cold / warm start** | RL from the released model, or after first learning from examples | runs 1–2 cold; run 3 warm |
| **oracle demos** | perfect episodes written by a script that follows the grader's logic | 235 demos taught the protocol |
| **reward hacking** | collecting reward without doing the task | run 1 kept looking things up and never answered |
| **signal** | a task's 8 attempts score differently, so GRPO has something to learn | 0 of 1,664 attempts rewarded → no signal |
| **entropy** | how varied the model's choices are; near 0, it repeats itself | 0.21 at the start; 0.003 after imitation |

---

<!-- _footer: "Our trainer: experiments/x3_mab_agentic/train_agentic_grpo.py, one RTX A5000 (24 GB): vLLM 0.31 for rollouts, JAX 0.11.2 for the update. LoRA r32 / α32, lr 3e-5, β = 0 (no reference model)." -->

<!-- _class: figure -->

## How agentic RL runs: vLLM plays the episodes, JAX learns from them

![w:1180](figures/exp-agent-loop.svg)

> One step: 8 tasks × 8 episodes, each a conversation with the health record, up to 8 turns. The grader scores each finished episode; the policy gradient flows only through the tokens the model wrote. The new weights go back to vLLM before the next step, so every episode is sampled from the current policy (on-policy).

<!--
Why two engines: generation dominates RL time (56% of a MedCalc step with our own sampler), and vLLM batches the 64 episodes as they diverge in length. JAX keeps the update transparent. The check that the two copies agree: vLLM's log-prob of each sampled token versus JAX's, mean gap 0.01-0.03 nats per token.
Token in, token out: the trainer sends vLLM token ids and receives token ids, so the sequence it trains on is exactly what was sampled; no re-tokenizing the text.
-->

---

<!-- _footer: "A real episode: Qwen3-1.7B on official task7_28 (most recent glucose of patient S6534835), passed. Token counts with the Qwen3 tokenizer. Training caps environment replies at 3,000 characters." -->

<!-- _class: dense small -->

## One episode, token by token: what the model writes is all that is trained

| Turn | Who | What (abridged) | Tokens | In the loss? |
|---|---|---|---|---|
| 0 | prompt | rules, 9 FHIR function specs, context, question: *"What is the most recent CBG of the patient S6534835?"* | 3,309 | no: context |
| 1 | **model** | `GET https://…/fhir/Observation?code=GLU&patient=S6534835` | **43** | **yes** |
| 2 | environment | *"Here is the response from the GET request:"* the patient's glucose records as FHIR JSON | 2,435 | no: masked |
| 3 | **model** | `FINISH([140.0])` | **9** | **yes** |

- **52 of 5,796 content tokens are the model's.** Every one of them gets the episode's advantage; the 5,744 others are only context.
- **Reward:** the official grader checks 140.0 against the latest glucose in the record, and the lookup gate checks turn 1 → **1.0**.
- **If it had looped** GET after GET without FINISH, the reward is 0. Our first run paid partial credit for lookups without a FINISH, and the policy learned exactly that loop (next slide).

<!--
This is the credit-assignment problem of agentic RL in one picture: one number at the end, spread over the few tokens the model chose, across turns separated by thousands of tokens the environment chose.
-->

---

<!-- _footer: "Our runs: vLLM, greedy, thinking off, official harness, 300 tasks. Lookup-gated = official pass and a successful GET of the patient's record or lab. Sampling check: 208 training tasks × 8 at T = 1." -->

<!-- _class: dense small -->

## The 0.6B's 6% is a guess, and RL has nothing to learn from

| Agent | Official | Lookup-gated |
|---|---|---|
| do-nothing | 23.3% | **0%** |
| Qwen3-0.6B | 6.3% / 5.3% (two runs) | **0%** |
| Qwen3-1.7B | 9.0% | **9.0%** |
| Qwen3-4B | 8.3% | **1.7%** |

- **Every 0.6B pass is `FINISH([-1])`,** after a failed lookup (17) or none (2). It writes the literal `{api_base}` placeholder into the URL, and −1 ("not available") happens to be right for 19 of 30 magnesium questions.
- **It knows what to ask, not how.** In its lab-lookup samples at T = 1, the commonest request is the right query with the placeholder (**162×**); with the real URL it asks for the first listed function, Condition (**83×**); never both at once.
- **Shortcut-proof reward:** **0 of 1,664** samples rewarded, so **0 of 208** GRPO groups carry a signal. Cold-start GRPO cannot start.

<!--
This is the measured answer to "do we need a warm-up before RL?" at 0.6B: with an honest reward, yes, or a reward that credits the halves it gets right. The 1.7B is the only size whose passes are all real.
-->

---

<!-- _footer: "Our run grpo-shaped-lora32-t4k (A5000), stopped at step 10. Shaped reward: 1 for a gated pass; else +0.1 real URL, +0.1 right request, +0.2 successful lookup. Held-out: 92 tasks, greedy, our driver." -->

<!-- _class: dense small -->

## Run 1: GRPO learned the lookup, then learned to never answer

| Step | 1 | 3 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|
| right lookup made (% of 64 episodes) | 0 | 0 | 1.6 | 6.2 | 17.2 | 17.2 | 23.4 | **26.6** |
| episodes that end with FINISH | 61 | 61 | 59 | 36 | 16 | 2 | 0 | **0** |
| groups with a signal (of 8) | 5 | 7 | 1 | 6 | 6 | 7 | 7 | 4 |

- **The partial credit worked as a teacher:** from 0 of 1,664 samples to a correct lookup in a quarter of episodes in 10 steps.
- **Then the reward was hacked:** the credit did not require a FINISH, so the cheapest 0.4 was to keep issuing GETs until the round limit. Held-out official **19.6% → 0%**, lookup-gated **0% → 0%**: 85 of 92 episodes hit the limit.
- **Fix, one line:** partial credit only for an episode that ends with FINISH. Run 2 starts again from the base model.

<!--
The textbook lesson, measured: an RL agent optimizes the reward you wrote, not the one you meant. The 19.6% it lost was never skill: those passes were guesses. What it gained (the lookup) is real, and the held-out gated score is the number that would show it.
-->

---

<!-- _footer: "Our run sft-then-grpo-lora32 (A5000): 235 of 328 oracle demos kept (gated pass, no truncated reply, answer inside the trained window); SFT on LoRA r32, lr 1e-4, 2 epochs, 13 min; then 15 GRPO steps (finish-gated reward). Held-out: greedy, our driver (256 tokens per turn, replies capped at 3,000 characters)." -->

<!-- _class: dense small -->

## Warm start, then RL: real success 0% → 39% → 47%

| Held-out: 92 tasks of 32 unseen patients | Cold start (runs 1, 2) | After SFT warm-up | + 15 GRPO steps |
|---|---|---|---|
| **real success** (lookup-gated) | 0% | **39.1%** (36) | **46.7%** (43) |
| official success | 0% | 39.1%: every pass is real | 46.7%: every pass is real |
| **official protocol** (default template, 2,048 tokens per turn, full replies) | base model: 6.5% (0% real) | | **46.7%, all real** |

| | find patient | age | record BP | referral | Mg 24 h | Mg + order | avg glucose | glucose | K + order | HbA1C |
|---|---|---|---|---|---|---|---|---|---|---|
| after SFT | 8 / 9 | 0 / 9 | 9 / 9 | 9 / 9 | 3 / 9 | 1 / 10 | 0 / 10 | 1 / 9 | 0 / 9 | 5 / 9 |
| + GRPO | 8 / 9 | 0 / 9 | 9 / 9 | 9 / 9 | 0 / 9 | **9 / 10** | 2 / 10 | 1 / 9 | **4 / 9** | 1 / 9 |

- **SFT taught the protocol:** the real URL, the right resource, the exact payloads. That solves patient search, vitals and referrals for unseen patients.
- **RL then learned the conditionals:** "check the level, order only if low" went from 1 to 13 of 19. In their words, "SFT to inject codes, RL to learn conditionals." Most of those correct answers are "no order needed", the common case.
- **Not free:** it lost value-reporting tasks (Mg 24 h 3 → 0, HbA1C 5 → 1); net +7 (≥ 14 gained, 7 lost), not yet significant. **Age stays at 0**: 51 demos show the age, not the arithmetic.

<!--
The warm start is not distillation from a model: the demos come from the grader's own logic (the oracle that scored 300 / 300), replayed through the real environment. After SFT the policy is nearly deterministic (entropy 0.003-0.03 nats per token), so most of GRPO's groups have identical samples; only 1-5 of 8 carry a signal each step. Exploration is the next problem.
-->

---

<!-- _footer: "Our runs on one A5000: run 1 shaped reward without a FINISH requirement (stopped at step 10); run 2 with it (stopped at 11); run 3 the same reward after the SFT warm-up (15 steps, its step 0 is after SFT)." -->

<!-- _class: figure -->

## Three runs, one picture: RL needs a model that sometimes succeeds

![w:1180](figures/exp-x3-agentic.svg)

> Cold start, the agent never succeeds and RL has nothing to reinforce: run 1 hacks the reward, run 2 stays at 0%. After 235 demonstrations it starts at 39%, and GRPO climbs from there to 46.7% on patients it never saw.

---

<!-- _footer: "Our runs, Qwen3-0.6B, LoRA, one GPU each; MedCalc-Bench Verified full test (1,100), greedy; MedAgentBench: 92 held-out tasks of 32 unseen patients, greedy, our driver. Frontier: MedAgentBench paper (Claude 3.5 Sonnet v2, official 300 tasks); Anthropic healthcare post (MedCalc, Opus 4.5 with tools)." -->

<!-- _class: dense small -->

## What we found, on one page

| | MedCalc-Bench: one checked answer | MedAgentBench: an agent, many turns |
|---|---|---|
| **start** | 26.2% with the formula shown | **0% real** (its 6% was guessing) |
| **RL alone** (GRPO) | 32.4% (+6.1) | 0%: no signal, then reward hacking |
| **with a teacher or demos** | **OPD: 41.2% (+14.7)** | **SFT warm-up: 39.1% real; + GRPO: 46.7%** |
| **the frontier, for scale** | Claude Opus 4.5 with tools: 61.3% | Claude 3.5 Sonnet v2: 69.7% (official 300) |
| **cost** | about **$29** of rented GPUs for everything, failures included | |

- **At 0.6B, a teacher's per-token signal beats an outcome reward,** in the same number of steps.
- **For the agent, demos come first and RL second:** RL cannot reinforce a success it never sees; after the demos it adds the conditional decisions.
- **The ruler matters as much as the method:** each benchmark has a cheap shortcut (a constant guess scores 12.7% on MedCalc; doing nothing, 23.3% on MedAgentBench). On MedAgentBench it passed for skill until we checked.

---

<!-- _class: dense -->

## Questions to take home

- **Knowledge or skill?** RL did not teach the missing formulas (both no-formula conditions stayed under 9%). Should knowledge come from pretraining, from retrieval, or from a teacher?
- **What can RL add that a teacher cannot?** OPD lifted pass@1 by 24 points but pass@8 by only 10: mostly sharpening.
- **How do you write a reward that cannot be gamed?** Our partial credit was exploited within 10 steps.
- **After imitation, how does the model keep exploring?** Its entropy fell to 0.009: its 8 tries per task all looked alike.
- **Small model with tools, or a bigger model?** Which is cheaper for the same accuracy, and which is safer to let act in a health record?
