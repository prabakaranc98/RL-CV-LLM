<!-- @title -->

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

<!-- @outcomes -->

<!-- _class: dense -->

## What you will take away

- **What post-training is**, and its main tools: imitation (SFT), RL from human feedback, RL with a checker (GRPO), and learning from a teacher (distillation).
- **How RL changes a language model:** sample answers, score them, and nudge the probability of each token up or down.
- **When to use which:** RL sharpens what a model can already sometimes do; demos or a teacher bring in what it cannot.
- **What goes wrong:** shortcuts and reward hacking, and why you check the ruler before training.
- **How it runs:** a fast generator plus a trainer, in JAX, on one GPU.
- **Two worked examples on a 0.6B model:** clinical calculations, and an agent working in a health record.

<!-- @mindmap -->

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

<!-- @timeline -->

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

<!-- @recipe -->

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

<!-- @takeaways -->

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

<!-- @questions -->

<!-- _class: dense -->

## Questions to take home

- **Knowledge or skill?** RL did not teach the missing formulas (both no-formula conditions stayed under 9%). Should knowledge come from pretraining, from retrieval, or from a teacher?
- **What can RL add that a teacher cannot?** OPD lifted pass@1 by 24 points but pass@8 by only 10: mostly sharpening.
- **How do you write a reward that cannot be gamed?** Our partial credit was exploited within 10 steps.
- **After imitation, how does the model keep exploring?** Its entropy fell to 0.009: its 8 tries per task all looked alike.
- **Small model with tools, or a bigger model?** Which is cheaper for the same accuracy, and which is safer to let act in a health record?

<!-- @glossary -->

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
