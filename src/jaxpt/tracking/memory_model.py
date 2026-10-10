"""Where the GPU memory goes in one RL step on Tunix, estimated from shapes (GiB), to compare with what the
allocator reports (memory/* metrics in W&B).

Phases of a step, in order; only the static part stays resident, each other phase's transient memory is freed
before the next:
  static     policy fp32 weights + fp32 grads + AdamW (2 x fp32) + reference / teacher bf16 + sampler copy bf16
  rollout    KV cache [rows, prompt + completion] for every layer, plus one layer's prefill attention scores
  log-probs  reference / teacher forward: Tunix returns fp32 logits for every position, then log-softmax
  train      forward + backward on a micro-batch: logits, log-softmax and entropy over the vocabulary, the
             logits' gradient, remat block inputs, and one block's attention scores being recomputed
The lesson of X1 run 1: at 0.6B parameters and a 152K vocabulary, the [rows, tokens, vocab] tensors are bigger
than the whole model with its optimizer (9.5 GiB): one 8-row micro-batch of 2,048 tokens has 9.3 GiB of fp32 logits.
"""
from dataclasses import dataclass

GiB = 2**30


@dataclass(frozen=True)
class Model:
    params: float
    layers: int
    d: int
    heads: int
    kv_heads: int
    head_dim: int
    vocab: int = 151_936

    def kv_bytes_per_token(self, dtype_bytes=2):
        return self.layers * 2 * self.kv_heads * self.head_dim * dtype_bytes


QWEN3_0P6B = Model(596e6, 28, 1024, 16, 8, 128)
QWEN3_4B = Model(4.02e9, 36, 2560, 32, 8, 128)


def step_budget(policy, reference, prompt, completion, rollout_rows, logps_rows, train_rows, eval_rows=None):
    """GiB per phase. rows are sequences (prompts x G)."""
    S, C, V = prompt + completion, completion, policy.vocab
    f32 = 4
    static = {
        "policy weights fp32": policy.params * f32,
        "gradients fp32": policy.params * f32,
        "AdamW moments fp32": 2 * policy.params * f32,
        "reference / teacher bf16": reference.params * 2,
        "sampler copy bf16": policy.params * 2,
    }
    rollout = {
        "KV cache": rollout_rows * S * policy.kv_bytes_per_token(),
        "prefill attention scores (1 layer, fp32)": rollout_rows * policy.heads * prompt * S * f32,
    }
    logps = {
        "reference logits fp32, all positions": logps_rows * S * V * f32,
        "log-softmax over completion": logps_rows * C * V * f32,
    }
    train = {
        "logits fp32 + their gradient": 2 * train_rows * S * V * f32,
        "log-softmax + entropy over completion": 2 * train_rows * C * V * f32,
        "remat block inputs (bf16)": policy.layers * train_rows * S * policy.d * 2,
        "attention scores of 1 block (fp32)": train_rows * policy.heads * S * S * f32,
    }
    phases = {"static": static, "rollout": rollout, "log-probs": logps, "train micro-step": train}
    if eval_rows:
        phases["eval loss, unchunked (Tunix 0.1.7)"] = {"logits fp32 + log-softmax": eval_rows * (S + C) * V * f32}
    return {p: {k: v / GiB for k, v in items.items()} for p, items in phases.items()}


def report(name, budget):
    static = sum(budget["static"].values())
    lines = [f"{name}", f"  {'phase / tensor':52} {'GiB':>8}"]
    for phase, items in budget.items():
        total = sum(items.values())
        peak = total if phase == "static" else static + total
        lines.append(f"  {phase:52} {total:8.1f}   (peak with static: {peak:.1f})")
        lines += [f"    {k:50} {v:8.1f}" for k, v in items.items()]
    return "\n".join(lines)


if __name__ == "__main__":
    G = 8
    print(report("X1 GRPO, A40 run 1 (crashed): rollout 4 prompts, logps 2, train 1, eval 64 unchunked",
                 step_budget(QWEN3_0P6B, QWEN3_0P6B, 1024, 1024, 4 * G, 2 * G, 1 * G, eval_rows=64 * G)))
    print()
    print(report("X1 GRPO, A100: rollout 8 prompts, logps 2, train 1 (eval chunked to train size)",
                 step_budget(QWEN3_0P6B, QWEN3_0P6B, 1024, 1024, 8 * G, 2 * G, 1 * G)))
    print()
    print(report("X2 OPD, A100: teacher Qwen3-4B as reference; same micro-batches",
                 step_budget(QWEN3_0P6B, QWEN3_4B, 1024, 1024, 8 * G, 2 * G, 1 * G)))
