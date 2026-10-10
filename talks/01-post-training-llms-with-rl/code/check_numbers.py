"""Numbers that appear on slides and figures, computed (not typed) and checked by a second route.

Run:  python code/check_numbers.py
"""
import math

# KL values in the toy figure (figures/kl-bars): six responses, one reference, two policies
ref = [0.25, 0.20, 0.18, 0.15, 0.12, 0.10]
close = [0.28, 0.22, 0.17, 0.14, 0.11, 0.08]
far = [0.03, 0.04, 0.05, 0.05, 0.80, 0.03]
kl = lambda p, q: sum(a * math.log(a / b) for a, b in zip(p, q))
for d in (ref, close, far):
    assert abs(sum(d) - 1) < 1e-12
assert round(kl(close, ref), 3) == 0.006 and round(kl(far, ref), 2) == 1.23
# second route: KL = cross-entropy - entropy
H = lambda p: -sum(a * math.log(a) for a in p)
CE = lambda p, q: -sum(a * math.log(b) for a, b in zip(p, q))
assert abs((CE(far, ref) - H(far)) - kl(far, ref)) < 1e-12

# Qwen3-0.6B weight counts (config.json: d=1024, 28 layers, 16 q heads, 8 kv heads, head 128, ffn 3072, vocab 151936)
d, L, hq, hkv, dh, dff, V = 1024, 28, 16, 8, 128, 3072, 151_936
attn = d * hq * dh + 2 * d * hkv * dh + hq * dh * d          # W_Q, W_K, W_V, W_O
mlp = 3 * d * dff                                             # gate, up, down
norms = 2 * d + 2 * dh                                        # two RMSNorms + q_norm, k_norm
per_layer = attn + mlp + norms
non_emb = L * per_layer + d                                   # + final norm
emb = V * d                                                   # tied with the LM head
assert per_layer == 15_730_944 and round(mlp / per_layer, 2) == 0.60
assert round(non_emb / 1e9, 2) == 0.44                       # model card: 0.44B non-embedding
assert round((non_emb + emb) / 1e9, 1) == 0.6                # model card: 0.6B
# KV cache per token per layer: K and V, 8 heads of 128
assert 2 * hkv * dh == 2048 and 2 * hq * dh == 4096
print(f"KL close {kl(close, ref):.4f}, far {kl(far, ref):.4f}; per layer {per_layer:,}; "
      f"non-embedding {non_emb/1e9:.3f}B; total {(non_emb+emb)/1e9:.3f}B; all checks passed")

# Memory bill (figures/memory-bill): trained = 16 bytes/param (ZeRO: 2 + 2 + 12), frozen bf16 = 2 bytes/param
n_qwen = non_emb + emb
assert round(n_qwen / 1e9, 3) == 0.596
gb = lambda n, b: n * b / 1e9
ppo7 = 2 * gb(7e9, 16) + 2 * gb(7e9, 2); grpo7 = gb(7e9, 16) + gb(7e9, 2)
ppo06 = 2 * gb(n_qwen, 16) + 2 * gb(n_qwen, 2); grpo06 = gb(n_qwen, 16) + gb(n_qwen, 2)
assert (ppo7, grpo7) == (252.0, 126.0) and gb(7e9, 16) == 112.0 and gb(7e9, 2) == 14.0
assert round(ppo06, 1) == 21.5 and round(grpo06, 1) == 10.7
# second route: PPO = 2 x (16 + 2) bytes per parameter = 36 bytes/param
assert abs(ppo7 - 36 * 7e9 / 1e9) < 1e-9
# Feedback collection (figures/feedback-collection): a ranking of K responses gives K-choose-2 pairs
assert math.comb(4, 2) == 6 and math.comb(9, 2) == 36
print(f"memory: PPO 7B {ppo7:.0f} GB, GRPO 7B {grpo7:.0f} GB, PPO 0.6B {ppo06:.1f} GB, GRPO 0.6B {grpo06:.1f} GB; pairs ok")
