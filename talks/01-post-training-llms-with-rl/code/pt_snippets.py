"""Every code snippet shown on the slides of "Post-training LLMs with RL", with tests.

Run:  python code/pt_snippets.py      (JAX on CPU is enough)
The slide versions are copied from the functions below; keep the two in sync.
"""
import math

import jax
import jax.numpy as jnp


# --- slide: "Everything we train is built from token log-probabilities" -------------------------
def token_logprobs(logits, targets):
    """log pi(y_t | x, y_<t) for every position.
    logits: [B, T, V] scores for the next token; targets: [B, T] the token that came next."""
    logp = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(logp, targets[..., None], axis=-1)[..., 0]   # [B, T]


def response_logprob(logits, targets, mask):
    """log pi(y | x): sum over response tokens only (mask = 1 on the response, 0 elsewhere)."""
    return (token_logprobs(logits, targets) * mask).sum(axis=-1)          # [B]


# --- slide: "The GRPO advantage is the reward, standardized within the group" -------------------
def grpo_advantages(rewards, eps=1e-6):
    """rewards: [num_prompts, G], one row per prompt's group of G responses."""
    mean = rewards.mean(axis=1, keepdims=True)
    std = rewards.std(axis=1, keepdims=True)
    return (rewards - mean) / (std + eps)


# --- slide: PPO / GRPO clipped loss --------------------------------------------------------------
def clipped_pg_loss(logp, logp_old, adv, mask, eps=0.2):
    """Negative clipped surrogate, averaged over response tokens (token-level mean, as in DAPO;
    the original GRPO averages within each response first). All inputs: [B, T]."""
    ratio = jnp.exp(logp - logp_old)                        # rho_t
    unclipped = ratio * adv
    clipped = jnp.clip(ratio, 1 - eps, 1 + eps) * adv
    per_token = jnp.minimum(unclipped, clipped)
    return -(per_token * mask).sum() / mask.sum()


def kl_k3(logp, logp_ref):
    """Per-token KL estimate used by GRPO (DeepSeekMath eq. 4): ratio - log ratio - 1 >= 0."""
    log_ratio = logp_ref - logp
    return jnp.exp(log_ratio) - log_ratio - 1


# --- slide: "On-policy distillation: the teacher scores every token" ----------------------------
def opd_advantages(logp_student, logp_teacher):
    """Per-token advantage: teacher log-prob minus student log-prob of the sampled token, [B, T].
    Averaged over the student's samples, minus this is the reverse KL(student || teacher)."""
    return jax.lax.stop_gradient(logp_teacher - logp_student)


# --- slide: "A GRPO update in JAX" ----------------------------------------------------------------
def grpo_loss(params, batch, apply_fn, beta=0.0, eps=0.2):
    logits = apply_fn(params, batch["ids"])[:, :-1]          # scores for the next token
    logp = token_logprobs(logits, batch["ids"][:, 1:])        # log pi_theta(y_t | x, y_<t)
    adv = batch["adv"][:, None]                               # one advantage per response
    loss = clipped_pg_loss(logp, batch["logp_old"], adv, batch["mask"], eps)
    if beta > 0:                                              # optional KL penalty to pi_ref
        loss += beta * (kl_k3(logp, batch["logp_ref"]) * batch["mask"]).sum() / batch["mask"].sum()
    return loss


def make_train_step(apply_fn, lr=1e-2):
    @jax.jit
    def train_step(params, batch):
        loss, grads = jax.value_and_grad(grpo_loss)(params, batch, apply_fn)
        params = jax.tree.map(lambda p, g: p - lr * g, params, grads)   # plain SGD on the pytree
        return params, loss
    return train_step


# =================================================================================================
def _toy_model(vocab=11, d=8, seed=0):
    k1, k2 = jax.random.split(jax.random.key(seed))
    params = {"embed": 0.1 * jax.random.normal(k1, (vocab, d)), "out": 0.1 * jax.random.normal(k2, (d, vocab))}
    apply_fn = lambda p, ids: p["embed"][ids] @ p["out"]         # [B, T, V]
    return params, apply_fn


def test_all():
    # log-probs: agree with a direct softmax
    logits = jnp.array([[[2.0, 0.0, -1.0], [0.5, 0.5, 0.0]]])
    targets = jnp.array([[0, 2]])
    lp = token_logprobs(logits, targets)
    direct = jnp.log(jax.nn.softmax(logits, axis=-1))[0, [0, 1], [0, 2]]
    assert jnp.allclose(lp[0], direct, atol=1e-6)
    assert jnp.allclose(response_logprob(logits, targets, jnp.array([[0.0, 1.0]])), direct[1], atol=1e-6)

    # GRPO advantages: the slide's worked example and the zero-variance case
    adv = grpo_advantages(jnp.array([[1.0, 0.0, 0.0, 1.0], [1.0, 1.0, 1.0, 1.0]]))
    assert jnp.allclose(adv[0], jnp.array([1.0, -1.0, -1.0, 1.0]), atol=1e-5)
    assert jnp.allclose(adv[1], 0.0)

    # clipped loss: with rho = 1 it equals the plain policy-gradient objective (negated mean of adv)
    a = jnp.array([[1.0, -1.0]]); m = jnp.ones((1, 2)); z = jnp.zeros((1, 2))
    assert jnp.allclose(clipped_pg_loss(z, z, a, m), 0.0)
    # clipping: a large ratio with positive advantage is capped at 1 + eps
    big = jnp.log(jnp.array([[3.0, 1.0]]))
    assert jnp.allclose(clipped_pg_loss(big, z, jnp.array([[1.0, 0.0]]), jnp.array([[1.0, 0.0]])), -1.2, atol=1e-6)

    # k3 KL estimate: zero when equal, non-negative otherwise
    assert jnp.allclose(kl_k3(z, z), 0.0)
    assert bool((kl_k3(jnp.array([0.0, -1.0, -3.0]), jnp.array([-2.0, 0.0, -3.0])) >= 0).all())

    # OPD: exact reverse KL on a tiny distribution equals E_student[-A]
    p_s = jnp.array([0.5, 0.3, 0.2]); p_t = jnp.array([0.6, 0.3, 0.1])
    A = opd_advantages(jnp.log(p_s), jnp.log(p_t))
    assert math.isclose(float((p_s * -A).sum()), float((p_s * jnp.log(p_s / p_t)).sum()), rel_tol=1e-6)

    # one jit-compiled GRPO step on a toy model: finite loss, same pytree structure, loss moves
    params, apply_fn = _toy_model()
    ids = jnp.array([[1, 2, 3, 4], [1, 2, 5, 6], [1, 2, 3, 7], [1, 2, 8, 9]])
    logp0 = token_logprobs(apply_fn(params, ids)[:, :-1], ids[:, 1:])
    batch = {"ids": ids, "logp_old": logp0, "logp_ref": logp0,
             "adv": grpo_advantages(jnp.array([[1.0, 0.0, 1.0, 0.0]]))[0],
             "mask": jnp.array([[0, 1, 1]] * 4, dtype=jnp.float32)}
    step = make_train_step(apply_fn)
    p1, l1 = step(params, batch)
    _, l2 = step(p1, batch)
    assert jnp.isfinite(l1) and jax.tree.structure(p1) == jax.tree.structure(params)
    assert float(l2) < float(l1), (l1, l2)
    print("all snippet tests passed; loss", float(l1), "->", float(l2))


if __name__ == "__main__":
    test_all()
