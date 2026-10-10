"""The two training signals, checked on numbers small enough to verify by hand.

Runs the real loss code (Tunix's grpo_loss_fn and compute_advantages, our opd_loss_fn) with the model replaced by
a parameter that *is* the per-token log-prob, so jax.grad gives d loss / d log pi(token) directly.

  1. GRPO advantage: z-score within each group of G (ddof = 1); a group whose rewards are all equal gets 0.
  2. GRPO on-policy (ratio = 1), 'sequence-mean-token-mean':
         d loss / d log pi(o_i,t) = -A_i / |o_i| / n_sequences
     every token of a sequence shares its sequence's advantage, and a long answer gives each token less weight.
  3. OPD: d loss / d log pi(o_i,t) = -(log pi_T - log pi_theta)(o_i,t) / |o_i| / n_sequences
     tokens the teacher rates above the student are pushed up, tokens it rates below are pushed down, zero where
     they agree; no reward is involved.

Run on a pod (CPU is enough):  JAX_PLATFORMS=cpu pytest -q -s -p no:warnings tests/unit/test_rl_objectives.py
"""
import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from tunix.rl import algo_core, common
from tunix.rl.grpo.grpo_learner import GRPOConfig, TrainExample

from jaxpt.algorithms import opd

# 2 sequences, up to 4 completion tokens; the second sequence has 3 real tokens
MASK = jnp.array([[1, 1, 1, 1], [1, 1, 1, 0]], jnp.int32)
LOGP = jnp.array([[-0.5, -1.0, -2.0, -0.1], [-0.3, -0.7, -1.5, 0.0]], jnp.float32)


class LogProbs(nnx.Module):
    """Stands in for the policy: its one parameter is the per-token log-prob of the sampled tokens."""
    def __init__(self, logp):
        self.logp = nnx.Param(logp)


def fake_logps(graphdef, state, **_):
    logp = nnx.merge(graphdef, state).logp.value
    return logp, jnp.zeros(logp.shape + (5,))                 # logits only feed the entropy diagnostic


def grad_wrt_logp(loss_fn, example, cfg, monkeypatch):
    monkeypatch.setattr(common, "compute_per_token_logps", fake_logps)
    graphdef, state = nnx.split(LogProbs(LOGP))
    loss = lambda st: loss_fn(nnx.merge(graphdef, st), example, cfg, pad_id=0, eos_id=1)[0]
    return np.asarray(jax.grad(loss)(state)["logp"].value)


def example(advantages, ref=None):
    ids = jnp.ones((2, 4), jnp.int32)
    return TrainExample(prompt_ids=ids, prompt_mask=ids, completion_ids=ids, completion_mask=MASK,
                        advantages=advantages, ref_per_token_logps=ref, old_per_token_logps=None)


def test_grpo_advantages_are_group_z_scores():
    r = np.array([1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], float)        # 2 groups of 8
    a = algo_core.compute_advantages(r, num_generations=8)
    g = r[:8]
    np.testing.assert_allclose(a[:8], (g - g.mean()) / (g.std(ddof=1) + 1e-6), rtol=1e-6)
    assert np.all(a[8:] == 0), "a group with equal rewards must carry no signal"
    print(f"\n  group 1 rewards {g.astype(int).tolist()} -> advantages {np.round(a[:8], 3).tolist()}")


def test_grpo_gradient_is_shared_advantage_over_length(monkeypatch):
    A = jnp.array([1.5, -0.5])
    cfg = GRPOConfig(num_generations=2, beta=0.0, loss_agg_mode="sequence-mean-token-mean")
    cfg.temperature = 1.0
    g = grad_wrt_logp(algo_core.grpo_loss_fn, example(A), cfg, monkeypatch)
    lengths = np.asarray(MASK.sum(1))
    expected = -np.asarray(A)[:, None] / lengths[:, None] / 2 * np.asarray(MASK)
    np.testing.assert_allclose(g, expected, rtol=1e-6, atol=1e-7)
    print(f"\n  d loss / d log pi:\n{np.round(g, 4)}")


def test_opd_gradient_is_teacher_minus_student(monkeypatch):
    teacher = LOGP + jnp.array([[0.4, 0.0, -0.6, 0.2], [0.0, 1.0, -0.2, 0.0]])  # where the teacher agrees: 0
    cfg = GRPOConfig(num_generations=2, beta=1.0, policy_loss_fn="opd", loss_agg_mode="sequence-mean-token-mean")
    cfg.temperature = 1.0
    g = grad_wrt_logp(opd.opd_loss_fn, example(jnp.zeros(2), ref=teacher), cfg, monkeypatch)
    lengths = np.asarray(MASK.sum(1))
    expected = -np.asarray(teacher - LOGP) / lengths[:, None] / 2 * np.asarray(MASK)
    np.testing.assert_allclose(g, expected, rtol=1e-5, atol=1e-7)
    assert g[0, 1] == 0 and g[1, 0] == 0, "no push where teacher and student agree"
    print(f"\n  d loss / d log pi:\n{np.round(g, 4)}")


def test_grpo_opd_gradient_is_group_plus_teacher(monkeypatch):
    A = jnp.array([1.5, -0.5])
    teacher = LOGP + jnp.array([[0.4, 0.0, -0.6, 0.2], [0.0, 1.0, -0.2, 0.0]])
    cfg = GRPOConfig(num_generations=2, beta=1.0, policy_loss_fn="grpo_opd", loss_agg_mode="sequence-mean-token-mean")
    cfg.temperature = 1.0
    cfg.distill_weight = 0.5
    g = grad_wrt_logp(opd.grpo_opd_loss_fn, example(A, ref=teacher), cfg, monkeypatch)
    lengths = np.asarray(MASK.sum(1))[:, None]
    adv = np.asarray(A)[:, None] + 0.5 * np.asarray(teacher - LOGP)
    np.testing.assert_allclose(g, -adv / lengths / 2 * np.asarray(MASK), rtol=1e-5, atol=1e-7)
    print(f"\n  d loss / d log pi (A_group + 0.5 x teacher gap):\n{np.round(g, 4)}")
