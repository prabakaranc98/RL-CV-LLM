"""On-policy distillation (OPD) as a Tunix policy loss, registered as "opd".

The student samples its own completions (the same rollout as GRPO); the teacher scores the student's tokens.
Each token gets its own advantage, with no reward involved:

    A_t = log pi_T(y_t | x, y_<t) - log pi_theta(y_t | x, y_<t)          (stop-gradient)

which is minus the per-token reverse-KL estimate on the student's samples. The loss is the policy-gradient
surrogate  -A_t * pi_theta(y_t) / pi_old(y_t)  (the ratio is 1 on-policy, so the gradient is -A_t grad log pi_theta),
aggregated exactly as in GRPO. So X1 and X2 differ only in where the per-token signal comes from:
a group-relative outcome reward (GRPO) or the teacher's per-token log-probs (OPD).
"RL in mechanics (sample, score, policy gradient), distillation in what it learns."

Tunix plumbing: the teacher is passed to RLCluster as the REFERENCE model, so train_example.ref_per_token_logps
holds the teacher's log-probs. Tunix computes them only when algo_config.beta != 0; this loss never reads beta.

References: Agarwal et al., "On-Policy Distillation of Language Models" (GKD, arXiv:2306.13649);
Thinking Machines, "On-Policy Distillation" (2025).
"""
import jax
import jax.numpy as jnp
from flax import nnx
from tunix.rl import algo_core, common, function_registry


@function_registry.register_policy_loss_fn("opd")
def opd_loss_fn(model, train_example, algo_config, pad_id, eos_id, **kwargs):
    graphdef, state = nnx.split(model)
    logps, logits = common.compute_per_token_logps(
        graphdef, state, prompt_tokens=train_example.prompt_ids, completion_tokens=train_example.completion_ids,
        pad_id=pad_id, eos_id=eos_id, stop_gradient=False, return_logits=True, temperature=algo_config.temperature)
    logps = logps.astype(jnp.float32)
    teacher = train_example.ref_per_token_logps.astype(jnp.float32)
    mask = train_example.completion_mask
    mode = algo_config.loss_agg_mode

    advantage = jax.lax.stop_gradient(teacher - logps)
    old = (jax.lax.stop_gradient(logps) if train_example.old_per_token_logps is None
           else train_example.old_per_token_logps.astype(jnp.float32))
    ratio = jnp.exp(logps - old)
    loss = common.aggregate_loss(-advantage * ratio, mask, mode)

    mean = lambda x: algo_core.masked_mean(x, mask)
    aux = {
        "kl": mean(logps - teacher),                      # reverse KL(student || teacher), k1 estimate on samples
        "pg_clipfrac": jnp.float32(0.0),
        "entropy": common.aggregate_loss(algo_core.compute_entropy_from_logits(logits), mask, mode),
        "ppo_kl": mean(old - logps),
        "advantage/nonzero_frac": mean((jnp.abs(advantage) > 1e-8).astype(jnp.float32)),
        "log_ratio/abs_mean": mean(jnp.abs(logps - old)),
        "opd/teacher_logp": mean(teacher),
        "opd/student_logp": mean(jax.lax.stop_gradient(logps)),
    }
    return loss, aux


@function_registry.register_policy_loss_fn("grpo_opd")
def grpo_opd_loss_fn(model, train_example, algo_config, pad_id, eos_id, **kwargs):
    """GRPO + on-policy distillation in one update: each token's advantage is its answer's group advantage (from the
    checker) plus the teacher-minus-student log-prob of that token, weighted by algo_config.distill_weight (lambda):

        A_t = A_group + lambda * (log pi_T - log pi_theta)_t        (stop-gradient)

    the weighted combination of an RL reward and a distillation term (GKD, Agarwal et al. 2024; KDRL, arXiv:2506.02208).
    The verifier says whether an answer is right; the teacher says which tokens are good."""
    graphdef, state = nnx.split(model)
    logps, logits = common.compute_per_token_logps(
        graphdef, state, prompt_tokens=train_example.prompt_ids, completion_tokens=train_example.completion_ids,
        pad_id=pad_id, eos_id=eos_id, stop_gradient=False, return_logits=True, temperature=algo_config.temperature)
    logps = logps.astype(jnp.float32)
    teacher = train_example.ref_per_token_logps.astype(jnp.float32)
    mask = train_example.completion_mask
    mode = algo_config.loss_agg_mode
    lam = getattr(algo_config, "distill_weight", 1.0)

    group = jnp.asarray(train_example.advantages, jnp.float32)[:, None]
    distill = jax.lax.stop_gradient(teacher - logps)
    advantage = group + lam * distill
    old = (jax.lax.stop_gradient(logps) if train_example.old_per_token_logps is None
           else train_example.old_per_token_logps.astype(jnp.float32))
    ratio = jnp.exp(logps - old)
    loss = common.aggregate_loss(-advantage * ratio, mask, mode)

    mean = lambda x: algo_core.masked_mean(x, mask)
    aux = {
        "kl": mean(logps - teacher),
        "pg_clipfrac": jnp.float32(0.0),
        "entropy": common.aggregate_loss(algo_core.compute_entropy_from_logits(logits), mask, mode),
        "ppo_kl": mean(old - logps),
        "advantage/nonzero_frac": mean((jnp.abs(advantage) > 1e-8).astype(jnp.float32)),
        "log_ratio/abs_mean": mean(jnp.abs(logps - old)),
        "opd/teacher_logp": mean(teacher),
        "opd/student_logp": mean(jax.lax.stop_gradient(logps)),
    }
    return loss, aux
