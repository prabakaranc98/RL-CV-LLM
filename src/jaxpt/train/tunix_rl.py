"""Pieces shared by X1 (GRPO) and X2 (OPD) on Tunix: the MedCalc reward and metrics, prompt batches, the cluster.

Tunix's contract, as read from tunix/rl (0.1.7):
  - a batch is a dict of numpy arrays with "prompts"; the learner repeats every field G times (np.repeat);
  - reward_fn(prompts, completions, mode, **fields) -> list of floats, one per completion;
  - metric_fn(prompts, completions, rewards, advantages, **fields) -> {name: (value, aggregate_fn)},
    logged under the current mode (train / eval).
"""
import collections
import dataclasses
import json
import os
import statistics
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from ..envs import medcalc
from ..tracking.text import is_degenerate


def typical_answers(train):
    """Per calculator, the typical training answer (median for decimals, mode otherwise): the constant guess."""
    by_calc = collections.defaultdict(list)
    for e in train:
        by_calc[e.calculator].append(e)
    out = {}
    for calc, pool in by_calc.items():
        if pool[0].output_type == "decimal":
            out[calc] = str(statistics.median(float(x.answer) for x in pool))
        else:
            out[calc] = collections.Counter(x.answer for x in pool).most_common(1)[0][0]
    return out


def _same_value(a, b):
    try:
        return abs(float(a) - float(b)) <= 1e-3 * max(1.0, abs(float(b)))
    except (TypeError, ValueError):
        return a == b


def batches(examples, prompt_of, prompts_per_step, seed=0, epochs=1, deadline=None):
    """Shuffled batches {"prompts": [P] str, "idx": [P] int}; idx points into `examples`.
    Stops at `deadline` (time.time() value): the learner ends cleanly when its data runs out."""
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = rng.permutation(len(examples))
        for i in range(0, len(order) - prompts_per_step + 1, prompts_per_step):
            if deadline is not None and time.time() > deadline:
                return
            idx = order[i:i + prompts_per_step]
            yield {"prompts": np.array([prompt_of(examples[j]) for j in idx]), "idx": idx}


class MedCalcScorer:
    """Reward and metrics for Tunix, plus a JSONL log of every completion (the record of how answers change)."""

    def __init__(self, examples, typical, group, log_path, rollout_stats=None):
        self.examples, self.typical, self.group = examples, typical, group
        self.log = open(log_path, "a")
        self.rollout_stats = rollout_stats or (lambda: {})    # () -> the rollout's stats of the latest call
        self.step_of = lambda: 0                              # set to the cluster's global step after creation
        self.on_eval_step = None                              # called once per held-out eval, with the step
        self._eval_steps = set()
        self._last = {}

    def reward(self, prompts, completions, mode="train", idx=None, **_):
        if mode == "eval" and self.on_eval_step and int(self.step_of()) not in self._eval_steps:
            self._eval_steps.add(int(self.step_of()))
            self.on_eval_step(int(self.step_of()))
        rewards, infos = [], []
        for text, i in zip(completions, idx):
            r, info = medcalc.reward(text, self.examples[int(i)])
            rewards.append(r)
            infos.append(info)
        self._last = {"mode": mode, "infos": infos}
        step = int(self.step_of())
        for text, i, r, info in zip(completions, idx, rewards, infos):
            e = self.examples[int(i)]
            self.log.write(json.dumps({"step": step, "mode": mode, "idx": int(i), "id": e.id,
                                       "calculator": e.calculator, "reward": r, "parsed": info["parsed"],
                                       "gold": e.answer, "text": text}) + "\n")
        self.log.flush()
        return rewards

    def metrics(self, prompts, completions, rewards, advantages, idx=None, **_):
        G = self.group
        r = np.asarray(rewards, float).reshape(-1, G)
        infos = self._last.get("infos", [{}] * len(completions))
        parsed = [i.get("parsed") for i in infos]
        calcs = [self.examples[int(i)].calculator for i in idx]
        valid = [p is not None for p in parsed]
        typical = [p is not None and _same_value(p, self.typical[c]) for p, c in zip(parsed, calcs)]
        groups = [parsed[k:k + G] for k in range(0, len(parsed), G)]
        out = {
            "medcalc/pass@1": float(r.mean()),
            f"medcalc/pass@{G}": float((r.max(1) > 0).mean()),
            "medcalc/zero_advantage_groups": float((r.std(1) == 0).mean()),
            "medcalc/format_valid": float(np.mean(valid)),
            "medcalc/typical_value_rate": float(np.sum(typical) / max(1, np.sum(valid))),
            "medcalc/distinct_answers_per_group": float(np.mean([len({p for p in g if p is not None}) for g in groups])),
            "medcalc/degenerate": float(np.mean([is_degenerate(t) for t in completions])),
        }
        out.update({f"rollout/{k}": float(v) for k, v in self.rollout_stats().items()})
        mem = device_memory()
        out.update(mem)
        if os.environ.get("JAXPT_PRINT_MEMORY"):          # stdout survives a crash; W&B's buffer does not
            print(f"[memory] step {int(self.step_of())} {self._last.get('mode')}: "
                  + ", ".join(f"{k.split('/')[1]} {v:.1f}" for k, v in mem.items()), flush=True)
        return {k: (v, np.mean) for k, v in out.items()}


def device_memory():
    """GPU memory from the allocator (needs XLA_PYTHON_CLIENT_PREALLOCATE=false to mean anything):
    in use now, the peak since the process started, and the largest single allocation so far."""
    st = jax.devices()[0].memory_stats() or {}
    gb = lambda k: st.get(k, 0) / 2**30
    return {"memory/in_use_gib": gb("bytes_in_use"), "memory/peak_gib": gb("peak_bytes_in_use"),
            "memory/largest_alloc_gib": gb("largest_alloc_size"), "memory/limit_gib": gb("bytes_limit")}


def mesh():
    """One device, with the axis names Tunix's sharding rules use. Auto axis types, as Tunix expects:
    jax.make_mesh now defaults to Explicit, under which Tunix's embedding gather raises a ShardingTypeError."""
    return jax.make_mesh((1, 1), ("fsdp", "tp"), axis_types=(jax.sharding.AxisType.Auto,) * 2)


def load_tunix_qwen3(repo, param_dtype=jnp.float32, compute_dtype=jnp.bfloat16, size="0p6b_base", remat=False):
    """A Tunix Qwen3 from a Hugging Face checkpoint: param_dtype weights (fp32 master copy for training),
    compute_dtype activations. remat=True recomputes each decoder block in the backward pass: without it the
    plain attention keeps a [B, H, T, T] fp32 score tensor per layer (2 GB per layer for 8 rows of 2,048 tokens)."""
    from tunix.models.qwen3 import model as tmodel, params as tparams
    from ..models.load import fetch
    cfg = getattr(tmodel.ModelConfig, f"qwen3_{size}")()
    cfg = dataclasses.replace(cfg, dtype=compute_dtype, param_dtype=param_dtype,
                              remat_config=tmodel.RematConfig.BLOCK if remat else tmodel.RematConfig.NONE)
    return tparams.create_model_from_safe_tensors(fetch(repo), cfg, mesh=None, dtype=param_dtype)


def run_dir(experiment_dir, name):
    d = Path(experiment_dir) / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def micro_batch_eval(trainer, rows):
    """Run the actor trainer's eval-loss step in chunks of `rows` sequences.

    Tunix 0.1.7 computes the eval loss on the whole eval set in one call: 64 problems x 8 samples x 1,024 tokens of
    fp32 logits over 152K tokens is 297 GiB, which crashed X1 run 1 right after its first update. The rollouts and
    rewards of the eval set are unaffected; only the loss / KL / entropy diagnostics are chunked here."""
    run_eval = trainer._run_eval

    def chunks(eval_ds):
        for item in eval_ds:
            for ex in (item if isinstance(item, (list, tuple)) else [item]):
                n = ex.prompt_ids.shape[0]
                for i in range(0, n, rows):
                    yield jax.tree.map(lambda x: x[i:i + rows] if getattr(x, "ndim", 0) and x.shape[0] == n else x, ex)

    trainer._run_eval = lambda eval_ds, eval_step_fn: run_eval(chunks(eval_ds), eval_step_fn)


def separate_reference_cluster():
    """RLCluster for a reference that is a *different model* (OPD's teacher).

    With a LoRA actor and the reference on the same mesh, Tunix 0.1.7 assumes the reference is the actor's base and
    computes reference log-probs from the actor's weights without LoRA. Right for GRPO (reference = base), silently
    wrong for OPD (the "teacher" would be the student's own base); it surfaced only because the structure check
    crashes on a 4B teacher vs a 0.6B student. This subclass removes the reference from backbone sharing."""
    from tunix.rl import rl_cluster as rlc

    class SeparateReferenceCluster(rlc.RLCluster):
        def _init_backbone_sharing_map(self, actor, reference=None):
            super()._init_backbone_sharing_map(actor, reference)
            ref = rlc.Role.REFERENCE
            for role in list(self._backbone_sharing_map):
                self._backbone_sharing_map[role] = [r for r in self._backbone_sharing_map[role] if r != ref]
            self._backbone_sharing_map[ref] = []

    return SeparateReferenceCluster


def tolerate_structure_mismatch():
    """Tunix 0.1.7's is_sharing_weights / is_sharing_backbone compare two models' states with jax.tree.map and raise
    when the structures differ (a LoRA policy vs its dense sampler copy; a 0.6B student vs a 4B teacher) instead of
    answering False. Different structures cannot share weights, so False is the correct answer."""
    from tunix.rl import utils as rl_utils
    for name in ("is_sharing_weights", "is_sharing_backbone"):
        check = getattr(rl_utils, name)
        if getattr(check, "_jaxpt_safe", False):
            continue

        def safe(m1, m2, _check=check):
            try:
                return _check(m1, m2)
            except (ValueError, TypeError):
                return False

        safe._jaxpt_safe = True
        setattr(rl_utils, name, safe)
