"""Tunix's vanilla rollout (its native JAX sampler) plus stop strings.

Tunix's sampler stops only on EOS token ids. A base model prompted with a worked example does not emit EOS after
its answer: it goes on to write the next "Patient note:". vLLM handles this with stop strings (E0 used them).
Here the same rule is applied after sampling: each completion is cut before its first stop string, so the reward
and the policy gradient see only the answer. Tokens decoded past the stop are wasted compute, never signal.

Pass it to Tunix as the rollout engine:
    ClusterConfig(rollout_engine=functools.partial(StopStringRollout, stop=medcalc.STOP), ...)
"""
import dataclasses
import time

import jax.numpy as jnp
import numpy as np
from flax import nnx
from tunix.rl.rollout import base_rollout, vanilla_rollout

from ..models import lora
from .tunix_qwen3 import sampler_copy


def cut_at_stop(tokens, decode, stop):
    """Longest token prefix whose text ends before the first stop string. Returns (tokens, text, stopped)."""
    text = decode(list(tokens))
    hits = [i for i in (text.find(s) for s in stop) if i >= 0]
    if not hits:
        return tokens, text, False
    cut = min(hits)
    lo, hi = 0, len(tokens)              # binary search for the largest k with len(decode(tokens[:k])) <= cut
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if len(decode(list(tokens[:mid]))) <= cut:
            lo = mid
        else:
            hi = mid - 1
    return tokens[:lo], decode(list(tokens[:lo])), True


class StopStringRollout(vanilla_rollout.VanillaRollout):
    """Built by RLCluster for a custom rollout engine as cls(rollout_actor=, tokenizer=, mesh=, rollout_config=)."""

    def __init__(self, rollout_actor, tokenizer, mesh=None, rollout_config=None, stop=(), lora_scale=None):
        cfg = rollout_actor.config
        cache = base_rollout.CacheConfig(cache_size=rollout_config.kv_cache_size, num_layers=cfg.num_layers,
                                         num_kv_heads=cfg.num_kv_heads, head_dim=cfg.head_dim)
        self._lora_scale = lora_scale if lora.has_lora(rollout_actor) else None
        # under LoRA the base never changes: keep it once (bf16) and merge each synced A, B into it
        self._base = lora.base_params(rollout_actor, jnp.bfloat16) if self._lora_scale is not None else None
        super().__init__(sampler_copy(rollout_actor, lora_scale=lora_scale), tokenizer, cache)
        self.stop = tuple(stop)
        self.last = {}                   # stats of the latest generate() call, for logging

    def update_params(self, params, filter_types=None):
        if self._lora_scale is None:
            return super().update_params(params, filter_types)
        synced = lora.flatten(nnx.to_pure_dict(params) if isinstance(params, nnx.State) else params)
        full = lora.flatten(self._base)
        full.update({p: v for p, v in synced.items() if lora.is_lora_key(p[-1])})
        return super().update_params(lora.merge_lora(lora.unflatten(full), self._lora_scale, jnp.bfloat16), None)

    def generate(self, prompts, rollout_config, **kwargs):
        t0 = time.perf_counter()
        out = super().generate(prompts, rollout_config, **kwargs)
        seconds = time.perf_counter() - t0
        decoded = [len(t) for t in out.tokens]
        if self.stop:
            cut = [cut_at_stop(t, self._sampler.tokenizer.decode, self.stop) for t in out.tokens]
            tokens, text, stopped = map(list, zip(*cut))
            logprobs = None if out.logprobs is None else [lp[:len(t)] for lp, t in zip(out.logprobs, tokens)]
            out = dataclasses.replace(out, tokens=tokens, text=text, logprobs=logprobs)
        else:
            stopped = [False] * len(prompts)
        kept = [len(t) for t in out.tokens]
        self.last = {"stop_string_rate": float(np.mean(stopped)),
                     "hit_max_tokens_rate": float(np.mean([n >= rollout_config.max_tokens_to_generate for n in decoded])),
                     "decoded_tokens": int(sum(decoded)), "kept_tokens": int(sum(kept)), "seconds": seconds}
        return out
