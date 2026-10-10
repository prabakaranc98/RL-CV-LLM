"""Tunix's Qwen3 for the sampler: prefill computes logits for the last prompt position only.

Tunix's sampler asks for last-position-only logits when the model's __call__ accepts `decode_only_last_token`;
in Tunix 0.1.7 only Gemma 4 does. Without it, prefill materializes [B, prompt, 152K] logits: 18.6 GiB in bf16
for 32 prompts of 2,048 tokens, which ran the A40 out of memory in Block 2. The body is Tunix's Qwen3.__call__
with one added slice; training keeps Tunix's own class.
"""
import dataclasses

import jax
import jax.numpy as jnp
from flax import nnx
from tunix.models.qwen3 import model as tmodel

from ..models import lora


class Qwen3(tmodel.Qwen3):

    def __call__(self, input_tokens, positions, cache, attention_mask, output_hidden_states=False,
                 segment_ids=None, decode_only_last_token=False):
        new_cache = None if cache is None else {}
        x = self.embedder.encode(input_tokens)
        for i, layer in enumerate(self.layers):
            name = f"layer_{i}"
            layer_cache, x = layer(x, positions, cache[name] if cache else None, attention_mask,
                                   segment_ids=segment_ids)
            if cache is not None:
                new_cache[name] = layer_cache
        if decode_only_last_token:
            x = x[:, -1:]                     # prompts are left-padded, so the last position is real for every row
        x = self.final_norm(x)
        if output_hidden_states:
            self.sow(nnx.Intermediate, "all_hidden_states", x)
        logits = self.embedder.decode(x) if self.config.use_tied_embedding else self.lm_head(x)
        return jnp.astype(logits, jnp.float32), new_cache


def sampler_copy(model, dtype=jnp.bfloat16, lora_scale=None):
    """A copy of a Tunix Qwen3 as this class, weights cast to dtype. Decoding is memory-bound, so reading fp32
    weights every step would roughly double its time; Tunix's update_params casts each weight sync to dtype.
    A LoRA policy is copied as its dense equivalent W + (alpha / rank) A B (jaxpt.models.lora.merge_lora)."""
    # No remat in the sampler: there is no backward pass, and nnx.remat fails inside the decode while_loop.
    cfg = dataclasses.replace(model.config, remat_config=tmodel.RematConfig.NONE)
    graphdef, abstract = nnx.split(nnx.eval_shape(lambda: Qwen3(cfg, rngs=nnx.Rngs(0))))
    if lora.has_lora(model):
        state = abstract
        nnx.replace_by_pure_dict(state, lora.dense_params(model, lora_scale, dtype))
    else:
        _, state = nnx.split(model)
        state = jax.tree.map(lambda x: x.astype(dtype) if jnp.issubdtype(x.dtype, jnp.floating) else x, state)
    return nnx.merge(graphdef, state)
