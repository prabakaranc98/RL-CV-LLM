"""Per-token log-probs and entropy without materializing [rows, tokens, vocab] logits.

Tunix 0.1.7's common.compute_per_token_logps asks the model for fp32 logits at every position, slices the completion,
then log-softmaxes; the loss computes the entropy with two more softmaxes over the same tensor, and the backward pass
holds the logits' gradient. At 0.6B parameters and a 152K vocabulary these tensors are the largest thing in memory
(measured 44 GiB peak per 8-row micro-batch on the A100, against 11 GiB for the model with its optimizer).

Here the model returns its final hidden states instead (Tunix's output_hidden_states sow; the unused full-vocabulary
matmul is dead code and XLA drops it), and the LM head, log-softmax and entropy run over `chunk` positions at a time
under jax.checkpoint, so the forward and the backward each hold one [rows, chunk, vocab] block.

install() swaps it in for every Tunix caller (policy loss, reference / teacher log-probs, eval): they all look the
function up as common.compute_per_token_logps. It returns (log-probs, entropy) where Tunix returned (log-probs, logits),
so compute_entropy_from_logits is swapped for a pass-through on 2-D input. Sequence packing falls back to Tunix.
"""
import jax
import jax.numpy as jnp
from flax import nnx
from tunix.rl import algo_core, common

_tunix_logps = common.compute_per_token_logps
_tunix_entropy = algo_core.compute_entropy_from_logits


def chunk_stats(hidden, head, targets, temperature, chunk=128):
    """hidden [B, C, D] (final-normed), head [D, V], targets [B, C] -> (log p(target) [B, C], entropy [B, C]) in fp32."""
    B, C, D = hidden.shape
    n = -(-C // chunk)
    pad = n * chunk - C
    h = jnp.pad(hidden, ((0, 0), (0, pad), (0, 0))).reshape(B, n, chunk, D).swapaxes(0, 1)
    y = jnp.pad(targets, ((0, 0), (0, pad))).reshape(B, n, chunk).swapaxes(0, 1)

    @jax.checkpoint
    def one(hy):
        hc, yc = hy
        logits = jnp.dot(hc, head).astype(jnp.float32)                          # [B, chunk, V]
        if temperature not in (0.0, 1.0):
            logits = logits / temperature
        lse = jax.nn.logsumexp(logits, axis=-1)
        logp = jnp.take_along_axis(logits, yc[..., None], axis=-1)[..., 0] - lse
        entropy = lse - jnp.sum(jax.nn.softmax(logits, axis=-1) * logits, axis=-1)
        return logp, entropy

    logp, entropy = jax.lax.map(one, (h, y))                                     # [n, B, chunk]
    unchunk = lambda x: x.swapaxes(0, 1).reshape(B, n * chunk)[:, :C]
    return unchunk(logp), unchunk(entropy)


def lm_head(model):
    """The [D, V] output matrix, in the model's compute dtype (as Tunix's Embedder.decode / lm_head use it)."""
    dtype = model.config.dtype
    if model.config.use_tied_embedding:
        return model.embedder.input_embedding.value.astype(dtype).T
    return model.lm_head.w.value.astype(dtype)


def compute_per_token_logps(graphdef, state, prompt_tokens, completion_tokens, pad_id, eos_id,
                            stop_gradient=True, return_logits=False, segment_ids=None, segment_positions=None,
                            temperature=1.0, images=None, chunk=128, **kwargs):
    if segment_ids is not None or images is not None:
        return _tunix_logps(graphdef, state, prompt_tokens, completion_tokens, pad_id, eos_id,
                            stop_gradient=stop_gradient, return_logits=return_logits, segment_ids=segment_ids,
                            segment_positions=segment_positions, temperature=temperature, images=images, **kwargs)
    model = nnx.merge(graphdef, state)
    tokens, positions, mask, _ = common.process_ids(prompt_tokens, completion_tokens, pad_id, eos_id)
    model(tokens, positions=positions, cache=None, attention_mask=mask, output_hidden_states=True)
    hidden = jax.tree.leaves(nnx.pop(model, nnx.Intermediate))[-1]               # [B, S, D] after the final norm
    C = completion_tokens.shape[1]
    logp, entropy = chunk_stats(hidden[:, -C - 1:-1].astype(model.config.dtype), lm_head(model),
                                tokens[:, -C:], temperature, chunk)
    if stop_gradient:
        logp, entropy = jax.lax.stop_gradient(logp), jax.lax.stop_gradient(entropy)
    return (logp, entropy) if return_logits else logp


def _entropy(x):
    return x if x.ndim == 2 else _tunix_entropy(x)          # 2-D: already the per-token entropy


def install():
    common.compute_per_token_logps = compute_per_token_logps
    algo_core.compute_entropy_from_logits = _entropy


def uninstall():
    common.compute_per_token_logps = _tunix_logps
    algo_core.compute_entropy_from_logits = _tunix_entropy
