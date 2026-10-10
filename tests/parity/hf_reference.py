"""The Hugging Face model as ground truth for the JAX implementation.

Run the HF Qwen3 forward pass once (fp32, CPU, plain "eager" attention) and record the input and output
of each piece of one decoder layer. Each JAX piece is then fed the same input and must give the same output,
so an error is located in exactly one piece.
"""
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO = "Qwen/Qwen3-0.6B-Base"
TEXT = ("A 34-year-old female comes to the ED complaining of epigastric pain and intractable nausea. "
        "Her vitals are as follows: HR 97 beats/minute, BP 90/63 mm Hg. What is her mean arterial pressure?")


def capture(repo=REPO, text=TEXT, layer=0):
    """Returns a dict of numpy arrays: token ids, '<piece>.in' / '<piece>.out' for one layer, final logits."""
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32, attn_implementation="eager").eval()
    ids = tok(text, return_tensors="pt").input_ids
    block = model.model.layers[layer]
    pieces = {
        "embed": model.model.embed_tokens,
        "ln1": block.input_layernorm,            # RMSNorm before attention
        "q_norm": block.self_attn.q_norm,        # Qwen3's per-head RMSNorm on queries
        "k_norm": block.self_attn.k_norm,        # ... and on keys
        "attn": block.self_attn,
        "ln2": block.post_attention_layernorm,   # RMSNorm before the MLP
        "mlp": block.mlp,
        "layer": block,                          # the whole decoder layer
        "norm": model.model.norm,                # final RMSNorm
        "lm_head": model.lm_head,
    }
    rec = {}

    def recorder(name):
        def hook(module, args, kwargs, out):
            x = args[0] if args else kwargs["hidden_states"]
            rec[f"{name}.in"] = x.detach().numpy()
            rec[f"{name}.out"] = (out[0] if isinstance(out, tuple) else out).detach().numpy()
        return hook

    handles = [m.register_forward_hook(recorder(n), with_kwargs=True) for n, m in pieces.items()]
    with torch.no_grad():
        out = model(ids)
    for h in handles:
        h.remove()
    rec["ids"] = ids.numpy()
    rec["logits"] = out.logits.numpy()
    return rec


def compare(name, ours, ref, tol):
    """Print one parity line and return whether it passed. Errors are in the units of the tensor."""
    ours, ref = np.asarray(ours, np.float64), np.asarray(ref, np.float64)
    assert ours.shape == ref.shape, f"{name}: shape {ours.shape} vs {ref.shape}"
    max_abs = float(np.max(np.abs(ours - ref)))
    rel = max_abs / (float(np.max(np.abs(ref))) + 1e-12)
    ok = max_abs <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name:38s} max |diff| {max_abs:.2e}   relative {rel:.2e}   (tol {tol:.0e})")
    return ok
