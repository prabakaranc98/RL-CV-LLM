"""Export round trip: Hugging Face checkpoint -> Tunix Qwen3 -> export_hf tensors gives back every original tensor,
bit for bit (bf16 -> fp32 -> bf16 is lossless). This is what lets vLLM evaluate trained policies.

Run on a pod:  pytest -s -p no:warnings tests/unit/test_export_roundtrip.py
"""
import dataclasses
import glob
import os

import jax.numpy as jnp
import ml_dtypes
import numpy as np
from safetensors.numpy import load_file

from jaxpt.models.export import hf_tensors
from jaxpt.models.load import fetch

REPO = "Qwen/Qwen3-0.6B-Base"


def test_export_roundtrip_is_exact():
    from tunix.models.qwen3 import model as tmodel, params as tparams
    path = fetch(REPO)
    cfg = dataclasses.replace(tmodel.ModelConfig.qwen3_0p6b_base(), param_dtype=jnp.float32)
    model = tparams.create_model_from_safe_tensors(path, cfg, mesh=None, dtype=jnp.float32)
    ours = hf_tensors(model, ml_dtypes.bfloat16)
    ref = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        ref.update(load_file(f))
    assert set(ours) == set(ref), (set(ours) ^ set(ref))
    bad = [k for k in ref if ours[k].shape != ref[k].shape or not np.array_equal(ours[k].view(np.uint16), ref[k].view(np.uint16))]
    print(f"\n  {len(ref)} tensors, {sum(v.size for v in ref.values()):,} values; mismatches: {len(bad)}")
    assert not bad, bad[:5]
