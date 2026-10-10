#!/usr/bin/env bash
# One-time environment setup on a pod (safe to re-run). Run on the pod after `scripts/sync.sh push`:
#   bash /workspace/RL-CV-LLM/scripts/pod_setup.sh
# Lesson from the first pod: pip onto the network volume took ~25 min; uv onto local disk takes well under a minute.
# So the venv lives on local disk (/root/venv-jax; redo after a pod restart), and the model cache on the volume.
set -euo pipefail
WS=/workspace
REPO=$WS/RL-CV-LLM
VENV=/root/venv-jax

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
export PATH="$HOME/.local/bin:$PATH"
[ -d $VENV ] || uv venv -q $VENV --python 3.12
export VIRTUAL_ENV=$VENV

# JAX side: Tunix first, then the CUDA build of JAX (the order the Tunix quickstart gives).
# cuda13 wheels need driver >= 580.
uv pip install -q google-tunix optax flax
uv pip install -q -U "jax[cuda13]"
# Hugging Face reference for parity tests: CPU-only PyTorch is enough (fp32 forward of a 0.6B model).
uv pip install -q torch --index-url https://download.pytorch.org/whl/cpu
uv pip install -q transformers safetensors huggingface_hub jinja2 pyarrow pandas wandb pytest
uv pip install -q -e $REPO

# Shell defaults: model cache on the volume, venv and secrets on login.
mkdir -p $WS/hf
grep -q 'RL-CV-LLM env' ~/.bashrc 2>/dev/null || cat >> ~/.bashrc <<EOF
# RL-CV-LLM env
export HF_HOME=/workspace/hf PYTHONWARNINGS=ignore
# JAX caps its allocator at 75% of GPU memory unless told otherwise (found when OPD hit a 59.4 GiB limit on 80 GB)
export XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.95
. $VENV/bin/activate
[ -f /workspace/.secrets ] && . /workspace/.secrets
cd $REPO
EOF

$VENV/bin/python - <<'EOF'
import jax, jax.numpy as jnp
print("jax", jax.__version__, "devices:", jax.devices())
x = jnp.ones((4096, 4096), jnp.bfloat16)
print("bf16 matmul ok:", float((x @ x)[0, 0]))
EOF
