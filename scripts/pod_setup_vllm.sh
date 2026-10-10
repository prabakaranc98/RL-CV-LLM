#!/usr/bin/env bash
# vLLM for evaluation, in its own venv next to the JAX one (same versions as the dev pod's E0 runs).
#   bash /workspace/RL-CV-LLM/scripts/pod_setup_vllm.sh
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null
[ -d /root/venv-vllm ] || uv venv -q /root/venv-vllm --python 3.12
VIRTUAL_ENV=/root/venv-vllm uv pip install -q vllm==0.31.0 pyarrow pandas wandb huggingface_hub
VIRTUAL_ENV=/root/venv-vllm uv pip install -q -e /workspace/RL-CV-LLM --no-deps
/root/venv-vllm/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__)"
