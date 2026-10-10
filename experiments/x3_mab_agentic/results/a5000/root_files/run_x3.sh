#!/usr/bin/env bash
export HF_HOME=/workspace/hf PYTHONWARNINGS=ignore
. /workspace/.secrets
cd /root
/root/venv-jax/bin/python /workspace/RL-CV-LLM/experiments/x3_mab_agentic/train_agentic_grpo.py --run sft-then-grpo-lora32 --sft-demos /root/mab_train/demos.jsonl --sft-epochs 2 --sft-lr 1e-4 --steps 200 --tasks-per-step 8 --group 8 --eval-every 5 --hours 0.25 --wandb
echo "X3 DONE"
