#!/usr/bin/env bash
# Evaluate one policy with the E0 harness (vLLM, Verified test, greedy + pass@8) on all four conditions,
# so base, GRPO and OPD are read with the same ruler. Run on a pod with the vLLM venv:
#   bash scripts/eval_policy.sh <hub id or HF checkpoint dir> <name>
set -euo pipefail
MODEL=$1; NAME=$2
. /root/venv-vllm/bin/activate; . /workspace/.secrets; export HF_HOME=/workspace/hf
cd /workspace/RL-CV-LLM
for cond in "per-calculator values" "per-calculator note" "generic values" "generic note"; do
  set -- $cond
  start=$(date +%s)
  python experiments/x1_medcalc_grpo/e0_vllm.py --model "$MODEL" --name "$NAME" --data verified \
    --shots "$1" --input "$2" --wandb > "/workspace/eval_${NAME}_$1_$2.log" 2>&1
  echo "$NAME $1 $2 exit=$? wall=$(( $(date +%s) - start ))s" >> /workspace/eval_status.txt
done
echo "EVAL DONE $NAME" >> /workspace/eval_status.txt
