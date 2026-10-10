#!/usr/bin/env bash
# After a training run, on the same pod: greedy full-test accuracy on condition C at every exported checkpoint
# (the full-test hill-climbing curve), then the final policy on all four conditions with pass@k (eval_policy.sh).
#   bash scripts/eval_checkpoints.sh <run results dir> <name>
set -euo pipefail
RUN=$1; NAME=$2
. /root/venv-vllm/bin/activate; . /workspace/.secrets; export HF_HOME=/workspace/hf
cd /workspace/RL-CV-LLM
S=/workspace/eval_status.txt
for ck in $(ls -d "$RUN"/ckpt/step* | sort -V); do
  step=$(basename "$ck")
  start=$(date +%s)
  python experiments/x1_medcalc_grpo/e0_vllm.py --model "$ck/hf" --name "$NAME-$step" --data verified \
    --shots per-calculator --input values --passk-per-calc 0 > "/workspace/eval_${NAME}_${step}.log" 2>&1
  echo "$NAME $step C greedy exit=$? wall=$(( $(date +%s) - start ))s" >> $S
done
bash scripts/eval_policy.sh "$RUN/hf" "$NAME"
