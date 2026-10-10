#!/usr/bin/env bash
# Keep the pod in step with this repo.
#   scripts/sync.sh push      code -> pod   (src, tests, experiments configs, scripts; never notes/, .venv, results)
#   scripts/sync.sh pull      results <- pod (experiments/*/results)
#   scripts/sync.sh secrets   W&B key from .env -> /workspace/.secrets on the pod (never printed)
#   scripts/sync.sh ssh       open a shell on the pod
# The pod address lives in scripts/pod.env (git-ignored): POD_HOST=root@<ip>  POD_PORT=<port>
set -euo pipefail
cd "$(dirname "$0")/.."
. "${POD_ENV:-scripts/pod.env}"     # dev pod by default; POD_ENV=scripts/pod-train.env for the train pod
REMOTE=/workspace/RL-CV-LLM
SSH="ssh -p $POD_PORT -o StrictHostKeyChecking=accept-new"

case "${1:-}" in
  push)
    rsync -az --no-owner --no-group --delete -e "$SSH" \
      --exclude '.git/' --exclude '.venv/' --exclude 'notes/' --exclude 'talks/' --exclude '.env' \
      --exclude '__pycache__/' --exclude '*.egg-info/' --exclude 'wandb/' --exclude 'scripts/pod*.env' \
      --exclude 'experiments/*/results/' \
      ./ "$POD_HOST:$REMOTE/"
    echo "pushed code to $POD_HOST:$REMOTE" ;;
  pull)
    rsync -az --no-owner --no-group -e "$SSH" --include='*/' --include='results/***' --exclude='*' \
      "$POD_HOST:$REMOTE/experiments/" ./experiments/
    echo "pulled results into experiments/*/results" ;;
  secrets)
    set -a; . ./.env; set +a
    key="${WANDB_API_KEY:-${WAND_API_KEY:-}}"
    [ -n "$key" ] || { echo "no W&B key in .env"; exit 1; }
    printf 'export WANDB_API_KEY=%s\n' "$key" | $SSH "$POD_HOST" 'umask 077; cat > /workspace/.secrets'
    echo "W&B key written to /workspace/.secrets (mode 600)" ;;
  ssh)
    exec $SSH "$POD_HOST" ;;
  *)
    echo "usage: scripts/sync.sh push|pull|secrets|ssh"; exit 2 ;;
esac
