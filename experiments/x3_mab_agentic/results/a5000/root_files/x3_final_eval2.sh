#!/usr/bin/env bash
# Official-protocol re-score on the 92 held-out tasks, guarded: one server at a time, its served path verified.
export HF_HOME=/workspace/hf
for pair in "final:/root/x3_policy" "base:Qwen/Qwen3-0.6B"; do
  name=${pair%%:*}; model=${pair#*:}
  for i in $(seq 1 30); do used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits); [ "$used" -lt 1000 ] && break; sleep 2; done
  setsid /root/venv-vllm/bin/vllm serve "$model" --served-model-name policy --port 8031 --max-model-len 32768 --gpu-memory-utilization 0.85 > /root/vllm_final2_$name.log 2>&1 &
  pgid=$!
  ok=0
  for i in $(seq 1 120); do
    root=$(curl -s localhost:8031/v1/models | python3 -c "import json,sys; print(json.load(sys.stdin)[\"data\"][0][\"root\"])" 2>/dev/null)
    [ "$root" = "$model" ] && ok=1 && break
    sleep 3
  done
  if [ $ok = 1 ]; then
    echo "serving $root for $name"
    /root/venv-mab/bin/python /root/rollout_vllm.py --tasks /root/mab_train/heldout_official.json --samples 1 --temperature 0 \
      --max-tokens 2048 --url http://localhost:8031/v1/chat/completions --model policy --workers 16 --name heldout_official2_$name > /root/x3_final2_$name.log 2>&1
    grep heldout_official2 /root/x3_final2_$name.log
  else
    echo "server for $name did not come up with root $model"
  fi
  kill -9 -$pgid 2>/dev/null; sleep 5
done
echo "FINAL EVAL2 DONE"
