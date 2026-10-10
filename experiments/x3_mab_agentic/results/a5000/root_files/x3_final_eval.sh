#!/usr/bin/env bash
# After X3 run 2: score the final policy and the base on the 92 held-out tasks with the official chat protocol
# (default chat template, thinking off, greedy, 2,048 tokens per turn, untruncated replies), official + lookup-gated.
# (run 3 done)
export HF_HOME=/workspace/hf
serve() {  # $1 model path or id
  . /root/venv-vllm/bin/activate
  nohup vllm serve "$1" --served-model-name policy --port 8031 --max-model-len 32768 --gpu-memory-utilization 0.85 > /root/vllm_final_$2.log 2>&1 &
  echo $! > /root/vllm_final.pid
  for i in $(seq 1 120); do curl -s localhost:8031/v1/models | grep -q policy && break; sleep 5; done
  deactivate
}
for which in final base; do
  if [ $which = final ]; then serve /root/x3_policy final; else serve Qwen/Qwen3-0.6B base; fi
  /root/venv-mab/bin/python /root/rollout_vllm.py --tasks /root/mab_train/heldout_official.json --samples 1 --temperature 0 \
    --max-tokens 2048 --url http://localhost:8031/v1/chat/completions --model policy --workers 16 --name heldout_official_$which > /root/x3_final_$which.log 2>&1
  kill $(cat /root/vllm_final.pid); sleep 10
done
echo "FINAL EVAL DONE" >> /root/x3_run.log
