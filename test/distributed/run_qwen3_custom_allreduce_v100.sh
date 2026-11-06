#!/usr/bin/env bash
# 服务器专用：先回归 NCCL/Eager/Graph，再验收同一生产路径的 CUDA IPC 规约。
set -euo pipefail
TP_AR_MODEL="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

bash test/distributed/run_qwen3_cuda_graph_v100.sh "$TP_AR_MODEL"
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
TP_AR_COMMON=(--worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
              --build-cache build-process-tp-cu126/CMakeCache.txt --custom-allreduce)
TP_AR_SESSION=(--generate build-process-tp-cu126/fastllm_tp_generate
               --decode-check build-process-tp-cu126/fastllm_tp_decode_check)

python3 test/distributed/run_process_tp.py "${TP_AR_COMMON[@]}" --cuda \
  --check build-process-tp-cu126/fastllm_tp_check
python3 test/distributed/run_qwen3_decode.py "${TP_AR_COMMON[@]}" \
  --check build-process-tp-cu126/fastllm_tp_decode_check
python3 test/distributed/run_qwen3_decode.py "${TP_AR_COMMON[@]}" --cuda-graph \
  --check build-process-tp-cu126/fastllm_tp_decode_check
python3 test/distributed/run_qwen3_decode.py "${TP_AR_COMMON[@]}" --cuda-graph \
  --check build-process-tp-cu126/fastllm_tp_decode_check --model "$TP_AR_MODEL"
python3 test/distributed/run_qwen3_session.py "${TP_AR_COMMON[@]}" "${TP_AR_SESSION[@]}" --cuda-graph
python3 test/distributed/run_qwen3_session.py "${TP_AR_COMMON[@]}" "${TP_AR_SESSION[@]}" --cuda-graph --model "$TP_AR_MODEL"
python3 test/distributed/run_qwen3_entrypoint.py "${TP_AR_COMMON[@]}" --cuda-graph \
  --generate build-process-tp-cu126/fastllm_tp_generate \
  --package-root build-process-tp-cu126/process-tp-python --model "$TP_AR_MODEL"
echo "十四组完成：原 NCCL 七组；IPC 边界/故障；IPC Eager Decode 30；IPC Graph Decode 30/1；IPC Graph Session 96/16；正式入口 12。见各组 PASS 报告。"
