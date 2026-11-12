#!/usr/bin/env bash
# 只运行已有CUDA12.6产物；CUDA_VISIBLE_DEVICES由调用者按机器拓扑明确指定。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?请先按拓扑指定四张GPU，TP2自定义规约要求每对双向P2P及原生原子支持}"
DP_MODEL_PATH="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
if (( $# > 0 )); then shift; fi
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export NCCL_DEBUG=WARN
python3 test/distributed/benchmark_qwen3_dp.py \
  --model "$DP_MODEL_PATH" \
  --generate build-process-tp-cu126/fastllm_tp_generate \
  --worker build-process-tp-cu126/fastllm_tp_worker \
  --build-cache build-process-tp-cu126/CMakeCache.txt "$@"
