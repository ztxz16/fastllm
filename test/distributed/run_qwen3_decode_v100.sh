#!/usr/bin/env bash
# 仅供双 V100 服务器手动执行：会编译 C++/CUDA。
set -euo pipefail

TP_DECODE_MODEL_DIR="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
hash -r

# 先回归已经通过的五组验收，同时重编译更新后的 worker。
bash test/distributed/run_qwen3_prefill_v100.sh "$TP_DECODE_MODEL_DIR"
cmake --build build-process-tp-cu126 --target fastllm_tp_decode_check -j 4
TP_DECODE_COMMON_ARGS=(--check build-process-tp-cu126/fastllm_tp_decode_check
                      --worker build-process-tp-cu126/fastllm_tp_worker
                      --devices 0,1 --build-cache build-process-tp-cu126/CMakeCache.txt)
python3 test/distributed/run_qwen3_decode.py "${TP_DECODE_COMMON_ARGS[@]}"
python3 test/distributed/run_qwen3_decode.py "${TP_DECODE_COMMON_ARGS[@]}" --model "$TP_DECODE_MODEL_DIR"
echo "七组验收均通过；各组报告目录见上方 PASS 输出。"
