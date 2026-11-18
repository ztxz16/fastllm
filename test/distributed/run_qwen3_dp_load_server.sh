#!/usr/bin/env bash
# 服务器验收：编译原生配套产物，再执行真实Token、KV压力、HTTP及旧策略回归。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DP_PYTHON="${DP_PYTHON:-/venv/fastllm-reference-cu126/bin/python3}"
DP_BUILD_DIR="${DP_BUILD_DIR:-build-attention-ep-cu126}"
DP_MODEL_DIR="${DP_MODEL_DIR:-/home/models/Qwen3-1.7B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export USE_TORCH=0

cmake --build "$DP_BUILD_DIR" --target fastllm_tp_generate fastllm_tp_worker -j 4
"$DP_PYTHON" test/distributed/run_parallel_config.py --execution-tests
DP_COMMON=(--model "$DP_MODEL_DIR" --generate "$DP_BUILD_DIR/fastllm_tp_generate"
           --worker "$DP_BUILD_DIR/fastllm_tp_worker" --build-cache "$DP_BUILD_DIR/CMakeCache.txt")
for layout in '2 1' '4 1' '2 2'; do
    read -r DP_SIZE TP_SIZE <<< "$layout"
    "$DP_PYTHON" test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp "$DP_SIZE" --tp "$TP_SIZE"
    "$DP_PYTHON" test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp "$DP_SIZE" --tp "$TP_SIZE" --cuda-graph --prefix-cache
done
"$DP_PYTHON" test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp 2 --tp 1 --dp-routing round_robin
"$DP_PYTHON" test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp 2 --tp 1 --dp-routing prefix --prefix-cache
