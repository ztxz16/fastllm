#!/usr/bin/env bash
# 双 V100 服务器：回归常驻会话（含新增原生取消），再验证真实 HTTP/SSE 输出和生命周期。
set -euo pipefail
TP_HTTP_MODEL_DIR="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
hash -r
python3 -c 'import jinja2; from transformers import AutoTokenizer'

cmake -S . -B build-process-tp-cu126 \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 \
  --target fastllm_tp_worker fastllm_tp_generate fastllm_tp_decode_check -j 4

TP_HTTP_ARGS=(--generate build-process-tp-cu126/fastllm_tp_generate
              --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
              --build-cache build-process-tp-cu126/CMakeCache.txt)
python3 test/distributed/run_qwen3_session.py "${TP_HTTP_ARGS[@]}" \
  --decode-check build-process-tp-cu126/fastllm_tp_decode_check
python3 test/distributed/run_qwen3_session.py "${TP_HTTP_ARGS[@]}" \
  --decode-check build-process-tp-cu126/fastllm_tp_decode_check --model "$TP_HTTP_MODEL_DIR"
python3 test/distributed/run_qwen3_http.py "${TP_HTTP_ARGS[@]}" --model "$TP_HTTP_MODEL_DIR"
echo "三组验收完成：合成/真实模型常驻与取消/流式对照、HTTP/SSE 服务。报告目录见 PASS 输出。"
