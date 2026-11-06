#!/usr/bin/env bash
# 服务器专用：构建正式包入口，回归原生会话后验证模型工厂与 CLI。
set -euo pipefail
TP_ENTRY_MODEL="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
python3 -c 'import jinja2; from transformers import AutoTokenizer'

cmake -S . -B build-process-tp-cu126 \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
  -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 \
  --target fastllm_process_tp_package fastllm_tp_decode_check -j 4
TP_ENTRY_ARGS=(--generate build-process-tp-cu126/fastllm_tp_generate
               --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
               --build-cache build-process-tp-cu126/CMakeCache.txt)
python3 test/distributed/run_qwen3_session.py "${TP_ENTRY_ARGS[@]}" \
  --decode-check build-process-tp-cu126/fastllm_tp_decode_check
python3 test/distributed/run_qwen3_session.py "${TP_ENTRY_ARGS[@]}" \
  --decode-check build-process-tp-cu126/fastllm_tp_decode_check --model "$TP_ENTRY_MODEL"
python3 test/distributed/run_qwen3_entrypoint.py "${TP_ENTRY_ARGS[@]}" \
  --package-root build-process-tp-cu126/process-tp-python --model "$TP_ENTRY_MODEL"
echo "三组验收完成：合成/真实模型常驻回归、正式模型工厂及 CLI 入口（12 项）。"
