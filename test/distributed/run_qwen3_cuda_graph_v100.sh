#!/usr/bin/env bash
# 服务器专用：同一公开模型执行器的 Eager 回归、严格 Graph 数值/生命周期及正式入口验收。
set -euo pipefail
TP_GRAPH_MODEL="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
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
  --target fastllm_process_tp_package fastllm_tp_decode_check fastllm_tp_check -j 4
TP_GRAPH_COMMON=(--worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
                 --build-cache build-process-tp-cu126/CMakeCache.txt)
TP_GRAPH_SESSION=(--generate build-process-tp-cu126/fastllm_tp_generate
                  --decode-check build-process-tp-cu126/fastllm_tp_decode_check)

python3 test/distributed/run_process_tp.py "${TP_GRAPH_COMMON[@]}" --cuda \
  --check build-process-tp-cu126/fastllm_tp_check
python3 test/distributed/run_qwen3_session.py "${TP_GRAPH_COMMON[@]}" "${TP_GRAPH_SESSION[@]}"
python3 test/distributed/run_qwen3_decode.py "${TP_GRAPH_COMMON[@]}" --cuda-graph \
  --check build-process-tp-cu126/fastllm_tp_decode_check
python3 test/distributed/run_qwen3_decode.py "${TP_GRAPH_COMMON[@]}" --cuda-graph \
  --check build-process-tp-cu126/fastllm_tp_decode_check --model "$TP_GRAPH_MODEL"
python3 test/distributed/run_qwen3_session.py "${TP_GRAPH_COMMON[@]}" "${TP_GRAPH_SESSION[@]}" --cuda-graph
python3 test/distributed/run_qwen3_session.py "${TP_GRAPH_COMMON[@]}" "${TP_GRAPH_SESSION[@]}" --cuda-graph \
  --model "$TP_GRAPH_MODEL"
python3 test/distributed/run_qwen3_entrypoint.py "${TP_GRAPH_COMMON[@]}" --cuda-graph \
  --generate build-process-tp-cu126/fastllm_tp_generate \
  --package-root build-process-tp-cu126/process-tp-python --model "$TP_GRAPH_MODEL"
echo "七组完成：通信底座；Eager 84；Graph Decode 30/1；Graph Session 96/16；正式入口 12。报告目录见 PASS 输出。"
