#!/usr/bin/env bash
# 仅在双 V100 服务器运行：回归生成入口，并验收常驻会话及连续文本问答。
set -euo pipefail

TP_SESSION_MODEL_DIR="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
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
  --target fastllm_tp_worker fastllm_tp_generate fastllm_tp_decode_check fastllm_tp_sampling_check -j 4

TP_SESSION_ARGS=(--generate build-process-tp-cu126/fastllm_tp_generate
                 --decode-check build-process-tp-cu126/fastllm_tp_decode_check
                 --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
                 --build-cache build-process-tp-cu126/CMakeCache.txt)
python3 test/distributed/run_qwen3_generation.py "${TP_SESSION_ARGS[@]}" \
  --sampling-check build-process-tp-cu126/fastllm_tp_sampling_check
python3 test/distributed/run_qwen3_generation.py "${TP_SESSION_ARGS[@]}" \
  --sampling-check build-process-tp-cu126/fastllm_tp_sampling_check --model "$TP_SESSION_MODEL_DIR"
python3 test/distributed/run_qwen3_session.py "${TP_SESSION_ARGS[@]}"
python3 test/distributed/run_qwen3_session.py "${TP_SESSION_ARGS[@]}" --model "$TP_SESSION_MODEL_DIR"

python3 tools/qwen3_process_chat.py --model "$TP_SESSION_MODEL_DIR" \
  --interactive --max-new-tokens 64 --output test/distributed/results/text-session <<'PROMPTS'
请用一句话介绍你自己。
一加一等于几？
PROMPTS
echo "四组验收与常驻双请求文本生成已完成；报告目录见上方输出。"
