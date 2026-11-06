#!/usr/bin/env bash
# 仅在双 V100 服务器手动执行：编译并验证生成入口，然后做真实文本问答。
set -euo pipefail

TP_GENERATION_MODEL_DIR="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
hash -r
python3 -c 'from transformers import AutoTokenizer'

bash test/distributed/run_qwen3_decode_v100.sh "$TP_GENERATION_MODEL_DIR"
cmake --build build-process-tp-cu126 --target fastllm_tp_generate fastllm_tp_sampling_check -j 4
TP_GENERATION_ARGS=(--generate build-process-tp-cu126/fastllm_tp_generate
                   --sampling-check build-process-tp-cu126/fastllm_tp_sampling_check
                   --decode-check build-process-tp-cu126/fastllm_tp_decode_check
                   --worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
                   --build-cache build-process-tp-cu126/CMakeCache.txt)
python3 test/distributed/run_qwen3_generation.py "${TP_GENERATION_ARGS[@]}"
python3 test/distributed/run_qwen3_generation.py "${TP_GENERATION_ARGS[@]}" --model "$TP_GENERATION_MODEL_DIR"
python3 tools/qwen3_process_chat.py --model "$TP_GENERATION_MODEL_DIR" \
  --prompt '请用一句话介绍你自己。' --max-new-tokens 64 \
  --output test/distributed/results/text-generation
echo "九组验收与真实文本生成已完成；报告目录见上方输出。"
