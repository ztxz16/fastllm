#!/usr/bin/env bash
# 仅供双 V100 服务器手动执行：会编译 C++/CUDA，禁止在本地无编译授权时运行。
set -euo pipefail

# 在切换工作目录之前解析模型路径，相对路径以调用者当前目录为基准。
TP_PREFILL_MODEL_DIR="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TP_PREFILL_BUILD_DIR=build-process-tp-cu126
if [[ ! -f "$TP_PREFILL_MODEL_DIR/config.json" ]]; then
    echo "模型目录缺少 config.json：$TP_PREFILL_MODEL_DIR" >&2
    exit 1
fi

export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
hash -r
"$CUDACXX" --version
nvidia-smi
git log -1 --oneline

cmake -S . -B "$TP_PREFILL_BUILD_DIR" \
    -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build "$TP_PREFILL_BUILD_DIR" \
    --target fastllm_tp_worker fastllm_tp_check fastllm_tp_model_check \
             fastllm_tp_layer_check fastllm_tp_prefill_check -j 4

TP_PREFILL_COMMON_ARGS=(--worker "$TP_PREFILL_BUILD_DIR/fastllm_tp_worker"
                       --devices 0,1 --build-cache "$TP_PREFILL_BUILD_DIR/CMakeCache.txt")
python3 test/distributed/run_process_tp.py \
    --check "$TP_PREFILL_BUILD_DIR/fastllm_tp_check" "${TP_PREFILL_COMMON_ARGS[@]}" --cuda
python3 test/distributed/run_qwen3_device_weights.py \
    --check "$TP_PREFILL_BUILD_DIR/fastllm_tp_model_check" "${TP_PREFILL_COMMON_ARGS[@]}"
python3 test/distributed/run_qwen3_layer.py \
    --check "$TP_PREFILL_BUILD_DIR/fastllm_tp_layer_check" "${TP_PREFILL_COMMON_ARGS[@]}"
python3 test/distributed/run_qwen3_prefill.py \
    --check "$TP_PREFILL_BUILD_DIR/fastllm_tp_prefill_check" "${TP_PREFILL_COMMON_ARGS[@]}"
python3 test/distributed/run_qwen3_prefill.py \
    --check "$TP_PREFILL_BUILD_DIR/fastllm_tp_prefill_check" "${TP_PREFILL_COMMON_ARGS[@]}" \
    --model "$TP_PREFILL_MODEL_DIR"

echo "五组验收均通过；各组报告目录见上方 PASS 输出。"
