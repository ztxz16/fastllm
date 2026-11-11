#!/usr/bin/env bash
# 服务器专用：四卡 V100 的纯 DP、DP+TP、Graph/前缀缓存/自定义规约组合验收。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
DP_MODEL_PATH="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
if [[ -f /home/venvs/fastllm-dp/bin/activate ]]; then
    source /home/venvs/fastllm-dp/bin/activate
fi
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export USE_TORCH=0
"$CUDACXX" --version
python3 -c 'import jinja2; from transformers import AutoTokenizer'
python3 test/distributed/run_parallel_config.py --execution-tests --real-cuda --dp 2 --tp 2

cmake -S . -B build-process-tp-cu126 \
    -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 --target fastllm_process_tp_package -j 4
DP_COMMON=(--model "$DP_MODEL_PATH" --generate build-process-tp-cu126/fastllm_tp_generate
           --worker build-process-tp-cu126/fastllm_tp_worker
           --build-cache build-process-tp-cu126/CMakeCache.txt)
python3 test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp 2 --tp 1
python3 test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp 4 --tp 1 --cuda-graph --prefix-cache
python3 test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp 2 --tp 2
python3 test/distributed/run_qwen3_dp.py "${DP_COMMON[@]}" --dp 2 --tp 2 \
    --cuda-graph --prefix-cache --custom-allreduce
echo '五组完成：Python/设备回归、DP2×TP1、DP4×TP1 Graph+Prefix、DP2×TP2、DP2×TP2 Graph+Prefix+IPC。'
