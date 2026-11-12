#!/usr/bin/env bash
# 服务器执行：同时更新原生缓存目录协议的协调器和worker，再验证旧入口与新路由。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?请指定四张GPU的已验收UUID顺序}"
if (( $# != 2 )); then
    echo 'usage: bash run_qwen3_dp_routing_v100.sh MODEL REQUEST_JSON' >&2
    exit 2
fi
DP_ROUTING_MODEL="$(realpath -- "$1")"
DP_ROUTING_REQUEST="$(realpath -- "$2")"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export NCCL_DEBUG=WARN

python3 test/distributed/run_parallel_config.py --execution-tests --benchmark-tests --real-cuda --dp 2 --tp 2
cmake -S . -B build-process-tp-cu126 \
    -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 --target fastllm_process_tp_package -j 4

DP_ROUTING_COMMON=(--model "$DP_ROUTING_MODEL" --generate build-process-tp-cu126/fastllm_tp_generate
                   --worker build-process-tp-cu126/fastllm_tp_worker --build-cache build-process-tp-cu126/CMakeCache.txt)
# 保留默认轮转策略的真实模型/CLI/HTTP/取消回归。
python3 test/distributed/run_qwen3_dp.py "${DP_ROUTING_COMMON[@]}" --dp 4 --tp 1 --cuda-graph --prefix-cache
python3 test/distributed/run_qwen3_dp.py "${DP_ROUTING_COMMON[@]}" --dp 2 --tp 2 --cuda-graph --prefix-cache --custom-allreduce
# 每种组合包含目录生命周期验证，以及两个策略×并发1/4×五轮性能对照。
python3 test/distributed/benchmark_qwen3_dp_routing.py "${DP_ROUTING_COMMON[@]}" --request-file "$DP_ROUTING_REQUEST" --dp 4 --tp 1
python3 test/distributed/benchmark_qwen3_dp_routing.py "${DP_ROUTING_COMMON[@]}" --request-file "$DP_ROUTING_REQUEST" --dp 2 --tp 2
