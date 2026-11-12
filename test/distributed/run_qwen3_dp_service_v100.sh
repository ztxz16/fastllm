#!/usr/bin/env bash
# 已有CUDA12.6产物的服务器稳定性验收；不编译、不安装、不下载模型。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?请指定四张GPU的已验收UUID顺序}"
if (( $# != 1 )); then
    echo 'usage: bash run_qwen3_dp_service_v100.sh MODEL' >&2
    exit 2
fi
DP_SERVICE_MODEL="$(realpath -- "$1")"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export NCCL_DEBUG=WARN

python3 test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
DP_SERVICE_COMMON=(--model "$DP_SERVICE_MODEL" --generate build-process-tp-cu126/fastllm_tp_generate
                   --worker build-process-tp-cu126/fastllm_tp_worker --build-cache build-process-tp-cu126/CMakeCache.txt)
for dp_service_routing in round_robin prefix; do
    python3 test/distributed/run_qwen3_dp_service.py "${DP_SERVICE_COMMON[@]}" --dp 4 --tp 1 --routing "$dp_service_routing"
    python3 test/distributed/run_qwen3_dp_service.py "${DP_SERVICE_COMMON[@]}" --dp 2 --tp 2 --routing "$dp_service_routing"
done
