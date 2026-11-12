#!/usr/bin/env bash
# 复用已有V100原生产物：通用中文示例或用户JSONL，经正式服务验证两套配置。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?请指定已验收的四卡UUID顺序}"
if (( $# < 1 || $# > 2 )); then
    echo 'usage: bash run_qwen3_dp_requests_v100.sh MODEL [REQUESTS.jsonl]' >&2
    exit 2
fi
DP_REQUEST_MODEL="$(realpath -- "$1")"
DP_REQUEST_INPUT="$(realpath -- "${2:-test/distributed/data/qwen3_requests_example.jsonl}")"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export NCCL_DEBUG=WARN
for dp_request_binary in build-process-tp-cu126/fastllm_tp_generate build-process-tp-cu126/fastllm_tp_worker; do
    if [[ ! -x "$dp_request_binary" ]]; then
        echo "缺少已验收原生产物：$dp_request_binary" >&2
        exit 1
    fi
done
python3 test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
for dp_request_profile in dp4_tp1 dp2_tp2; do
    echo "RUN PROFILE: $dp_request_profile; requests: $DP_REQUEST_INPUT"
    python3 test/distributed/diagnose_qwen3_dp_lengths.py --model "$DP_REQUEST_MODEL" \
        --generate build-process-tp-cu126/fastllm_tp_generate --worker build-process-tp-cu126/fastllm_tp_worker \
        --build-cache build-process-tp-cu126/CMakeCache.txt --profile "$dp_request_profile" --requests "$DP_REQUEST_INPUT"
done
