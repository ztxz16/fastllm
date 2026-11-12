#!/usr/bin/env bash
# 只使用已有CUDA12.6产物；设备顺序由调用者指定，默认诊断DP1×TP2。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?请按已验收的拓扑指定GPU顺序}"
if (( $# < 2 )); then
    echo 'usage: bash diagnose_qwen3_tp_concurrency_v100.sh MODEL REQUEST_JSON [options]' >&2
    exit 2
fi
TP_DIAG_MODEL="$(realpath -- "$1")"
TP_DIAG_REQUEST="$(realpath -- "$2")"
shift 2
unset PYTHONHOME PYTHONPATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export NCCL_DEBUG=WARN
python3 test/distributed/diagnose_qwen3_tp_concurrency.py \
    --model "$TP_DIAG_MODEL" --request-file "$TP_DIAG_REQUEST" \
    --generate build-process-tp-cu126/fastllm_tp_generate \
    --worker build-process-tp-cu126/fastllm_tp_worker \
    --build-cache build-process-tp-cu126/CMakeCache.txt "$@"
