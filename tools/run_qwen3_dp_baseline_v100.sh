#!/usr/bin/env bash
# 已完成长Prompt验收的V100服务器：常驻正式服务，不编译、不安装。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
: "${CUDA_VISIBLE_DEVICES:?请指定已验收的四卡UUID顺序}"
if (( $# < 2 )); then
    echo 'usage: bash tools/run_qwen3_dp_baseline_v100.sh MODEL dp4_tp1|dp2_tp2 [--port 8080 ...]' >&2
    exit 2
fi
DP_BASELINE_MODEL="$(realpath -- "$1")"
DP_BASELINE_PROFILE="$2"
shift 2
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export PYTHONPATH="$PWD/tools"
export NCCL_DEBUG=WARN
exec python3 -m fastllm_pytools.dp_baseline --model "$DP_BASELINE_MODEL" --profile "$DP_BASELINE_PROFILE" "$@"
