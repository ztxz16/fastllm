#!/usr/bin/env bash
# 仅由用户在四卡V100服务器执行；后台调度独立批次，不编译、不下载。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export PYTHONNOUSERSITE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${ATTENTION_EP_CUDA_HOME:-/usr/local/cuda-12.6}"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
accept_python="${ATTENTION_EP_PYTHON:-/venv/fastllm-main-cu126/bin/python3}"
accept_build="${ATTENTION_EP_BUILD:-/home/project/vllm_fastllm/build-attention-ep-cu126}"
accept_model="${ATTENTION_EP_MODEL:-/home/models/Qwen1.5-MoE-A2.7B-Chat}"
[[ -x "$accept_python" ]] || { echo "缺少Python：$accept_python" >&2; exit 1; }
accept_dir="$("$accept_python" test/distributed/attention_ep_acceptance.py prepare \
    --build "$accept_build" --model "$accept_model" --python "$accept_python" \
    --reference-python "${ATTENTION_EP_REFERENCE_PYTHON:-$accept_python}" "$@")"
nohup "$accept_python" test/distributed/attention_ep_acceptance.py run "$accept_dir" \
    > "$accept_dir/stdout.log" 2> "$accept_dir/stderr.log" < /dev/null &
accept_pid=$!
printf '%s\n' "$accept_pid" > "$accept_dir/pid"
printf '后台任务 PID=%s\n日志目录：%s\n计划：%s/plan.md\n' "$accept_pid" "$accept_dir" "$accept_dir"
printf '查看进度：tail -f %q\n完成后查看：cat %q\n' "$accept_dir/stdout.log" "$accept_dir/exit.code"
