#!/usr/bin/env bash
# 服务器专用：完整回归后验证四种执行组合的跨请求前缀快照。
set -euo pipefail
TP_PREFIX_MODEL="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

bash test/distributed/run_qwen3_custom_allreduce_v100.sh "$TP_PREFIX_MODEL"
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES=0,1
export USE_TORCH=0
TP_PREFIX_COMMON=(--worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
                  --build-cache build-process-tp-cu126/CMakeCache.txt)
TP_PREFIX_CHECK=(--check build-process-tp-cu126/fastllm_tp_decode_check)
TP_PREFIX_SESSION=(--generate build-process-tp-cu126/fastllm_tp_generate
                   --decode-check build-process-tp-cu126/fastllm_tp_decode_check --prefix-cache)

for TP_PREFIX_GRAPH in 0 1; do
  for TP_PREFIX_CUSTOM in 0 1; do
    TP_PREFIX_FLAGS=()
    if [[ "$TP_PREFIX_GRAPH" == 1 ]]; then TP_PREFIX_FLAGS+=(--cuda-graph); fi
    if [[ "$TP_PREFIX_CUSTOM" == 1 ]]; then TP_PREFIX_FLAGS+=(--custom-allreduce); fi
    python3 test/distributed/run_qwen3_prefix_cache.py "${TP_PREFIX_COMMON[@]}" \
      "${TP_PREFIX_CHECK[@]}" "${TP_PREFIX_FLAGS[@]}"
    python3 test/distributed/run_qwen3_prefix_cache.py "${TP_PREFIX_COMMON[@]}" \
      "${TP_PREFIX_CHECK[@]}" "${TP_PREFIX_FLAGS[@]}" --model "$TP_PREFIX_MODEL"
  done
done

python3 test/distributed/run_qwen3_session.py "${TP_PREFIX_COMMON[@]}" "${TP_PREFIX_SESSION[@]}" \
  --cuda-graph --custom-allreduce
python3 test/distributed/run_qwen3_session.py "${TP_PREFIX_COMMON[@]}" "${TP_PREFIX_SESSION[@]}" \
  --cuda-graph --custom-allreduce --model "$TP_PREFIX_MODEL"
python3 test/distributed/run_qwen3_entrypoint.py "${TP_PREFIX_COMMON[@]}" --prefix-cache \
  --cuda-graph --custom-allreduce --generate build-process-tp-cu126/fastllm_tp_generate \
  --package-root build-process-tp-cu126/process-tp-python --model "$TP_PREFIX_MODEL"
echo "二十五组完成：原十四组；前缀四种组合各合成10/真实1；前缀+Graph+IPC会话96/16；正式入口13。报告见各组 PASS 输出。"
