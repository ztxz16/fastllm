#!/usr/bin/env bash
# 服务器专用：验证FT共享页管理器及四种执行组合的分页前缀缓存。
set -euo pipefail
TP_PREFIX_MODEL="$(cd -- "${1:-/home/models/Qwen3-1.7B}" && pwd -P)"
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

bash test/distributed/run_qwen3_custom_allreduce_v100.sh "$TP_PREFIX_MODEL"
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
export USE_TORCH=0
TP_PREFIX_COMMON=(--worker build-process-tp-cu126/fastllm_tp_worker --devices 0,1
                  --build-cache build-process-tp-cu126/CMakeCache.txt)
TP_PREFIX_CHECK=(--check build-process-tp-cu126/fastllm_tp_decode_check)
TP_PREFIX_SESSION=(--generate build-process-tp-cu126/fastllm_tp_generate
                   --decode-check build-process-tp-cu126/fastllm_tp_decode_check --prefix-cache)

cmake --build build-process-tp-cu126 --target fastllm_tp_paged_cache_check -j 4

python3 test/distributed/run_ft_paged_cache.py --cuda \
  --check build-process-tp-cu126/fastllm_tp_paged_cache_check \
  --build-cache build-process-tp-cu126/CMakeCache.txt

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
python3 test/distributed/run_qwen3_http.py "${TP_PREFIX_COMMON[@]}" --prefix-cache \
  --generate build-process-tp-cu126/fastllm_tp_generate --model "$TP_PREFIX_MODEL"
TP_PREFIX_DP_FLAGS=(--dp 2 --tp "${TP_PREFIX_DP_TP:-1}" --prefix-cache --dp-routing prefix --cuda-graph)
if [[ "${TP_PREFIX_DP_TP:-1}" == 2 ]]; then TP_PREFIX_DP_FLAGS+=(--custom-allreduce); fi
python3 test/distributed/run_qwen3_dp.py --model "$TP_PREFIX_MODEL" \
  --generate build-process-tp-cu126/fastllm_tp_generate --worker build-process-tp-cu126/fastllm_tp_worker \
  --build-cache build-process-tp-cu126/CMakeCache.txt "${TP_PREFIX_DP_FLAGS[@]}"
for TP_PREFIX_MOE_GRAPH in 0 1; do
  python3 test/distributed/run_qwen_moe_full.py --binary build-process-tp-cu126/fastllm_moe_generate \
    --worker build-process-tp-cu126/fastllm_tp_worker --build-cache build-process-tp-cu126/CMakeCache.txt \
    --synthetic-case "f16_bias_ep2_graph${TP_PREFIX_MOE_GRAPH}_prefix1"
done
echo "FT页管理器、分页前缀四种执行组合、会话、正式入口、HTTP、DP前缀路由与MoE回归完成；以各组结构化报告为准。"
