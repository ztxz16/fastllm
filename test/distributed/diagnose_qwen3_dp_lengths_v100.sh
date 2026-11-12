#!/usr/bin/env bash
# 仅供服务器执行：更新长Prompt原生产物，再做数值边界与混合长度服务验收。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?请指定已验收的四卡UUID顺序，TP2每对必须支持双向P2P/native atomics}"
if (( $# != 1 )); then
    echo 'usage: bash diagnose_qwen3_dp_lengths_v100.sh MODEL' >&2
    exit 2
fi
DP_LENGTH_MODEL="$(realpath -- "$1")"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export NCCL_DEBUG=WARN

echo 'STEP 1/5: Python回归（详细用例写入报告目录）'
python3 test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
echo 'STEP 2/5: 编译长Prompt原生产物'
cmake -S . -B build-process-tp-cu126 -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build build-process-tp-cu126 --target fastllm_process_tp_package fastllm_tp_prefill_check \
    fastllm_tp_decode_check fastllm_tp_sampling_check -j 4
DP_LENGTH_COMMON=(--worker build-process-tp-cu126/fastllm_tp_worker --build-cache build-process-tp-cu126/CMakeCache.txt)
echo 'STEP 3/5: 长Prefill、生成与Prefix Cache数值/边界检查（含独立CPU参考）'
python3 test/distributed/run_qwen3_prefill.py "${DP_LENGTH_COMMON[@]}" --long-context --check build-process-tp-cu126/fastllm_tp_prefill_check
python3 test/distributed/run_qwen3_generation.py "${DP_LENGTH_COMMON[@]}" --long-context \
    --generate build-process-tp-cu126/fastllm_tp_generate --decode-check build-process-tp-cu126/fastllm_tp_decode_check \
    --sampling-check build-process-tp-cu126/fastllm_tp_sampling_check
python3 test/distributed/run_qwen3_prefix_cache.py "${DP_LENGTH_COMMON[@]}" --long-context --cuda-graph --custom-allreduce \
    --check build-process-tp-cu126/fastllm_tp_decode_check
echo 'STEP 4/5: DP4×TP1混合长度'
python3 test/distributed/diagnose_qwen3_dp_lengths.py "${DP_LENGTH_COMMON[@]}" --model "$DP_LENGTH_MODEL" \
    --generate build-process-tp-cu126/fastllm_tp_generate --dp 4 --tp 1
echo 'STEP 5/5: DP2×TP2混合长度'
python3 test/distributed/diagnose_qwen3_dp_lengths.py "${DP_LENGTH_COMMON[@]}" --model "$DP_LENGTH_MODEL" \
    --generate build-process-tp-cu126/fastllm_tp_generate --dp 2 --tp 2
