#!/usr/bin/env bash
# 仅供服务器执行：默认构建并验收；--reuse-build复用已验收产物，仅测三档并发。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
: "${CUDA_VISIBLE_DEVICES:?请指定已验收的四卡UUID顺序，TP2每对必须支持双向P2P/native atomics}"
if (( $# < 1 || $# > 2 )) || { (( $# == 2 )) && [[ "$2" != --reuse-build ]]; }; then
    echo 'usage: bash diagnose_qwen3_dp_lengths_v100.sh MODEL [--reuse-build]' >&2
    exit 2
fi
DP_LENGTH_MODEL="$(realpath -- "$1")"
DP_LENGTH_REUSE_BUILD="${2:-}"
unset PYTHONHOME PYTHONPATH LD_LIBRARY_PATH LD_PRELOAD
source /home/venvs/fastllm-dp/bin/activate
export PYTHONNOUSERSITE=1 USE_TORCH=0 TOKENIZERS_PARALLELISM=false
export CUDA_HOME=/usr/local/cuda-12.6
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:/usr/local/nvidia/lib:/usr/local/nvidia/lib64"
export NCCL_DEBUG=WARN

if [[ "$DP_LENGTH_REUSE_BUILD" == --reuse-build ]]; then
    for dp_length_binary in build-process-tp-cu126/fastllm_tp_generate build-process-tp-cu126/fastllm_tp_worker; do
        if [[ ! -x "$dp_length_binary" ]]; then
            echo "缺少可执行产物：$dp_length_binary；请先完成长Prompt服务器编译与验收" >&2
            exit 1
        fi
    done
    if [[ ! -r build-process-tp-cu126/CMakeCache.txt ]]; then
        echo '缺少 build-process-tp-cu126/CMakeCache.txt，无法记录编译架构' >&2
        exit 1
    fi
fi

echo 'STEP 1/5: Python回归（详细用例写入报告目录）'
python3 test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
DP_LENGTH_COMMON=(--worker build-process-tp-cu126/fastllm_tp_worker --build-cache build-process-tp-cu126/CMakeCache.txt)
if [[ "$DP_LENGTH_REUSE_BUILD" == --reuse-build ]]; then
    echo 'STEP 2/5: 复用已验收的长Prompt原生产物，不编译'
    echo 'STEP 3/5: 跳过已通过的数值/边界验收'
else
    echo 'STEP 2/5: 编译长Prompt原生产物'
    cmake -S . -B build-process-tp-cu126 -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
        -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
    cmake --build build-process-tp-cu126 --target fastllm_process_tp_package fastllm_tp_prefill_check \
        fastllm_tp_decode_check fastllm_tp_sampling_check -j 4
    echo 'STEP 3/5: 长Prefill、生成与Prefix Cache数值/边界检查（含独立CPU参考）'
    python3 test/distributed/run_qwen3_prefill.py "${DP_LENGTH_COMMON[@]}" --long-context --check build-process-tp-cu126/fastllm_tp_prefill_check
    python3 test/distributed/run_qwen3_generation.py "${DP_LENGTH_COMMON[@]}" --long-context \
        --generate build-process-tp-cu126/fastllm_tp_generate --decode-check build-process-tp-cu126/fastllm_tp_decode_check \
        --sampling-check build-process-tp-cu126/fastllm_tp_sampling_check
    python3 test/distributed/run_qwen3_prefix_cache.py "${DP_LENGTH_COMMON[@]}" --long-context --cuda-graph --custom-allreduce \
        --check build-process-tp-cu126/fastllm_tp_decode_check
fi
echo 'STEP 4/5: DP4×TP1混合长度，C1/C4/C8各五轮'
python3 test/distributed/diagnose_qwen3_dp_lengths.py "${DP_LENGTH_COMMON[@]}" --model "$DP_LENGTH_MODEL" \
    --generate build-process-tp-cu126/fastllm_tp_generate --dp 4 --tp 1
echo 'STEP 5/5: DP2×TP2混合长度，C1/C2/C4各五轮'
python3 test/distributed/diagnose_qwen3_dp_lengths.py "${DP_LENGTH_COMMON[@]}" --model "$DP_LENGTH_MODEL" \
    --generate build-process-tp-cu126/fastllm_tp_generate --dp 2 --tp 2
