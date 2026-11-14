#!/usr/bin/env bash
# 仅由用户在四卡V100服务器执行；本地助手不得执行本脚本。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
moe_stage="${1:-synthetic}"
moe_build="${ATTENTION_EP_BUILD:-build-attention-ep-cu126}"
moe_model="${ATTENTION_EP_MODEL:-/home/models/Qwen1.5-MoE-A2.7B-Chat}"
moe_python="${ATTENTION_EP_PYTHON:-/venv/main/bin/python3}"
[[ -x "$moe_python" ]] || { echo "无效Python：$moe_python" >&2; exit 1; }
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export PYTHONNOUSERSITE=1 TOKENIZERS_PARALLELISM=false
export CUDA_HOME="${ATTENTION_EP_CUDA_HOME:-/usr/local/cuda-12.6}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
moe_common=(--binary "$moe_build/fastllm_moe_generate" --worker "$moe_build/fastllm_tp_worker"
            --build-cache "$moe_build/CMakeCache.txt")

case "$moe_stage" in
    build)
        [[ -x "$CUDACXX" ]] || { echo "缺少CUDA编译器：$CUDACXX" >&2; exit 1; }
        "$moe_python" - "$moe_build/CMakeCache.txt" "$CUDACXX" <<'PY'
from pathlib import Path
import sys
cache = Path(sys.argv[1])
if cache.exists():
    compiler = next((x.split('=', 1)[1] for x in cache.read_text().splitlines()
                     if x.startswith('CMAKE_CUDA_COMPILER:')), None)
    if compiler != sys.argv[2]:
        sys.exit('构建目录使用其他CUDA编译器；请设ATTENTION_EP_BUILD为新的目录。')
PY
        "$CUDACXX" --version
        nvidia-smi
        cmake -S . -B "$moe_build" -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
            -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
        cmake --build "$moe_build" --target fastllm_process_tp_package fastllm_moe_layer_check \
            fastllm_tp_decode_check fastllm_tp_prefill_check fastllm_tp_model_check fastllm_tp_layer_check \
            fastllm_tp_shard_check -j "${ATTENTION_EP_JOBS:-4}"
        ;;
    synthetic)
        "$moe_python" test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
        "$moe_python" test/distributed/run_qwen_moe_layer.py --check "$moe_build/fastllm_moe_layer_check" \
            --worker "$moe_build/fastllm_tp_worker" --devices 0,1,2,3 --build-cache "$moe_build/CMakeCache.txt"
        "$moe_python" test/distributed/run_qwen_moe_full.py "${moe_common[@]}"
        ;;
    reproduce)
        [[ $# -ge 2 ]] || { echo 'reproduce需要小模型case名称及可选重复次数' >&2; exit 2; }
        "$moe_python" test/distributed/run_qwen_moe_full.py "${moe_common[@]}" \
            --synthetic-case "$2" --repeat "${3:-10}"
        ;;
    real)
        "$moe_python" test/distributed/run_qwen_moe_full.py "${moe_common[@]}" --model "$moe_model" \
            --reference-python "${ATTENTION_EP_REFERENCE_PYTHON:-$moe_python}"
        ;;
    performance)
        "$moe_python" test/distributed/run_qwen_moe_full.py "${moe_common[@]}" --model "$moe_model" \
            --reference-python "${ATTENTION_EP_REFERENCE_PYTHON:-$moe_python}" --performance
        ;;
    http)
        "$moe_python" test/distributed/run_qwen_moe_http.py "${moe_common[@]}" --model "$moe_model"
        ;;
    regression)
        moe_dense=(--worker "$moe_build/fastllm_tp_worker" --devices 0,1 --build-cache "$moe_build/CMakeCache.txt")
        "$moe_python" test/distributed/run_qwen3_decode.py --check "$moe_build/fastllm_tp_decode_check" "${moe_dense[@]}"
        "$moe_python" test/distributed/run_qwen3_decode.py --check "$moe_build/fastllm_tp_decode_check" "${moe_dense[@]}" --cuda-graph
        "$moe_python" test/distributed/run_attention_ep_dense_regression.py --build "$moe_build"
        ;;
    *) echo 'usage: bash test/distributed/run_attention_ep_v100.sh {build|synthetic|reproduce CASE [REPEAT]|real|http|performance|regression}' >&2; exit 2 ;;
esac
if [[ "$moe_stage" == reproduce ]]; then
    echo 'PASS: reproduce（仅定向复现，不代表完整验收通过）；结果见本次summary.json。'
else
    echo "PASS: $moe_stage；GPU验收结果以本次生成的summary.json为准。"
fi
