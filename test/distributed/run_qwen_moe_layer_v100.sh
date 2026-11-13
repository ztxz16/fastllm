#!/usr/bin/env bash
# 仅供服务器手动执行：编译阶段 2 C++/CUDA，并用自动生成的小模型验收，无需下载权重。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.."

moe_build="${ATTENTION_EP_BUILD:-build-process-tp-cu126}"
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export PYTHONNOUSERSITE=1

# 默认沿用当前已激活的 Python 环境；显式指定无效路径时直接报错。
if ! moe_python="$(command -v -- "${ATTENTION_EP_PYTHON:-python3}")" || [[ ! -f "$moe_python" || ! -x "$moe_python" ]]; then
    echo "找不到可执行的 Python：${ATTENTION_EP_PYTHON:-python3}" >&2
    echo '请激活 Python 3.10+ 环境，或将 ATTENTION_EP_PYTHON 设为有效的解释器路径；unset ATTENTION_EP_PYTHON 可恢复自动选择。' >&2
    exit 1
fi
if ! "$moe_python" -c 'import sys
if sys.version_info < (3, 10):
    sys.exit("需要 Python 3.10+，实际版本：" + sys.version)
print("验收 Python：" + sys.executable + " (" + sys.version.split()[0] + ")", flush=True)
'; then
    echo 'Python 启动或版本检查失败，请通过 ATTENTION_EP_PYTHON 指定可用的 Python 3.10+。' >&2
    exit 1
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.6}"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1,2,3}"
IFS=',' read -r -a moe_devices <<< "$CUDA_VISIBLE_DEVICES"
if [[ ${#moe_devices[@]} -ne 4 ]]; then
    echo '需要四张可见 GPU；CUDA_VISIBLE_DEVICES 可使用编号或完整 UUID。' >&2
    exit 1
fi

"$moe_python" test/distributed/run_parallel_config.py --execution-tests --benchmark-tests
"$moe_python" test/distributed/run_parallel_config.py --real-cuda --dp 1 --tp 1 --attention-dp 4 --ep 4
"$CUDACXX" --version
nvidia-smi
git log -1 --oneline
cmake -S . -B "$moe_build" -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
cmake --build "$moe_build" --target fastllm_tp_worker fastllm_moe_layer_check \
    fastllm_tp_shard_check fastllm_tp_model_check fastllm_tp_layer_check -j "${ATTENTION_EP_JOBS:-4}"

# 权重读取、GPU 上传和稠密单层是本次复用的组件，先执行对应既有回归。
"$moe_python" test/distributed/run_qwen3_shards.py --check "$moe_build/fastllm_tp_shard_check" --build-cache "$moe_build/CMakeCache.txt"
moe_common=(--worker "$moe_build/fastllm_tp_worker" --devices 0,1 --build-cache "$moe_build/CMakeCache.txt")
"$moe_python" test/distributed/run_qwen3_device_weights.py --check "$moe_build/fastllm_tp_model_check" "${moe_common[@]}"
"$moe_python" test/distributed/run_qwen3_layer.py --check "$moe_build/fastllm_tp_layer_check" "${moe_common[@]}"
"$moe_python" test/distributed/run_qwen_moe_layer.py --check "$moe_build/fastllm_moe_layer_check" \
    --worker "$moe_build/fastllm_tp_worker" --devices 0,1,2,3 --build-cache "$moe_build/CMakeCache.txt"
echo '阶段 2 与既有权重/单层回归均通过；报告路径见各组输出。'
