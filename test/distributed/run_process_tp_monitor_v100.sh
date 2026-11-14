#!/usr/bin/env bash
# 仅在服务器执行：编译并验收退出监控、心跳和进程生命周期，不需要模型或 PyTorch。
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
TP_MONITOR_BUILD="${TP_MONITOR_BUILD:-build-process-tp-cu126}"
TP_MONITOR_PYTHON="${TP_MONITOR_PYTHON:-python3}"
TP_MONITOR_CUDA="${TP_MONITOR_CUDA:-/usr/local/cuda-12.6}"
TP_MONITOR_CUSTOM="${TP_MONITOR_CUSTOM:-0}"
case "$TP_MONITOR_CUSTOM" in
    0|1) ;;
    *) echo "TP_MONITOR_CUSTOM 只接受 0 或 1" >&2; exit 2 ;;
esac

export CUDA_HOME="$TP_MONITOR_CUDA"
export CUDACXX="$CUDA_HOME/bin/nvcc"
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES-0,1}"
export USE_TORCH=0
"$CUDACXX" --version
nvidia-smi
git log -1 --oneline

TP_MONITOR_CONFIGURE=(-S . -B "$TP_MONITOR_BUILD" -DUSE_CUDA=ON -DUSE_NUMAS=OFF
                      -DBUILD_PROCESS_TP=ON "-DCMAKE_CUDA_COMPILER=$CUDACXX" -DCUDA_ARCH=70)
# 旧目录切换编译器时，CMake 可能清空缓存并按默认选项重新配置，但仍返回成功。
# 检查最终缓存，必要时重新应用一次全部参数；不能在 CPU/非 TP 配置上继续构建。
for TP_MONITOR_ATTEMPT in 1 2; do
    cmake "${TP_MONITOR_CONFIGURE[@]}"
    if awk -F= '
        $1 ~ /^USE_CUDA:/ { cuda = ($2 == "ON") }
        $1 ~ /^USE_NUMAS:/ { numas = ($2 == "OFF") }
        $1 ~ /^BUILD_PROCESS_TP:/ { tp = ($2 == "ON") }
        $1 ~ /^CUDA_ARCH:/ { arch = ($2 == "70") }
        END { exit !(cuda && numas && tp && arch) }
    ' "$TP_MONITOR_BUILD/CMakeCache.txt"; then
        break
    fi
    if [[ "$TP_MONITOR_ATTEMPT" == 2 ]]; then
        echo "CMake 配置仍不符合 CUDA/TP/SM70 要求，停止构建；请检查 $TP_MONITOR_BUILD/CMakeCache.txt" >&2
        exit 1
    fi
    echo "CMake 最终缓存与请求不一致，重新应用 CUDA/TP/SM70 配置。" >&2
done

# 同时更新正式入口及打包目录，避免测试产物已更新但推理入口仍链接旧进程底座。
cmake --build "$TP_MONITOR_BUILD" --target fastllm_process_tp_package fastllm_tp_check -j 4

TP_MONITOR_ARGS=(--worker "$TP_MONITOR_BUILD/fastllm_tp_worker"
                 --check "$TP_MONITOR_BUILD/fastllm_tp_check" --devices 0,1 --cuda
                 --build-cache "$TP_MONITOR_BUILD/CMakeCache.txt")
"$TP_MONITOR_PYTHON" test/distributed/run_process_tp.py "${TP_MONITOR_ARGS[@]}"
if [[ "$TP_MONITOR_CUSTOM" == 1 ]]; then
    # 只有所选卡对支持双向 P2P/native atomics 才能通过，失败不静默降级。
    "$TP_MONITOR_PYTHON" test/distributed/run_process_tp.py "${TP_MONITOR_ARGS[@]}" --custom-allreduce
fi
