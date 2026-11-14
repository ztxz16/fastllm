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

cmake -S . -B "$TP_MONITOR_BUILD" -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_PROCESS_TP=ON \
    -DCMAKE_CUDA_COMPILER="$CUDACXX" -DCUDA_ARCH=70
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
