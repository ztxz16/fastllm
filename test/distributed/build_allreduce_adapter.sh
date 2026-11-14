#!/usr/bin/env bash
# 仅在服务器显式执行；运行 benchmark_allreduce.py 不会调用本脚本。
set -euo pipefail
AR_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AR_BUILD="${AR_BUILD:-$AR_ROOT/build-allreduce-cu126}"
AR_PYTHON="${AR_PYTHON:-python3}"
AR_CUDA="${AR_CUDA:-/usr/local/cuda-12.6}"
AR_NCCL_ROOT="${AR_NCCL_ROOT:-}"
if [[ -z "$AR_NCCL_ROOT" ]]; then
  AR_NCCL_ROOT="$("$AR_PYTHON" - <<'PY'
import importlib.metadata
for name in ('nvidia-nccl-cu12', 'nvidia-nccl-cu11', 'nvidia-nccl-cu13'):
    try:
        print(importlib.metadata.distribution(name).locate_file('nvidia/nccl'))
        break
    except importlib.metadata.PackageNotFoundError:
        pass
PY
)"
fi
cmake -S "$AR_ROOT/test/distributed/allreduce_native" -B "$AR_BUILD" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=70 \
  -DCMAKE_CUDA_COMPILER="$AR_CUDA/bin/nvcc" -DCUDAToolkit_ROOT="$AR_CUDA" \
  -DNCCL_ROOT="$AR_NCCL_ROOT"
cmake --build "$AR_BUILD" --target fastllm_allreduce_adapter -j "${AR_JOBS:-4}"
