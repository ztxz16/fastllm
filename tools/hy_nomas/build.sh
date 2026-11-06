#!/usr/bin/env bash
set -euo pipefail
repo=$(cd "$(dirname "$0")/../.." && pwd)
out=${1:-"$repo/build-hy-nomas"}
# One set of source files and precision options for both build entry points.
arch=${CUDA_ARCH:-120}
cmake -S "$repo" -B "$out" -DUSE_CUDA=ON -DBUILD_HY_NOMAS=ON \
  -DUSE_NUMAS=OFF -DCUDA_ARCH="${arch#sm_}" -DCMAKE_BUILD_TYPE=Release
cmake --build "$out" --target fastllm_nomas -j "${BUILD_JOBS:-2}"
