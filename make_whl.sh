#!/usr/bin/env bash
# Build a CPU + CUDA 12 wheel inside an explicitly selected Linux userspace.
set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash make_whl.sh [options]

Run inside Ubuntu 20.04 (see whl_docker/Dockerfile) for Ubuntu 20.04/22.04
compatibility. This script does not install toolchains or change the host OS.

  --nightly                 Build ftllm-nightly instead of ftllm
  --build-dir PATH          Separate CPU/CUDA build trees (default: build-wheel)
  --output-dir PATH         Wheel and audit JSON (default: build-fastllm/tools/dist)
  --cpu-jobs N              CPU build jobs (default: min(nproc, 32))
  --cuda-jobs N             CUDA build jobs (default: min(nproc, 12))
  --nvcc-threads N          NVCC threads per job (default: 2)
  --split-compile N         NVCC optimizer threads (default: 2)
  --cuda-root PATH          CUDA 12.9 installation (default: /usr/local/cuda-12.9)
  --cuda-arch LIST          Semicolon-separated CMake architecture list
  --compression MODE        size, speed, balance, default, none, legacy (default: size)
  --max-glibc VERSION       Reject newer glibc dependencies (default: 2.31)
  --no-strip                Keep debugging and static symbol tables
  -h, --help                Show this help

CXX, PYTHON, CMAKE and STRIP select tools. CPU_JOBS, CUDA_JOBS, NVCC_THREADS,
NVCC_SPLIT_COMPILE, CUDA_ROOT and CUDA_ARCH_LIST provide option defaults.
Existing build trees are reused; choose a new --build-dir for a clean build.
CUDA runtime packages remain pip dependencies. The wheel uses linux_x86_64;
the GLIBC audit alone is not a manylinux certification.
EOF
}
die() { echo "ERROR: $*" >&2; exit 1; }
need_value() { [[ $# -ge 2 && -n "$2" ]] || die "$1 requires a value"; }

SOURCE_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
BUILD_DIR="$SOURCE_DIR/build-wheel"
OUTPUT_DIR=""
NIGHTLY=0
CPUS=$(getconf _NPROCESSORS_ONLN)
CPU_JOBS=${CPU_JOBS:-$(( CPUS < 32 ? CPUS : 32 ))}
CUDA_JOBS=${CUDA_JOBS:-$(( CPUS < 12 ? CPUS : 12 ))}
NVCC_THREADS=${NVCC_THREADS:-2}
NVCC_SPLIT_COMPILE=${NVCC_SPLIT_COMPILE:-2}
CUDA_ROOT=${CUDA_ROOT:-/usr/local/cuda-12.9}
CUDA_ARCH_LIST=${CUDA_ARCH_LIST:-'60-real;70-real;75-real;80-real;86-real;89-real;90-real;100-real;120'}
COMPRESSION=size
MAX_GLIBC=2.31
DO_STRIP=1
while (($#)); do
    case "$1" in
        --nightly) NIGHTLY=1; shift ;;
        --no-strip) DO_STRIP=0; shift ;;
        -h|--help) usage; exit 0 ;;
        --build-dir|--output-dir|--cpu-jobs|--cuda-jobs|--nvcc-threads|--split-compile|--cuda-root|--cuda-arch|--compression|--max-glibc)
            need_value "$@"
            case "$1" in
                --build-dir) BUILD_DIR=$2 ;; --output-dir) OUTPUT_DIR=$2 ;;
                --cpu-jobs) CPU_JOBS=$2 ;; --cuda-jobs) CUDA_JOBS=$2 ;;
                --nvcc-threads) NVCC_THREADS=$2 ;; --split-compile) NVCC_SPLIT_COMPILE=$2 ;;
                --cuda-root) CUDA_ROOT=$2 ;; --cuda-arch) CUDA_ARCH_LIST=$2 ;;
                --compression) COMPRESSION=$2 ;; --max-glibc) MAX_GLIBC=$2 ;;
            esac
            shift 2 ;;
        *) die "Unknown argument: $1" ;;
    esac
done
[[ $(uname -s) == Linux && $(uname -m) == x86_64 ]] || die 'This release script targets Linux x86_64.'
for value in "$CPU_JOBS" "$CUDA_JOBS" "$NVCC_THREADS" "$NVCC_SPLIT_COMPILE"; do
    [[ $value =~ ^[1-9][0-9]*$ ]] || die 'Job/thread counts must be positive integers.'
done
[[ $MAX_GLIBC =~ ^[0-9]+\.[0-9]+$ ]] || die 'Use --max-glibc MAJOR.MINOR.'
case "$COMPRESSION" in size|speed|balance|default|none|legacy) ;; *) die 'Invalid compression mode.' ;; esac
PYTHON=${PYTHON:-python3}
CMAKE=${CMAKE:-cmake}
STRIP=${STRIP:-strip}
if [[ -z ${CXX:-} ]]; then
    for candidate in g++-11 g++-10 g++; do
        if command -v "$candidate" >/dev/null; then CXX=$candidate; break; fi
    done
fi
CXX=$(command -v "${CXX:-g++}") || die 'C++ compiler not found.'
for tool in "$PYTHON" "$CMAKE" readelf; do command -v "$tool" >/dev/null || die "Missing $tool"; done
[[ $DO_STRIP == 0 ]] || command -v "$STRIP" >/dev/null || die "Missing $STRIP"
"$PYTHON" -c 'import setuptools, wheel' || die 'Install setuptools and wheel in the build Python.'
"$PYTHON" - "$MAX_GLIBC" <<'PY'
import os, sys
host = os.confstr('CS_GNU_LIBC_VERSION').split()[-1]
if tuple(map(int, host.split('.'))) > tuple(map(int, sys.argv[1].split('.'))):
    sys.exit('Build glibc %s exceeds target %s. Use the Ubuntu 20.04 Docker/chroot '
             'environment, or explicitly change --max-glibc for a newer-system wheel.' % (host, sys.argv[1]))
PY
CUDA_COMPILER="$CUDA_ROOT/bin/nvcc"
[[ -x $CUDA_COMPILER ]] || die "nvcc not found: $CUDA_COMPILER"
CUDA_VERSION=$("$CUDA_COMPILER" --version)
[[ $CUDA_VERSION == *'release 12.9,'* ]] || die 'This all-architecture release recipe requires CUDA 12.9.'
BUILD_DIR=$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$BUILD_DIR")
OUTPUT_DIR=${OUTPUT_DIR:-"$SOURCE_DIR/build-fastllm/tools/dist"}
OUTPUT_DIR=$("$PYTHON" -c 'import os,sys; print(os.path.abspath(sys.argv[1]))' "$OUTPUT_DIR")
[[ $BUILD_DIR != "$SOURCE_DIR" && $BUILD_DIR != / ]] || die 'Use a dedicated build directory.'
mkdir -p "$BUILD_DIR" "$OUTPUT_DIR"
run() { "$@" 2>&1 | tee -a "$BUILD_DIR/build.log"; }
trap 'echo "Build failed at line $LINENO (exit $?). See $BUILD_DIR/build.log" >&2' ERR
export FASTLLM_NIGHTLY=$NIGHTLY
export CUDA_HOME=$CUDA_ROOT
export PATH="$CUDA_ROOT/bin:$PATH"
START_SECONDS=$SECONDS
printf 'Source: %s\nBuild: %s\nCompiler: %s\nCPU/CUDA jobs: %s/%s\n' "$SOURCE_DIR" "$BUILD_DIR" "$CXX" "$CPU_JOBS" "$CUDA_JOBS"

COMMON=(-DMAKE_WHL_X86=ON -DUSE_NUMAS=ON -DCMAKE_CXX_COMPILER="$CXX"
        '-DCMAKE_BUILD_RPATH=$ORIGIN' '-DCMAKE_INSTALL_RPATH=$ORIGIN'
        -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON)
run "$CMAKE" -S "$SOURCE_DIR" -B "$BUILD_DIR/cpu" "${COMMON[@]}" -DUSE_CUDA=OFF
run "$CMAKE" --build "$BUILD_DIR/cpu" --target fastllm_tools --parallel "$CPU_JOBS"
printf 'CPU build: %s seconds\n' "$((SECONDS - START_SECONDS))"
CUDA_START_SECONDS=$SECONDS
run "$CMAKE" -S "$SOURCE_DIR" -B "$BUILD_DIR/cuda" "${COMMON[@]}" \
    -DUSE_CUDA=ON -DCUDA_ARCH="$CUDA_ARCH_LIST" -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH_LIST" \
    -DCMAKE_CUDA_HOST_COMPILER="$CXX" -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
    -DFASTLLM_CUDA_COMPRESSION="$COMPRESSION" \
    "-DCMAKE_CUDA_FLAGS=--threads=$NVCC_THREADS --split-compile=$NVCC_SPLIT_COMPILE"
run "$CMAKE" --build "$BUILD_DIR/cuda" --target fastllm_tools --parallel "$CUDA_JOBS"
printf 'CUDA build: %s seconds\n' "$((SECONDS - CUDA_START_SECONDS))"
run "$PYTHON" "$SOURCE_DIR/tools/wheel/package_wheel.py" \
    --source-dir "$SOURCE_DIR" --build-dir "$BUILD_DIR" --output-dir "$OUTPUT_DIR" --max-glibc "$MAX_GLIBC" \
    --cxx "$CXX" --strip "$STRIP" --do-strip "$DO_STRIP"
printf 'Completed in %s seconds. Artifacts: %s\n' "$((SECONDS - START_SECONDS))" "$OUTPUT_DIR"
