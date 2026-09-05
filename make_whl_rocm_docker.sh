#!/usr/bin/env bash
set -euo pipefail

repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
ubuntu_version=22.04
rocm_version=10.0.0
image_name=
base_image=
skip_image_build=false
wheel_args=()

usage() {
    cat <<'HELP'
Usage: bash make_whl_rocm_docker.sh [Docker options] [wheel-builder options]

Docker options:
  --ubuntu VERSION       Ubuntu 22.04 (default) or 24.04
  --rocm-version VERSION ROCm Python SDK version (default: 10.0.0)
  --image NAME           Override the builder image name
  --base-image NAME      Use an alternate Ubuntu base image or registry
  --skip-image-build     Use an already built image
  --help                Show this help

Wheel options are passed to make_whl_rocm.sh, including:
  --architectures 'gfx1100;gfx942'   Default: all targets supported by the SDK/compiler
  --list-architectures             List SDK/compiler targets without compiling
  --jobs N                         Default: min(12, CPU count)
  --skip-build                     Repackage a matching existing Docker build
  --dist-dir PATH                  Path inside the mounted source checkout

Default output: build-rocm-docker-<Ubuntu version>/dist/
The host needs Docker access; no GPU, AMD driver or host ROCm SDK is required.
HELP
}

while (($#)); do
    case "$1" in
        --ubuntu|--rocm-version|--image|--base-image)
            if (($# < 2)) || [[ "$2" == --* ]]; then
                echo "Missing value for $1" >&2
                exit 2
            fi
            case "$1" in
                --ubuntu) ubuntu_version=$2 ;;
                --rocm-version) rocm_version=$2 ;;
                --image) image_name=$2 ;;
                --base-image) base_image=$2 ;;
            esac
            shift 2 ;;
        --skip-image-build) skip_image_build=true; shift ;;
        --help|-h) usage; exit 0 ;;
        --) shift; wheel_args+=("$@"); break ;;
        *) wheel_args+=("$1"); shift ;;
    esac
done
case "$ubuntu_version" in
    22.04|24.04) ;;
    *) echo 'Supported Ubuntu versions: 22.04, 24.04' >&2; exit 2 ;;
esac
image_name=${image_name:-ftllm-rocm-builder:ubuntu${ubuntu_version}-rocm${rocm_version}}
build_uid=${SUDO_UID:-$(id -u)}
build_gid=${SUDO_GID:-$(id -g)}

if ! "$skip_image_build"; then
    build_args=()
    if [[ -n "$base_image" ]]; then build_args+=(--build-arg "BASE_IMAGE=$base_image"); fi
    docker build --platform linux/amd64 \
        --build-arg "UBUNTU_VERSION=$ubuntu_version" \
        --build-arg "ROCM_VERSION=$rocm_version" \
        "${build_args[@]}" --tag "$image_name" "$repo_dir/whl_docker_rocm"
fi

# Keep container CMake caches separate from native SDK builds. Files are owned
# by the invoking user, including when this wrapper is launched through sudo.
exec docker run --rm --platform linux/amd64 \
    --user "$build_uid:$build_gid" \
    --mount "type=bind,src=$repo_dir,dst=/workspace/fastllm" \
    "$image_name" \
    --build-dir "/workspace/fastllm/build-rocm-docker-$ubuntu_version" \
    --dist-dir "/workspace/fastllm/build-rocm-docker-$ubuntu_version/dist" \
    "${wheel_args[@]}"
