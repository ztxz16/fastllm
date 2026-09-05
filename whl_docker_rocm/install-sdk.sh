#!/usr/bin/env bash
set -euo pipefail

rocm_version=${1:?Pass the ROCm Python SDK version}
if [ "$(dpkg --print-architecture)" != amd64 ]; then
    echo 'The ROCm wheel builder currently requires Linux x86_64.' >&2
    exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential ca-certificates git libnuma-dev python3-venv
rm -rf /var/lib/apt/lists/*

python3 -m venv /opt/rocm-venv
sdk_python=/opt/rocm-venv/bin/python
"$sdk_python" -m pip install --no-cache-dir 'pip==26.2.1'
"$sdk_python" -m pip install --no-cache-dir \
    'cmake==4.4.3' 'ninja==1.13.2' 'setuptools==84.0.0' 'wheel==0.48.0' \
    "rocm[libraries,devel]==${rocm_version}" \
    --extra-index-url https://stable.repo.amd.com/rocm/whl-next/
"$sdk_python" -m pip check
sdk_root=$("$sdk_python" -m rocm_sdk path --root)
"$sdk_root/lib/llvm/bin/clang++" --version

# GPU device packages and kernel drivers are only needed on inference hosts.
