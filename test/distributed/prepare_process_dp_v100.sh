#!/usr/bin/env bash
# 仅供用户在新的 Ubuntu 24.04 / V100 服务器执行；安装依赖和模型，不编译。
set -euo pipefail
if [[ "${EUID}" != 0 ]]; then
    echo '请在服务器以 root 执行此环境准备脚本。' >&2
    exit 1
fi
export DEBIAN_FRONTEND=noninteractive
apt-get update
# 只装进程后端所需的 CUDA 12.6 编译器、运行时和 cuBLAS 开发文件。
apt-get install -y git cmake g++ python3-venv \
    cuda-nvcc-12-6 cuda-cudart-dev-12-6 libcublas-dev-12-6
apt-mark unhold libnccl2 libnccl-dev
apt-get install -y --allow-downgrades \
    libnccl2=2.22.3-1+cuda12.6 libnccl-dev=2.22.3-1+cuda12.6
apt-mark hold libnccl2 libnccl-dev
ldconfig

python3 -m venv /home/venvs/fastllm-dp
source /home/venvs/fastllm-dp/bin/activate
python3 -m pip install huggingface_hub transformers jinja2
mkdir -p /home/models
hf download Qwen/Qwen3-1.7B --local-dir /home/models/Qwen3-1.7B
/usr/local/cuda-12.6/bin/nvcc --version
dpkg-query -W libnccl2 libnccl-dev
df -h /
echo '环境准备完成；下一步执行 run_qwen3_dp_v100.sh。'
