# syntax=docker/dockerfile:1
# 构建上下文必须是 export_env.py export 生成的目录。
FROM vastai/base-image:cuda-13.0.3-auto
USER root
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
ENV PYTHONNOUSERSITE=1

# 仅用于启动 aria2 的安装允许包管理器下载；后续安装包统一由 aria2 获取。
RUN apt-get update && \
    if ! command -v aria2c >/dev/null; then \
        DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends aria2; \
    fi

# bind mount 避免将安装包和 venv 压缩包另存一份到最终镜像层。
RUN --mount=type=bind,target=/fastllm-export \
    DEBIAN_FRONTEND=noninteractive FASTLLM_IMAGE_BUILD=1 /usr/bin/python3 /fastllm-export/export_env.py \
        restore --context /fastllm-export

COPY manifest.json packages.lock system-packages.txt /opt/fastllm-image/
COPY export_env.py /opt/fastllm-image/export_env.py

ENV CUDA_HOME=/usr/local/cuda-12.6 \
    CUDACXX=/usr/local/cuda-12.6/bin/nvcc \
    ATTENTION_EP_CUDA_HOME=/usr/local/cuda-12.6 \
    ATTENTION_EP_PYTHON=/venv/main/bin/python3 \
    PYTHONNOUSERSITE=1
ENV PATH="/venv/main/bin:/usr/local/cuda-12.6/bin:${PATH}" \
    LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:${LD_LIBRARY_PATH}"

# Vast 基础镜像用此摘要识别向工作卷同步的 Python/Conda 环境。
RUN if command -v env-hash >/dev/null; then env-hash > /.env_hash; fi

WORKDIR /workspace
