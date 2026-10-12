# 使用 Ubuntu 20.04 作为基础镜像
FROM ubuntu:20.04
ARG BUILD_JOBS=32
ARG NCCL_VERSION=2.28.3

# 设置非交互式安装环境
ENV DEBIAN_FRONTEND=noninteractive

# 安装基础工具和GCC版本
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    software-properties-common \
    wget \
    ca-certificates \
    && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
    && apt-get update \
    && apt-get install -y \
    g++-10 \
    g++-11 \
    && rm -rf /var/lib/apt/lists/*

# 设置G++多版本配置
RUN update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-10 50 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 60 \
    && update-alternatives --set g++ /usr/bin/g++-11

# 添加 NVIDIA 仓库密钥和源（用于安装 CUDA 12.9）
RUN apt-get update && apt-get install -y gnupg2 \
    && wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb \
    && dpkg -i cuda-keyring_1.0-1_all.deb \
    && rm cuda-keyring_1.0-1_all.deb \
    && apt-get update

# 安装 CUDA 12.9 toolkit（包含 nvcc）
RUN apt-get install -y cuda-toolkit-12-9

# 配置 CUDA 环境变量（默认使用 12.9）
ENV PATH=/opt/binutils-2.38/bin:/usr/local/cuda-12.9/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}

# 创建默认 CUDA 软链接
RUN ln -sf /usr/local/cuda-12.9 /usr/local/cuda

# 验证安装
RUN g++ --version
RUN nvcc --version

# Python 3.8 用于构建，运行 wheel 需要 Python >= 3.10。
RUN apt-get update && apt-get install -y libnuma-dev python3-pip \
    && python3 -m pip install cmake==3.25.0 setuptools==75.3.2 wheel==0.45.1

# -----------------------------------------------------------------
# 新增部分：编译安装 Binutils 2.38 以升级 'as' (汇编器)
# -----------------------------------------------------------------
# 1. 安装编译 binutils 依赖的 texinfo 和 bison (虽然后面编译时不包含文档可以跳过texinfo，但为了保险起见加上)
RUN apt-get update && apt-get install -y texinfo bison flex xz-utils
# 2. 下载 Binutils 2.38 源码，编译并安装
RUN cd /tmp \
    && wget --timeout=30 --tries=3 https://mirrors.kernel.org/gnu/binutils/binutils-2.38.tar.xz \
    && echo 'e316477a914f567eccc34d5d29785b8b0f5a10208d36bbacedcc39048ecfe024  binutils-2.38.tar.xz' | sha256sum -c - \
    && tar -xf binutils-2.38.tar.xz && mkdir binutils-build && cd binutils-build \
    && ../binutils-2.38/configure --prefix=/opt/binutils-2.38 --disable-multilib --disable-werror --disable-nls --enable-gold \
    && make -j"${BUILD_JOBS}" \
    && make install \
    && cd /tmp \
    && rm -rf binutils-2.38 binutils-2.38.tar.xz binutils-build
# 3. 验证 'as' 版本 (预期输出包含 2.38)
RUN as --version

# 固定已验证的便携版 NCCL；避免引入新系统 glibc 依赖。
RUN python3 -m pip install --no-deps --target /opt/nccl "nvidia-nccl-cu12==${NCCL_VERSION}" \
    && ln -s /opt/nccl/nvidia/nccl/include/nccl.h /usr/local/cuda-12.9/include/nccl.h \
    && ln -s /opt/nccl/nvidia/nccl/lib/libnccl.so.2 /usr/local/cuda-12.9/lib64/libnccl.so.2 \
    && ln -s libnccl.so.2 /usr/local/cuda-12.9/lib64/libnccl.so
ENV NCCL_HOME=/opt/nccl/nvidia/nccl
