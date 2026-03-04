# 使用 Ubuntu 20.04 作为基础镜像
FROM ubuntu:20.04

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
ENV PATH=/usr/local/cuda-12.9/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/cuda-12.9/lib64:${LD_LIBRARY_PATH}

# 创建默认 CUDA 软链接
RUN ln -sf /usr/local/cuda-12.9 /usr/local/cuda

# 验证安装
RUN g++ --version
RUN nvcc --version

RUN apt-get update
RUN apt-get install cmake -y
RUN apt-get install libnuma-dev -y
RUN apt-get install python3-pip -y
RUN pip install setuptools wheel

RUN apt remove --purge cmake -y
RUN pip install cmake==3.25.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

# -----------------------------------------------------------------
# 新增部分：编译安装 Binutils 2.38 以升级 'as' (汇编器)
# -----------------------------------------------------------------
# 1. 安装编译 binutils 依赖的 texinfo 和 bison (虽然后面编译时不包含文档可以跳过texinfo，但为了保险起见加上)
RUN apt-get update && apt-get install -y texinfo bison flex
# 2. 下载 Binutils 2.38 源码，编译并安装
RUN cd /tmp \
    && wget https://ftp.gnu.org/gnu/binutils/binutils-2.38.tar.gz \
    && tar -xzf binutils-2.38.tar.gz \
    && cd binutils-2.38 \
    && ./configure --prefix=/usr --disable-multilib --disable-werror --enable-gold \
    && make -j$(nproc) \
    && make install \
    && cd /tmp \
    && rm -rf binutils-2.38 binutils-2.38.tar.gz
# 3. 验证 'as' 版本 (预期输出包含 2.38)
RUN as --version

# 安装最新 CUDA 12.9 对应 NCCL
RUN apt-get update && apt-get install -y "libnccl2=*+cuda12.9" "libnccl-dev=*+cuda12.9" \
    && rm -rf /var/lib/apt/lists/*
# 创建 NCCL 符号链接以便 CUDA 12.9 找到
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnccl* /usr/local/cuda-12.9/lib64/ 2>/dev/null || true \
    && ln -sf /usr/include/nccl.h /usr/local/cuda-12.9/include/ 2>/dev/null || true
# 设置 NCCL 环境变量
ENV NCCL_HOME=/usr
ENV NCCL_INCLUDE_DIR=/usr/include
ENV NCCL_LIB_DIR=/usr/lib/x86_64-linux-gnu
# 验证 NCCL 安装
RUN ls -la /usr/include/nccl.h && ls -la /usr/lib/x86_64-linux-gnu/libnccl*