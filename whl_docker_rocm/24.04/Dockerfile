# 使用 Ubuntu 20.04 作为基础镜像
FROM ubuntu:24.04

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

# 验证安装
RUN g++ --version

RUN apt-get update
RUN apt-get install cmake -y
RUN apt-get install libnuma-dev -y
RUN apt-get install python3-pip -y

RUN wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/noble/amdgpu-install_6.3.60303-1_all.deb
RUN apt install ./amdgpu-install_6.3.60303-1_all.deb -y
RUN amdgpu-install --usecase=hiplibsdk,rocm,dkms -y
RUN apt-get install python-is-python3 -y
RUN apt-get install python3.12-venv -y
RUN python3 -m venv ~/ftllm
#RUN source ~/ftllm/bin/activate
#RUN pip install setuptools wheel -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

#RUN apt remove --purge cmake -y
#RUN pip install cmake==3.25.0 -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
