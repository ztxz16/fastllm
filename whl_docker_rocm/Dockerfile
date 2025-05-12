# 使用 Ubuntu 22.04 作为基础镜像
FROM ubuntu:22.04

# 设置非交互式安装环境
ENV DEBIAN_FRONTEND=noninteractive

# 安装基础工具和GCC版本
RUN apt-get update && apt-get install gcc g++ make -y
# 验证安装
RUN g++ --version

RUN apt-get update
RUN apt-get install cmake -y
RUN apt-get install libnuma-dev -y
RUN apt-get install python3-pip -y
RUN apt-get install wget -y

RUN wget https://repo.radeon.com/amdgpu-install/6.3.3/ubuntu/jammy/amdgpu-install_6.3.60303-1_all.deb
RUN apt install ./amdgpu-install_6.3.60303-1_all.deb -y
RUN amdgpu-install --usecase=hiplibsdk,rocm,dkms -y
RUN apt-get install python-is-python3 -y
RUN pip install setuptools wheel -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
RUN apt install libstdc++-12-dev
