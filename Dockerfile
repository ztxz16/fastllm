# syntax=docker/dockerfile:1-labs
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04

# Update Apt repositories
RUN apt-get update 

# Install and configure Python
RUN apt-get -y --no-install-recommends install wget build-essential python3.10 python3-pip
RUN update-alternatives --install /usr/bin/python  python /usr/bin/python3.10 1
RUN pip install setuptools streamlit-chat

ENV WORKDIR /fastllm

# Install cmake
RUN wget -c https://cmake.org/files/LatestRelease/cmake-3.27.0-linux-x86_64.sh && bash ./cmake-3.27.0-linux-x86_64.sh  --skip-license --prefix=/usr/

WORKDIR $WORKDIR
ADD . $WORKDIR/

RUN mkdir $WORKDIR/build && cd build && cmake .. -DUSE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=native && make -j && cd tools && python setup.py install

CMD /fastllm/build/webui -p /models/chatglm2-6b-int8.flm
