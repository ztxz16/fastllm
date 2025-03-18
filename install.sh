#!/bin/bash
folder="build-fastllm"

# 创建工作文件夹
if [ ! -d "$folder" ]; then
    mkdir "$folder"
fi

# 如果有-DUSE_ROCM=ON参数，检查ROCm是否安装
if [[ "$@" == *"-DUSE_ROCM=ON"* ]]; then
    if [ ! -d "/opt/rocm" ]; then
        echo "ROCm is not installed, please install ROCm first."
        exit -1
    fi
    export CC=/opt/rocm/bin/amdclang
    export CXX=/opt/rocm/bin/amdclang++
fi

cd $folder
cmake .. "$@"
make -j$(nproc)

#编译失败停止执行
if [ $? != 0 ]; then
    exit -1
fi

cd tools
pip install .[all]
#python3 setup.py sdist build
#python3 setup.py bdist_wheel
#python3 setup.py install --all