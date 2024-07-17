#!/bin/bash
folder="build-fastllm"

# 创建工作文件夹
if [ ! -d "$folder" ]; then
    mkdir "$folder"
fi

cd $folder
cmake .. "$@"
make -j$(nproc)

if [ $? == 0 ]; then
cd tools
python3 setup.py sdist build
python3 setup.py bdist_wheel
python3 setup.py install
fi