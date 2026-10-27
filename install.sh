#!/bin/bash
folder="build-fastllm"

# 创建工作文件夹
if [ ! -d "$folder" ]; then
    mkdir "$folder"
fi

cd $folder
cmake .. "$@"
make -j16

#编译失败停止执行
if [ $? != 0 ]; then
    exit -1
fi

# Python 文件更新不一定触发 C++ 重新链接，安装前同步最新工具代码。
cmake -E copy_directory ../tools/fastllm_pytools tools/ftllm || exit 1

cd tools
pip install .[all]
#python3 setup.py sdist build
#python3 setup.py bdist_wheel
#python3 setup.py install --all
