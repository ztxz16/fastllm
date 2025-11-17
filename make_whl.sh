#!/bin/bash
folder="build-fastllm"

# 创建工作文件夹
rm -rf "$folder"
mkdir "$folder"
cd $folder

# cpu
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DMAKE_WHL_X86=ON -DUSE_CUDA=OFF -DUSE_NUMAS=ON
make fastllm_tools -j$(nproc)
if [ $? != 0 ]; then
    exit -1
fi
cp tools/ftllm/libfastllm_tools.so tools/ftllm/libfastllm_tools-cpu.so

# cuda-10
#rm -rf CMakeCache.txt CMakeFiles
#cmake .. -DMAKE_WHL_X86=ON -DUSE_CUDA=ON -DCUDA_ARCH=70 -D CMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc
#make fastllm_tools -j$(nproc)
#if [ $? != 0 ]; then
#    exit -1
#fi
#cp tools/ftllm/libfastllm_tools.so tools/ftllm/libfastllm_tools-cu10.so

# cuda-11
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DMAKE_WHL_X86=ON -DUSE_CUDA=ON -DUSE_NUMAS=ON -DCUDA_ARCH="52;53;70" -D CMAKE_CXX_COMPILER=g++-10 -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-10 -D CMAKE_CUDA_COMPILER=/usr/local/cuda-11.3/bin/nvcc
make fastllm_tools -j$(nproc)
if [ $? != 0 ]; then
    exit -1
fi
cp tools/ftllm/libfastllm_tools.so tools/ftllm/libfastllm_tools-cu11.so

# cuda-12
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DMAKE_WHL_X86=ON -DUSE_CUDA=ON -DUSE_NUMAS=ON -DCUDA_ARCH="52;53;70;89" -D CMAKE_CXX_COMPILER=g++-11 -D CMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 -D CMAKE_CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
make fastllm_tools -j$(nproc)
if [ $? != 0 ]; then
    exit -1
fi

cd tools
ldd ftllm/libfastllm_tools.so | grep '=>' | awk '{print $3}' | grep 'libnuma' | xargs -I {} cp -n {} ftllm/.
python3 setup.py sdist build
python3 setup.py bdist_wheel --plat-name manylinux2014_$(uname -m)
#python3 setup.py install --all