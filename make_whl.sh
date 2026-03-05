#!/bin/bash

NIGHTLY=0
for arg in "$@"; do
    case $arg in
        --nightly)
            NIGHTLY=1
            shift
            ;;
    esac
done

export FASTLLM_NIGHTLY=$NIGHTLY

folder="build-fastllm"

# 创建工作文件夹
rm -rf "$folder"
mkdir "$folder"
cd $folder

# cpu
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DMAKE_WHL_X86=ON -DUSE_CUDA=OFF -DUSE_NUMAS=ON
make fastllm_tools -j30
if [ $? != 0 ]; then
    exit -1
fi
cp tools/ftllm/libfastllm_tools.so tools/ftllm/libfastllm_tools-cpu.so

# cuda-10
#rm -rf CMakeCache.txt CMakeFiles
#cmake .. -DMAKE_WHL_X86=ON -DUSE_CUDA=ON -DCUDA_ARCH=70 -D CMAKE_CUDA_COMPILER=/usr/local/cuda-10.1/bin/nvcc
#make fastllm_tools -j30
#if [ $? != 0 ]; then
#    exit -1
#fi
#cp tools/ftllm/libfastllm_tools.so tools/ftllm/libfastllm_tools-cu10.so

# cuda-12
rm -rf CMakeCache.txt CMakeFiles
CUDA_ARCH_LIST="70;75;80;89;90;100;120"
if [ -x /usr/local/cuda/bin/nvcc ]; then
    CUDA_COMPILER=/usr/local/cuda/bin/nvcc
elif [ -x /usr/local/cuda-12.9/bin/nvcc ]; then
    CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
elif [ -x /usr/local/cuda-12.1/bin/nvcc ]; then
    CUDA_COMPILER=/usr/local/cuda-12.1/bin/nvcc
else
    echo "nvcc not found in /usr/local/cuda*/bin"
    exit -1
fi

cmake .. \
    -DMAKE_WHL_X86=ON \
    -DUSE_CUDA=ON \
    -DUSE_NUMAS=ON \
    -DCUDA_ARCH="${CUDA_ARCH_LIST}" \
    -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCH_LIST}" \
    -DCMAKE_CXX_COMPILER=/usr/bin/g++-11 \
    -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-11 \
    -DCMAKE_CUDA_COMPILER="${CUDA_COMPILER}"
make fastllm_tools -j30
if [ $? != 0 ]; then
    exit -1
fi

cd tools
ldd ftllm/libfastllm_tools.so | grep '=>' | awk '{print $3}' | grep 'libnuma' | xargs -I {} cp -n {} ftllm/.
python3 setup.py sdist build
python3 setup.py bdist_wheel --plat-name manylinux2014_$(uname -m)
#python3 setup.py install --all