source ~/ftllm/bin/activate
pip install setuptools wheel -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple

#!/bin/bash
folder="build-fastllm-rocm"

# 创建工作文件夹
rm -rf "$folder"
mkdir "$folder"
cd $folder

# cpu
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DMAKE_WHL_X86=ON -DUSE_ROCM=OFF -DUSE_NUMAS=ON
make fastllm_tools -j$(nproc)
if [ $? != 0 ]; then
    exit -1
fi
cp tools/ftllm/libfastllm_tools.so tools/ftllm/libfastllm_tools-cpu.so

# cuda-11
rm -rf CMakeCache.txt CMakeFiles
cmake .. -DMAKE_WHL_X86=ON -DUSE_ROCM=ON -DUSE_NUMAS=ON -DROCM_ARCH="gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101"
make fastllm_tools -j$(nproc)
if [ $? != 0 ]; then
    exit -1
fi

cd tools
ldd ftllm/libfastllm_tools.so | grep '=>' | awk '{print $3}' | grep 'libnuma' | xargs -I {} cp -n {} ftllm/.
python3 setup_rocm.py sdist build
python3 setup_rocm.py bdist_wheel --plat-name manylinux2014_$(uname -m)