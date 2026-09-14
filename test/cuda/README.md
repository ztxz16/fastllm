# CUDA 分页与草稿采样回归检查

`test_paged_int_params.cpp` 检查分页上传的 0/256/512/1024/2048/4096 边界、最大元数据长度、越界保护、4097 页回退，以及修改主机数组后的 Graph 重放。`test_mtp_draft_sampling.cpp` 检查草稿频率与返回的 q 一致、拒绝验证符合目标 p，包括温度和 top-p 过滤。

在已编译出 `libfastllm_tools.so` 的仓库根目录执行，按实际位置设置 `FASTLLM_TEST_LIBRARY_DIR` 和 `FASTLLM_TEST_CUDA_DIR`：

```sh
FASTLLM_TEST_LIBRARY_DIR="$PWD/build"
FASTLLM_TEST_CUDA_DIR=/usr/local/cuda
for test_name in test_paged_int_params test_mtp_draft_sampling; do
    g++ -O2 -std=c++17 -DUSE_CUDA -pthread -Iinclude -Ithird_party/json11 \
        -I"$FASTLLM_TEST_CUDA_DIR/include" "test/cuda/$test_name.cpp" \
        -L"$FASTLLM_TEST_LIBRARY_DIR" -lfastllm_tools \
        -L"$FASTLLM_TEST_CUDA_DIR/lib64" -lcudart \
        -Wl,-rpath,"$FASTLLM_TEST_LIBRARY_DIR" -Wl,-rpath,"$FASTLLM_TEST_CUDA_DIR/lib64" \
        -o "/tmp/$test_name"
    "/tmp/$test_name"
done
```

编译兼容性检查使用仍支持 sm_60 的 nvcc（本次验证为 CUDA 12.4），不需要 GPU：

```sh
python3 test/cuda/check_paged_params_build.py --nvcc /usr/local/cuda-12.4/bin/nvcc
```

它编译完整 sm_60 attention 翻译单元，并直接提取生产分页上传代码，编译同时包含 sm_60/sm_75 的目标，防止重新引入超出旧架构 4 KiB 参数上限的内核。
