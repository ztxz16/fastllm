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

真实模型的单请求/批处理切换回归覆盖两条采样请求，以及采样/贪心混合请求。指定已安装本次构建的 Python 环境和模型路径，保证两张 GPU 空闲：

```sh
python test/cuda/test_mtp_scheduler_transition.py \
    -p /path/to/Qwen3.8-27B-FP8 --tp 2 --mtp 3 --max_batch 2 --threads 8 \
    --cuda_embedding --kv_cache_dtype float16 --prefix_cache false --cache_history false \
    --max_context_length 8192 --chunked_prefill_size 2048
```

测试应输出最后的 `PASS: MTP sampling and mixed requests across scheduler transitions`。开启 CUDA Graph 时同时检查日志确实捕获了 `batch=2` 的 MTP 验证图，避免将普通解码回退误记为批处理验证通过。

编译兼容性检查使用仍支持 sm_60 的 nvcc（本次验证为 CUDA 12.4），不需要 GPU：

```sh
python3 test/cuda/check_paged_params_build.py --nvcc /usr/local/cuda-12.4/bin/nvcc
```

它编译完整 sm_60 attention 翻译单元，并直接提取生产分页上传代码，编译同时包含 sm_60/sm_75 的目标，防止重新引入超出旧架构 4 KiB 参数上限的内核。
