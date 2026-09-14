# CUDA 分页与草稿采样回归检查

`test_paged_int_params.cpp` 检查分页上传的 0/256/512/1024/2048/4096 边界、最大元数据长度、越界保护、4097 页回退，以及修改主机数组后的 Graph 重放。MTP 分布检查使用 `test/basic/test_cuda_mtp_gumbel.cpp` 和 `test/basic/test_cuda_mtp_rejection.cpp`，覆盖实际草稿 q、完整分布拒绝采样、温度、联合 top-k/top-p、残差、bonus 和 Graph 随机性。旧候选集 MTP 采样接口及其专用测试已移除。

`test/basic/test_qwen35_mtp_sampling.cpp` 直接通过模型 `ForwardGPU` 验证目标采样和贪心草稿接受，包括重复状态的退出概率、目标分布之外的草稿、bonus、混合请求长度和无草稿请求。对应 CTest 为 `qwen35_mtp_sampling` 和 `qwen35_mtp_sampling_batch`，无需额外的公开采样辅助接口。

在已编译出 `libfastllm_tools.so` 的仓库根目录执行，按实际位置设置 `FASTLLM_TEST_LIBRARY_DIR` 和 `FASTLLM_TEST_CUDA_DIR`：

```sh
FASTLLM_TEST_LIBRARY_DIR="$PWD/build"
FASTLLM_TEST_CUDA_DIR=/usr/local/cuda
for test_source in test/cuda/test_paged_int_params.cpp test/basic/test_cuda_mtp_gumbel.cpp test/basic/test_cuda_mtp_rejection.cpp; do
    test_name=$(basename "$test_source" .cpp)
    g++ -O2 -std=c++17 -DUSE_CUDA -pthread -Iinclude -Ithird_party/json11 \
        -I"$FASTLLM_TEST_CUDA_DIR/include" "$test_source" \
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

测试应输出最后的 `PASS: MTP sampling and mixed requests across scheduler transitions`。采样请求默认使用随机草稿和完整分布拒绝采样，并在单请求与批处理切换时保持实际 q 同步。开启 CUDA Graph 时同时检查日志确实捕获了 `batch=2` 的 MTP 验证图，避免将普通解码回退误记为批处理验证通过。

编译兼容性检查使用仍支持 sm_60 的 nvcc（本次验证为 CUDA 12.4），不需要 GPU：

```sh
python3 test/cuda/check_paged_params_build.py --nvcc /usr/local/cuda-12.4/bin/nvcc
```

它编译完整 sm_60 attention 翻译单元，并直接提取生产分页上传代码，编译同时包含 sm_60/sm_75 的目标，防止重新引入超出旧架构 4 KiB 参数上限的内核。
