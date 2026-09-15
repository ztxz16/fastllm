# 双卡 Ampere prefill 可选加速

以下两个开关默认关闭，只有值为 `1` 才启用，在通信组初始化时读取。

| 开关 | 行为 |
|---|---|
| `FASTLLM_TP2_WHT6_ALLREDUCE=1` | 将 rank-local FP16 contribution 做 32 点 WHT 和 6-bit 量化，通过主机固定内存中转，解码求和后再加未压缩的 residual。属于有损通信。 |
| `FASTLLM_TP2_MLP_OVERLAP=1` | 将符合条件的 MLP 按 512 token 分块，使 down projection 后的归约与下一块计算重叠。WHT6 开启时使用 WHT6，否则使用原始 NCCL FP16 归约。 |

两者均仅适用于 TP=2、两张卡都为 SM80/SM86 的 eager 执行。CUDA Graph capture、其他架构及不满足条件的调用继续使用已有路径。关闭开关时，不分配相应 GPU 工作区或创建额外 CUDA stream，也不改变原 NCCL 归约函数。

WHT6 仅接管 1–32 MiB 的 FP16 contribution，保留原 Marlin 权重布局。它改变归约数值结果，应针对实际任务验证质量；默认行为不包含此量化。

MLP overlap 当前限定无 bias、FP16 activation/residual、FP8 E4M3 Marlin 权重、128×128 block scales，以及每卡 `tokens=2048, hidden=5120, intermediate=8704` 的 dense MLP。工作区约 300.5 MiB/卡，只在首次符合条件的调用中分配，跟随通信组释放。它不预取或反量化下一层权重。

原始 NCCL overlap 保留 rank 0 先以 FP16 加 residual、再进行 FP16 AllReduce 的顺序。分块 GEMM 仍可能因计算顺序产生舍入差异。两卡须以相同顺序调用集合通信；开始提交后发生错误会终止当前路径，不切换后端继续执行。

## 默认 FP8 linear 路径

SM75/80/86 的大 prefill 可使用已有 Marlin 布局反量化 + cuBLAS 路径，仍受既有 `FASTLLM_CUDA_FP8_PREFILL_CUBLAS` 开关与工作区条件控制。该扩展只作用于 FP8，共用状态入口不会将 NVFP4 扩展到 SM80/86。

SM80/86 的 2–3 token 宽扩展矩阵采用 Marlin 布局多行 GEMV：`N>=16384, K>=4096, K<=N/2`。单 token GEMV 保持原实现，其他小 batch 和中间尺寸继续使用 Marlin GEMM。没有新增 GEMV 行数环境开关。

## 回归测试

配置 `-DUSE_CUDA=ON -DUNIT_TEST=ON` 后构建相关 target，再运行：

```bash
ctest --test-dir build-fastllm --output-on-failure \
  -R 'cuda_fp8_(sm75_prefill|mlp_prefill|marlin_multirow)|tp2(WHT6AllReduce|_mlp_overlap)'
```

测试覆盖独立 CPU 反量化参考、cuBLAS 与 Marlin 路由、CUDA Graph、prefill 前后 decode、WHT6 编码及通信缓冲区复用，以及 NCCL/WHT6 两种 MLP overlap 后端。历史名称 `cuda_fp8_sm75_prefill_*` 的测试同时适用于 SM80/86。
