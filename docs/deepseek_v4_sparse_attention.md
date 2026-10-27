# DeepSeek V4 Sparse Attention Decode

[Benchmark](benchmarks/deepseek_v4.md) · [部署指南](deepseek.md)

## 接口与调度

Triton 路径保留模型的缓存布局：BF16 query `[1, 1, local_heads, 512]`、FP32 window KV `[1, 128, 512]`、BF16 compressed KV `[1, capacity, 512]`，以及 FP32 attention sink。
通用 split-K 按 batch/head/split 并行，默认每段处理 8 个候选，输出 FP32 numerator、max 和 denominator，再合并 sink。位置和有效长度来自 graph-safe `decode_meta`。
工作区按设备复用；Graph capture 期间不分配或释放地址。Triton 编译、元数据校验、载入或 launch 失败时回退原生 CUDA。

SM120/SM121、D=512 且 local heads ≤16 时优先使用 tensor-core 路径：每块处理对齐到 16 的 heads 和 16 个候选，QK/PV 使用 BF16 dot，softmax 和 split/merge 保留 FP32，默认 8 warps。失败时依次尝试通用 Triton、原生 CUDA。
原生 CUDA 承担更宽的兼容范围，包括 FP32/FP16/BF16 query、不同 compressed KV dtype、batch 和最大 1024 head dim。

需要启用总 Triton 开关：

```bash
FASTLLM_CUDA_TRITON=1
```

相关环境变量：

| 环境变量 | 默认值 | 作用 |
|---|---:|---|
| `FASTLLM_CUDA_TRITON_DEEPSEEK_V4_SPARSE_DECODE` | `1` | sparse decode Triton 总开关 |
| `FASTLLM_CUDA_TRITON_DSV4_SPARSE_SM120` | `1` | SM12x 专用路径 |
| `FASTLLM_CUDA_TRITON_DSV4_SPARSE_SM120_SPLIT_SIZE` | `16` | SM12x candidate tile，可选 16/32/64 |
| `FASTLLM_CUDA_TRITON_DSV4_SPARSE_SM120_SPLIT_WARPS` | `8` | SM12x split kernel warps |
| `FASTLLM_CUDA_TRITON_DSV4_SPARSE_SPLIT_SIZE` | `8` | 通用路径 split size，可选 8/16/32/64 |
| `FASTLLM_CUDA_TRITON_DSV4_SPARSE_SPLIT_WARPS` | `4` | 通用 split kernel warps |
| `FASTLLM_CUDA_TRITON_DSV4_SPARSE_MERGE_BLOCK_D` | `32` | merge kernel 的输出维 tile |
| `FASTLLM_CUDA_TRITON_DSV4_SPARSE_MERGE_WARPS` | `4` | merge kernel warps |

推荐 TP8 decode 同时开启：

```bash
FASTLLM_CUDA_GRAPH=1
FASTLLM_CUDA_CUSTOM_ALLREDUCE=1
FASTLLM_CUDA_TRITON=1
```

## 数值与验证范围

通用 split/merge 的独立 FP32 参考测试最大误差不超过 1.49e-8。SM120 使用 BF16 tensor-core 归约，不保证与 FP32 路径逐位一致；需要 FP32 split-K 时设置 `FASTLLM_CUDA_TRITON_DSV4_SPARSE_SM120=0`。
已记录的全模型短上下文验证覆盖 16 步，top-1 相同，decode logits 余弦相似度为 0.997749–0.999921；这不是长上下文或广泛质量评估。
发布的性能只覆盖 TP8、128 输入 / 256 输出，不应外推到更长候选列表、其他 batch 或混合专家放置。
