# DeepSeek V4 Sparse Attention Decode

本文记录 DeepSeek V4 / DeepSeek V4 Flash 在 FastLLM 上的 sparse attention
decode 实现、与 vLLM/FlashInfer SM120 实现的差异，以及 TP8 SM120 的优化结果。

## Kernel 对比

### vLLM

本机 vLLM 0.26 在 SM120 上选择
`DeepseekV4FlashInferSM120Attention`，实际调用 FlashInfer 的
`sparse_mla_decode_dsv4_kernel` 和 split merge kernel。它不是普通的
FlashAttention kernel，而是 DeepSeek V4 专用 sparse MLA decode：

- query head 按 16 对齐，候选 token 以 64 个为一块；
- 主核使用 8 个 math warp 和 1 个 IO warp，双缓冲 KV；
- 448 维 NoPE 使用 FP8 block-scaled MMA，64 维 RoPE 使用 BF16 MMA；
- 每个 token 的 KV 为 FlashMLA footer 格式：448 B FP8 NoPE、128 B BF16
  RoPE、7 B UE8M0 scale 和 1 B padding，逻辑大小 584 B；
- 候选维做 split-K，主核写 `mid_out`/`mid_lse`，第二个 kernel merge；
- 由 sparse indexer 提供 top-k indices 和有效长度。

### FastLLM 优化前

FastLLM 的缓存 ABI 是：

- query：BF16 `[1, 1, local_heads, 512]`；
- sliding window KV：FP32 `[1, 128, 512]`；
- compressed KV：BF16 `[1, capacity, 512]`；
- attention sink：FP32 `[local_heads]`。

原 Triton kernel 每个 local head 只启动一个 program，在一个 CTA 内串行遍历
全部 window 和 compressed 候选，并维护 512 维 FP32 online-softmax accumulator。
TP8 每个 rank 只有 8 个 local heads，因此每层只有 8 个 CTA。SM120 实测约
37--38 registers/thread、无 spill，瓶颈是候选维没有并行展开，而不是寄存器
spill。

原生 CUDA fallback 支持更宽的输入范围，包括 FP32/FP16/BF16 query、不同
compressed KV dtype、batch 和最大 1024 head dim。因而 FastLLM sparse core 的
兼容性主要由原生 fallback 提供；Triton 路径是受控的 fast path。

## 第一阶段：通用 split-K fast path

通用 Triton 路径保持现有缓存 ABI，不改变模型侧缓存的所有权和布局：

1. split kernel 的 grid 为 `batch x head x split`，每个 program 默认处理 8 个
   候选，并输出 FP32 partial numerator、max 和 denominator；
2. merge kernel 在 FP32 中合并所有 partial 和 learned attention sink；
3. 两个 kernel 都读取 graph-safe `decode_meta`，只处理当前实际存在的
   window/compressed splits；
4. scratch buffer 按 device 复用，CUDA Graph capture 期间不分配或释放地址；
5. Triton 编译、metadata 校验、载入或 launch 任一失败，都返回原生 CUDA
   fallback。

生产形状 `H=8, D=512, window=128, compressed_capacity=64` 的隔离测试中，
原串行 kernel 为 98.83 us，通用 split-8 为约 12.4 us，约快 8 倍。

## 第二阶段：SM120 tensor-core fast path

SM120/SM121、`D=512` 且 local heads 不超过 16 时，dispatcher 默认先尝试
`sm120_tensorcore`：

- 一个 program 处理 16 个对齐后的 head 和 16 个候选；
- QK 与 PV 使用 BF16 tensor-core `dot`，softmax 和 split/merge 状态仍是 FP32；
- 默认 8 warps、split size 16；编译结果为约 40 registers/thread、32 KiB
  dynamic shared memory、无 spill；
- 外部 ABI、scratch 和 merge 与通用路径相同，因此失败时可先降级到通用
  Triton，再降级到原生 CUDA。

该实现借鉴 vLLM 的 head padding、candidate tiling、tensor-core QK/PV 和
split/merge 调度，但暂未复制 FlashInfer 的 FP8 packed KV、TMA gather 和
warp-specialized IO pipeline。

## Dispatch 和开关

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

## 性能结果

### Nsight Systems，8 层短上下文稳态 decode

相同机器、TP8、CUDA Graph node trace：

| sparse attention core | 每层平均 GPU 时间 |
|---|---:|
| FastLLM 原串行 Triton | 81.859 us |
| vLLM FlashInfer SM120 主核 + merge | 13.660 us |
| FastLLM SM120 split + merge | 5.796 us |

FastLLM 新主核平均 4.079 us，merge 平均 1.716 us。相对旧 FastLLM core
约快 14.1 倍；在该短上下文口径下也比 vLLM core 低约 2.36 倍。这个结论不应
外推到 top-k=512/1024 的长上下文，因为当前 FastLLM cache 每 token 的读取量
更大。

### 全模型 TP8，input 128 / output 256

固定 `FASTLLM_CUDA_GRAPH=1` 和 `FASTLLM_CUDA_CUSTOM_ALLREDUCE=1`：

| 指标 | 优化前 | SM120 fast path | 变化 |
|---|---:|---:|---:|
| TPOP | 13.51 ms/token | 10.01 ms/token | -25.91% |
| decode throughput | 74.03 tok/s | 99.93 tok/s | +34.99% |
| total throughput | 66.32 tok/s | 86.57 tok/s | +30.53% |
| total time | 3.8600 s | 2.9572 s | -23.39% |
| TTFT | 415.27 ms | 405.47 ms | -2.36% |

## 数值回归

- operator-level 通用 split/merge 对 FP32 reference 的最大误差不超过
  `1.49e-8`；
- 8 层模型的 generic 和 SM120 fast path 在 4 个步骤中生成 token 全相同；
- 全模型 16-step 与优化前生成 token 16/16 相同，prefill bit-exact；decode
  logits cosine 为 `0.997749--0.999921`，每一步 top-1 相同；
- 与 vLLM 在相同 cached context 的第二步 logits 相比，cosine 从旧 FastLLM
  的 `0.988647` 提升到 `0.992436`。

SM120 路径的 BF16 tensor-core reduction 顺序与原 FP32 serial reduction 不同，
所以不承诺 bit-exact decode logits。需要严格复现旧 FastLLM logits 时，可设置
`FASTLLM_CUDA_TRITON_DSV4_SPARSE_SM120=0` 使用通用 FP32 split-K 路径。

## 后续重点

短上下文下 core 已不再慢于 vLLM。下一阶段应面向长上下文和语义完整性：

1. 接入真正的 sparse top-k index/lens；当前模型能读取 `index_topk=512`，但
   FastLLM decode core 仍遍历全部可用 compressed blocks；
2. 为 SM120 增加 584 B/token 的 FP8 packed KV ABI，减少相对 FP32 window
   cache 约 3.5 倍、相对 BF16 compressed cache 约 1.75 倍的流量；
3. 再实现 TMA/cp.async bulk gather、双缓冲和 IO/math warp specialization；
4. 用 512/1024 candidates、不同 C4/C128 layer 和 batch 做单独的长上下文
   correctness、带宽和 occupancy 回归。

在完成 packed cache 与 indexer 前，当前结果应理解为“保持 FastLLM ABI 的
SM120 计算核优化”，而不是对 FlashInfer sparse MLA 的逐字复制。
