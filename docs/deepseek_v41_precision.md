# DeepSeek V4.1 参考精度模式

`FASTLLM_DSV41_REFERENCE_MATH=1` 用于对齐官方模型的 CPU PyTorch 参考计算。必须在加载模型前设置；默认关闭。

```bash
export FT_NUMAS=1
export FASTLLM_DSV41_REFERENCE_MATH=1
export FASTLLM_DEEPSEEK_V41_BLAS_LIBRARY=/path/to/torch/lib/libtorch_cpu.so

numactl -C 0-31 -m 0 ftllm server /path/to/DeepSeek-V4.1-Flash \
    --device cuda --moe_device cpu --cuda_shared_expert false \
    --threads 30 --dtype float16 --kv_cache_dtype bfloat16 --max_batch 1
```

使用包含本改动的构建。数学库应与生成参考张量的 CPU PyTorch 环境一致；此模式通过该库的 SLEEF/VML 入口计算 CPU 路由和 SwiGLU。不设置数学库时使用标准库函数，不能据此预期逐值一致。

该模式保留 FP8 共享专家的原始分块 scale，对 FP8/NVFP4 专家逐 block-32 累加。CUDA 侧保留 FP32/BF16 存储边界，并匹配参考的归一化归约、mHC、RoPE、分组投影、64 槽在线 softmax 和 FP32 输出头。CUDA graph 在此模式下关闭。CPU 使用 AVX-512 并行计算独立输出，无 AVX-512 时保留标量回退；CUDA 并行计算注意力槽位、输出通道和归约中的独立分量，保持参考累加顺序。单请求 2–512 token 的 CPU prefill，在专家权重为受支持的 block-32 FP8/NVFP4 时按专家合并成最多 4-token 的计算任务，并按原专家顺序还原每个 token 的输出；其他形状继续走既有路径。HC 的 4×4 Sinkhorn 迭代保存在 warp 寄存器中，单 token 投影合并读取连续 K 值，保持原有 FP32 舍入。

2026-09-12 在 EPYC 9374F、RTX 4090 D、单卡注意力和 CPU MoE 上，对完整 40 层模型执行了 5-token Prefill 和两步固定输入 Decode：120 个层末输出、3 组 logits 均逐值一致。另有 40 层独立 MoE 对比全部一致。这是上述输入和参考环境的验证结果；尚未据此验收长上下文、多请求、多卡、量化 KV cache 或官方 GPU 内核的逐值精度。

`deepseekV41PrecisionRegression` 覆盖激活量化、RoPE 存储边界、64 槽注意力及扩容缓存、mHC FP32 舍入、混合专家和 block-32 累加。普通模式与参考模式应分别运行：

```bash
FT_NUMAS=1 numactl -C 0-31 -m 0 ./deepseekV41PrecisionRegression --cuda
FT_NUMAS=1 FASTLLM_DSV41_REFERENCE_MATH=1 \
    numactl -C 0-31 -m 0 ./deepseekV41PrecisionRegression --cuda
```

2026-09-12 优化后，在上述 EPYC 9374F + RTX 4090 D 配置下，使用 30 线程、`FT_NUMAS=1 numactl -C 0-31 -m 0`、单请求和 BF16 KV，实测如下。每组运行 3 轮、每轮 32 步 decode，取中位数；排除模型加载、tokenization 和 HTTP 开销，关闭 dump、前缀缓存及 CUDA graph。

| 输入 token | Prefill token/s | Prefill 秒 | Decode token/s |
| ---: | ---: | ---: | ---: |
| 128 | 22.42 | 5.71 | 8.10 |
| 512 | 20.58 | 24.88 | 7.50 |

相对于上一轮参考模式（prefill 约 13.96 token/s、decode 约 5.10–5.37 token/s），prefill 再提速 1.47–1.61 倍，decode 再提速 1.47–1.51 倍。优化后重新加载完整模型，既有 120 个层末输出和 3 组 logits 仍逐值一致；两种较长测速输入的生成 token 序列也与优化前相同。较长输入的结果属于吞吐和生成序列回归，没有扩大上文的逐值精度验收范围。


## 允许舍入差异时的速度配置

在单 GPU 注意力、CPU 路由专家上，可使用普通数学模式并将共享专家放到 GPU：

```bash
export FT_NUMAS=1
export FASTLLM_DSV41_REFERENCE_MATH=0
export FASTLLM_DSV41_CUDA_GRAPH=0
export FASTLLM_DSV41_DISABLE_PREFIX_CACHE=1

numactl -C 0-31 -m 0 ftllm server /path/to/DeepSeek-V4.1-Flash \
    --device cuda --moe_device cpu --cuda_shared_expert true \
    --threads 30 --dtype float16 --kv_cache_dtype bfloat16 --max_batch 1
```

GPU 共享专家保留 gate/up 的 BF16 舍入与截断、激活的 BF16 舍入和后续 FP8 量化。在 graph 关闭且路由专家位于 CPU 时，提前复制 CPU 分支的输入，使共享专家的 GPU 计算可与 CPU 计算重叠。参考模式或多卡情况下，共享专家由 MoE 路径统一计算。

普通模式的单 token mHC 将独立点积和平方和分配到多个 CUDA block，保留各输出原有的累加和归约顺序。CPU MoE 在满足上文权重和形状约束时，也可在普通模式按专家合并最多 8-token 的任务；参考模式继续采用最多 4-token 的任务并保留其数学计算方式。可分别设置 `FASTLLM_DSV41_LEGACY_HCMIX_DECODE=1`、`FASTLLM_DSV41_DISABLE_SHARED_OVERLAP=1`、`FASTLLM_DSV41_DISABLE_BATCHED_MOE=1`，回到对应改动前的路径进行对照。

普通模式的 CPU NVFP4 block-32 矩阵计算，在多个 token 间复用权重解码与 scale，并展开固定大小循环以减少寄存器溢出。各 token 保留独立累加和原有归约；仅在不会溢出的输入和 scale 范围内，将精确的 2 的幂缩放移到 scale 上，极端输入及 E8M0 最小 scale 保留原缩放顺序（含 DAZ 非规格化数处理）。单 token 仍走既有内核。对于至少 256-token 的合并 MoE，输入量化与输出归并按 token 分给线程池，每个 token 内部的专家累加顺序不变。设置 `FASTLLM_DSV41_DISABLE_NVFP4_TOKEN_REUSE=1` 或 `FASTLLM_DSV41_DISABLE_PARALLEL_MOE_ROWS=1` 可分别禁用这两项优化进行对照；参考模式不启用这两项优化。

该配置接受与参考模式的数值差异。基础问答、计算和较长文本检查不能证明广泛能力等价；固定输入 logits 相对参考模式仍有明显差异。要求逐值对齐时，应继续使用参考模式。上述速度路径的验收范围是单卡、单请求、BF16 KV、关闭 graph 和缓存，并非长上下文、多请求或多卡的完整精度评测。


2026-09-12 在 EPYC 9374F + RTX 4090 上，用上述 NUMA/线程配置、FP16 加载、GPU 共享专家、CPU 路由专家、BF16 KV，通过 HTTP 固定生成 64 token、各运行 3 轮，中位数如下。首 token 时间包含接口开销；输入速度是输入 token 数除以该时间，不是剥离接口开销的算子计时。

| 输入 token | 首 token 秒 | 输入 token/s（估算） | Decode token/s |
| ---: | ---: | ---: | ---: |
| 113 | 1.73 | 65.44 | 19.83 |
| 498 | 6.65 | 74.85 | 19.08 |

相对于本轮前的 2.78/13.40 秒，首 token 时间分别缩短约 38%/50%；历史 PR 为 2.53/7.04 秒，两个输入长度的 prefill 均已达到其历史速度，decode 保持在约 19–20 token/s。上述结果属于普通数学模式。

最终普通/参考模式 CUDA 联合算子回归分别通过 454/394 项，CPU 构建分别通过 226/110 项，包含极小 scale 的 DAZ 边界和 257-token 的并行 MoE 路由。完整模型固定 5-token prefill 与两步 decode，120 个层末输出和 3 组 logits 相对此前已修复的普通速度模式全部逐值一致。24 项基础质量检查和完整中文数据分析通过，6 条测速文本与 25 条质量文本均与本轮前同一 HTTP 请求的输出完全相同。这些结果是相对既有普通模式的回归，不是与官方参考模式的广泛能力等价证明。

## 8K 配置及新版 PR 整合复验

2026-09-13，以 PR #727 的 `fd873f3a` 为底座融合上述本地修复后，重新构建并验证。普通模式在单 RTX 4090、CPU 路由专家、GPU 共享专家、FP16 加载、BF16 KV、30 线程及 NUMA 0 绑定下，24 项基础质量检查、完整中文数据分析和 8207-token 三位置 JSON 检索均通过，检索的 9 个数字全部正确，输出正常结束。

该 GPU 共享专家配置处理 8K 输入时，启动命令应显式加上 `--chunked_prefill_size 512`。此前本地修复版在 4096 分块下出现 CUDA OOM；本轮采用 512 分块完成验证，没有重试 4096 分块。继续保留 `FT_NUMAS=1 numactl -C 0-31 -m 0`、关闭 graph 和前缀缓存等上述测试设置。

本轮通过 HTTP 每组运行 3 轮、固定生成 64 token，中位数如下。输入速度为输入 token 数除以首条文本到达时间的估算，包含接口开销；每轮前缀缓存命中均为 0。

| 实际输入 token | 首 token 秒 | 输入 token/s（估算） | Decode token/s |
| ---: | ---: | ---: | ---: |
| 113 | 1.66 | 67.99 | 19.93 |
| 498 | 7.27 | 68.50 | 19.22 |
| 8178 | 109.28 | 74.84 | 19.06 |

8K prefill 仍慢于本机未经修复的 PR 配置（CPU 共享专家、4096 分块，首 token 69.24 秒），但该原版未通过本轮基础质量、长文分析和 8K 检索检查。两套配置的性能对比不能用于隔离单项修复的收益。

整合版普通/参考模式 CUDA 联合算子回归分别通过 454/394 项。另行启用参考模式及原数学库，完整 40 层的 5-token prefill 和两步固定输入 decode 共 120 个层末输出、3 组 logits、6,122,240 个 FP32 值与既有官方 CPU 参考结果逐值一致。这项参考模式结果不代表普通速度模式逐值等价，也没有扩大到 8K 的逐值精度或真实 TP 验证。
