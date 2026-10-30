# Clover-35B-A3B-Preview 单卡部署

此路径使用 FastLLM 的原生 C++ / CUDA 前向和生成循环，支持原版 BF16 checkpoint。GPU 计算不需要安装 SGLang 或 Triton。

使用包含 Clover 支持的 FastLLM CUDA 构建：

```bash
ftllm server /path/to/Clover-35B-A3B-Preview --auto \
  --model_name clover --host 127.0.0.1 --port 8080
```

`--auto` 会选择一张显存足够的 SM80 或更新架构 GPU，检查权重、KV、FP32 循环状态、推测快照和工作空间的容量。默认使用模型的 131072-token 上下文、最多 4096-token 分块 prefill、16-token 自推测块；一次执行一个请求。可用 `--device cuda:0` 指定可见设备编号，用 `--max_context_length` 缩短上下文。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"clover","messages":[{"role":"user","content":"计算 6×7，只给出答案。"}],"max_tokens":256,"temperature":0,"stream":true}'
```

## 生成方式

默认自动部署沿用官方 SGLang Clover 路径的自推测方式：双向 preview 生成候选，因果前向验证，再提交接受前缀。默认接受规则为 `threshold`，`accept_ratio=1`、`accept_delta=0.6`。这是官方的阈值接受算法，不是保持目标分布的 rejection sampling。

请求中的 `clover_options` 可覆盖生成方式：

```json
{
  "clover_options": {
    "mode": "self_speculative",
    "block_size": 16,
    "accept_rule": "exact"
  }
}
```

`exact` 要求候选 token 等于验证行的 argmax。支持的块宽是 1、2、4、8、16、32。

如需官方 Hugging Face 的低置信度迭代填充方式：

```bash
ftllm server /path/to/Clover-35B-A3B-Preview --auto --speculative_algorithm off
```

或者在单个请求中设置：

```json
{
  "clover_options": {
    "mode": "low_confidence",
    "block_size": 32,
    "threshold": 0.95
  }
}
```

这里的 `off` 切换到模型原生的低置信度扩散生成，并非把 Clover 改成普通自回归模型。

默认采样参数读取 checkpoint 的 `generation_config.json`；原版配置为贪婪、`top_k=0`、`top_p=1`、`temperature=1`、`repetition_penalty=1.05`、`presence_penalty=0.9`、`frequency_penalty=0`。显式传入正温度可启用随机采样，`temperature=0` 为贪婪。惩罚只统计已输出 token。随机数实现与 PyTorch 不同，相同 seed 不保证跨框架生成相同序列。

## NUMA 混合推理

编译时启用 `USE_CUDA` 和 `USE_NUMAS`，可将路由专家放在主存，主干、router 和共享专家仍在单张 GPU 上计算。此路径使用原生 BF16 专家，不需要额外的算子库。

```bash
ftllm server /path/to/Clover-35B-A3B-Preview \
  --device cuda:0 --moe_device numa --moe_cuda_cache 0 \
  --cuda_embedding --threads 32 --max_batch 1 \
  --max_context_length 4096 --chunked_prefill_size 256 \
  --speculative_algorithm clover --speculative_num_draft_tokens 16
```

线程数需按 CPU 核数调整。`--moe_cuda_cache 0` 使用纯 NUMA 路由专家；显式设置如 `--moe_cuda_cache 3g` 可启用 GPU 专家缓存和动态分流。该值限制专家缓存及临时上传区，**不是进程总显存上限**；还需为主干权重、KV、FP32 循环状态、推测快照和工作区留出空间。当前 `--auto` 仍选择全部权重驻留 GPU 的方案，混推使用上述显式配置。

CPU 按整个请求块归并同一专家的路由；缓存与动态分流复用通用混推的热度策略、搬运成本判断和整专家调度。同一专家在当前块内统一分配到 CPU 或 GPU，临时上传一次，最后按原专家编号归约。CPU/GPU 都保留 gate/up、SwiGLU 和加权专家结果的 BF16 舍入边界，但矩阵乘累加顺序不同，完整模型不保证与全部 GPU 驻留路径逐 token 相同。NUMA 层不捕获包含 CPU 工作的 MoE 子图。

当前主存同时保留原始专家与 NUMA 分片，原版模型需要约 120 GiB 的专家存储，另加加载暂存与运行工作区。缓存是否提速取决于命中率、CPU、PCIe 和请求块大小，应以实际负载测量为准。

## 实现范围

- 权重、激活和注意力 KV 为 BF16，GDN 循环状态保持 FP32。
- Prefill 默认最多 4096 token 一块，可用 `--chunked_prefill_size` 显式调整上限。大块按 512 对齐，最后不足 512 的尾块单独计算，保持尾块的算子分派；块间可响应取消请求。
- GDN 直接读取卷积输出中的 Q/K/V，避免拆分复制；支持连续和跨行布局，布局优化不改变原有逐 token 状态和快照的运算顺序。
- 长 prefill 的 GDN 投影拆分使用 16 字节向量搬运，门控计算保持原有 FP32 运算；短块、未对齐地址或行跨度使用标量路径。
- 较宽的长 prefill 因果卷积在每个线程中复用八个相邻 token 的输入，保留各乘积的 BF16 舍入和 tap 累加顺序；短块及逐 token 快照仍使用原路径，历史状态在卷积完成后单独提交。
- 至少 2048 token、至少 16 个 value head 且不需要逐 token 快照时，GDN 可使用 SM80+ 的 64-token 分块算法：BF16 Tensor Core 矩阵乘、FP32 循环状态和 FP64 门控前缀和。它与原 FP32 逐 token 递推数学等价，但舍入路径不同，不保证逐值相同。硬件、共享内存或形状不满足要求时沿用原递推内核；preview、验证和快照回滚不切换到该算法。
- 分块 GDN 的工作区由模型持有，各层复用并按需增长；该 checkpoint 的 4096-token 块需要约 419 MiB，包含在 `--auto` 的 3 GiB 工作区与分配器预留中。无新增环境变量或外部算子库依赖。
- GDN 的 QK/KK、输出和 W/U 重建矩阵乘在 warp 内复用寄存器片段，减少重复共享内存读取，保持各输出的累加顺序。
- 分块状态更新提前异步搬运下一块的 W/K/U，通过共享内存按 16 字节合并写回历史状态和增量，并合并重复同步；FP32 状态更新顺序与 BF16 舍入保持不变。
- 原 GDN 递推内核在共享内存中复用 32-token 输入，并用八线程组计算一个 value 行，保留 FP32 递推和归约顺序。128/256 维归一化在行数较多时使用单 warp 一行，保持原归约顺序。
- SM90 的 prefill 和 MoE 图内专家计算可使用原生 TMA 搬运与融合 gate/up；小块 eager 解码及其他架构沿用通用 CUDA 路径。
- 至多 32 行、hidden 维度按 128 对齐的 SM90 专家 gate/up 使用 128 元素的 K 块，并由两个 warp 分别搬运 64 元素片段，减少循环与搬运发射的开销。Down 投影和较大块继续使用原分块方式。
- SM90 大块 prefill 在至少 2048 行、平均每个专家至少 64 条路由时使用原生 WGMMA，三阶段共享内存流水允许下一组矩阵乘与搬运重叠。专家结果成对写回，未对齐的 BF16 视图保留标量写入；保留 gate/up、路由加权结果的 BF16 舍入位置和专家求和顺序，无新增运行库或环境变量。较小块继续使用原有 GEMM。
- 专家输入重排按行进行 16 字节向量复制，未对齐的 BF16 输入可使用标量读取，避免大量仅搬运 512 字节的小线程块。
- 长 prefill 并行整理专家路由，以 warp 为单位搬运输入、向量化归约输出；保持每个输出元素按专家编号累加。小块解码继续使用原有调度方式。
- 专家归约同时完成 shared expert 的 sigmoid 门控、乘法和相加，省去两次独立 kernel 发射；保留 sigmoid、shared 乘积、路由专家总和及最终相加各自的 BF16 舍入。通用与 SM90 路径共用该归约，可省略 shared 参数以仅计算路由专家。
- Preview 只读已提交状态；部分接受从对应 token 的状态快照恢复，避免重复计算。
- 支持流式 / 非流式文本 API、EOS、停止字符串、取消请求和运行时错误返回。
- 当前不支持多卡、跨请求前缀缓存、其他权重精度、多模态、LoRA 或约束解码。默认 `--max_batch 1`；自推测模式可设 `--max_batch 2`～`8`，相同 16/32-token 块宽的请求合批执行 MoE 和输出 head；attention 投影每组最多 32 行，状态计算与 prefill 保持按请求执行。不同请求的 KV、循环状态、coreset 和接受前缀相互独立；低置信度扩散模式仍限单请求。
- 推测预览和验证使用模型持有的 MoE 子图，切换块宽会释放旧图；捕获失败回退到 eager。已有的 `FASTLLM_CUDA_GRAPH=0` 可关闭子图。当前未启用整模型 CUDA Graph。
- SM90 及以上且实际加载的 PTX 支持时，GDN、归一化、旋转位置编码和采样辅助算子使用 CUDA PDL 隐藏相邻 kernel 的发射开销。每个 kernel 在读取前驱输出前等待依赖完成，计算及舍入顺序不变；旧架构或旧 PTX 保留普通 stream 发射，不增加环境变量或外部依赖。
- MoE 子图回放时，归一化直接写入图的输入缓冲，残差加法直接读取图输出；视图仅在当前 forward 内使用，避免每层输入、输出各一次复制，计算和舍入顺序不变。
- Attention 和 MoE 后的残差加法与下一次 RMSNorm 融合，先保留 BF16 残差舍入，再按原有 FP32 归约顺序计算归一化；2048 维路径在寄存器中复用残差，其他宽度使用通用 CUDA 实现。

不同框架的 BF16 计算顺序会累积误差，完整模型不保证与 HF 或 SGLang 逐 token 相同。

## 参考实现

数学定义和 checkpoint 来自 [DiffuSpace Clover](https://github.com/DiffuSpace/Clover-35B-A3B-Preview)（Apache-2.0），权重来自 [ModelScope](https://modelscope.cn/models/Diffuspace/Clover-35B-A3B-Preview)。实现参考了 checkpoint 中的 `modeling_clover_moe.py`、`sampling_clover_moe.py`，以及 [SGLang Clover PR](https://github.com/sgl-project/sglang/pull/46927) 的生成与融合边界。

CUDA kernel 在本仓库内实现；注意力复用仓库已有的 FlashInfer 代码。没有新增外部 GPU 算子库依赖。
