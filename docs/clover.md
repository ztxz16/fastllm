# Clover-35B-A3B-Preview 单卡部署

此路径使用 FastLLM 的原生 C++ / CUDA 前向和生成循环，支持原版 BF16 checkpoint。GPU 计算不需要安装 SGLang 或 Triton。

使用包含 Clover 支持的 FastLLM CUDA 构建：

```bash
ftllm server /path/to/Clover-35B-A3B-Preview --auto \
  --model_name clover --host 127.0.0.1 --port 8080
```

`--auto` 会选择一张显存足够的 CUDA GPU，检查权重、KV、FP32 循环状态、推测快照和工作空间的容量，同时为主存前缀缓存估算容量。默认使用模型的 131072-token 上下文、最多 4096-token 分块 prefill、16-token 自推测块；一次执行一个请求。可用 `--device cuda:0` 指定可见设备编号，用 `--max_context_length` 缩短上下文。

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"clover","messages":[{"role":"user","content":"计算 6×7，只给出答案。"}],"max_tokens":256,"temperature":0,"stream":true}'
```

## 架构与精度

SM80 及更新架构默认使用 BF16。SM75 及以下自动使用 FP16 权重、激活、卷积历史和注意力 KV；归一化权重、GDN 循环状态及矩阵乘累加保留 FP32。现代 GPU 也可以显式指定 `--atype float16` 使用 FP16 路径。SM90 的 FP16 路由专家复用 TMA 搬运与 Tensor Core 内核，保留 FP16 舍入边界和 FP32 累加；不满足架构或形状条件时回退到通用算子。FP16 与 BF16 不保证生成相同 token。

路由专家使用公共 CUDA FP16/BF16 MoE 后端，Clover 仅生成路由索引与分数。SM70/75 的 4～32 行专家计算复用 cuBLAS 批量矩阵乘、通用 FP32 SwiGLU 和精度转换；专家指针表在 GPU 上更新，支持动态路由的图回放。较大块及不满足对齐条件的形状保留通用分组矩阵乘，更低架构使用普通 CUDA 专家计算。cuBLAS 路径保留 FP32 累加和 FP16 存储舍入边界，但归约顺序不同，不保证与原路径生成相同 token。注意力在 SM75 及以上复用 FlashInfer，SM70 及以下使用有界工作区的在线 softmax 内核。FP16 GDN prefill 使用因果递推，不调用 BF16 专用内核。此 FP16 路径当前用于全 GPU 驻留，NUMA 混推仍使用 BF16。

需为目标架构编译，例如 `-DUSE_CUDA=ON -DCUDA_ARCH=75`。SM70 及以下使用仍支持对应架构的 CUDA 12.x 工具链；CUDA 13 不再生成这些架构的代码。FP16 与 BF16 都是两字节权重，精度切换不会降低模型的显存容量要求。

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

工具调用使用 checkpoint 的聊天模板及 Qwen XML 解析器，支持流式返回、多个工具调用和工具结果回填。工具名称、参数和 `tool_choice` 由 API 解析校验；Clover 的块生成后端尚未实现逐 token 约束采样。

此 checkpoint 只接受文本，图片输入会返回请求错误。

跨请求前缀缓存默认开启，使用 `--prefix_cache false` 关闭。Clover 与 Qwen3.5 共用前缀匹配及快照淘汰逻辑；快照同时包含已提交的 Attention KV、卷积历史和 FP32 GDN 状态。完整提示词命中时还会复用未施加采样惩罚的末尾 logits，采样参数和惩罚计数仍由新请求独立初始化。`usage.prompt_tokens_details.cached_tokens` 返回实际复用的输入 token 数。

快照保存在主存，恢复时复制到请求独立的显存；KV 按新增区间保存，多个快照共享不可变区间。沿用 `--prefix_cache_snapshot_interval_pages`（默认 16 页）、`--prefix_cache_snapshot_max_per_request`（默认 4）和 `--prefix_cache_snapshot_max_records`。未指定总快照上限时，Clover 使用 `max(8, max_batch × 每请求快照上限)`，`--auto` 会将其计入主存预算；显式上限仍然生效。连续请求按实际 token 前缀关联，优先回收同一分支的冗余旧快照，再淘汰其他未引用的历史。活动请求持有的检查点不可被其他请求回收；容量不足时跳过保存。保存位置为达到间隔的 prefill 分块末尾、完整提示词末尾及可安全提交的完成位置；只能复用这些位置上的完整 GDN 状态，不能任意截断递归状态。停止或输出上限截断验证块时，不保存超出实际返回文本的状态。缓存作用域为当前模型实例，支持 BF16、FP16 和 NUMA 混推；不跨进程持久化。

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

- 权重、激活和注意力 KV 使用 BF16 或 FP16，GDN 循环状态保持 FP32。
- Prefill 默认最多 4096 token 一块，可用 `--chunked_prefill_size` 显式调整上限。大块按 512 对齐，最后不足 512 的尾块单独计算，保持尾块的算子分派；块间可响应取消请求。
- GDN 直接读取卷积输出中的 Q/K/V，避免拆分复制；支持连续和跨行布局，布局优化不改变原有逐 token 状态和快照的运算顺序。
- 长 prefill 的 GDN 投影拆分使用 16 字节向量搬运，门控计算保持原有 FP32 运算；短块、未对齐地址或行跨度使用标量路径。
- 较宽的长 prefill 因果卷积在每个线程中复用八个相邻 token 的输入，保留各乘积的 BF16 舍入和 tap 累加顺序；多请求的 preview 和验证也复用八个相邻 token 的输入与权重，并保留逐 token 快照；较小形状使用原路径，历史状态在卷积完成后单独提交。
- 至少 2048 token、至少 16 个 value head 且不需要逐 token 快照时，GDN 可使用 SM80+ 的 64-token 分块算法：BF16 Tensor Core 矩阵乘、FP32 循环状态和 FP64 门控前缀和。它与原 FP32 逐 token 递推数学等价，但舍入路径不同，不保证逐值相同。硬件、共享内存或形状不满足要求时沿用原递推内核；preview、验证和快照回滚不切换到该算法。
- 分块 GDN 的工作区由模型持有，各层复用并按需增长；该 checkpoint 的 4096-token 块需要约 419 MiB，包含在 `--auto` 的 3 GiB 工作区与分配器预留中。无新增环境变量或外部算子库依赖。
- GDN 的 QK/KK、输出和 W/U 重建矩阵乘在 warp 内复用寄存器片段，减少重复共享内存读取，保持各输出的累加顺序。
- 分块状态更新提前异步搬运下一块的 W/K/U，通过共享内存按 16 字节合并写回历史状态和增量，并合并重复同步；FP32 状态更新顺序与 BF16 舍入保持不变。
- 原 GDN 递推内核在共享内存中复用输入，并用八线程组计算一个 value 行；SM90 的紧凑快照验证使用四线程组及 16-token 输入块，扩大 Q/K 复用范围，保留 FP32 递推和归约顺序。128/256 维归一化在行数较多时使用单 warp 一行，保持原归约顺序。
- 公共 MoE 后端为 SM90 的 prefill 和图内专家计算保留 TMA 搬运与融合 gate/up；小块 eager 解码及其他架构按形状选择通用实现。
- 至多 32 行、hidden 维度按 128 对齐的 SM90 专家 gate/up 使用 128 元素的 K 块，并由两个 warp 分别搬运 64 元素片段，减少循环与搬运发射的开销。Down 投影和较大块继续使用原分块方式。
- SM90 大块 prefill 在至少 2048 行、平均每个专家至少 64 条路由时使用原生 WGMMA，三阶段共享内存流水允许下一组矩阵乘与搬运重叠。专家结果成对写回，未对齐的 BF16 视图保留标量写入；保留 gate/up、路由加权结果的 BF16 舍入位置和专家求和顺序，无新增运行库或环境变量。较小块继续使用原有 GEMM。
- 专家输入重排按行进行 16 字节向量复制，未对齐的 BF16 输入可使用标量读取，避免大量仅搬运 512 字节的小线程块。
- 长 prefill 并行整理专家路由，以 warp 为单位搬运输入、向量化归约输出；保持每个输出元素按专家编号累加。小块解码继续使用原有调度方式。
- 专家归约同时完成 shared expert 的 sigmoid 门控、乘法和相加，省去两次独立 kernel 发射；保留 sigmoid、shared 乘积、路由专家总和及最终相加各自的 BF16 舍入。通用与 SM90 路径共用该归约，可省略 shared 参数以仅计算路由专家。
- Preview 只读已提交状态；部分接受恢复对应前缀的卷积及 GDN 状态，并截断 Attention KV 的逻辑长度。
- Attention KV 在 256 token 以内按 256 预留，之后按 1024 token 扩容，减少连续缓冲的分配、搬运和释放；预留容量不超过有效上下文按 256 向上取整的预算，逻辑长度与回滚规则不变。
- 支持流式 / 非流式文本 API、EOS、停止字符串、取消请求和运行时错误返回。
- 当前不支持多卡、BF16/FP16 以外的权重精度、多模态、LoRA 或约束解码。默认 `--max_batch 1`；自推测模式可设 `--max_batch 2`～`64`，相同 16/32-token 块宽的请求合批执行输入投影、GDN、MoE 和输出 head；输出投影保留 32 行分块的归约顺序，SM90 BF16 使用同一 cuBLASLt 算法合并发射；其他路径保留原分块。Q/K 归一化、RoPE 和 KV 追加合批执行，SM90 full attention 复用 FlashInfer 合批内核，其他架构保留逐请求内核；prefill 按请求执行。不同请求的可写 KV、循环状态、coreset 和接受前缀相互独立；低置信度扩散模式仍限单请求。64 是调度和算子的容量上限，实际可用并发还取决于显存、上下文和推测块宽；`--auto` 为每个活跃请求完整计算 KV、循环状态和回滚快照，并计入 MoE 图与 logits 工作区。显存不足时需减少并发或缩短上下文，不会自动截断请求。
- 至少 8 个请求、16/32-token 验证块的 GDN 快照保存初始 FP32 状态及逐 token 更新记录；部分接受时按原有舍入顺序重建状态，在 CTA 内复用 K、残差和衰减系数，避免为每个 token 写出完整矩阵。较小批次保留完整快照。`--auto` 仍按完整快照保守估算容量，以覆盖批次缩小和混合块宽时的回退。
- 推测预览和验证在至多 32 个请求时使用模型持有的 MoE 子图，更大批次走 eager；切换合批形状会释放旧图，捕获失败回退到 eager。已有的 `FASTLLM_CUDA_GRAPH=0` 可关闭子图。当前未启用整模型 CUDA Graph。
- SM90 及以上且实际加载的 PTX 支持时，GDN、归一化、旋转位置编码和采样辅助算子使用 CUDA PDL 隐藏相邻 kernel 的发射开销。每个 kernel 在读取前驱输出前等待依赖完成，计算及舍入顺序不变；旧架构或旧 PTX 保留普通 stream 发射，不增加环境变量或外部依赖。
- MoE 子图回放时，归一化直接写入图的输入缓冲，残差加法直接读取图输出；视图仅在当前 forward 内使用，避免每层输入、输出各一次复制，计算和舍入顺序不变。
- Attention 和 MoE 后的残差加法与下一次 RMSNorm 融合，先保留 BF16 残差舍入，再按原有 FP32 归约顺序计算归一化；2048 维路径在寄存器中复用残差，其他宽度使用通用 CUDA 实现。

不同框架的 BF16 计算顺序会累积误差，完整模型不保证与 HF 或 SGLang 逐 token 相同。

## 参考实现

数学定义和 checkpoint 来自 [DiffuSpace Clover](https://github.com/DiffuSpace/Clover-35B-A3B-Preview)（Apache-2.0），权重来自 [ModelScope](https://modelscope.cn/models/Diffuspace/Clover-35B-A3B-Preview)。实现参考了 checkpoint 中的 `modeling_clover_moe.py`、`sampling_clover_moe.py`，以及 [SGLang Clover PR](https://github.com/sgl-project/sglang/pull/46927) 的生成与融合边界。

CUDA kernel 在本仓库内实现；注意力复用仓库已有的 FlashInfer 代码。没有新增外部 GPU 算子库依赖。
