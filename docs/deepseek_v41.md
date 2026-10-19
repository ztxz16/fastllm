# DeepSeek-V4.1-Flash 支持说明

[返回 README](../README.md) · [DeepSeek-V4 部署指南](deepseek.md) · [混合推理](mixforward.md)

本文记录 FastLLM 对 DeepSeek-V4.1 系列（`model_type = deepseek_v41`，目前为 DeepSeek-V4.1-Flash）的支持范围、
架构差异、启动方式与验证方法。

面向自动化 agent 的 `--auto` 启动、参数覆盖和验收流程见 [模型部署速查](model-deployment-agent.md)。

实现与回归入口：

- `include/models/deepseekv41.h` / `src/models/deepseekv41.cpp`：模型（继承 `DeepSeekV4Model`，复用其 MoE / HC-post / WoA / 采样等基础设施）
- `src/models/deepseekv41_vision.cpp`：视觉编码器（ViT + aligner）与图文前向 `ForwardMultimodal`
- `src/devices/cpu/deepseekv41ops.cpp`：V4.1 专用算子的 CPU 参考实现
- `src/devices/cuda/models/deepseekv41-kernels.cu`：对应的 CUDA kernel（面向 SM86 等无 FP8 tensor core 的设备；
  稀疏注意力与 indexer 打分在 SM80+ 上走 BF16 mma，其余算子为 FP32）
- `src/models/deepseekv41_dspark.cpp`：DSpark 投机解码（草稿层 `mtp.*`、校验与回滚）
- `src/devices/cuda/models/deepseekv41-dspark-kernels.cu`：DSpark 草稿侧 markov head 整条链的融合 kernel
- `tools/fastllm_pytools/deepseek_v41_engram.py`：Engram 哈希元数据生成
- `tools/fastllm_pytools/encoding_dsv41.py`：官方 V4.1 prompt 编码（vendored）
- `tools/fastllm_pytools/deepseek_v41_multimodal.py`：图像动态分辨率预处理（移植自官方 `image_processor.py`）与图像占位符展开
- `test/basic/deepseek_v41_reference.py`：与官方 `inference/model.py` 的端到端数值对齐测试
- `test/basic/deepseek_v41_vision_reference.py`：图文输入的端到端对齐测试（含真实 ViT 权重验证）
- `test/basic/deepseek_v41_dspark.py`：DSpark 的精确性（开 / 关输出一致）与回滚测试
- `test/basic/test_deepseek_v41_cpu_fixture.py`：CPU-only 的固定 fixture 对齐回归（只依赖 numpy，可进 CI）
- `test/ops/deepseekV41OpsRegression.cpp`：11 个 V4.1 专用算子的 CPU / CUDA 一致性测试

## 相对 DeepSeek-V4 的架构变化

| 项目 | DeepSeek-V4 | DeepSeek-V4.1 |
| --- | --- | --- |
| Hyper-Connections | 每个子层自己算 pre/post/comb 并立即使用 | pre 由上一个子层计算、下一个子层使用；最后一层 FFN 的 pre 用于 lm_head 前的折叠 |
| 压缩注意力 | 每层独立 compressor（ratio 4 / 128，带 ape、overlap） | `compress_ratios` 取 0 / 1 / 2；只有 `kv_source_layer_ids` 中的层拥有 compressor 与压缩 KV cache，其后各层直接读取（跨层共享） |
| Indexer | 每个 CSA 层有独立的 128 维 compressor + Hadamard 旋转 | key 由 compressor latent 经 `wk + k_norm` 派生；只有 `index_source_layer_ids` 层运行 indexer，其它层复用最近的 top-k |
| 两级 top-k | 无 | `candidate_source_layer_id` 层先按 `candidate_block_size` 分块选出 `candidate_topk_blocks` 个块，之后的 index source 只在候选块内选 top-k |
| Engram | 无 | `engram_layer_ids` 层前对 residual stream 做 n-gram 哈希查表（每层约 100 GB 的 FP8 表）并门控写回 |
| 路由 | 前若干层 hash 路由 | 全部 noaux_tc，`gate.bias`；图像 token 使用 `gate.bias_vl` |
| q 处理 | 额外 RMS 归一 | 无 |
| 权重格式 | FP8 128x128 块 scale | 稠密与共享专家 FP8 32x32 块 UE8M0 scale；路由专家 FP4（沿 K 每 32 个一组 UE8M0 scale） |
| 附加模块 | MTP / DSpark | DSpark 草稿层（mtp.*，markov head 为 embed + head、草稿层 128 专家 top-3、目标层取自主干）、视觉编码器（vision.* / aligner.* / image_start / image_end / image_newline） |

## 当前支持范围

已实现并通过数值对齐测试：

- 文本推理（prefill + decode，含分块 prefill），CUDA 与 CPU 两套算子路径；
- 多请求批量 decode：多个请求的 token 拼成一个序列共享一次前向（Linear / MoE / Engram 查表按整批执行，
  注意力与压缩 KV 按请求分别执行），见下文"多请求与前缀缓存"；
- 前缀缓存 / 多轮对话复用（`--cache_history true`）；
- 跨层共享压缩 KV（ratio 1 / 2）、两级 indexer top-k、Engram、Hyper-Connections、sqrt-softplus 路由；
- 与官方实现一致的 FP8 / FP4 伪量化（窗口 KV、压缩 KV、indexer q/k）；
- 真实 checkpoint 的权重格式（FP8 32x32、FP4 路由专家、FP8 + UE8M0 的 Engram 表）；
- 图像输入（OpenAI 接口的 `image_url`）：ViT + aligner、图像 token 的 `gate.bias_vl` 路由与 Engram 掩码，详见下文；
- DSpark 投机解码（`mtp.*` 草稿层），详见下文"DSpark 投机解码"。
- 张量并行（`--tp 2`、`--tp 4`、`--tp 8`），详见下文"张量并行"。

尚未实现：

- 视觉编码器与 DSpark 草稿层的张量并行及 CUDA Graph；主干 decode / DSpark 校验已支持分段图；
- DSpark 与批量 decode 的组合（批内不产生候选，只保持草稿缓存同步）、DSpark 与图文请求的组合。

## Engram 元数据

Engram 的哈希基于 tokenizer 归一化（NFKC / NFD / 去重音 / 小写 / 空白折叠）后的"压缩 token id"。
该映射依赖 HuggingFace `tokenizers` 的 normalizer，因此由 Python 侧生成、C++ 侧加载：

```bash
python -m ftllm.deepseek_v41_engram /path/to/DeepSeek-V4.1-Flash
# 生成 /path/to/DeepSeek-V4.1-Flash/engram_meta.json（压缩词表大小应为 99092）
```

通过 `ftllm` 启动时自动读取模型目录的 `engram_meta.json`，缺失时自动生成；模型目录只读时写到
`~/.cache/fastllm/engram/`，C++ 侧自动从该缓存目录读取。素数桶布局由 C++ 侧按官方算法推导，
并与 `engram_num_embeddings` 做一致性校验，无需指定元数据环境变量。

Engram 表（两层，各约 100 GB）以 FP8 + UE8M0 scale 原样保存；查表在 CPU 完成，
`wkv` 投影与门控在 GPU 完成。默认 `--ngram_device cpu` 将表常驻内存；
`--ngram_device disk` 与 Qwen4 PLE 共用通用磁盘权重加载和 `EmbeddingDirect` 算子，
按行读取权重及 scale，不加载整张表。磁盘读取可能增加延迟，访问过的数据仍可能占用操作系统页缓存。

### Engram 的分段计时与几个开关

查表发生在模型代码里而不是算子里，`FASTLLM_PRINT_PROFILE` 看不到它，需要单独计时：

```bash
FASTLLM_DSV41_ENGRAM_PROFILE=1 FASTLLM_CUDA_SYNC=1 ftllm server ...
```

会按 decode / prefill 分桶打印四段的平均耗时：`hash`（算行号）、`gather`（读表 +
FP8→BF16）、`wkv`（投影）、`apply`（门控写回）。后两段在 GPU 上异步下发，
不加 `FASTLLM_CUDA_SYNC=1` 只能量到 kernel launch 的时间。`=2` 额外逐次打印，
`FASTLLM_DSV41_ENGRAM_PROFILE_EVERY=N` 控制中途汇总的频率（默认 64 次，0 表示只在退出时打印）。

Engram 配置如下；默认值见表。权重格式与数学边界应分别核对：

| 开关 | 作用 |
|---|---|
| `FASTLLM_DSV41_ENGRAM_POOL`（默认开，=0 关） | 查表改用 fastllm 的常驻线程池，替掉每次调用现场 create/join 最多 32 个 `std::thread` 的写法。输出逐位相同，prefill 收益最明显。 |
| `FASTLLM_DSV41_ENGRAM_PREFETCH=1` | 跨层预取。后台线程提前计算下一层的 n-gram 行号；常驻表还会在缓存预算内预读表行，磁盘模式只预计算行号。后台线程只看历史窗口的快照，不引用请求状态。 |
| `FASTLLM_DSV41_ENGRAM_MADVISE=random / hugepage / both` | 仅作用于常驻表：`random` 设置 `MADV_RANDOM`；`hugepage` 使用匿名 mmap + `MADV_HUGEPAGE` 分配，减少页表与 TLB 开销，并省掉 `std::vector` 的清零。 |
| `FASTLLM_DSV41_ENGRAM_WKV_FP8`（默认开，=0 关） | `layers.{1,14}.engram.wkv.weight` 在真实权重里本来就是 F8_E4M3 + UE8M0 块 scale（`[25600, 6144]`，block 32x32），默认会被解量化成启动 dtype（float16），每层 157 MB 变 314 MB。打开后按原样保留 FP8，**不做任何重量化**——权重数值就是 checkpoint 里的那份，比解成 float16 还少一次舍入；省下每层 157 MB 显存与同样多的每步带宽。只有伴随的 `.scale` 张量存在时才切换，权重是 BF16 的迷你模型不受影响。 |

## 自动配置（推荐）

自动化部署建议先把本页和 [Agent 部署速查](model-deployment-agent.md) 交给 agent。对 V4.1 Flash，自动规划器会根据资源尝试 `fast_prefill`、DSpark、Engram 放置和长上下文预算；主机内存不足时会把 Engram/专家回退到磁盘，并在日志中给出实际配置。

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash-1017 \
  --auto --model_name deepseek-v4.1-flash-1017 \
  --host 0.0.0.0 --port 8080
```

如果需要模拟较小机器，使用 `--auto_gpu_memory_limit 20g` 和
`--auto_host_memory_limit 128g`；它们限制自动规划预算，省略时自动读取实际硬件。

## 手动配置参考

下面以单张 CUDA GPU + 大内存主机为例（专家与 Engram 表放在 CPU 内存；显存需求取决于上下文和 chunk）：

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device cuda --moe_device numa \
  --dtype float16 \
  --chunked_prefill_size 4096
```

- 没有 FP8 tensor core 的 GPU（如 SM86）请使用 `--dtype float16`，稠密 FP8 权重会在加载时按 32x32 块 scale 反量化；
- `--kv_cache_dtype` 控制长期 KV 缓存（压缩 KV + indexer key）的存储精度，见下面的“KV 缓存存储精度”一节。
  默认 BF16，可选 `fp8_e4m3`（1650 B/token）与 `fp4_e2m1`（890 B/token，与官方一致，且数值无损）；
- `ftllm` launcher 的自动配置按 config 计算 V4.1 的常驻内存下界（FP4 专家约 289 GB + Engram 表约 203 GB +
  稠密部分），权重文件不全时也不会低估；主机内存不足以放下专家与 Engram 表时会退到 `moe_device=disk`；
- 单路 CPU 机器可用 `--moe_device cpu`；
- 内存需求：Engram 表约 200 GB + 路由专家（FP4）约 270 GB + 加载临时空间；
- 首次启动会生成 `engram_meta.json`（约 1 分钟）并读入两张 Engram 表。

CPU / NUMA 专家使用 FastLLM 自有线程池，由 `--threads` 控制；CLI 会自动设置 `FT_THREADS`，无需重复指定。

### 导出 Q2_K/Q4_K 混合量化

`tools/deepseek_v41_export_q2.py` 按分片导出可独立加载的 FastLLM 模型目录，配置见
`example/quant/deepseekv41/Q2_K_MIXED.json`：主干路由专家的 `w1/w3` 使用 Q2_K，`w2` 使用 Q4_K；
其它线性权重使用 FP16，模型映射保护的权重保留指定精度。Engram 表和 DSpark 草稿权重逐字节复制。
混合分片会先分离这些原样保留的张量，导出不依赖 `FASTLLM_DSPARK_TOKENS` 设置。
原始 Flash checkpoint 的路由专家已经是 FP4，因此这里是从 FP4 再量化，未使用校准集或 importance matrix。

```bash
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  python tools/deepseek_v41_export_q2.py \
  --model /path/to/DeepSeek-V4.1-Flash \
  --output /path/to/DeepSeek-V4.1-Flash-Q2_K-Mixed \
  --threads 28
```

中断后使用相同命令并增加 `--resume`；脚本检查源配置、量化配置和分片信息，跳过已经完成的分片。
输出是 FastLLM 的扩展 safetensors 格式，需使用支持导出激活元数据和 V4.1 GGML NUMA 激活转换的 FastLLM：
独立 scale 会内嵌到转换后的权重，`config.json` 中的 `fastllm_activation_quantized_linears` 保留原模型的激活量化边界。

单 NUMA 混推启动示例（CPU 核号应按机器的 NUMA 拓扑调整）：

```bash
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash-Q2_K-Mixed \
  --device cuda:0 --moe_device numa --ngram_device cpu \
  --threads 28 --dtype float16 --kv_cache_dtype bfloat16 \
  --chunked_prefill_size 4096 --max_batch 1 --dspark 5 \
  --fast_prefill --cache_history true --host 0.0.0.0 --port 8080
```

### Q2 混合量化的缓存与计算

支持 `--moe_cuda_cache` 的 NUMA Q2_K_R4/Q4_K_R4 专家直接借用已注册的主机权重，避免第二份完整快照。
普通 decode 与 2–8 行 DSpark 验证保留 block-32 FP8 激活量化、GGML Q8 编码、SwiGLU 限幅、路由权重位置和逐专家 BF16 舍入。
命中率应区分 cache lookup 与全部 route：CPU 分流未必查询 CUDA 缓存，lookup 命中率不是全部专家驻留率。

长 prefill 在 BF16、block-32 激活、Q2_K gate/up + Q4_K down 以及兼容形状和设备上使用 grouped MMQ，支持 NUMA R4 布局。
输入按 token 量化后按 route 分发；scale 和 min 修正保留 FP32。至少 1024 行时使用 64-row 计算块，其余使用小块，尾部访问限于当前专家。
down 权重在独立 stream 上传，与 gate/up 计算重叠；不支持的布局或显存不足时保留回退。
不同矩阵累加路径可能改变 BF16/FP8 舍入及后续路由，功能检查不替代模型质量评估。

Q2 混合量化也可以使用磁盘专家和有容量上限的 CPU 专家缓存。Engram 同时放磁盘、关闭 DSpark 的示例：

```bash
FT_NUMAS=1 OPENBLAS_NUM_THREADS=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash-Q2_K-Mixed \
  --device cuda:0 --moe_device disk --moe_cpu_cache 64g --ngram_device disk \
  --threads 28 --dtype float16 --kv_cache_dtype bfloat16 \
  --chunked_prefill_size 4096 --max_batch 1 --dspark 0 --mtp 0 \
  --fast_prefill --cache_history true --host 0.0.0.0 --port 8080
```

`64g` 表示 64 GiB 专家缓存容量，不是进程总内存上限。磁盘缓存路径为 V4.1 的 GGML 路由专家和
FP16 共享专家保留 block-32 激活量化、BF16 中间舍入及路由权重位置；这两种专家在 CPU 上计算。
专家缓存未命中时默认使用 direct I/O，避免另在系统页缓存中保留整份专家；Engram 行读取仍使用系统页缓存。
磁盘 MoE 已用独立标量计算验证，不能直接沿用 NUMA 配置的整模型精度评测结论。


磁盘路径直接读入对齐专家缓冲，保留不对齐文件偏移、分散 gate/up 和文件尾部的普通读取回退。
CPU 缓存按热度和最后访问时间管理，4096 次访问后衰减；预算限制专家 payload，不限制整个进程 RSS。
另有最多 32 MiB 的读取缓冲复用池。2–8 行 MoE 验证使用双专家预取，保持计算、缓存准入及归约顺序，异常退出也等待全部预取完成。
未消费专家的暂存空间不计入活跃缓存预算；该模型单个 Q2 路由专家约 13.7 MiB。

部署速度、内存和适用范围见[Benchmark](benchmarks/deepseek_v41.md)。

## 张量并行

支持 GPU 共享专家、单 token 异步调度，以及分段和单请求整图 CUDA Graph。吞吐收益取决于 GPU、卡间通信和
CPU 专家占比，应在相同配置下分别测量 decode 与首 token 延迟。

```bash
FASTLLM_CUDA_GRAPH=1 ftllm server /path/to/DeepSeek-V4.1-Flash \
  --tp 2 --moe_device numa --cuda_shared_expert true --dtype float16
```

`--tp 2` 会把主 device 归一化成 `multicuda:0,1` 并设置 `FASTLLM_TP`（触发加载期的权重切分）。
用户显式给出的 `--moe_device`（如 `numa`）不受影响：路由专家仍在 CPU / NUMA 上算一份，
结果广播回两张卡。直接写 `--device multicuda:0,1` 也能启用，只是稠密权重改为在首次使用时切分，
显存峰值更高。

### 切什么、为什么

沿用 DeepSeek-V4 的切法，只切 **query head** 这一条线：

| 权重 | 切法 | 说明 |
| --- | --- | --- |
| `attn.wq_b` | 按行切（`linearRow`） | 输出 q 的 head 维分片，`Reshape` 会把 tpAxis 从最后一维换算到 head 维 |
| `attn.attn_sink` | 按 head 切 | 与 q 的分片区间一致 |
| `attn.wo_a` | 按 head 组切（`linearRow`） | 区间必须对齐到 `o_group`（每组 `num_attention_heads / o_groups` 个 head） |
| `attn.wo_b` | 按列切（`linearColumn`） | 输入分片，输出 all-reduce 回复制布局 |
| `ffn.shared_experts.gateup` / `w2` | 行切 / 列切 | 与其它 MoE 模型一致 |
| `head.weight` | 按行切 | 两卡各算一半词表，`LLMSamplingBlock` 已有分片 logits 的贪心 / 汇聚路径 |

其余全部**复制**（每卡一份完整副本）：`wq_a`、`wkv`、`compressor.*`、`indexer.*`、`ffn.gate`、
各 RMSNorm 权重，以及跨层共享的压缩 KV、indexer key、滑窗 KV 缓存。

为什么跨层共享的压缩 KV 与两级 indexer 选复制而不是切分 + all-gather：

- **压缩 KV 与 head 无关**。V4.1 的注意力是 MLA 式的：每个位置只有一份 `head_dim` 的 latent，
  所有 query head 共享它。沿 head 切它没有意义，沿序列切则每一层注意力都要 all-gather 整段 KV，
  通信量正比于上下文长度 × 层数。复制的代价只是一份显存（压缩后本身就比滑窗 KV 小 1~2 倍）。
- **压缩 KV 是跨层共享的**。`kv_source_layer_ids` 的层算出的 cache 会被后面若干层直接读，
  切分后每个消费层都要重新 all-gather，而不是只在生产层付一次代价。
- **两级 indexer 的候选块也是跨层共享的**。`candidate_source_layer_id` 选出的候选块被之后所有
  index source 层复用。indexer 的分数矩阵是 `[token, m]`（1M 上下文时每 token 有 50 万个候选），
  切 index head 意味着要对这个矩阵做 all-reduce 才能取 top-k——通信量比整个注意力还大。
  而且 top-k 必须在两张卡上**逐位一致**：卡 0 和卡 1 选到不同的 KV 块，
  两边算出的注意力就不是同一个函数的两个分片了。复制 indexer 既省通信又天然保证一致。
- **路由分数变换与专家选择同理**逐卡各算一份，保证两卡选到同一组专家。
- **Engram 表在 CPU**，查表结果按 token 复制到每张卡。

### 约束

TP 要求 query heads 均匀分片，且每个分片包含完整的 `o_group`。
真实模型 64 头、`o_groups=8`，TP=2/4/8 分别对应每卡 32/16/8 个 head；
稀疏 Attention 的 MMA 和标量回退路径均支持尾部不足一个 tile 的 head，
不会读取或写入填充 head 对应的全局内存。不满足分组对齐条件时模型会打印说明并**整体退回单卡**（撤销注意力与 head 的 TP 权重注册，
把 device map 改回 `cuda:<第一张卡>`），而不是做"只切 FFN"的半张量并行。

八卡 GPU 专家配置示例（Engram 表仍常驻 CPU 内存）：

```bash
FASTLLM_CUDA_GRAPH=1 ftllm server /path/to/DeepSeek-V4.1-Flash \
  --tp 8 --dtype auto --ngram_device cpu --chunked_prefill_size 512
```

默认 MoE 设备跟随主设备，使用八卡张量并行。此命令不启用近似 `--fast_prefill`。

视觉编码器（ViT + aligner）与 DSpark 草稿层仍在单卡上算。`--tp 2 --dspark 5` 可以组合使用：
目标模型按上述方式张量并行，草稿层在第一张卡运行，共享的 `head.weight` 沿用两卡词表分片，
完整 logits 汇总到第一张卡进行候选生成与校验。校验回滚同时截断两张卡上的缓存副本。

TP 主模型特征传回草稿 GPU 时显式指定设备；回归覆盖两种首卡顺序、预填充、不同接受长度及缓存副本一致性。

### 混合推理的 decode 调度

TP 的共享专家在各卡复制计算，保留完整点积后的 BF16 舍入边界；混合执行的路由专家仍使用配置的 CPU / NUMA 后端。
输入先暂存，再发射 GPU 共享专家，使两条分支重叠。单 token TP 默认使用常驻 worker 与流事件，
保持 Graph 和段外算子使用的 GPU 地址稳定，并在采样前等待所有 logits 分片的生产完成。

CPU 专家结果上传、共享结果相加与 HC post 在一次 worker 调用中执行，保留 AddTo 的中间舍入。
普通单 token 和 DSpark verify 共用该路径。输入直接 D2H，CPU MoE 输出保持二维以复用 GPU 副本；每次上传覆盖新结果。
不支持的布局、类型或设备使用原路径。同步 H2D 保证 CPU 源在回调返回后即可复用。

DSpark 提交 KV 时复用同一套异步 TP 派发，在整轮提交结束后同步两张卡。
部分接受时，截取的 KV 行保留到同步结束，避免逐层同步以及临时缓冲提前释放。
NUMA 的多行 BF16 输入在至少 8192 个元素时按完整 block-32 分配激活量化任务，保持原有量化结果；
单行和较小输入保持串行，避免线程派发开销。
这些优化不改变 prefill 的动态专家分配。

比较 TP 与按层分卡时，应保持共享专家位置、CPU 线程、NUMA、KV 类型和 prefill 分块相同。
NUMA 按多个 LLC 分散绑核时，TP 执行线程保留启动时的 CPU 亲和性，避免把专家正在使用的核
误判为空闲核。不需要额外设置绑核环境变量。

## 快速 prefill（`--fast_prefill`）

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash --device cuda --moe_device numa \
  --fast_prefill
```

默认关闭，也支持 `--fast-prefill` 写法，启动时以参数为准。
开启后使用 decoder SWA bounded replay：每个文本 prefill chunk 完整计算到最后一个
KV source 层，后续层只计算各请求末尾 `sliding_window` 个 token。V4.1-Flash 对应
第 0–20 层处理完整 chunk，第 21–39 层处理末尾至多 128 token。
长度不超过窗口的片段、包含图像嵌入或图像掩码的前向保持完整计算。

压缩 KV 和 indexer key 仍完整保留，RoPE、缓存槽位和请求长度使用原始绝对位置。
支持分块 prefill、混合批次、前缀缓存恢复及 DSpark；DSpark 的 main hidden 同步取尾部，
verify 始终计算全部候选位置。单 token decode 和 CUDA Graph 的执行范围不变。

这是**近似计算**：后段层的局部 attention 在保留窗口的起点截断，可能改变 logits，
差异不局限于浮点舍入。开启后应按实际任务验证质量，并固定输入与 chunk 配置对比速度。


## decode 与 DSpark 校验的 CUDA Graph

`FASTLLM_CUDA_GRAPH=1` 在启动预热时捕获普通 decode 图。按照 `max_batch` 预热 1、2、4 等二次幂，以及配置的最大 batch；未预热的 batch 或失效图使用 eager，不在普通请求内重新捕获。DSpark 目标验证仍保留独立的分段图。

GPU 执行模式下，单请求文本 decode 满足以下条件时，会进一步组成每卡一张整图：

- `--ngram_device cuda`，全部 Engram 表均已成功放到 GPU；
- 普通 BF16 KV，未开启 DSpark，兼容的 TP head 分片；
- 当前支持的 512 维 Full Attention、128 维 Indexer、四路 HC 布局。

整图包含各层投影、归一化、动态 KV 更新、Indexer、Attention、路由专家、共享专家、Engram GPU 计算及 LM head。每个 decode step 每张卡只调用一次 `cudaGraphLaunch`，上下文长度和 KV 地址通过固定元数据缓冲更新，不随上下文增长重新捕获。请求准备、embedding 输入准备、Engram 地址计算和采样仍在图外；不能把它理解为整个 HTTP 请求只有一个 GPU 操作。

启动日志出现以下信息，才表示整图已就绪：

```text
DeepSeek-V4.1 full decode graph ready: batch=1, ranks=8, one graph per rank/step
```

例如八卡单并发服务：

```bash
FASTLLM_CUDA_GRAPH=1 ftllm server /path/to/DeepSeek-V4.1-Flash \
  --tp 8 --ngram_device cuda --dtype auto --atype float32 --max_batch 1
```

这里 `atype=float32` 是模型前端接受的设置；主干内部仍按原模型使用 BF16 激活及相应量化边界。`dtype=auto` 保留带缩放的 FP8/FP4 源权重。

其他已预热的 batch、CPU Engram 或混合专家模式使用可捕获的分段图，其动态操作保留在图外。普通多 token prefill、按层分卡及带同步 profiling/dump 的路径使用 eager。工作区由模型持有，不能并发取得锁时走 eager；设备、精度或权重布局变化后重新校验，捕获失败时各 TP rank 一致回退。按层部署应保持 Graph 关闭，避免 CUDA embedding 放置影响跨卡访问。

排查参数包括 `FASTLLM_DSV41_CUDA_GRAPH_WARMUP`（默认 2）和 `FASTLLM_DSV41_CUDA_GRAPH_DEBUG`。故障注入和回放掩码仅供回归测试使用，不应作为部署调优配置。

## 按层切分

```bash
FASTLLM_CUDA_GRAPH=0 ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device "{'cuda:0':1,'cuda:1':1}" --moe_device numa
```

用普通的 device map 就能把层平均分到两张卡上（`SelectDeviceFromMap` 按权重划分层区间），
V4.1 的 `ForwardSegments` 每层都会 `ApplyDeviceMap`，隐藏状态在 stage 边界由执行器自动搬运。
每卡只保存自己那一半的层权重，**每卡显存约减半**；单请求延迟基本不变（两张卡是串行执行的，
不是真正的流水线——调度器没有 micro-batch，所以并发请求也不会形成 stage 级重叠）。

### 切分点必须落在 kv source 层的边界上

V4.1 的压缩 KV 是跨层共享的：`kv_source_layer_ids` 的层算出的 cache 会被之后若干层直接读。
如果 source 层和消费层落在不同卡上，执行器会在每次访问时把整段压缩 KV 搬到另一张卡，
下一步 source 层追加时再搬回来——每个 token 都要来回搬整段缓存。

真实模型（40 层）的 source 布局正好让**均分**成为合法切分点：

| source 层 | 服务的层 |
| --- | --- |
| 2 | 2–7 |
| 8 | 8–13 |
| 14 | 14–19 |
| 20 | 20–39（同时是 `candidate_source_layer_id`，候选块被 24/28/32/36 复用） |

`{'cuda:0':1,'cuda:1':1}` 得到的切分点正好是 20，stage 0 = 0–19、stage 1 = 20–39，
每个 (source, 消费者) 对都在同一张卡上，Engram 层 [1, 14] 也都在 stage 0。
**不要为了平衡显存把切分点挪到 21**（`head.weight` 1.3 GB 在 stage 1，会让 stage 1 略重）：
挪一层就会把 source 20 和它的 19 个消费层拆到两张卡上。

### 与张量并行的取舍

| | 张量并行 | 按层切分 |
| --- | --- | --- |
| 权重分布 | 注意力与 head 分片，共享专家复制；路由专家使用配置的 EP/混合后端 | 整层归属一张卡 |
| 计算方式 | 同一层由多卡同时计算，需要集合通信 | 单请求依次经过各卡上的层 |
| Decode 调度 | 支持常驻 worker 和符合条件的整图 | stage 边界传递隐藏状态 |
| 约束 | head 均分，每个分片包含完整的 o_group | 切分点必须落在 kv source 边界 |

两种部署方式不能直接以旧的逐算子调度耗时比较。实际速度取决于卡间互联、
GPU/CPU 专家布局、Engram 放置和 CUDA Graph 是否生效，应在相同负载下测量。

## prefill 的多卡专家流（`FT_MOE_ASSIST_DEVICES`）

`--moe_device numa` 下，一个 prefill 分块要把**全部**路由专家的权重过一遍，专家权重是从主机内存
流式送到 GPU 的，不需要张量并行分片。所以瓶颈是「主机内存 -> GPU」这条链路的带宽，而不是算力：
多一张卡就多一条链路。但 `GetNumasMoeCudaAssistDevices()` 原本只返回承载稠密层的那张卡，
模型放得下单卡的机器上第二张卡在 prefill 期间完全闲置。

```bash
FT_MOE_ASSIST_DEVICES=0,1 ftllm server ... --device cuda --moe_device numa
```

把额外的 CUDA 设备加进专家流。默认为空，行为不变。每张卡拿到一份输入激活的副本、一组不相交的
专家，各自算出一份 partial，最后在 root 卡（产出这一层激活的那张）上相加。

### 输入搬运与归约重叠

输入副本由对应 GPU worker 在自己的 stream 搬运，优先使用 peer copy；不可用时等待 D2H 事件后经 pinned host staging 上传。
各 GPU 的 partial 同样在 worker 内搬到 root 侧独立缓冲，与 CPU 和 root GPU 的专家计算重叠。root stream 等待事件后按顺序相加，完成同步后才释放 partial。

输入搬运与 partial 归约始终使用重叠路径，无需设置环境变量。
没有 peer 通路时自动使用 pinned host 中转。下面两项调度策略仍默认关闭。

| 变量 | 作用 |
| --- | --- |
| `FT_MOE_ASSIST_BALANCE=1` | 按各卡实测的「每专家毫秒」分配 GPU 专家，而不是按 route 数均分 |
| `FT_EXPERT_LIMIT_AUTO=1` | 用真实层反馈出的 CPU / GPU 速度算 expertLimit，取代单专家合成 benchmark |

事件、归约缓冲、pinned 中转缓冲都按设备缓存在每层的 MoE manager 上，跨层复用。
设置 `FASTLLM_PROFILE_NUMAS_MOE=1 FASTLLM_PROFILE_DETAIL=1` 可统一查看每层的
CPU/GPU 专家划分、各卡 route 数，以及 stage / limit / prep / cpu / join / reduce 耗时。

### expertLimit 的选择

`expertLimit` 是「一个专家至少要有多少 route 才值得送上 GPU」的阈值：低于它的专家留在 CPU 上算，
和 GPU 并行，理想情况下两边同时结束。默认由 `MoeExpertSpeedEstimator` 在第一次 prefill 时跑一个
合成 benchmark 推出来——**每轮只跑一个专家、每轮 sync**。这样量到的 GPU 每专家耗时里含着
无法摊掉的固定开销（workspace 准备、stream 创建、启动延迟），也拿不到真实层里跨专家的
「copy(i+1) 与 compute(i) 重叠」，会系统性高估 GPU，把过多 route 留在 CPU 上。

`FT_EXPERT_LIMIT_AUTO=1` 改成用真实层反馈的两个系数直接算 makespan 最优点：

- 每张卡的「每专家毫秒」= worker 线程墙钟 ÷ 分到的专家数（EMA）。因为是两张卡并发跑时量的，
  跨专家流水重叠、两卡争抢主存带宽的影响都已经算进去。
- CPU 的「每 route 毫秒」= CPU 专家段墙钟 ÷ 落在 CPU 上的 route 数（EMA）。

然后枚举阈值 t，用与实际分配一致的贪心把 GPU 专家摊到各卡上，取 `max(cpuMs, gpuMs)` 最小的 t。
样本不足（前几层）时先将 route 最少的两个专家分给 CPU，其余交给 GPU，以收集两侧耗时。
`FT_EXPERT_LIMIT=<n>` 的显式覆盖优先级最高，
两种自动估计都不会执行。

动态 prefill 分配可能改变专家归约和舍入路径。数值比较应使用相同输入及生成历史，结合 logits
误差与任务结果判断；不要仅为逐字节复现而关闭动态分配。生成历史分歧之后的 logits 不能直接比较。

## KV 缓存存储精度（`--kv_cache_dtype`）

长期 KV 缓存只有两部分：**压缩 KV**（`kv_source_layer_ids` 各层，每 `compress_ratio` 个 token 一行，
`head_dim` = 512）与 **indexer key**（同样的层，`index_head_dim` = 128）。滑窗缓存长度固定为
`sliding_window`，不随上下文增长，不计入每 token 开销。

写进缓存之前，这两者都已经过与官方一致的伪量化（`DeepSeekV41RotaryQuant` 的 `quantMode`）：

| 张量 | 伪量化网格 | 分组 | scale 编码 |
| --- | --- | --- | --- |
| 压缩 KV | FP4 E2M1 | 每 16 个通道 | E4M3 |
| indexer key | FP4 E2M1 | 每 32 个通道 | UE8M0（2 的幂） |
| 滑窗 KV | FP8 E4M3 | 每 32 个通道 | UE8M0 |

也就是说值本来就落在 FP4 / FP8 网格上，**按 FP4 存储是无损的**。三档存储的行布局与开销：

| `--kv_cache_dtype` | 压缩 KV 行 | indexer key 行 | 每 token | 相对 BF16 | 精度 |
| --- | --- | --- | --- | --- | --- |
| 默认（BF16） | 1024 B | 256 B | **3200 B** | 1.00x | 基准 |
| `fp8_e4m3` | 528 B（512 E4M3 + 16 UE8M0） | 132 B | **1650 B** | 0.52x | 压缩 KV 有不超过 2^-4 的相对舍入 |
| `fp4_e2m1` | 288 B（256 打包 E2M1 + 32 E4M3） | 68 B（64 打包 E2M1 + 4 UE8M0） | **890 B** | 0.28x | **逐 bit 无损** |

890 B/token 与官方实现一致。每 token 2.5 行压缩 KV（层 2/8/14 的 `compress_ratio` 是 2，层 20 是 1）：
`2.5 x 288 + 2.5 x 68 = 890`。

- **FP4 无损的原因**：`DeepSeekV41QuantizeKV` 的块 scale 推导（`DeepSeekV41BlockScale`）与伪量化
  (`DeepSeekV41FakeQuantRow`) 是同一份代码，分组大小与 scale 编码也完全对齐，所以对已经伪量化过的行
  是幂等的；而 `q * scale`（q 至多 3 个有效位、scale 是 E4M3 或 2 的幂）在 BF16 上也是精确的。
  实测：开启伪量化时 FP4 存储与 BF16 存储的 logits **逐 bit 相同**（`test/basic/deepseek_v41_kv_cache_dtype.py`）。
- **FP8 为什么有损**：压缩 KV 的网格是「FP4 值 x E4M3 scale」，而 FP8 存储用的是 2 的幂块 scale，
  两者的网格对不上，会引入一次真实的舍入。它的用途是在不支持 FP4 打包读取的场合省一半显存。
- 滑窗 KV 在 `fp8_e4m3` 与 `fp4_e2m1` 两档下**都按 FP8 存储**（它本来就在 FP8 网格上，改 FP4 会真的损失精度），
  三种行布局可以在同一次前向里混用，读路径按行宽自动识别。
- 缓存行统一放在 `INT8` 的 `Data` 里，形状 `[b, rows, rowBytes]`；FP4 打包为一个字节两个 code，
  低 4 位是偶数下标。CUDA 侧在把候选装进共享内存时就地解成 BF16 片段，再进 `mma.sync`，
  `FASTLLM_DSV41_LEGACY_ATTN` / `FASTLLM_DSV41_LEGACY_INDEXER` 的标量回退路径同样支持。

### 怎么选

- **长上下文 / 高并发**：用 `fp4_e2m1`。同样显存能装的上下文约为 BF16 的 3.6 倍，且数值与 BF16 完全一致，
  没有任何精度代价；长上下文下稀疏注意力每步读取的字节数同比下降，decode 也有正收益。
- **默认（BF16）**：短上下文、显存不紧张时省掉打包/解包的一点开销。
- `fp8_e4m3`：介于两者之间，但既比 FP4 大又比 FP4 差，现在没有明显适用场景，保留是为了兼容既有部署。

## 多请求与前缀缓存

### 批量 decode

`DeepSeekV41Model::ForwardSegments` 把若干"片段"（请求状态 + 起始位置 + token 数）拼成一个 token 流：
embedding、Hyper-Connections、各 Linear、路由、MoE 与 Engram 查表按整批执行，RoPE、压缩 KV 追加、
indexer top-k、稀疏注意力与滑窗写入按片段分别执行。通用调度器的多请求 `ForwardBatch` 直接走这条路径，
单请求 `Forward` 是它的单片段特例；含长 prefill 片段的混合批次退回为逐个前向。

每个请求的缓存（`DeepSeekV41RequestState`）与 `ResponseContext` 绑定，请求结束或 abort 时释放。
在迷你模型上，8 并发 decode 与逐个请求相比，token 序列的差异仅来自 GEMM 按 batch 选核带来的
BF16 舍入（同一请求换不同的批次伙伴 / 批内位置，logits 逐 bit 一致）；单卡 3090Ti 上迷你模型的
decode 吞吐从 209 tok/s（1 并发）提高到 573 tok/s（8 并发）。

### 前缀缓存

OpenAI 接口在模型实例内保存工具轮次的原始输出（含思考和 DSML，最多 128 轮、8 MiB，支持流式）。
续轮的调用 ID、顺序、函数名、参数值和正文匹配时恢复原文，避免 DSML 拼写、JSON 格式等变化破坏 token 前缀。
匹配忽略 JSON 空格、对象键顺序和纯空白正文；客户端可省略思考内容，但修改思考、切换思考模式或
模板要求丢弃思考时不复用。不接受客户端原始模板覆盖；未命中仍按普通模板编码和 prefill。

启动时加 `--cache_history true`。请求结束时把每层 `windowKV`、`compressedKV`、`indexK`、`rawTail`
与 Engram 历史快照到 CPU 内存（LRU，默认保留 8 条），新请求按最长公共前缀查找并恢复，只对新增 token
做 prefill。多轮对话中只要客户端原样回传上一轮的回复，通常就是精确命中。

DSML 块前的两个模板分隔换行不作为 assistant 正文返回，避免续轮重复添加。
流式解析暂存这两个换行；普通正文末尾的换行在流结束时照常输出。

恢复长度受模型结构约束：滑窗缓存是只保留最后 `window_size` 个位置的环形缓冲，因此只能恢复到记录
长度 T 或 T-1（记录不超过 `window_size` 时可以任意截断）；ratio-2 压缩层凑不满一组的原始尾块只在
精确命中时可用，其它情况要求恢复长度为偶数。不满足约束的候选会被跳过（退回完整 prefill）。

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSV41_DISABLE_PREFIX_CACHE` | 关闭前缀缓存 |
| `FASTLLM_DSV41_PREFIX_CACHE_DEBUG` | 打印命中 / 记录日志 |
| `FASTLLM_DSV41_PREFIX_CACHE_MIN_TOKENS` | 最短命中长度（默认 16） |
| `FASTLLM_DSV41_PREFIX_CACHE_MAX_RECORDS` | 最多保留的记录数（默认 8，也可用 `FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS`） |

迷你模型上精确命中与 T-1 截断命中后的贪心输出与不中断的原请求逐 bit 一致；800 token 提示词的
首 token 延迟从 39–54 ms 降到 6 ms。

## DSpark 投机解码

`config.json` 的 `text_config` 里 `dspark_block_size > 0` 且 checkpoint 带 `mtp.*` 权重时可以开启。
启动加 `--speculative_algorithm dspark --dspark 5`。`--dspark N` 指每轮最多校验 N 个候选，
运行时草稿长度取 `max(dspark_block_size, N)`：小于训练块时只校验前缀，
大于训练块时扩展整个草稿前向，包含双向注意力和 Markov 链。
本 checkpoint 训练块为 5，已验证 `--dspark 7`；全部接受时一轮最多输出 8 个 token。
更长草稿的速度取决于接受率与校验成本，训练配置本身不需要修改。

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device cuda --moe_device numa --dtype float16 \
  --speculative_algorithm dspark --dspark 5
```

`ftllm` 的自动配置（launcher）在 `enable_speculative_decoding` 时会识别 V4.1 的内置 DSpark，
按 checkpoint 的训练 block size 填 `--draft_tokens`。不加 `--dspark` / `--draft_tokens` 时
不加载 `mtp.*`。原始量化 checkpoint 的草稿原始张量（含 scale）约 7.39 GiB；
实际占用还受加载 dtype、权重转换和运行缓冲影响。

### 结构

`mtp.0/1/2` 是三个与主干同构的草稿层：`compress_ratios` 在这三层是 0（纯滑窗注意力），
MoE 是 `dspark_n_routed_experts` = 128 专家 top-3，embedding 与 lm_head 与主干共享。
另外 `mtp.0` 有 `main_proj` / `main_norm`，`mtp.2` 有 `norm`、`markov_head`（`embed` + `head`，
秩 256）与 `confidence_head`（输入 `dim + 256`）。

草稿层的滑窗 KV 不来自草稿 token 自身，而来自目标模型：`dspark_target_layer_ids`（[37, 38, 39]）
各层 attention 的输入（对 4 份 Hyper-Connections 取均值）拼成 `3 * dim` 后经 `main_proj` / `main_norm`
得到 `main_x`，每个草稿层再用自己的 `wkv` / `kv_norm` 把它写进一行滑窗 KV。也就是说每个已提交
位置在草稿侧只有一行 KV，草稿模型不需要重跑主干。

一次 proposal：把 `block_size` 个位置的输入 token 置为 `dspark_noise_token_id`（第 0 个位置放锚点
token，即目标模型刚产出、还没进 KV 缓存的那个 token），一次前向产出 `block_size` 组 logits；
再用 markov head 逐位置做 bigram 修正，得到 `block_size` 个候选 token；
`confidence_head` 对每个位置给出一个 sigmoid 后的置信度。

贪心请求取 argmax；采样请求在 GPU 保存完整草稿分布 q，通过共用的 MTP 采样与拒绝采样内核校验，
不再限制草稿支持集为 64 项。Markov 修正使用上一步实际抽到的 token；草稿与目标均使用请求的
`temperature`、`top_k`、`top_p`，无工具掩码时与普通 CUDA 采样一致。
工具名、参数名约束按位置更新：在允许集合内取 top-k 并重新归一化后取 top-p，草稿、校验和 bonus
使用各自前缀；拒绝和待发队列不会错误推进约束状态。强制候选测试使用 one-hot q，无需额外开关。

### 校验与回滚

候选与锚点拼成一个 `1 + N` 长度的片段一次喂给目标模型（`ForwardSegments` 天然支持一次多 token），
贪心请求逐位置比较 argmax，第一个不匹配处截断。采样请求复用 `qwen3_5.cpp` 的链式拒绝采样：
实际草稿概率为 q，目标条件概率为 p，以 `min(1, p(token)/q(token))` 接受候选；首次拒绝时从
归一化的 `max(p-q, 0)` 抽取替代 token，全部接受时从额外目标行抽取 bonus token。
接受 n 个候选时这一轮提交 n + 1 个 token、产出 n + 1 个输出
token：第一个立刻返回，其余进入请求的待发队列，调度器之后每轮直接出队，不再前向。

校验前向按完整 block 更新缓存，接受长度确定后要把多算的部分退回：

- 滑窗环形缓冲：写入**延后**到接受长度确定之后。片段内的位置在稀疏注意力里一律从 `chunkKV` 读取，
  推迟写入不改变本次前向的任何结果，因此也不需要为回滚保存被覆盖的旧行；
- 压缩 KV 与 indexer key：按接受后的长度重新截断行数，并用保存下来的 compressor 原始输入流
  （旧 `rawTail` + 本次新行）重建 `rawTail`；
- Engram 历史与各层 `totalLen`：截断到接受后的长度。

拒绝采样在目标条件概率相同的前提下保持目标输出分布，不要求草稿分布等于目标分布。
这不代表开启与关闭 DSpark 的原始 logits 逐 bit 一样，也不保证相同随机种子给出相同序列。
一次多 token 与逐 token 前向会选择不同的 GEMM / MoE 路径，浮点舍入可能改变 logits，
也可能翻转接近并列的 argmax；分布测试与浮点前向误差需要分别验证。

### 限制

- 支持简单贪心及 CUDA 上的 temperature / top-k / top-p 采样，也支持默认工具名、参数名约束。
  重复惩罚、工具内容采样、没有前缀状态的独立 token 白名单、`output_logits`、正的
  `output_token_least`、非有限采样参数会退回普通解码；CPU 采样也走普通路径。
  与普通解码一样，`do_sample=true`、正温度且 `top_k<=1` 时将 top-k 规范化为 5；
- 只在**单请求**前向里产生候选。批量 decode 的那一轮不投机，但仍然采集 main hidden，
  让草稿滑窗跟上目标缓存；已经校验通过的 token 在批量路径里也能正常出队；
- 图文请求不投机；
- 前缀缓存命中恢复出来的前缀没有草稿侧的滑窗（`main_x` 无法从目标缓存反推），
  草稿注意力只看得到恢复之后新增的位置，接受率会在最初的 `window_size` 个 token 内偏低；
- 校验提交在首个 EOS / stop token 处截断，避免结束时缓存超出真实输出。
  请求若因长度上限在待发队列还没取完时结束，这一轮多算的 token 仍可能让缓存长度超过
  `allTokens`，该请求的前缀缓存记录会被跳过；
- 支持与主干张量并行组合（`--tp 2`），草稿层仍在第一张卡运行；目标校验支持分段 CUDA Graph，草稿层仍逐算子执行。

### 调试环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSPARK_TOKENS` | 每轮校验的候选数（由 `--dspark` / `--draft_tokens` 设置，不要直接设） |
| `FASTLLM_DSPARK_CONFIDENCE_THRESHOLD` | 置信度低于该值的候选之后不再校验；0 表示总是用满 block |
| `FASTLLM_DSPARK_STATS` | 默认 `1`，打印总体及逐位置接受率；`0` 关闭，`2` 打印详细分段耗时与逐轮记录 |
| `FASTLLM_DSPARK_STATS_EVERY` | 每累计 N 轮校验打印一次（默认 32，0 表示只在退出时打印） |
| `FASTLLM_DSPARK_STATS_WARMUP` | 统计前跳过的轮数（默认 3）。第一次 decode 含 CUDA context / 显存池 / 权重量化缓存的一次性开销，会把均值拉偏 |
| `FASTLLM_DSPARK_DISABLE_FUSED_MARKOV` | 关掉 markov head 的融合 kernel，退回通用算子（对拍 / 排查用；两条路径输出逐 bit 一致） |
| `FASTLLM_DSPARK_PROBE_EVERY` | 每 N 轮故意不带候选走一次普通单 token 前向，给"校验 N 个候选"提供同等缓存状态下的对照基线。默认 0（关闭），只在诊断时打开 |
| `FASTLLM_DSPARK_STATS_FILE` | 每轮校验追加一行 `轮次 候选数 接受数`（测试用） |
| `FASTLLM_DSPARK_FORCE_DRAFTS` | 用文件里的候选替换模型的候选，构造指定的接受长度（测试用） |
| `FASTLLM_DSV41_DEBUG_STATE` | 每次前向后导出各层缓存长度（对比回滚是否正确） |

## 图像输入

`config.json` 顶层的 `vision_n_layers > 0` 时加载视觉编码器：32 层 ViT（1024 维、16 头、patch 14、2D RoPE、SwiGLU MLP）
+ aligner（3x3 下采样后 `w1 -> GELU -> w2` 投影到 5120 维）+ 三个学习到的分隔符嵌入。权重保持 float16（不参与低比特量化），
以 float32 激活在执行器所在设备上计算，长序列注意力按 1024 个 query 分块。

处理流程（与官方 `image_processor.py` / `model.py` 一致）：

1. Python 侧（`deepseek_v41_multimodal.py`）：`encode_messages(..., return_multi_modal_data=True)` 渲染带
   `<｜deepseek_image｜>` 占位符的 prompt；每张图按动态分辨率规则缩放 / 灰色补边（`vision_min_pixels` 295936、
   `vision_max_n_token` 1024），切成 `n_vit_h x n_vit_w` 个 patch，占位符展开为
   `[image_start] + ([IMAGE] * n_llm_w + [image_newline]) * n_llm_h + [image_end]`（每个位置都是 `image_token_id`），
   patch 与每张图的 `(起始位置, n_vit_h, n_vit_w)` 作为 payload 传给 C++；
2. C++ 侧：请求创建时把多模态输入记到请求状态（`OnResponseContextCreated` / `ForwardMultimodal`），
   第一个 prefill 块对每张图做 ViT + aligner，把结果与分隔符嵌入组成 span 嵌入缓存起来；之后每个 prefill 块
   （无论由调度器还是模型自己按 `chunked_prefill_size` 切分，span 可以跨块）把与 span 重叠的位置换成图像嵌入，
   图像 span 内的 token 路由使用 `gate.bias_vl`、Engram 对其不做查表也不参与 n-gram；decode 步骤与纯文本相同。

用法：与其它多模态模型相同，OpenAI 接口的 `image_url` 支持 http(s)、`data:image/...;base64` 与 `file://`：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "v41", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}]}]}'
```

限制：

- 只支持图像，不支持视频；一张图最多 1024 个 token（约 9216 个 ViT patch），多张图按出现顺序对应；
- 带图请求不与其它请求合并 prefill；图像 span 必须落在 prompt 内（Python 侧展开时保证）；
- 前缀缓存 / 多轮复用尚未实现，每个请求都会重新编码图像；
- 21 环境下服务模式若加载不到 HF tokenizer，会退回 fastllm 原生 tokenizer 编码 prompt（两者对占位符的 id 相同）。


## 数值验证

参考精度模式及验收边界见[精度说明](deepseek_v41_precision.md)。同一数学模式的回归、官方参考对齐和生成质量属于不同检查，不能相互替代。

### CPU-only 固定 fixture 回归（无需 torch）

`test/basic/deepseek_v41_reference.py` 需要 torch、transformers 与官方 `inference/` 代码，进不了 CI。
`test/basic/test_deepseek_v41_cpu_fixture.py` 把「一个微型 V4.1 checkpoint + 官方实现算出的参考 logits」
固化在 `test/basic/deepseek_v41_fixture.npz`（约 2.2 MB）里，只依赖 numpy 与 fastllm 的 Python 包：

```bash
PYTHONPATH=build/tools python test/basic/test_deepseek_v41_cpu_fixture.py
```

追加 `--ngram-device disk` 可用同一份 fixture 校验磁盘 Engram 加载与按行读取。

退出码 0 表示通过，1 表示失败；fixture 缺失时打印重新生成的命令并以 0 退出（跳过）。
fixture 里的模型是 5 层、dim 128、词表 128、2 专家，覆盖 V4.1 的全部结构特性：
`compress_ratios = (0, 2, 2, 1, 1)`（三种压缩层）、`kv_source_layer_ids = (1, 3)`（层 2 / 4 跨层复用压缩 KV）、
`index_source_layer_ids = (1, 3, 4)` 且 `candidate_source_layer_id = 3`（两级 top-k）、
`engram_layer_ids = (1, 4)`、`hc_mult = 2`、`window_size = 8`（48 token 的 prefill 让滑窗环形缓冲绕多圈）。

比较方式：逐步（prefill + 4 步 decode）比较 logits 的余弦相似度（>= 0.9975）与
`max|diff|` 占 logits 值域的比例（<= 4%），并要求贪心 token 与参考一致；
参考侧是官方实现的 bf16 前向，噪声底约为值域的 2–3%（fastllm 自己的 CPU 与 CUDA 路径之间也有约 1.5%），
所以容差按值域折算而不是给绝对值。fixture 生成时会在多个提示词种子里挑一个每步 top1/top2 间距都
远大于该噪声的，避免 argmax 因舍入翻转；万一仍然翻转，测试会按"参考侧两者间距落在容差内"判为并列而不算失败。

重新生成 fixture（需要 torch + 官方 inference 代码 + 一张 GPU）：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_fixture_gen.py \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference \
  --work-dir /tmp/v41-micro --out test/basic/deepseek_v41_fixture.npz
```

生成脚本自带一个 128 词的迷你 tokenizer，不依赖真实 checkpoint 的 tokenizer.json。

### 算子级 CPU / CUDA 一致性

`test/ops/deepseekV41OpsRegression.cpp`（`cmake -DUNIT_TEST=ON`）用同一份输入分别在 CPU 与 CUDA 上跑
11 个 V4.1 专用算子并比较输出，覆盖随机输入与边界情况：

```bash
./build/deepseekV41OpsRegression            # 全部算子
./build/deepseekV41OpsRegression --list     # 列出用例
./build/deepseekV41OpsRegression IndexerTopK CandidateBlocks   # 只跑指定算子
ctest -R deepseekV41Ops                     # 无 CUDA 设备时以 77 跳过
```

## Decode 算子与兼容性

单 token indexer 使用 head 作为 MMA 矩阵行。注意力将输出维度分成四份；候选较多且临时空间
充足时，先并行计算 QK，再保持原有的 64 槽在线 softmax、BF16 概率舍入和 PV 顺序。
BF16 5120 维、1–8 行 RMSNorm 与四路 HC Finish 使用保持归约顺序的专用内核。

MMA 路径检查设备及实际加载内核的架构、线程数和共享内存限制；不满足时使用原 FP32 标量路径。
非 decode 形状使用通用 MMA，注意力临时内存不足时退回无需该空间的融合 MMA；执行错误正常上报。
RMSNorm 其他类型、维度和批量保留原路径，不满足向量读取对齐时使用标量读取。
CUDA 构建仍需包含目标显卡支持的代码镜像，并使用支持该架构的工具链。

其他加速路径也保留回退：

- BF16×FP8 block-32 Marlin 要求 SM80+、匹配的尺寸和可无损表示的 E8M0 scale；否则保留原始行优先权重，使用通用 GEMV/cuBLAS 路径。
- FP8 GEMV 的单行循环展开仅针对 SM90，其余架构使用原循环。
- MoE 的 V4.1 量化、路由权重和 FP32 求和边界独立于加速后端；不支持 Marlin 时使用普通 CUDA 路径。不能捕获的路径保持 eager。
- GPU Engram 要求各 rank 可互访且显存充足，否则保留 CPU 查表；不改变原始表的数值格式。
- 整图不支持的形状、KV 类型、执行模式或设备退回分段图/eager。启动后失效的普通 decode 图不在请求内重新捕获。

压缩 Attention 保留原有 PV 累加顺序；未保留改变 FP32 结合顺序的候选分块优化。整图回归对普通和压力输入均要求与 eager 结果逐位一致。

```bash
./build/deepseekV41OpsRegression
ctest --test-dir build -R 'deepseekV41(Precision|Tp|DecodeOptimization|FullDecode)' --output-on-failure
python test/basic/test_deepseek_v41_tp_graph.py --binary ./build/deepseekV41TpGraphRegression
```

Graph fixture 需要 PyTorch、safetensors、numpy 和两张 CUDA GPU；覆盖不同长度的连续请求、
启动预热后的 eager / Graph 切换和共享专家重叠，以及 DSpark 的 1～8 token 图重放、目标层特征与部分接受后的 KV 回滚。
`--hc-mult 2` 可补测两路 HC，默认四路。
算子回归检查通用路径与 decode 对齐、独立 CPU RMSNorm 参考、量化 KV、远端 logits 同步、
缓冲复用、并发 Graph 及捕获期间临时空间不足时的回退。

## 调试环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSV41_LEGACY_ATTN` | 稀疏注意力退回 FP32 标量 kernel（对比 / 排查用） |
| `FASTLLM_DSV41_LEGACY_INDEXER` | indexer 打分退回 FP32 标量 kernel |
| `FASTLLM_DSV41_LEGACY_HCMIX` | HC 混合系数退回旧 kernel |
| `FASTLLM_DSV41_LEGACY_FP4_UNPACK` | 量化 KV 缓存行退回逐元素解包（不再按 4 字节一组批量展开） |
| `FASTLLM_DSV41_LEGACY_TOPK` | indexer top-k 退回旧的「全 visible 扫描 + 8 轮 4-bit radix」kernel |
| `FASTLLM_DSV41_LEGACY_ROTARY` | 旋转 / 伪量化退回旧的「一行一个 block + 共享内存」kernel |
| `FASTLLM_DSV41_DISABLE_HCPRENORM` | 不融合 HcApplyPre 与 RMSNorm，退回两个算子分开做 |
| `FASTLLM_DSV41_INDEX_SCORE_MB` | indexer 分数矩阵的显存预算（MB，默认 128），决定 token 维分块大小 |
| `FASTLLM_DSV41_INDEX_CHUNK` | 直接指定 indexer 的 token 分块大小（覆盖上面的预算推算） |
| `FASTLLM_DSV41_ENGRAM_PROFILE` | Engram 查表 / 转换 / 投影分段计时（见 "Engram 元数据"） |
| `FASTLLM_DSV41_ENGRAM_POOL` | Engram 查表改用常驻线程池 |
| `FASTLLM_DSV41_ENGRAM_PREFETCH` | 跨层预取下一个 Engram 层的行号与表行 |
| `FASTLLM_DSV41_ENGRAM_MADVISE` | Engram 表的内存访问提示（random / hugepage / both） |
| `FASTLLM_DSV41_ENGRAM_WKV_FP8` | `engram.wkv` 保留 checkpoint 里的 FP8 精度，不解量化 |
| `FASTLLM_DSV41_DISABLE_FAKE_QUANT` | 关闭 FP8 / FP4 伪量化（仅用于对齐调试） |
| `FASTLLM_DSV41_DISABLE_CUDA_ROUTE` | 路由退回 CPU 参考实现 |
| `FASTLLM_DSV41_DUMP_DIR` | 把每层中间张量写到该目录（对齐调试） |
| `FASTLLM_DSV41_DISABLE_TP_ATTENTION` | 张量并行时不切分注意力 head（排查用，注意力改为每卡各算一份） |
| `FASTLLM_CUDA_GRAPH` 等 | decode 与 DSpark 校验的 CUDA Graph，见对应章节 |
| `FASTLLM_TRACE_OPS` | 逐算子打印"算子名 / 落在哪个设备 / 权重名"（排查 TP 落点用） |
| `FASTLLM_DSV41_DISABLE_PREFIX_CACHE` 等 | 前缀缓存相关，见"多请求与前缀缓存" |
| `FASTLLM_DSPARK_*` | DSpark 投机解码相关，见"DSpark 投机解码" |
| `FT_MOE_ASSIST_DEVICES` / `FT_MOE_ASSIST_BALANCE` / `FT_EXPERT_LIMIT_AUTO` | NUMA MoE 的多卡专家流，见"prefill 的多卡专家流" |
| `FASTLLM_PROFILE_NUMAS_MOE=1 FASTLLM_PROFILE_DETAIL=1` | NUMA MoE 分阶段耗时、CPU/GPU 专家划分及各卡 route 数 |

## DSpark 与专家缓存验证

CUDA + NUMA 混合专家缓存会自动处理 2–8 行 verify，包含 TP + CUDA Graph 路径，配置和命中率口径见
[CUDA 专家缓存](cuda-expert-cache.md#deepseek-v41)。HC mix 支持 1–8 行小批量；
WoA 对小于 16 行的批次复用权重，保持各输出的累加顺序，其他形状走已有路径。
草稿的稠密层、路由专家、共享专家与 HC post 复用主干的量化和 BF16 舍入语义。

`deepseekV41SamplingRegression` 检查实际 CUDA 拒绝采样器的输出分布、部分拒绝、
bonus token 和缓存回滚；`deepseekV41OpsRegression` 检查小批量算子与原归约路径；
`cuda_dsv41_moe_cache_test --dual` 检查独立数值参考和跨卡缓存切换。
`test/basic/test_deepseek_v41_tp_graph.py --binary build-fastllm/deepseekV41TpGraphRegression --expert-cache`
使用 NVFP4 专家检查 TP verify 缓存、1–6 行 Graph 重放、主模型特征与 KV 回滚。
编程模板在 top-p 截断后可能只剩一个候选，因此高接受率本身不能证明使用了贪心验证；
评估时应记录采样参数、拒绝轮数和代码功能结果，分别检查采样校正与前向浮点误差。

### Decode 投影与准备操作

以下优化自动按输入条件选择，无需新增环境变量；不符合条件时继续使用原路径：

- **mHC finish 与 Norm 合并下发**：BF16、HC=4、少于 16 个 token、偶数隐藏维度 4096–8192。
  当前子层的 Sinkhorn 与使用上一子层 pre 系数的 Norm 在独立 CTA 中并行。
  保持原 FP32 归约树、折叠后的 BF16 舍入及 Sinkhorn 顺序。
- **Full Decode 的旋转准备**：独立的 Q/KV 描述符、输出旋转和窗口写入分别共用一次 kernel 下发。
  位置、量化和窗口更新规则保持不变。
- **Engram WKV 单 token 计算分片**：BF16 输入和原始 FP8 权重、对齐的输出通道、可用 NCCL TP 组。
  每个 rank 只计算自己的输出通道，随后 AllGather BF16 结果。每个点积仍计算完整 K，
  没有跨卡部分和归约。权重继续保留复制布局；这项优化减少计算和读取，不减少权重显存。
  Prefill、多 token、其它 dtype、非对齐尺寸及不支持的权重布局使用原投影路径。
- **Router 准备融合**：不超过 8 个 token、BF16 输入、FP32 权重、至多 1024 个专家时，
  直接在寄存器中转换输入，并融合温度缩放与 sqrt(softplus)。保留原 FP32 点积归约顺序，
  专家 Top-K、偏置、归一化和路由规则不变。
- **O_A 单 token 归约**：各输出行使用独立 warp，保留原 256 线程二叉树求和顺序，
  减少 block 同步。其余 token 数和不适用的分组布局继续走已有实现。

这些路径使用通用 CUDA 实现，不依赖 Hopper 专用指令。CUDA 回归
`deepseekV41DecodeProjections` 检查 mHC、Router 和 O_A 的精确结果及拒绝路径；
`deepseekV41EngramProjection` 检查多卡输出顺序、较大 batch 回退及非法分片边界；
`deepseekV41FullDecode` / `deepseekV41FullDecodeStress` 覆盖旋转、KV 生命周期和图重放。
MoE 专家计算与通信调度未在这些优化中调整。
