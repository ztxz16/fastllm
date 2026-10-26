# 手动部署与完整参数参考

[English](deployment_en.md) · [文档索引](README.md)

日常部署先使用 `ftllm server /path/to/model --auto`，常用覆盖参数见 [`--auto` 调节指南](auto.md)。本文保留手动设备配置、模型专属参数、下载导出和源码安装说明。

- [典型部署配置](#典型部署配置)
- [完整参数参考](#常用参数)
- [模型格式、下载与导出](#模型格式下载与导出)
- [源码安装](#源码安装)

## 当前模型能力

以下列出主要模型系列的能力与限制，具体部署参数见对应模型指南。

| 模型系列 | 当前重点能力 |
| --- | --- |
| Qwen | Qwen4-Exp / Qwen3.8-Flash-Next 文本解码、QSA、PLE n-gram 和 CPU/CUDA/NUMA 混合推理；Qwen3.8-Flash-Next MTP；Qwen3.5/3.6/3.8 MTP 和 DFlash2 |
| DeepSeek | DeepSeek-V4 / V4-Flash、稀疏注意力、内置 DSpark、多卡 CUDA 与 CPU/NUMA 混合 MoE |
| Kimi | Kimi-K3、KDA/MLA、外部 DSpark，以及 CUDA、NUMA、CPU/GPU 专家和磁盘专家 |
| GLM | GLM-5 DSA、GLM-5.3-Flash KDA 与分页缓存、GLM-5.2 量化 KV-B CPU 推理 |
| 其他 | Dots3-Note、Laguna、HY-V3、Step3.5/3.7、MiniMax-M2、Gemma4 等 |

Qwen4-Exp / Qwen3.8-Flash-Next 当前不加载视觉权重；Qwen3.8-Flash-Next 可通过 `--mtp` 按需加载 MTP 权重并启用推测解码。早期模型的兼容信息仍可在[支持模型列表](models.md)中查询；最新适配和限制以[版本日志](version.md)为准。

## 典型部署配置

### 多卡张量并行

`--tp 0,1` 显式使用 0、1 号 GPU；`--tp 2` 表示使用前两张可见 GPU；`--tp auto` 自动使用检测到的 GPU。

~~~bash
ftllm server /data/models/my-model \
  --device cuda --tp 0,1 \
  --max_batch 16 --gpu_mem_ratio 0.9
~~~

### GPU + NUMA 混合 MoE

~~~bash
ftllm server /data/models/my-moe-model \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192
~~~

内存不足时还可以把少量专家层放到磁盘，例如 `--moe_device "{'cuda':1,'numa':8,'disk':1}"`。磁盘路径依赖 SSD 随机读取性能，详细配置参见[混合推理指南](mixforward.md)。

### 长上下文与前缀缓存

`--max_context_length`（别名 `--max-context-length`）设置单会话输入与输出合计上限。扩大模型声明窗口时，还需要有效的 `--rope_scaling`（别名 `--rope-scaling`），接受 `yarn` 或 JSON；只缩小窗口时可省略 RoPE 参数。配置在加载时生效，不修改模型的 `config.json`。

例如，将 Qwen3-0.6B 扩展到 65536 token，并开启前缀缓存。其 YaRN 原始长度为 32768，应显式指定，不能用配置声明的 40960 代替：

~~~bash
ftllm server /data/models/Qwen3-0.6B \
  --device cuda --max_batch 1 --tokens 65536 \
  --max_context_length 65536 \
  --rope_scaling '{"rope_type":"yarn","factor":2,"original_max_position_embeddings":32768}' \
  --chunked_prefill_size 8192 \
  --prefix_cache true
~~~

Qwen3.8-27B-FP8 可以使用已知原始长度的 `yarn` 简写。下面配置双卡、FP4 KV 和 1,000,000 token 的目标窗口，解析得到 original=262144、factor=4：

~~~bash
ftllm server /data/models/Qwen3.8-27B-FP8 \
  --tp 2 --kv_cache_dtype fp4 \
  --max_context_length 1000000 --rope_scaling yarn
~~~

`--tokens` 是所有会话共享的 KV 池容量，未设置时自动预算。显式目标超过 RoPE 覆盖范围或 warmup 校准容量会启动失败；`/v1/models` 返回实际窗口、模型原声明和用户目标。上述 1M 命令需要足够显存，本机双 24GB 的测试配置无法容纳，完整 1M 输入尚未实测。

当前扩展支持 Qwen2、Qwen3、Qwen3.5 布局，以及基于 Qwen3.5 架构的 Qwen3.8；Launcher 高级参数中的「RoPE 扩展」使用相同配置。HF 目录和 GGUF 文件均可显式设置 `--rope_scaling`，仍受模型布局限制。GGUF、FLM 和自定义 GraphLLM 在 CLI 中仅设置长度时保留只缩小的行为；FLM 和自定义 GraphLLM 暂不支持 RoPE 扩展。更多参数、适配范围和验证结果见[上下文扩展说明](context-length-extension.md)。

### 投机解码

以下功能只适用于结构和 checkpoint 匹配的模型：

~~~bash
# 以 Qwen3.5 为例使用内置 MTP，每轮最多配置 8 个 draft token
ftllm server /data/models/qwen3.5 --mtp 4

# 给不含 MTP 权重的 Qwen3.5 GGUF 挂载独立 MTP 模块
ftllm server /data/models/qwen3.5.gguf \
  --device cuda --cuda_embedding \
  --draft /data/models/qwen3.5-fp8/mtp.safetensors \
  --draft_tokens 5

# DeepSeek-V4 内置 DSpark
ftllm server /data/models/deepseek-v4 --dspark 7

# Qwen3.8 + 独立 DFlash2 draft checkpoint
ftllm server /data/models/qwen3.8 \
  --tp 2 \
  --draft /data/models/qwen3.8-dflash2 \
  --draft_tokens 7
~~~

`--draft` 会根据 draft checkpoint 自动识别 MTP、DFlash2 或 DSpark。独立 MTP 当前用于 Qwen3.5 GGUF，路径可指向包含 `config.json` 和 `mtp.safetensors` 的目录，也可直接指向 `mtp.safetensors`；省略 `--draft_tokens` 时默认使用 5。DFlash2 的 `--draft_tokens` 表示实际 draft token 数，不包含 anchor token。

DFlash2 的完整配置和验证结果见 [Qwen3.8 DFlash2 文档](benchmarks/qwen38_27b_dflash2.md)。

### Qwen4 PLE 磁盘模式

Qwen4-Exp / Qwen3.8-Flash-Next 的 PLE 表较大。主机内存不足时可按需从 checkpoint 读取：

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

磁盘模式会降低常驻内存，但增加随机 I/O，建议使用高速 SSD。更多限制见 [Flash-Next 用户手册](qwen3.8-flash-next/README.md)。

## 常用参数

CLI 会持续演进，`ftllm <command> --help` 是当前安装版本的最终依据。下面列出部署中最常用的参数。

### 模型、设备与精度

<a id="3-如何设定运行设备"></a>

| 参数 | 说明 |
| --- | --- |
| `model` / `-p, --path` | Hugging Face 仓库 ID、本地 HF 目录、FastLLM 模型文件或配置文件 |
| `--device` | 主计算设备，常用值为 `cpu`、`cuda`、`numa` |
| `--vision_device` | Qwen3.5 视觉编码器设备：`auto`（默认）、`cpu`、`cuda`、`cuda:N`；多卡 CUDA 跟随普通 TP，`cuda:N` 不单独覆盖；`cpu` 时视觉塔权重常驻内存，省显存但编码变慢 |
| `--tp` | CUDA 张量并行设备；支持 `0,1`、`2` 或 `auto` |
| `--moe_device` | MoE 专家层设备，可使用 `cpu`、`cuda`、`numa`、`disk` 或按比例组合 |
| `--moe_device_layers` | 仅让最后 N 个 MoE 层使用 `--moe_device`；`-1` 表示全部 |
| `-t, --threads` | CPU/NUMA 线程数；`-1` 表示自动选择 |
| `--dtype` | 加载 HF 权重时的权重类型；默认 `auto`，已量化模型通常不应覆盖 |
| `--moe_dtype` | 单独设置 MoE 权重类型 |
| `--atype` / `--moe_atype` | 设置普通层和 MoE 层的激活类型 |
| `--kv_cache_dtype` | KV Cache 类型：`auto`、`float16`、`bfloat16`、`fp8_e4m3` 或 `fp4`，需模型与后端支持 |
| `--dtype_config` | 动态量化配置文件，参见[动态量化说明](dtype_config.md) |
| `--triton` | 启用可用的 Triton CUDA 算子 |

### 显存、上下文与调度

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--gpu_mem_ratio` | `0.9` | 可用于模型与缓存的 GPU 显存比例 |
| `--auto_gpu_memory_limit` | 未设置 | `--auto` 时限制每张可见 GPU 的规划显存，例如 `20g`；未设置则读取实际可用显存 |
| `--auto_host_memory_limit` | 未设置 | `--auto` 时限制自动规划使用的主机内存总量，例如 `128g`；未设置则读取实际可用内存 |
| `--force_ddr_limit` | 未设置 | 强制指定最终主存规划预算，例如 `114g`，可超过自动检测上限，不再扣除 CMA 或自动预留；与 `--auto_host_memory_limit` 互斥，不修改系统内存硬限制，见[预算说明](hybrid-auto.md) |
| `--moe_cuda_cache` / `--moe-cuda-cache` | `0` | GPU 专家权重缓存预算，例如 `3g`（3 GiB）；`0` 关闭。配合 CUDA 主计算与 CPU/NUMA/disk 专家使用，KV Cache、普通权重及工作区另算；见[专家缓存说明](cuda-expert-cache.md) |
| `--moe_cpu_cache` / `--moe-cpu-cache` | `0` | `--moe_device disk` 的专家内存缓存总预算，例如 `32g`；可与显存专家缓存同时使用，见[磁盘多级缓存](cuda-expert-cache.md#disk-backed-multilevel-cache) |
| `--kv_cache_limit` | `auto` | KV Cache 最大使用量 |
| `--tokens` | 自动 | 用于计算 Paged KV Cache 容量的总 token 数 |
| `--page_size` | 后端决定 | Paged KV Cache 每页 token 数；多卡默认通常为 16 |
| `--max_batch` | 自动 | 每轮最多同时推理的请求数 |
| `--max_context_length` / `--max-context-length` | 自动 | 单会话输入与输出合计上限；HF 显式目标需通过 RoPE 与 KV 容量检查 |
| `--rope_scaling` / `--rope-scaling` | 沿用模型配置 | RoPE 扩展，接受 `yarn` 或 JSON；仅对已适配的 HF 模型布局生效 |
| `--chunked_prefill_size` | 关闭/模型决定 | 分块 Prefill 的切片大小，例如 `8192` |
| `--fast_prefill` / `--no_fast_prefill` | 关闭；V4.1 的 `--auto` 默认开启 | 开启 / 关闭 DeepSeek V4.1 快速 Prefill，省略显式设置则跟随默认值 |
| `--prefix_cache` | 模型/环境决定 | 是否开启前缀缓存，使用 `true` 或 `false` |
| `--cuda_slab` | `0` | CUDA 权重 slab 大小（MB）；`0` 为关闭 |
| `--ngram_device` | `cpu` | Qwen4 PLE 表放在 `cpu` 或 `disk` |

表中为 CLI 默认值；`--auto` 会为未指定的选项选择模型专属配置。资源上限是规划预算，不是运行时内存硬限制。覆盖规则和缓存预算含义见 [`--auto` 调节指南](auto.md)。

### 解码、模板与工具调用

<a id="工具调用"></a>

| 参数 | 说明 |
| --- | --- |
| `--enable_thinking` | 控制模型的思考模板开关，需要模型支持 |
| `--mtp` | 支持 MTP 的模型每轮生成的 draft token 数，`0` 关闭，当前最大为 8 |
| `--mtp_fp8_draft_head` / `--mtp-fp8-draft-head` | Qwen3.5 系列多卡 MTP 的 FP8 draft 输出头开关：`1` 开启，`0` 复用原输出头以节省显存，可能降低生成速度；不改变 MTP draft 数。显式参数优先于 `FASTLLM_MTP_FP8_DRAFT_HEAD`，未指定时沿用环境变量，默认开启 |
| `--dspark` | 启用模型内置 DSpark，并设置每轮 draft token 数 |
| `--draft` / `--draft_model_path` | 外部 MTP/DSpark/DFlash draft checkpoint；根据配置自动识别算法，MTP 可直接指定 `mtp.safetensors` |
| `--draft_tokens` | 每轮最多使用的 draft token 数；未指定时读取 draft 配置 |
| `--tool_call_parser` | 工具调用解析器；默认 `auto` |
| `--chat_template` | 自定义 Jinja chat template 文件 |
| `--cache_dir` | 在线模型的本地缓存目录 |
| `--ori` | 读取部分 GGUF 时指定原模型配置和 tokenizer 目录 |
| `--mmproj` | Qwen3.5 / DeepSeek V4.1 GGUF 的配套视觉模块；见 [Qwen GGUF 多模态](qwen3.md#gguf-multimodal) 和 [DeepSeek V4.1](deepseek_v41.md) |

Qwen4 CUDA 推理将 MTP 草稿专家常驻 GPU：单卡使用该卡，PP 使用最后一个 CUDA 阶段，TP 在各卡常驻对应分片。草稿专家独立于 `--moe_cuda_cache` 的目标模型专家缓存预算，需要额外显存；CPU 推理沿用 MoE 设备配置。

Qwen3.5 系列的 MTP 和 DFlash 草稿支持以下设置。启用相应的草稿算法后，兼容路径默认使用 NVFP4；环境变量可在启动前覆盖默认值：

| 环境变量 | 默认值 | 说明 |
| --- | --- | --- |
| `FASTLLM_DRAFT_QUANT` | `nvfp4` | `off` 关闭 NVFP4 转换；`nvfp4_head` 只转换独立草稿输出头；`nvfp4` 转换草稿主干和输出头。目标模型输出头保持原权重，不支持的设备、类型或形状保留原路径 |
| `FASTLLM_CUDA_NVFP4_SWIGLU_MULTIROW` | `1` | 允许多行 NVFP4 Linear + SwiGLU 融合，`0` 关闭。当前多行内核仅支持 SM75、FP16 输入、M=2～8，并检查形状、布局和临时空间；其他情况回退 |
| `FASTLLM_TP_NVFP4_MLP_SWIGLU` | `1` | TP MLP 尝试 NVFP4 Linear + SwiGLU 融合，`0` 关闭；由底层能力检查选择内核，不支持时执行原 Linear + SwiGLU |
| `FASTLLM_TP_NATIVE_GREEDY` | `1` | Qwen3.5 TP 贪心采样支持原生类型 logits；eager 推测验证省去 FP16→FP32 转换，`0` 关闭。随机采样、返回 logits、Graph 和 GPU token handoff 仍保持 FP32 路径 |
| `FASTLLM_DFLASH_BATCH_PREFIX_SNAPSHOTS` | `1` | DFlash2 CUDA 批量验证按各请求接受长度恢复线性状态，避免拒绝后的主模型重算。max_batch≤4 使用逐位置状态快照；更大配置保存紧凑激活、批量恢复卷积/GDN 状态，实际 batch 缩小时仍沿用该路径，无 16 路上限。自动预留恢复缓冲与草稿滑窗 KV 显存，相应减少主模型 KV 容量。`0` 回退到完整前缀重算；单请求路径不变。不限定 SM；状态算子已在 SM75 验证到 32 路 |
| `FASTLLM_MTP_BATCH_SAMPLING` | `1` | Qwen3.5 CUDA MTP 将多请求的草稿采样和主模型拒绝采样合批，直接读取各请求独立的草稿缓存，合并结果回读。混合贪心/采样或不兼容验证长度保留原路径；`0` 恢复逐请求采样。保持采样分布，不保证与逐请求路径随机输出逐 token 相同 |
| `FASTLLM_MTP_FP8_MARLIN` | `1` | Qwen3.5 TP MTP 在初始化时为符合已有 Marlin 条件的 FP8 草稿分片准备计算布局，避免多行草稿首次调用错过布局转换。不改变权重量化格式，保留现有架构/后端选择；`0` 恢复延迟准备。已在 SM75/TP2 验证 |
| `FASTLLM_CUDA_GDN_SEQUENCE_PREPARE` | `1` | 批量短序列 GDN 在 K/V 维度均为 128、batch≥4 的 eager 路径预计算 Q/K 归一化与门控系数，避免各状态分块重复计算。保留逐 token FP16 状态舍入及前缀快照；小批量、其他 V 维度和 Graph 保留旧路径。`0` 关闭；已在 SM75/TP2 验证 |
| `FASTLLM_MTP_DRAFT_TOKEN_IDS` | 未设置 | 可选的单卡或多卡 MTP 草稿词表 token ID 文件，仅在启用 NVFP4 转换时使用；未设置或为 `0` 时使用完整词表。多卡按原词表分片筛选并映射回全局 token ID，对齐填充只重复已有候选。筛选词表只用于贪心草稿，随机采样使用完整输出头，目标模型仍使用完整词表验证 |
| `FASTLLM_DFLASH_DRAFT_TOKEN_IDS` | 未设置 | DFlash2 贪心请求的可选 token ID 文本清单（空白分隔）。单卡支持原编码 GGUF 或 NVFP4 草稿头，多卡支持 NVFP4；每个连续词表分片至少需有 selector top-k 个候选。对齐填充在 top-k 前移除，选择后恢复原 token ID。完整目标头保留，缩表额外占用草稿头显存，覆盖不足可能降低接受率；启用前应验证任务速度与输出一致性。随机采样使用完整词表；非法或不支持的清单回退完整词表。`FASTLLM_DRAFT_QUANT=off` 时不启用 |
| `FASTLLM_DFLASH_ATTENTION` | 未设置：SM75 开，其余关 | DFlash FP16 融合滑窗 attention 的统一开关：`0` 关闭，`1` 在支持的设备上开启。要求 head_dim=128、query 数 1～16、query 数×GQA 分组数≤64、query 数≤窗口≤4096，且 FlashInfer 可用；不匹配时回退原路径。SM80 及以上的 Q64 路径尚无实机正确性或速度验证；SM70 及以下始终保持原路径 |

DFlash2 的动态卷积、QKV 和 Gateup 准备会根据设备、类型和形状自动选择融合实现，不支持时回退到常规算子；TP selector 投影固定在 rank 0 输出头工作流中提前提交。旧开关 `FASTLLM_CUDA_DFLASH_FUSED_CONV`、`FASTLLM_CUDA_DFLASH_FUSED_QKV_PREPARE`、`FASTLLM_CUDA_DFLASH_FUSED_GATEUP_PREPARE` 和 `FASTLLM_DFLASH_TP_EARLY_SELECTOR` 已移除，设置它们不再影响执行路径。

未覆盖的设置使用表中默认值；内核本身的能力限制仍然有效。单卡和多卡均可使用草稿 NVFP4；多卡在切分后转换符合条件的分片。MTP 限于稠密模型、FP16 计算；DFlash 的量化 Linear 通过 FP16 适配后恢复原激活类型。量化会影响草稿接受率，独立输出头和视图副本也会改变显存占用，不保证所有模型提速；目标模型权重与完整词表验证保持原样。可用 `FASTLLM_DRAFT_QUANT=off` 保留原草稿精度，既有多卡 FP8 草稿输出头开关仍有效。

NVFP4 小矩阵解码默认在 SM75、68 个 SM 的设备（如 RTX 2080 Ti）上启用已验证的调优，覆盖 M=1～8、N×K 为 17408×5120 的融合 SwiGLU，以及 5120×8704、5120×3072 的 Linear。运行时还需满足线程块驻留条件；其他架构、形状和原始 M>8 的 prefill 保持原路径。可用 `FASTLLM_CUDA_NVFP4_SM75_DECODE_TUNE=0` 关闭，`1` 显式开启全部，或用 `linear` / `swiglu` 仅开启对应部分。多行 SwiGLU 的融合入口仍由 `FASTLLM_CUDA_NVFP4_SWIGLU_MULTIROW` 单独控制。

### API Server

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8080` | Server 端口；WebUI 默认端口为 1616 |
| `--model_name` | 自动 | API 中校验和返回的部署名称 |
| `--api_key` | 空 | 非空时开启 Bearer API Key 校验 |
| `--temperature` / `--top_p` / `--top_k` | 模型默认 | 覆盖服务端默认采样参数 |
| `--repeat_penalty` | 模型默认 | 覆盖重复惩罚参数，也支持 `--repetition_penalty` |
| `--hide_input` | 关闭 | 不在服务日志中显示请求内容 |
| `--startup-progress` | `off` | 设置为 `ndjson` 时向 stderr 输出模型加载与就绪事件 |

查看完整参数：

~~~bash
ftllm --help
ftllm server --help
ftllm bench --help
ftllm download --help
~~~

## 模型格式、下载与导出

### 支持的输入格式

- Hugging Face 原始 Safetensors 权重，包括模型自带的 FP16、BF16 或 FP8 权重。
- 已量化的 AWQ 模型。
- FastLLM 导出的定精度或动态量化模型。
- 部分 GGUF 格式；已适配的模型可直接读取内置配置和 tokenizer，也可通过 `--ori` 指定原模型目录。Qwen3.5 架构族的 GGUF 使用独立视觉模块时，还需指定 `--mmproj` 并提供匹配的视觉配置，见 [GGUF 多模态](qwen3.md#gguf-multimodal)。

量化格式是否可用取决于模型结构、设备和对应 kernel。首次部署建议保留 `--dtype auto`；对于已经量化的 checkpoint，不要再次指定在线量化类型。

### 下载模型

~~~bash
ftllm download <repo-id> --local-dir /data/models/model-name
~~~

只下载配置和 tokenizer 文件时可以排除权重：

~~~bash
ftllm download <repo-id> \
  --exclude "*safetensors*" \
  --local-dir /data/models/model-config
~~~

### 导出模型

在线量化会增加每次启动的加载时间。可以预先导出 FastLLM 格式：

~~~bash
ftllm export /data/models/source-model \
  -o /data/models/source-model-int4 \
  --dtype int4 -t 16
~~~

MoE 模型可以分别设置普通层和专家层精度：

~~~bash
ftllm export /data/models/source-moe \
  -o /data/models/source-moe-mixed \
  --dtype float16 --moe_dtype int4 -t 16
~~~

动态量化配置见[动态量化说明](dtype_config.md)。

## 源码安装

源码构建需要 C++17 编译器、Make 和 CMake；建议 GCC/G++ 9.4+、CMake 3.23+。Linux NUMA 构建通常还需要 `libnuma-dev`。CUDA 构建请预先安装兼容的 CUDA Toolkit 和 NCCL。

~~~bash
# Ubuntu/Debian 基础依赖
sudo apt-get install -y build-essential cmake libnuma-dev

# NVIDIA CUDA
bash install.sh -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"

# 指定 CUDA 架构，例如 Ada 使用 89
bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"

# CPU-only
bash install.sh
~~~

更多平台说明：

- [ROCm 编译与 wheel 打包](rocm.md)
- [TFACC 平台](tfacc.md)
- [示例程序、Android 和其他平台](../example/README.md)
- [编译与运行 FAQ](faq.md)
