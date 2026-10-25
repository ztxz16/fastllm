# 使用 `--auto` 部署与调节

[English](auto_en.md) · [文档索引](README.md)

支持自动规划的模型先从这条命令开始：

```sh
ftllm server /path/to/model --auto
```

当前运行时自动规划面向 Linux NVIDIA CUDA。`--auto` 根据模型元数据、可见 GPU、可用显存和主存，选择设备布局、专家放置与缓存、上下文、并发、线程和 prefill 分块。它做容量与延迟估算，不会反复加载模型测速，也不保证对所有请求最快。支持范围和各模型默认值见[自动规划原理](hybrid-auto.md)；需要实测搜索的 DeepSeek V4.1 场景另见 [`--autotune`](autotune.md)。

## 按模型选择入口

| 模型 | 部署方式 |
| --- | --- |
| Qwen3 稠密模型 | 支持 `--auto`，不使用 MoE 专家缓存 |
| Qwen3.5 架构的稠密 HF 模型（含 Qwen3.8-27B-FP8） | 优先 `--auto`；其他量化格式需分别验证，见 [Qwen 指南](qwen3.md) |
| Qwen4 / Qwen3.8-Flash-Next | 优先 `--auto`，见 [Flash-Next 指南](qwen3.8-flash-next/README.md) |
| GLM-5.3-Flash | 优先 `--auto`，见 [GLM 指南](glm5.md) |
| DeepSeek-V4.1 | 优先 `--auto`，见 [V4.1 指南](deepseek_v41.md) |
| DeepSeek-V4、Kimi-K3、Dots3-Note、Laguna、GLM-5/5.2 | 尚未接入运行时自动规划，使用各自指南的手动命令 |

支持模型推理不等于支持 `--auto`。CPU-only、Windows 和 ROCm 使用对应的手动部署方式。

## 覆盖规则

- **只填写需要调整的参数，其余交给 `--auto`。** CLI 显式参数优先；JSON 配置中的值也算显式设置，命令行优先于 JSON。不要直接复制手动配置的全部默认值。
- `0`、`false`、`off` 可以是关闭指令，不能等同于省略。下面表格标明支持关闭的参数。
- 显式上下文、并发、分块和缓存需通过预算校验；无可行方案时会报错，不会为满足预算悄悄缩小你指定的数值。
- 显存与内存上限是**规划上限**，不是进程或系统的内存硬限制。`g`、`m` 按 GiB、MiB 解释；实际运行仍受模型、请求形状和分配器影响。
- 进程继承的环境变量仍生效。`CUDA_VISIBLE_DEVICES` 控制可见 GPU；`FT_THREADS`、`FASTLLM_NUMA_THREADS` 的原生优先级可能覆盖 CLI 线程数；`FASTLLM_TP` 也可能固定拓扑。

## 常用参数与影响

| 参数与示例 | 对象与影响 |
| --- | --- |
| `--max_context_length 32768` | 每个会话输入与输出合计上限。调小可减少 KV 预算，为专家缓存或权重腾出显存，但长请求可能超出窗口。调大需要容量和模型/RoPE 支持；它不是每次请求的输出长度。 |
| `--max_batch 2` | 每轮最多同时推理的请求数。可能提高并发总吞吐，也增加 KV/工作区预算、减少专家缓存，单请求延迟不一定更低。 |
| `--tokens 65536` | 所有会话共享的总 token 容量。自动模式未指定时按上下文 × 并发预留；显式缩小会限制可同时容纳的会话长度，不能绕过实际 KV 需求。 |
| `--auto_gpu_memory_limit 20g` | 每张可见 GPU 的规划显存上限。可能减少缓存、分块或 GPU 驻留权重；不改变 GPU 架构、算力和带宽。 |
| `--auto_host_memory_limit 120g` | 整个规划可用的主存上限。可能减少常驻专家或磁盘后端主存缓存，必要时转向 disk；不会限制操作系统页缓存或其他进程。 |
| `--force_ddr_limit 120g` | 直接指定最终主存规划预算为 120 GiB，替代自动检测和预留后的预算；与 `--auto_host_memory_limit` 互斥。只影响规划，不限制系统实际内存使用。 |
| `--gpu_mem_ratio 0.9` | 显存预算比例，与显存上限共同约束规划。降低可留更多余量；提高可能扩大缓存但增加显存紧张风险。DeepSeek V4.1 自动混推目前默认 0.95，其他场景通常 0.9。 |
| `--threads 32` | CPU/NUMA 计算线程数。需结合物理核、内存带宽和 NUMA 拓扑；过多线程可能争抢带宽、增加同步。不等于进程 CPU 亲和性限制。 |
| `--chunked_prefill_size 4096` | 每块输入处理的 token 数。大块通常利于长输入吞吐，但占用更多工作区，可能挤占专家缓存或增加并发等待；小块省空间，也可能降低 prefill 吞吐。 |
| `--moe_cuda_cache 7g` | 混推时每卡专家显存缓存，`0` 关闭。大缓存可能减少专家搬运和 CPU 计算，取决于命中率；普通权重、KV、工作区另算。显式缓存也限制自动放置，例如 DeepSeek V4.1 不再自动改选 NUMA/disk 混合分界。 |
| `--kv_cache_dtype fp4` | KV 存储类型，可选 FP16/BF16/FP8/FP4 等，需模型/后端支持。低精度可能省显存，但不保证更快或精度不变。DeepSeek V4.1 原生伪量化 KV 的匹配 FP4 打包有特殊无损存储语义，不能推广到其他模型。 |
| `--prefix_cache true` | 复用跨请求的相同前缀，主要加快重复请求的 prefill；不同前缀或缓存失效时没有同样收益。`false` 关闭，单独关闭它不等于关闭全部历史复用。 |

KV 类型通常建议省略，保留模型专属默认值。显式 `--kv_cache_dtype auto` 会覆盖自动规划器选出的存储类型，不能简单当成“没填”。

## 推测解码与快速 Prefill

| 参数与示例 | 影响与适用范围 |
| --- | --- |
| `--speculative_algorithm off` | 关闭全部推测解码。适合普通解码基线或接受率很低的请求；保留草稿路径时也可用它明确关闭。 |
| `--dspark 5` | 启用内置 DSpark 并指定每轮草稿数，`0` 关闭。需要对应模型和草稿权重；更多草稿会增加 draft/verify 成本，额外接受的 token 足以抵消成本时才更快。 |
| `--speculative_dspark_confidence_threshold 0.5` | DSpark 草稿置信度门槛，范围 `[0,1]`。提高会减少低置信度草稿；降低可能增加验证工作。`0` 关闭门槛，**不会强制接受草稿**，最终仍由目标模型验证。 |
| `--mtp 3` | 支持 MTP 的模型的草稿数，当前范围 `0–8`，`0` 关闭。不能与 DSpark 同时启用。 |
| `--mtp_min_p 0.5` | 支持该能力的 Qwen4/GLM-5.3 贪心 MTP 草稿概率门槛，`0` 关闭。调高减少低置信度草稿，不保证更快。 |
| `--draft /path/to/draft` | 指定受支持的外部草稿模型，算法、格式与目标模型必须匹配。增加权重和工作区需求，不能用任意文件补齐 DSpark/MTP。 |
| `--fast_prefill` / `--no_fast_prefill` | 开启 / 关闭 DeepSeek V4.1 的快速 Prefill。该模型的 `--auto` 默认开启；删除这两个参数即可恢复自动设置。使用 JSON 配置时，也需删除显式的 `fast_prefill` 字段。 |

DeepSeek V4.1 通常自动启用 DSpark 5，但没有内置 DSpark 权重的 GGUF 不会仅因 `--auto` 开启。带受支持内置 MTP 权重的 Qwen3.5 架构模型可自动启用 MTP。以实际日志为准，不要假设所有模型都启用同一种推测解码。显式 MTP/DSpark 数值（包括 `0`）会覆盖模型默认的推测策略。

## 设备、磁盘与专家放置

这些选项更容易限制规划器的搜索范围，通常先尝试上下文、并发与资源上限。

| 参数与示例 | 影响与注意事项 |
| --- | --- |
| `CUDA_VISIBLE_DEVICES=0,1` | 只让服务看到选定 GPU，进程内部卡号重新从 0 编排。仍由 `--auto` 选择可见卡中的方案。 |
| `--device cudapp=0,1 --tp off` | 明确选择两卡 PP；`--tp 2` 则请求两卡 TP。通信、权重分配和并发收益不同；DeepSeek V4.1 当前自动搜索单卡和 PP，TP 需显式请求。 |
| `--moe_device numa` / `cpu` / `disk` | 固定主机专家后端，可能排除可行或更快的放置方案。受支持的混推实现中 CPU/NUMA 共用计算、缓存与动态分流能力，具体格式仍需对应后端支持。 |
| `--moe_device_layers -1` | 所有专家层使用指定后端；正数表示末尾多少层使用该后端。不是 GPU 驻留层数。固定它会缩小 GPU 常驻前缀的搜索范围。 |
| `--moe_cpu_cache 64g` | disk 后端的专家主存缓存总预算，`0` 关闭。CPU/NUMA 常驻专家不靠此参数扩容；缓存越小，冷请求和未命中时越依赖磁盘。 |
| `--auto_disk_cache_ratio 0.5` | DeepSeek V4.1 自动 NUMA/disk 分界规划的推荐最低主存缓存覆盖比例，范围 `[0,1]`。增大倾向于为 disk 专家保留更多缓存；减小允许更多 NUMA 常驻层。`0` 取消目标，`1` 也不等于禁止磁盘；无法满足时会保留 disk 回退并输出说明。不是 GPU 缓存比例或实测命中率。 |
| `--ngram_device cpu` / `disk` | PLE/Engram 大表常驻或按需读取，独立于专家后端；`moe_device=numa` 同时 `ngram_device=disk` 不代表专家走磁盘。 |
| `--prefix_cache false --cache_history false` | 同时关闭前缀和历史复用，才允许自动规划考虑支持范围内的 Qwen4 精确 KV streaming。不保证一定选择 streaming，也不是其他模型的通用省显存开关。 |

GPU/主存能容纳专家时，自动规划优先避免磁盘专家。“没有 disk 专家”不意味着所有文件都不再读盘：加载、PLE/Engram、操作系统换页属于不同环节。缓存热度、替换频率等 `--moe_cache_*` 高级策略见[专家缓存文档](cuda-expert-cache.md)，一般先保留默认值。

## 常见命令

```sh
# 给其他程序留出资源，其余自动选择
ftllm server /path/to/model --auto \
  --auto_gpu_memory_limit 20g --auto_host_memory_limit 120g

# 32K 单会话窗口、最多 2 个并发请求
ftllm server /path/to/model --auto --max_context_length 32768 --max_batch 2

# 单卡普通解码，不启用推测算法
CUDA_VISIBLE_DEVICES=0 ftllm server /path/to/model --auto --speculative_algorithm off

# 支持内置 DSpark 的 DeepSeek V4.1：固定 5 个草稿，关闭快速 Prefill
ftllm server /path/to/deepseek --auto --dspark 5 --no_fast_prefill
```

API 的 `max_tokens`、`temperature`、`top_p` 等请求参数影响单次生成，不等于容量预算。比较速度时固定请求、输出长度与并发，区分冷缓存和重复请求，分别看 prefill、单请求 decode 和并发总吞吐。真实生成的 token/专家路由可能不同，小幅波动不应直接归因于某个参数。

## 启动器自动模式

运行 `ftllm`，新增启动项先在顶部选择模型文件或目录，再填写基础配置并选择“自动配置”。资源限制、视觉模块和模型参数默认折叠；不填写覆盖项就保持纯自动规划：

- **资源限制（可选）**：显存比例对应 `gpu_mem_ratio`；内存上限对应 `force_ddr_limit`，例如 `120g`，仅用于规划预算。
- **视觉模块（可选）**：展开后选择与 GGUF 主模型匹配的视觉模块文件，对应 `--mmproj`；留空不加载外部视觉模块。
- **模型参数**：默认折叠，只提供 Prefill Chunk（`chunked_prefill_size`）、最大并发（`max_batch`）、自定义参数和自定义环境变量。

Chunk 和最大并发留空即自动规划。其他选项可以直接填写到“自定义参数”，例如 `--dspark 5`；参数附加到命令末尾，带空格的值使用引号。环境变量填写 `KEY=VALUE`，用空格或换行分隔。这两项在自动和自定义模式下都生效。

例如，在“资源限制”填写显存比例 `0.9`、内存上限 `120g`，在“模型参数”填写 Chunk `4096`、最大并发 `2`，对应：

```sh
ftllm server /path/to/model --auto --gpu_mem_ratio 0.9 \
  --force_ddr_limit 120g --chunked_prefill_size 4096 --max_batch 2
```

调大 Chunk 可能提高 prefill 吞吐，但增加工作区占用；调大最大并发可能提高总吞吐，但也增加缓存预算。需要快速 Prefill 开关等其他选项时，在“自定义参数”填写 `--fast_prefill` 或 `--no_fast_prefill`；删除这些参数恢复默认。

旧自动配置中的显存、内存绝对上限会保留为“自定义参数”里的 `--auto_gpu_memory_limit`、`--auto_host_memory_limit`，不会改成显存比例或强制主存预算。

自动与自定义模式的 Chunk、最大并发和显存比例分别保存，避免切换模式时误用隐藏值。设置随启动项保存，可在预览中检查最终命令，重启模型服务后生效。

## 查看最终配置

启动日志中的 `[auto]` JSON 比预览命令更完整，重点检查：

| 字段 | 用途 |
| --- | --- |
| `device`、`tp`、`mode` | 实际 GPU 与并行方式 |
| `max_context_length`、`total_tokens`、`max_batch` | 单会话窗口、共享容量与并发 |
| `chunked_prefill_size`、`threads`、`kv_cache_dtype` | 输入分块、线程和 KV 存储 |
| `moe_device`、`resident_layers`、`moe_device_layers` | 专家后端及驻留层分界 |
| `moe_cuda_cache`、`moe_cuda_cache_per_device`、`moe_cpu_cache` | 缓存预算；含义依单一后端或 NUMA/disk 混合放置而异，见规划原理文档 |
| `gpu_budget_bytes` / `gpu_estimated_bytes`、`host_budget_bytes` / `host_estimated_bytes` | 预算与估计用量，不是运行中测得的峰值 |
| `automatic.dspark`、`automatic.mtp`、`automatic.fast_prefill` | 推测解码与快速 Prefill 的最终配置 |

`automatic.cuda_graph=true` 表示自动策略允许使用 Graph，运行时仍可能因后端、设备布局或请求形状回退。缺失的模型专属字段不应当作通用功能已开启。
