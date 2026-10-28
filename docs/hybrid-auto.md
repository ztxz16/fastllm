# 混合推理自动配置

日常部署、参数取舍和启动器操作见 [`--auto` 参数调节指南](auto.md)。本文说明模型默认值、容量预算和后端细节。

使用 `--auto` 可以一次选择完整的 CUDA 运行配置：

```sh
ftllm server /path/to/model.gguf --auto
```

支持 Linux NVIDIA CUDA 上的 dense Qwen3、Qwen3.5 架构的稠密 HF 模型，以及 Qwen4、GLM-5.3、DeepSeek-V4.1 混推模型。未启用 `--auto` 时，原有数值和默认行为保持不变。

Qwen3.8-27B-FP8 已验证 `--auto`：优先选择纯 GPU TP，默认并发 4、模型原生上下文和内置 MTP5，显存不足时降低上下文。并发 MTP 验证默认采用 eager CUDA 路径，以避开批量验证 CUDA Graph 的非法内存访问；可用 `FASTLLM_QWEN35_MTP_VERIFY_CUDA_GRAPH=1` 显式覆盖。其他 Qwen3.5 架构 checkpoint 的量化格式和 MTP 权重仍需分别验证。

直接运行 `ftllm` 打开的 Launcher 中，新增启动项默认选择“自动配置”。先在顶部选择模型文件或目录，再填写基础配置。资源限制、视觉模块和模型参数默认折叠：显存比例对应 `--gpu_mem_ratio`，内存上限对应 `--force_ddr_limit`，视觉模块文件对应 `--mmproj`；模型参数只保留 Chunk、最大并发、自定义参数和自定义环境变量。每次启动重新读取硬件余量，留空的参数继续自动规划。其他选项可以填写到自定义参数；自定义环境变量和启动进程继承的环境均生效。界面保留自动和自定义两种模式，旧的长上下文／高并发配置按自定义模式打开并保留参数。具体操作见[启动器指南](launcher.md)。

自动配置设置中可以填写可选的“显存上限（每卡）”和“内存上限”。它们分别对应
`--auto_gpu_memory_limit` 与 `--auto_host_memory_limit`，接受 `20g`、`128g` 等二进制容量；留空时继续读取启动时的实际可用显存和主机内存。显存上限作用于每张可见 GPU，主机内存上限作用于整个自动规划预算。这是规划上限，不会限制操作系统或其它进程的实际内存分配；规划结果不足以容纳模型时会在加载前报错。

Dense Qwen3 按实际 GQA 头数、KV、dense 权重及 tied output head 副本预算，不分配专家缓存；上下文取 128K 与模型原生窗口的较小值。目前 Qwen3 的 CUDA Graph 预热会触发 deferred pool 的 stream-capture-unsupported，因此此模型的 `--auto` 默认使用普通 CUDA 执行。显式 `FASTLLM_CUDA_GRAPH` 仍优先；其他模型的 Graph 默认策略不变。

Qwen3.5 稠密 HF 模型按全注意力层的 GQA KV、线性注意力层的常驻状态、原生 FP8 权重和独立 MTP 层预算。先尝试原生上下文与纯 GPU TP2，必要时尝试 TP4 和较短上下文；显式 `--tp`、`--max_batch`、`--max_context_length`、`--tokens`、`--mtp 0` 或 `--speculative_algorithm off` 优先。内置 `mtp.safetensors` 且配置声明一个草稿层时默认启用 MTP5。默认 `--cuda_embedding`，不自动启用视觉输入。

HF safetensors 目录支持索引引用的多级分片和模型目录内的符号链接分片；`dtype=fastllm` 的版本 1 GGUF 量化张量读取序列化类型头，按保存的量化字节数规划，不按默认 FP16 展开。索引中的绝对路径、`..` 路径和指向模型目录外的符号链接仍会拒绝。

## 单参数自动选择

DeepSeek-V4.1 默认采用不超过模型原生窗口的 **1M（1,048,576 token）上下文、单请求**，开启 **DSpark 5**（置信度阈值默认 0.5）和 **fast_prefill**。没有内置 DSpark 权重的 GGUF 不自动启用 DSpark。Qwen3.5 稠密 HF 模型默认并发 4、尽量使用原生上下文；其他模型保持 **128K 上下文、单请求**。按代表性的 4K 输入、512 输出估计交互延迟。读取可见 CUDA 设备的可用显存、SM 数、内存带宽参数、PCIe 条件和 peer access 能力，比较单卡、PP、双卡等比例 TP，并联合选择专家驻留、显存缓存、主存缓存和 chunk。权重可全驻留且预算允许时，也可选择全部专家放 GPU。CPU 线程依据进程可用的物理核数选择，并预留启动和 IO 线程。

DeepSeek-V4.1 的 `--auto` 默认使用 **`kv_cache_dtype=fp4`**：长期压缩 KV 和 indexer key 在原模型计算中已经经过 FP4 伪量化，按匹配的网格、分组和 scale 打包是无损存储；滑窗使用其原有 FP8 网格。Flash-1017 的长期 KV 为 **890 B/token**，1M 有效载荷约 **0.869 GiB**，加上当前分配器的扩容容量和新旧缓冲重叠，规划预留约 **1.59 GiB**。此前统一选择 16 位 KV 会将这两项分别放大到 3.125 / 5.69 GiB。显式 `--kv_cache_dtype`（包括 `float16`、`bfloat16`、`fp4_e2m1`、`auto`）优先；启用 `FASTLLM_DSV41_DISABLE_FAKE_QUANT` 或不满足打包维度时不自动启用 FP4。其他模型的存储默认不变。

DeepSeek-V4.1 的 `--auto` 默认开启快速 Prefill；`--fast_prefill` 显式开启，`--no_fast_prefill` 关闭，省略显式设置则跟随自动规划。`--dspark 0` 或 `--speculative_algorithm off` 关闭推测解码。显式上下文、草稿数、阈值和 JSON 配置中的 `fast_prefill: false` 均优先；`--threads 64` 可固定线程数。预算不足时先调整自动 chunk、缓存和专家放置，仍放不下则报错，不自动缩短 1M 上下文。

DeepSeek-V4.1 的显存估算按 KV source 层统计共享的压缩 KV 和 index key，并计入原生缓存扩容的容量余量及新旧缓冲重叠。Indexer 打分按原生分块预算估算（默认 128 MiB，遵循 `FASTLLM_DSV41_INDEX_SCORE_MB` 和 `FASTLLM_DSV41_INDEX_CHUNK`），不直接预留完整的 `chunk × context` 分数矩阵。原模型的 FP8 投影按实际保留的精度及 scale 预算，输出头、路由和其他强制展开的投影仍按原生精度计入。DeepSeek-V4.1 的自动 chunk 候选为 **8192、4096、2048、1024、512、256**；其他模型的候选上限保持 4096。候选不超过上下文长度；显式指定数值时不会自动降低。

规划日志分别输出 `kv_logical_bytes_per_device`（有效载荷）、`kv_capacity_bytes_per_device`（含增长余量的容量）和 `kv_reallocation_bytes_per_device`（扩容时的旧缓冲重叠），避免将有效 KV 大小和规划峰值混为一谈。GGUF NUMA 模型开启 DSpark 多卡分流时，辅助卡即使没有本地 NUMA 专家，也保留专家上传缓冲预算。

DeepSeek-V4.1 原生 FP8/MXFP4 模型在单请求、非 TP、BF16、NUMA/磁盘专家和 `fast_prefill` 路径下，按 attention、WoA、MoE 分阶段的临时缓冲峰值估算工作区。主机专家使用复用的 256 行工作缓冲，避免按 chunk 为每个路由重复计入 GPU 中间张量；WoA 保留 FP32 fallback 的权重展开和分块空间。调试 dump、其他精度、GPU 驻留专家等路径继续采用通用保守估算。CUDA graph 隐式启用的 GPU embedding 也计入预算。

DeepSeek 混推的 `--auto` 在上下文预算不变的前提下，先争取 chunk 达到 **512**，再优先将**每卡专家缓存补到 3 GiB**，随后选择最大的可行 chunk，并将剩余显存分给缓存。候选 chunk 为 8192、4096、2048、1024、512、256，不超过上下文长度；预算不足以满足目标时尽量接近，不因此拒绝可行方案，也不为凑缓存把已可行的 512 降到 256。全部专家驻留 GPU 时无需额外缓存。显式 chunk、缓存、驻留和内存限制优先。这一排序同时用于各拓扑内部和跨拓扑选择；DeepSeek V4.1 自动拓扑仅考虑单卡及多卡 PP，不再自动选择 TP（包括容量不足时的回退）；显式 `--tp` 或现有 `FASTLLM_TP` 设置仍优先。主机专家方案会比较正常 embedding 与 `low_gpu_mem`，为 chunk 和专家缓存腾出显存；全部专家可放 GPU 时保留正常模式。DeepSeek V4.1 的 `--auto` 混推候选默认使用 `gpu_mem_ratio=0.95`，全部专家驻留 GPU 和其他模型仍使用原默认值；显式设置该参数时始终优先。规划值同步用于运行时。此策略保留 512 MiB guard、完整上下文 KV 及扩容重叠预算，不把短请求的实测显存当作 1M 上下文上限。

主存优先使用 NUMA（构建支持时），否则使用 CPU；超大 PLE/Engram 表优先按需读取，减少启动搬运和主存占用，主存仍放不下时再考虑磁盘专家缓存。CUDA Graph 默认开启，原有 `FASTLLM_CUDA_GRAPH` 显式设置仍优先。原有运行时检查继续决定 Graph 能否实际捕获。

Qwen4 混推默认保留前缀缓存和历史对话复用，后续请求可复用已计算的相同文本前缀。自动规划不会为了扩大专家缓存而关闭复用；完整 KV 放不下时，调整专家驻留、缓存和设备配置，仍无可行方案则在加载前报错，不自动缩短上下文。

只有显式设置 `--prefix_cache false --cache_history false` 时，自动规划才考虑精确 QSA KV streaming；对应 JSON 配置和现有 `FASTLLM_PREFIX_CACHE` 环境变量也遵循这一规则。该模式不复用跨请求的前缀和历史，但保留原生注意力选页。适用范围为现有支持的 Qwen4 FP16 QSA、单请求混推路径，可使用单卡、PP 或 TP；不启用其他模型的近似 KVMem。跳过 warmup、指定其他精度、多请求或指定不等于单请求上下文的总 token 容量时，不自动启用。显式 `--kvmem` 仍需配合原有手动参数。

允许 streaming 时，上下文容量达到 64K 后优先使用每层 32K 显存池，将腾出的显存用于专家缓存或驻留。若各卡仍能容纳自己负责的全部专家，完整显存 KV 也参与延迟比较。32K 池不可行时再尝试 16K、8K 池；较短上下文仅在完整 KV 放不下时考虑 streaming。所有方案保留逻辑上下文容量；完整 FP16 KV 放在 pinned 主存，索引仍覆盖全部上下文。预算包括主存 KV、GPU 页表和 prefill 临时使用的一整层 KV，保留原有显存安全余量。主存专家加 KV streaming 的优先级高于磁盘专家；主存不足或 streaming 不可行时仍可退回完整显存 KV。最终 `[auto]` 日志中的 `automatic.kv_streaming` 和 `kvmem` 给出实际选择。

这些是快速估计规则，不是实测吞吐或完整调优：不会为了挑选配置反复加载模型。缓存局部性、PCIe 和算子效率的估计有误差；先应用上述 KV 容量策略，再比较预计延迟。估计延迟相差不超过 10% 时，优先选择较少设备和集合通信的方案；同一拓扑在显存能容纳全部专家时优先保留完整 KV。实际请求也可能更适合另一种模式。Qwen3.5 稠密 HF 模型自动搜索双卡和四卡等比例 TP；DeepSeek V4.1 暂不自动搜索 TP；其余模型的自动 TP 搜索仍限双卡、相近规格。多卡 PP 和显式等比例 TP 继续沿用现有预算校验。

Qwen4 的纯自动选型优先采用可行的单卡或 PP 方案，减少混推解码中的逐层 TP 归并；同一主存后端没有可行 PP 配置时，仍可退回 TP。避免磁盘专家的优先级高于 PP 偏好：先搜索主存专家后端的驻留、缓存和 chunk 组合，包括可行的 TP，全部放不下才考虑磁盘专家；PLE/Engram 按需读取独立于专家后端。显式 `--tp` 优先，其他模型继续按延迟估计选择。

Qwen4 自动混推在主存充足时优先将目标专家全部放在主存，把可用显存用于动态专家缓存；主存不足时仍可选择部分整层驻留，避免过早使用磁盘专家。能全驻留 GPU 的模型仍保留全 GPU 方案。PP 的自动缓存按每张卡的权重、KV、MTP 和工作区分别核算，不再由最紧张的卡统一限制；`moe_cuda_cache_per_device` 给出各卡字节预算，`moe_cuda_cache` 为其中的最大值。显式缓存和显式驻留仍优先；TP 缓存仍使用统一的每卡预算，其他模型保持原有选择规则。

显存预算保留 `gpu_mem_ratio` 对应的预留量（DeepSeek V4.1 的 `--auto` 混推默认总显存的 5%，其他场景默认 10%），并另外计入每卡 512 MiB 的 Graph、库句柄和分配器保护空间；不会将实测剩余显存全部分配给专家缓存。Qwen 的预算还包括 Graph 默认启用的 CUDA embedding，GGUF 词嵌入按实际展开后的 FP32 大小计算；多卡 PP MTP 自动采用 CPU embedding 时免去对应显存开销。这是容量估计，不是运行时显存硬上限；CUDA 分配器可能保留可复用的空闲缓冲，因此驱动看到的显存峰值可能高于规划估算。实际余量仍取决于请求形状和运行时分配。

主存预算从检测到的可用主存中扣除总主存的 10%，预留量最低 1 GiB、最高 4 GiB；显式 `--auto_host_memory_limit` 仍可进一步收紧预算。加载暂存、运行工作区和 KV 镜像另计入预计用量。这是规划预算，不是进程内存硬上限。

需要突破自动检测上限时，可显式指定最终主存预算：

```sh
ftllm server /path/to/model.gguf --auto --force_ddr_limit 114g
```

`--force_ddr_limit`（别名 `--force-ddr-limit`）将最终预算直接设为指定值。`114g` 即 114 GiB，不再受检测到的可用主存或 cgroup 余量约束，也不再扣除 CMA 和自动预留；权重、KV、工作区、加载暂存和专家缓存的预计用量仍必须装进这个预算。它与只允许收紧预算的 `--auto_host_memory_limit` 互斥；省略或设为 `auto` 时保持原有检测行为，支持与其他内存参数相同的容量单位。

该参数仅改变规划，不改变真实可分配内存、CMA 配置或系统／容器硬限制；预算过大仍可能导致分配失败或 OOM。`[auto]` 中的 `host_budget_bytes` 是实际采用的预算，`automatic.force_ddr_limit` 记录显式指定值。逐项自动预算和 `--autotune` 的候选规划也使用此值；autotune 实测时的低内存和 swap 保护仍保留。

DeepSeek-V4.1 的单卡／PP 自动规划在无法让专家全部常驻显卡和主存时，会沿用已选的 GPU 驻留前缀、chunk 和上下文，再估算 `GPU → NUMA → disk` 的连续层分界。`--auto_disk_cache_ratio`（也可写 `--auto-disk-cache-ratio`）控制剩余 disk 专家权重可被主存缓存覆盖的推荐最低比例，默认 `0.5`，范围为 `0` 到 `1`：

```sh
ftllm server /path/to/model.gguf --auto --auto_disk_cache_ratio 0.5
```

计算比例为 **disk 主存专家缓存字节数 / 剩余 disk 专家权重预估字节数**，不是 GPU 缓存比例或实测命中率。在满足该比例、原有显存和主存预算的方案中，选择最多 NUMA 层；较小的值允许更多 NUMA 常驻层，较大的值给 disk 留更多主存缓存。`0` 取消最低覆盖比例，仍受容量预算约束。若没有任何混合分界能满足推荐比例，则保留原有 disk 回退并打印说明；`1` 不会强制启动一个装不下的方案。

预算同时包含两套后端的独立上传环、disk 主存 staging、加载暂存、KV、工作区与安全余量。剩余显存给 NUMA 和 disk 的独立专家缓存，其中 NUMA 缓存不超过每卡剩余缓存预算的一半或 3 GiB。`[auto]` 的 `numa_layers`、`moe_device_layers`、`disk_expert_bytes`、`disk_cpu_cache_ratio` 给出实际分界和覆盖率，`moe_cuda_cache_per_device` 是 NUMA 缓存，`moe_cuda_cache` 是 disk 缓存。能全部常驻时仍采用原有方案；显式指定 `moe_device`、`moe_device_layers`、`moe_cuda_cache` 或 `moe_cpu_cache` 时不自动改成混合放置。未编译 NUMA、其他模型及显式 TP 保持原行为。该规则只读元数据并估算容量，不运行测速，也不保证混合方案一定更快。

显式参数优先，包括关闭缓存的 `0`、全部主存专家的 `--moe_device_layers -1`、`--tp off`、线程数、精度和阈值。除上述带内置草稿权重的 Qwen3.5 稠密 HF 模型外，仅添加 `--auto` 不会启用 MTP，需指定 MTP 或提供草稿模型。若指定并发而未指定总 token 容量，会按上下文乘以并发预留。不会为了通过预算缩小用户指定的上下文或并发。JSON 配置中的设置也作为显式值处理；`--auto` 下同时给出的命令行设置优先于 JSON。

使用 `--auto --draft /path/to/mtp.gguf` 指定 Qwen4 外部 MTP GGUF 时，默认最多生成 3 个草稿 token，加上 anchor 为 4 行目标验证，匹配现有验证 CUDA Graph 的宽度。显式正数 `--mtp`、`--draft_tokens` 优先；阈值默认关闭，可用 `--mtp_min_p` 指定。保留草稿路径但关闭推测解码时，使用 `--speculative_algorithm off`。

Qwen4 混推开启 MTP 且自动选到多卡 PP 时，若未显式指定 embedding 策略，会启用现有 `low_gpu_mem` 配置，使用 CPU embedding，避免目标首级与草稿末级之间反复搬运整份词嵌入权重。显式 `--cuda_embedding`、`--low_gpu_mem` 或 JSON 中对应的布尔值保持优先；单卡、TP、纯 GPU 和其他模型不应用这条规则。

Qwen4 的跨卡 prefill 流水线要求驻留专家与对应的普通层位于同一张 GPU。跨阶段驻留时，保留原有双卡布局并采用串行分块 prefill；解码配置不变。全部主存专家或同阶段驻留仍可使用流水线。

已有线程环境变量仍遵循原生 NUMA 运行时的优先级：`FT_THREADS` 优先于 `FASTLLM_NUMA_THREADS`，后者按有效 NUMA 节点数换算总线程数。未显式指定线程时，自动规划采用同样的顺序；这些环境变量也可能覆盖命令行线程数。

```sh
# 自动选择其余参数，固定 16K 上下文和双请求容量
ftllm server /path/to/model.gguf --auto --max_context_length 16384 --max_batch 2

# 固定 PP 模式，其余自动选择
ftllm server /path/to/model.gguf --auto --device cudapp=2
```

启动打印 `[auto]`，给出最终模式、设备、线程、chunk、驻留、缓存、上下文及内存预算；`model.hybrid_auto_plan` 保存同一结果。权重精度、路由数学和采样算法保持原有实现。

## 不依赖 NUMA 的 Qwen4 GGUF prefill 分流

单卡 CUDA 配合 `--moe_device cpu` 时，Qwen4 的普通 CPU 常驻 GGUF 专家也能在 prefill 中分流到 CPU/GPU，不要求 `USE_NUMAS`。Windows 可以继续使用 `-DUSE_NUMAS=OFF`；无需开启 NUMA 分配、绑核或专家显存缓存。GPU 常驻层维持原有放置，engram 设备配置不变。

此路径支持 33–4096 行的 FP32/FP16/BF16 激活和现有 CUDA GGUF 反量化格式。根据每个专家的路由量与 CPU/GPU 两支的耗时反馈选择阈值，GPU 临时上传高路由量专家，CPU 同时计算其余专家，最后合并两支结果。CPU 重排权重先完成重排再交给 GPU 上传，gate/up 继续使用普通 CPU 的分半布局。当前入口限于无专家 bias 的串行 Qwen4 混推；TP、NUMA 已注册权重、磁盘专家和小批量 decode 保持各自原有路径。

`FT_GPU_PREFILL=0` 关闭这一分流；`FT_EXPERT_LIMIT=<n>` 显式指定 GPU 专家的最低路由量（`0` 让所有活跃专家走 GPU），不设置时使用耗时反馈。临时上传不写入专家缓存。CPU 与 CUDA 的 GGUF 激活精度不同，合并结果不保证逐位一致。

诊断时设置 `FASTLLM_PROFILE_HOST_MOE=1`，可记录每层两支的专家数、路由数和耗时；可用 `FASTLLM_PROFILE_HOST_MOE_PATH` 指定日志文件。Windows 构建还可设置 `FASTLLM_NVTX=1`，在 Nsight Systems 中查看 `host_moe_prefill/layer/<id>`、`host_moe_prefill/cpu` 和 `host_moe_prefill/gpu` 范围，确认 CPU 层的 GPU 专家计算及两支重叠情况。

## 不依赖 NUMA 的 Qwen4 GGUF decode 分流与缓存

普通 CPU 常驻 GGUF 专家也可使用单行 FP32 decode 的 CPU/GPU 动态调度。缓存命中的专家在 GPU 计算，未命中的专家根据各层 CPU、传输和 GPU 计算的实测耗时分配到 CPU 或 GPU 临时工作区；临时 GPU 计算与缓存准入互相独立。GPU 专家缓存沿用全局热度、容量和替换策略，在完整 token 结束后准入热点专家。

此路径在 `USE_NUMAS=OFF` 的 Windows 构建中可用。CPU 权重先完成原地 R4 重排，按需复制到有大小上限的页锁定上传暂存区，再传到显卡，不注册整份模型的内存页，也不创建第二份完整专家权重。普通 CPU gate/up 的分半行序与 NUMA 的交错行序分别描述，显卡缓存中的记录还原为原始 GGUF 布局。上传暂存区由缓存或临时执行工作区持有，等待传输完成后复用，权重仍由模型持有。

`--moe_cuda_cache 512m` 指定显卡专家缓存预算；`0` 保留动态调度和临时 GPU 计算，但不保存常驻专家。显存不足时需减少 GPU 常驻专家层，为缓存和上传工作区留出空间。单卡 Qwen4 的 Q2_0、IQ2_S、IQ2_XXS、IQ1_M GGUF 自动内存规划也按这一份 CPU 权重预算。其他模型、TP、NUMA 已注册布局及不同激活类型继续使用各自已有路径。

默认启用；`FASTLLM_HOST_MOE_DECODE=0` 旁路普通 CPU 权重的 decode 分流和缓存执行，供同进程对照使用（已经分配的缓存仍占显存）。`FASTLLM_PROFILE_HOST_MOE_DECODE=1` 输出每层的 topk、命中、GPU/CPU 路由、临时上传路由、槽位及 CPU 耗时，可用 `FASTLLM_PROFILE_HOST_MOE_DECODE_PATH` 指定文件。`FASTLLM_NVTX=1` 提供 `host_moe_decode/layer/<id>`、`host_moe_decode/cpu` 和 `host_moe_decode/gpu_submit` 范围。请求之间可读取 `get_moe_cuda_cache_stats` 和 `get_moe_cuda_cache_route_stats`，取前后差值确认缓存命中和分流；统计读取会同步 GPU，不放入测速区间。

CPU 协调任务复用每个推理调用线程自己的专用 worker，不占用其内部矩阵乘法线程池，也不逐层创建和销毁线程。GPU 提交异常时仍等待 CPU 任务结束，再向调用者传播异常。

Windows WDDM 可能将异步 kernel 留在软件提交队列，直到 CPU 结果汇合后的同步才实际执行。普通 CPU/GPU 混合 decode 在 GPU 分支提交后调用非阻塞的 `cudaStreamQuery`，让已提交工作及时进入设备队列；临时上传存在时也提交其 copy stream。它不等待 GPU 完成，不替代最终结果依赖。只在 CPU 与 GPU 都有专家路由时执行；`FASTLLM_HOST_MOE_WDDM_FLUSH=0` 可关闭此提交步骤作对照，不影响 NUMA 构建设置。

排查并行时，区分 `host_moe_decode/gpu_submit` 的主机提交与设备 kernel 的实际起止时间。`host_moe_decode/flush_gpu`、`wake_cpu`、`wait_cpu` 标记队列提交及协调等待；`cpu_moe/dot` 位于 CPU 线程池实际的 GGUF 矩阵乘函数内，可与 CUDA `Q8Projection` kernel 对齐检查计算区间重叠，避免仅凭两个提交范围判断并行。

## 混合推理共用的 prefill 专家缓存

CUDA/NUMA 混推在有专家显存缓存时，默认让 prefill 和 decode 复用同一份缓存与热度策略。这个运行时能力也适用于手动配置，不依赖 `--auto`。Qwen 标准 MoE、DeepSeek-V4.1、GLM-5.3 共用权重传输层：原生布局一致时直接借用显存记录；布局不一致时在 GPU 上还原原有 NUMA 字节布局，包括 gate/up 行序和 FP8/NVFP4 scale。未命中的专家继续临时上传，符合热度策略的记录写入有容量上限的缓存。

原生格式的缓存已有 decode 历史时，prefill 保留已有记录、decode 热度和更新周期，只按当前路由热度填充空槽。已满缓存的替换交给后续 decode 观察决定，避免长提示词覆盖 decode 工作集、导致 prefill 提升却生成变慢。冷缓存仍可按原有 prefill 热度策略装入专家；GGUF 保留原有阶段策略。

支持现有共享 NUMA 缓存注册的标准 FP8（block128/per-channel）、NVFP4（float-scale/planar/compact）、DeepSeek-V4.1 FP8 block128/NVFP4 block32，以及 GLM-5.3 原生 FP8 block128/compact NVFP4。GGUF 保留原有 grouped prefill 路径。没有可用缓存或布局不受支持时，继续执行原有主存路径。

调度器根据整层 CPU/GPU 并行执行的实测耗时调整分流；原生 DeepSeek-V4.1 和 GLM-5.3 也使用这一反馈路径。原生逐专家路径中，低路由量的已驻留专家仍可分配到 CPU，避免缓存命中引入过多小算子开销；分配到 GPU 的已驻留专家使用拥有其记录的设备。显式 `FT_EXPERT_LIMIT` 固定 CPU/GPU 分流阈值，但不固定多个 GPU 之间的专家分配；后者仍可依据实测耗时调整，BF16 分组归约不保证跨次逐位一致。辅助 GPU 可以读取已有缓存，写入仍遵循原有 owner/admission 规则。

专家缓存优化无需开启 `--fast_prefill`。缓存命中仍需要计算；需要布局还原时还会有 GPU 内存读写，因此 prefill 加速幅度取决于缓存覆盖率、PCIe 搬运与算子耗时。

排查或对照测试可设置 `FASTLLM_MOE_PREFILL_CACHE=0`，同时旁路 prefill 的驻留调度、缓存读取和写入，不关闭 decode 缓存；正常运行无需设置。在两次请求之间读取 `ftllm.llm.get_moe_cuda_cache_prefill_stats(device=0)` 并相减，可得到共享原生路径的调用数、GPU 专家/路由数、命中专家/路由数、避免上传的字节数、实际上传字节数和新写入专家数。统计读取会同步 GPU，不能放入吞吐计时区间；不包含独立 grouped GGUF 后端，GPU 内部布局还原也不计入上传字节。

## DeepSeek 混推的分块预填充与校验

DeepSeek-V4.1 手动启动时加 `--fast_prefill` 开启快速 Prefill；使用 `--auto` 时默认开启，可加 `--no_fast_prefill` 关闭。启动器中展开“模型参数”，在“自定义参数”填写 `--fast_prefill` 或 `--no_fast_prefill`；删除对应参数恢复默认。

`FASTLLM_DSV41_HYBRID_PP=1` 显式启用混推跨块流水线，默认关闭。它要求 CUDA PP 层放置、NUMA 专家、`fast_prefill`、单请求 greedy，以及各级独立拥有其使用的共享 KV/index 来源；例如已验证的双卡 `--device cudapp=14:26 --tp off`。专家显存缓存、DSpark、历史复用或不兼容的层切分等情况会沿用普通路径。该入口不用于纯 GPU 或 TP 执行。

DeepSeek-V4.1 的 GGUF 专家缓存支持 `IQ1_S`、`IQ1_M`、`Q2_0`、`Q2_K`、`Q3_K`、`Q4_K` 的 gate/up、down 组合及不同层的混合格式，保留原量化字节。NUMA 的交错行和 R4 重排在入缓存时恢复为标准量化排列，无需 FP16 展开或额外主存权重副本；原 `Q2_K_R4`／`Q4_K_R4` 快速路径保留。部分专家层驻留 GPU 时，仅将其余 NUMA 层注册到缓存。开启 Graph 的单 token 解码和 DSpark 2～8 token 验证均可使用缓存。`--auto` 另外预留 GGUF 主存 prefill 的原始／还原上传环（3 × 16 × 2 个专家记录）和 256 MiB 预取空间及分配对齐余量，避免它们与 decode 缓存争用显存。

DeepSeek-V4.1 原生 FP8/NVFP4 专家缓存随 `--moe_cuda_cache` 启用，不需要额外的频率策略开关。单卡 DSpark 的 2～8 token 校验在 Graph 开启时也使用现有缓存和动态分流，保持图段的缓冲地址；无可用 GPU 专家时保留 NUMA 回退。`get_moe_cuda_cache_route_stats` 同时统计普通解码与校验的路由，需在请求结束后读取。DeepSeek-V4.1 的 `--auto` 支持并默认开启内置 DSpark：所有启用草稿层的权重、BF16 滑窗状态、目标层 hidden 特征和验证临时空间均计入预算。草稿稠密部分按最后一张 PP 卡核算（TP 使用保守复制上界），草稿专家沿用最后 N 个主干层的专家放置；不会为纯滑窗草稿分配整段 1M KV。外置 DSpark 仍需手动配置。

## DeepSeek FP8 WoA 的通用 prefill GEMM

DeepSeek-V4/V4.1 的 CUDA WoA 投影在 token 数不小于 16 时，默认尝试分组 GEMM，并按设备能力使用 Tensor Core；混合推理和全 GPU 推理共用入口，不依赖 `--auto`、专家缓存或 `fast_prefill`。支持可变 token 数、batch、分组数和输出 rank，以及 FP8 block32、block128 和矩形量化块；非对齐的 K/N、token 尾块和不足 128 个元素的输入量化尾块均有边界处理，不要求固定 SM 数。原有 SM120 的 7/8-token 特化仍保留；其他小批量走原有归约路径，奇数 rank/量化行块走逐行归约。

WoA 根据实际设备能力和数值条件选择计算路径，所有路径都保留已有的 FP8 权重解码到 FP16 的舍入：

- SM80 及以上，BF16 输入且所有权重 scale 均为 float32 正规数表示的 2 的整数幂或正零时，解码结果可以精确表示为 BF16。直接使用 BF16 Tensor Core、FP32 累加并输出 BF16，省去输入展开和独立输出转换。每次调用检查实际 scale，不使用可能失效的指针缓存。
- SM80 及以上的其余 BF16/FP16 输入，将输入和解码权重精确展开到 FP32，使用 TF32 Tensor Core 和 FP32 累加，再舍入到 BF16。FP16 权重与 BF16/FP16 输入都能精确表示为 TF32，不额外损失乘法输入精度。
- SM75 使用 pedantic FP32 cuBLAS GEMM；未量化的 FLOAT32 输入也使用该路径，避免 TF32 截断。更早设备或 cuBLAS 拒绝时继续使用通用归约实现。

SM120 上仍保留原有 WoA 输入 E4M3/UE8M0 量化。行宽能被 128 整除时，每个 warp 处理一组 128 个元素，用 shuffle 完成最大值归约、一次计算并广播 scale，再进行四元素 FP8 转换；非对齐行继续走有边界检查的实现。GEMM 的累加顺序与原有 SIMT 归约不同，结果允许 BF16 精度内的数值差异，不承诺投影逐位一致。SM120 以下仍按原有行为不执行这一步输入量化。

展开后的权重只在本次调用中使用，不建立跨请求的常驻权重缓存。真实 G=8、N=1024、K=4096 形状的 BF16 路径只需 64 MiB 权重工作区，直接读取输入、写入输出，默认每次 GEMM 处理最多 4096 行。TF32/FP32 路径另需 128 MiB 权重工作区，token 工作区按最多 1024 行、通常不超过 160 MiB 分块（单行超过该预算时仍保留一行）。它们由现有 CUDA 内存池管理和复用，不能等同于请求结束立即归还驱动的显存。输入量化和最终输出的现有存储另计。

无需新增启动参数。A/B 排查可设置以下环境变量，恢复时取消变量：

- `FASTLLM_DSV4_DISABLE_CUDA_WOA_GEMM=1`：关闭全部 WoA GEMM，回退到 SIMT；不关闭专家缓存或输入量化。
- `FASTLLM_DSV4_DEBUG_CUDA_WOA_GEMM=1`：输出命中形状及 cuBLAS 拒绝状态。

构建 `deepseekV4WoAGemmRegression` 后运行可检查 FP64 独立参考、小批量回退、奇数维度、矩形量化块、输入范围、输入量化和真实投影形状；附加 `--bench` 输出SIMT 回退与自动 GEMM 路径的含同步交替测速中位数。旧架构验证需使用对应的 `compute_75/80/89` 编译 WoA 并测试对应能力分支；在新 GPU 上 JIT 执行旧 PTX 只能验证兼容路径，不能代表旧 GPU 的实际吞吐，也不能替代完整模型在旧 GPU 上的部署验证。

## 逐项自动预算

Qwen4、GLM-5.3 和 DeepSeek-V4.1 的混合推理命令支持以下可选值：

| 参数 | `auto` 的含义 |
| --- | --- |
| `--moe_cuda_cache auto` | 扣除权重、KV、固定状态、工作区和预留空间，计算每卡专家缓存上限 |
| `--moe_cpu_cache auto` | `moe_device=disk` 时按有效主存余量计算专家缓存上限；CPU/NUMA 常驻专家不另分配此缓存 |
| `--moe_device_layers auto` | 加载前规划 GPU 驻留前缀，将后面的专家层留在指定的主机设备 |
| `--chunked_prefill_size auto` | 在 4096、2048、1024、512、256 中选择满足内存预算的切片；DeepSeek-V4.1 所有主机专家后端额外优先考虑 8192，且不超过上下文长度 |
| `--auto_gpu_memory_limit 20g` | 将自动规划的每张 GPU 显存预算限制为不超过 20 GiB |
| `--auto_host_memory_limit 128g` | 将自动规划的主机内存预算限制为不超过 128 GiB |
| `--force_ddr_limit 114g` | 强制将最终主存预算设为 114 GiB，允许超过自动检测上限；与 `--auto_host_memory_limit` 互斥 |

只有写成 `auto` 的参数才参与自动选择。原有数值及默认值保持不变。
`moe_device_layers` 的数值仍表示**末尾使用 `moe_device` 的层数**，不是 GPU 驻留层数；`-1` 表示全部专家层使用该设备。

```sh
ftllm server /path/to/model.gguf \
  --device cuda --tp 2 --moe_device numa --ngram_device disk \
  --tokens 16384 --max_context_length 16384 --max_batch 1 \
  --moe_cuda_cache auto --moe_device_layers auto \
  --chunked_prefill_size auto
```

PP 使用 `--device cudapp=2` 并省略 `--tp`。也可以只自动设置缓存，保留明确的 chunk 和驻留配置：

```sh
ftllm server /path/to/model.gguf \
  --device cudapp=2 --moe_device numa --ngram_device disk \
  --tokens 16384 --max_batch 1 --chunked_prefill_size 512 \
  --moe_cuda_cache auto
```

主存不足时，可显式选择磁盘专家及自动主存缓存。当前 Qwen4 GGUF 磁盘专家的 GPU 缓存尚不支持，自动 CUDA 缓存会设为 0 并说明原因；主存缓存仍按预算分配：

```sh
ftllm server /path/to/model.gguf \
  --device cuda --moe_device disk --ngram_device disk \
  --tokens 16384 --max_batch 1 --chunked_prefill_size 512 \
  --moe_cuda_cache auto --moe_cpu_cache auto
```

HF packed FP4 在 `auto`/`nvfp4` 下按配对的权重与 scale 元数据核算，包含运行时 scale、行对齐和布局转换开销，已合入的 scale 不重复计费。E4M3 取带行全局 scale/对齐的紧凑布局与 FP32 block scale 回退中较大的预算；E8M0 保留 block16 兼容布局的容量上界，以覆盖 CPU 能力或开关导致的回退，不再统一将 packed payload 翻倍。未识别的布局和显式精度转换仍采用保守上界。

规划器读取 GGUF 全部分片的张量目录，或 HF safetensors 的索引和头部，不读取权重正文。GPU 预算采用 CUDA 的逻辑设备编号，因此遵循 `CUDA_VISIBLE_DEVICES`；主存预算同时检查 Linux `MemAvailable`、当前及各级可见父 cgroup 的内存限制与占用，不将 swap 计为可用内存。

上下文、KV 精度、MTP、阈值和用户指定的并发数不参与自动降级。`--tokens` 是整体 KV token 容量；未指定时，按 `--max_context_length` 或模型原生窗口预留。自动规划下未指定 `--max_batch` 时取 1，并打印结果。大窗口放不下会在加载前报错，需要用户明确调整参数。

联合规划优先选择可容纳的 chunk，再比较 GPU 上能存放的专家总量；容量接近时优先减少主存专家。PP 按普通层和驻留专家各自的分配规则核算，TP 对专家分片核算、对公共权重和状态保守预留。Qwen 的常驻 GPU 草稿专家及 GLM 的主机草稿专家分别计入。逐项自动预算仍使用统一的每卡缓存上限；统一 `--auto` 的 Qwen4 主存专家 PP 使用上述分卡预算。两种方式都不自动设置缓存环境变量。

启动时的 `[hybrid auto]` 输出包括具体参数、每卡预算及估计占用、主存预算及估计占用。Python 返回的模型还可通过 `model.hybrid_auto_plan` 查看结果。估计包含加载中间缓冲和运行工作区，不能视作实测峰值；其他进程在规划后占用内存仍可能导致分配失败。原有运行时容量检查继续生效。

这是一套内存容量规划规则，不是吞吐最优配置搜索。它复用现有缓存、动态 CPU/GPU 分流、Graph 和 TP/PP 执行路径，不改变算子、路由数学或采样策略。改变缓存或驻留配置可能选用现有的不同计算后端，其数值特性与手动使用相同参数一致。

当前支持 Linux、等比例 TP、单卡及加权 CUDA PP 设备映射。CUDA 自动缓存按模型支持现有的 GGUF、FP8、packed FP4 专家格式；GLM/DeepSeek 的主存专家 CUDA 缓存要求 NUMA 和 GGUF/native FP8 E4M3/native packed FP4，普通 CPU 专家仍使用显式缓存参数。HF 精度转换按保守上界核算。除统一 `--auto` 的 Qwen4 精确 KV streaming 外，显式 KVMem、自定义 KV 字节上限、异比例 TP、视觉、自定义模型/权重转换、LoRA、RoPE 扩展及 MTP/内置 DeepSeek-V4.1 DSpark 以外的推测算法暂不参与自动规划，仍可使用原有显式参数。自动规划也不能与每卡缓存环境变量覆盖同时使用。逐项自动预算要求混合设备配置；统一 `--auto` 可以在预算允许时选择专家全驻留 GPU，但不选择纯 CPU 执行。
