# 混合推理自动配置

使用 `--auto` 可以一次选择完整的 CUDA 运行配置：

```sh
ftllm server /path/to/model.gguf --auto
```

支持 Linux NVIDIA CUDA 上的 dense Qwen3，以及 Qwen4、GLM-5.3、DeepSeek-V4.1 混推模型。未启用 `--auto` 时，原有数值和默认行为保持不变。

Qwen3.8 27B 使用 Qwen3.5 架构，目前不属于上述 dense Qwen3 支持范围；请使用[手动 CUDA/TP 配置](qwen3.md)，或 Launcher 原有的长上下文/高并发推荐。

直接运行 `ftllm` 打开的 Launcher 中，新增启动项默认选择“自动配置”，只需选择本地模型并填写模型名、监听地址、端口等服务信息。启动命令使用 `--auto`，每次启动重新读取硬件余量；表单中原有的手动推理参数和环境变量不会带入这个模式。原有长上下文、高并发和自定义模式仍保留，已保存的配置不会自动切换。

自动配置设置中可以填写可选的“显存上限（每卡）”和“内存上限”。它们分别对应
`--auto_gpu_memory_limit` 与 `--auto_host_memory_limit`，接受 `20g`、`128g` 等二进制容量；留空时继续读取启动时的实际可用显存和主机内存。显存上限作用于每张可见 GPU，主机内存上限作用于整个自动规划预算。这是规划上限，不会限制操作系统或其它进程的实际内存分配；规划结果不足以容纳模型时会在加载前报错。

Dense Qwen3 按实际 GQA 头数、KV、dense 权重及 tied output head 副本预算，不分配专家缓存；上下文取 128K 与模型原生窗口的较小值。目前 Qwen3 的 CUDA Graph 预热会触发 deferred pool 的 stream-capture-unsupported，因此此模型的 `--auto` 默认使用普通 CUDA 执行。显式 `FASTLLM_CUDA_GRAPH` 仍优先；其他模型的 Graph 默认策略不变。

## 单参数自动选择

DeepSeek-V4.1 默认采用不超过模型原生窗口的 **1M（1,048,576 token）上下文、单请求**，开启 **DSpark 7**（置信度阈值默认 0.5）和 **fast_prefill**。其他模型保持 **128K 上下文、单请求**。按代表性的 4K 输入、512 输出估计交互延迟。读取可见 CUDA 设备的可用显存、SM 数、内存带宽参数、PCIe 条件和 peer access 能力，比较单卡、PP、双卡等比例 TP，并联合选择专家驻留、显存缓存、主存缓存和 chunk。权重可全驻留且预算允许时，也可选择全部专家放 GPU。CPU 线程依据进程可用的物理核数选择，并预留启动和 IO 线程。

DeepSeek-V4.1 的 `fast_prefill` 沿用已有近似注意力语义，可能改变 logits。可用 `--no_fast_prefill` 关闭；`--dspark 0` 或 `--speculative_algorithm off` 关闭推测解码。显式上下文、草稿数、阈值和 JSON 配置中的 `fast_prefill: false` 均优先；`--threads 64` 可固定线程数。预算不足时先调整自动 chunk、缓存和专家放置，仍放不下则报错，不自动缩短 1M 上下文。

主存优先使用 NUMA（构建支持时），否则使用 CPU；超大 PLE/Engram 表优先按需读取，减少启动搬运和主存占用，主存仍放不下时再考虑磁盘专家缓存。CUDA Graph 默认开启，原有 `FASTLLM_CUDA_GRAPH` 显式设置仍优先。原有运行时检查继续决定 Graph 能否实际捕获。

Qwen4 混推在上下文容量达到 64K 时，优先使用每层 32K 显存池的精确 QSA KV streaming，将腾出的显存用于专家缓存或驻留。计入完整 KV、工作区和安全余量后，若各卡仍能容纳自己负责的全部专家（驻留或缓存），则完整显存 KV 也参与延迟比较。32K 池不可行时再尝试 16K、8K 池；较短上下文仅在完整 KV 放不下时考虑 streaming。所有方案保留逻辑上下文容量；完整 FP16 KV 放在 pinned 主存，索引仍覆盖全部上下文。预算包括主存 KV、GPU 页表和 prefill 临时使用的一整层 KV，保留原有显存安全余量。主存专家加 KV streaming 的优先级高于磁盘专家；主存不足或 streaming 不可行时仍可退回完整显存 KV。若仍放不下，加载前报错，不自动缩短上下文。

自动 KV streaming 仅用于现有支持的 Qwen4 FP16 QSA、单请求混推路径，可使用单卡、PP 或 TP；不启用其他模型的近似 KVMem。它关闭 prefix/history 复用，但保留原生注意力选页；模型检测自动填入的历史缓存默认值不会阻止规划。用户通过命令行、JSON 或现有 prefix-cache 环境变量显式开启复用，或跳过 warmup、指定其他精度、多请求、指定不等于单请求上下文的总 token 容量时，不自动启用。显式 `--kvmem` 仍需配合原有手动参数。最终 `[auto]` 日志中的 `automatic.kv_streaming` 和 `kvmem` 给出实际选择。

这些是快速估计规则，不是实测吞吐或完整调优：不会为了挑选配置反复加载模型。缓存局部性、PCIe 和算子效率的估计有误差；先应用上述 KV 容量策略，再比较预计延迟。估计延迟相差不超过 10% 时，优先选择较少设备和集合通信的方案；同一拓扑在显存能容纳全部专家时优先保留完整 KV。实际请求也可能更适合另一种模式。自动 TP 搜索目前限双卡、相近规格；多卡 PP 和显式等比例 TP 继续沿用现有预算校验。

Qwen4 的纯自动选型优先采用可行的单卡或 PP 方案，减少混推解码中的逐层 TP 归并；同一主存后端没有可行 PP 配置时，仍可退回 TP。避免磁盘专家的优先级高于 PP 偏好：先搜索主存专家后端的驻留、缓存和 chunk 组合，包括可行的 TP，全部放不下才考虑磁盘专家；PLE/Engram 按需读取独立于专家后端。显式 `--tp` 优先，其他模型继续按延迟估计选择。

Qwen4 自动混推在主存充足时优先将目标专家全部放在主存，把可用显存用于动态专家缓存；主存不足时仍可选择部分整层驻留，避免过早使用磁盘专家。能全驻留 GPU 的模型仍保留全 GPU 方案。PP 的自动缓存按每张卡的权重、KV、MTP 和工作区分别核算，不再由最紧张的卡统一限制；`moe_cuda_cache_per_device` 给出各卡字节预算，`moe_cuda_cache` 为其中的最大值。显式缓存和显式驻留仍优先；TP 缓存仍使用统一的每卡预算，其他模型保持原有选择规则。

显存预算保留 `gpu_mem_ratio` 对应的预留量（默认总显存的 10%），并另外计入每卡 512 MiB 的 Graph、库句柄和分配器保护空间；不会将实测剩余显存全部分配给专家缓存。Qwen 的预算还包括 Graph 默认启用的 CUDA embedding，GGUF 词嵌入按实际展开后的 FP32 大小计算；多卡 PP MTP 自动采用 CPU embedding 时免去对应显存开销。这是容量估计，不是运行时显存硬上限；CUDA 分配器可能保留可复用的空闲缓冲，因此驱动看到的显存峰值可能高于规划估算。实际余量仍取决于请求形状和运行时分配。

显式参数优先，包括关闭缓存的 `0`、全部主存专家的 `--moe_device_layers -1`、`--tp off`、线程数、精度和阈值。仅添加 `--auto` 不会启用 MTP，需指定 MTP 或提供草稿模型。若指定并发而未指定总 token 容量，会按上下文乘以并发预留。不会为了通过预算缩小用户指定的上下文或并发。JSON 配置中的设置也作为显式值处理；`--auto` 下同时给出的命令行设置优先于 JSON。

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

## 混合推理共用的 prefill 专家缓存

CUDA/NUMA 混推在有专家显存缓存时，默认让 prefill 和 decode 复用同一份缓存与热度策略。这个运行时能力也适用于手动配置，不依赖 `--auto`。Qwen 标准 MoE、DeepSeek-V4.1、GLM-5.3 共用权重传输层：原生布局一致时直接借用显存记录；布局不一致时在 GPU 上还原原有 NUMA 字节布局，包括 gate/up 行序和 FP8/NVFP4 scale。未命中的专家继续临时上传，符合热度策略的记录写入有容量上限的缓存。

原生格式的缓存已有 decode 历史时，prefill 保留已有记录、decode 热度和更新周期，只按当前路由热度填充空槽。已满缓存的替换交给后续 decode 观察决定，避免长提示词覆盖 decode 工作集、导致 prefill 提升却生成变慢。冷缓存仍可按原有 prefill 热度策略装入专家；GGUF 保留原有阶段策略。

支持现有共享 NUMA 缓存注册的标准 FP8（block128/per-channel）、NVFP4（float-scale/planar/compact）、DeepSeek-V4.1 FP8 block128/NVFP4 block32，以及 GLM-5.3 原生 FP8 block128/compact NVFP4。GGUF 保留原有 grouped prefill 路径。没有可用缓存或布局不受支持时，继续执行原有主存路径。

调度器根据整层 CPU/GPU 并行执行的实测耗时调整分流；原生 DeepSeek-V4.1 和 GLM-5.3 也使用这一反馈路径。原生逐专家路径中，低路由量的已驻留专家仍可分配到 CPU，避免缓存命中引入过多小算子开销；分配到 GPU 的已驻留专家使用拥有其记录的设备。显式 `FT_EXPERT_LIMIT` 固定 CPU/GPU 分流阈值，但不固定多个 GPU 之间的专家分配；后者仍可依据实测耗时调整，BF16 分组归约不保证跨次逐位一致。辅助 GPU 可以读取已有缓存，写入仍遵循原有 owner/admission 规则。

该优化保留现有投影、激活、量化和归并算子，不启用 `fast_prefill` 的近似注意力。缓存命中仍需要计算；需要布局还原时还会有 GPU 内存读写，因此 prefill 加速幅度取决于缓存覆盖率、PCIe 搬运与算子耗时。

排查或对照测试可设置 `FASTLLM_MOE_PREFILL_CACHE=0`，同时旁路 prefill 的驻留调度、缓存读取和写入，不关闭 decode 缓存；正常运行无需设置。在两次请求之间读取 `ftllm.llm.get_moe_cuda_cache_prefill_stats(device=0)` 并相减，可得到共享原生路径的调用数、GPU 专家/路由数、命中专家/路由数、避免上传的字节数、实际上传字节数和新写入专家数。统计读取会同步 GPU，不能放入吞吐计时区间；不包含独立 grouped GGUF 后端，GPU 内部布局还原也不计入上传字节。

## DeepSeek 混推的分块预填充与校验

启用 `--fast_prefill` 后，DeepSeek-V4.1 对满足 bounded replay 条件的单请求分块预填充，只在最后一块执行后段 SWA 层与输出头；前面的块更新长期 KV。末块不超过 SWA 窗口、增量续写、图像输入或 DSpark 等情况沿用原路径。该优化沿用 `fast_prefill` 的近似注意力语义；该模型的 `--auto` 默认开启，手动启动仍需显式开启。

`FASTLLM_DSV41_HYBRID_PP=1` 显式启用混推跨块流水线，默认关闭。它要求 CUDA PP 层放置、NUMA 专家、`fast_prefill`、单请求 greedy，以及各级独立拥有其使用的共享 KV/index 来源；例如已验证的双卡 `--device cudapp=14:26 --tp off`。专家显存缓存、DSpark、历史复用或不兼容的层切分等情况会沿用普通路径。该入口不用于纯 GPU 或 TP 执行。

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
| `--chunked_prefill_size auto` | 在 4096、2048、1024、512、256 中依次选择满足内存预算的切片，且不超过上下文长度 |
| `--auto_gpu_memory_limit 20g` | 将自动规划的每张 GPU 显存预算限制为不超过 20 GiB |
| `--auto_host_memory_limit 128g` | 将自动规划的主机内存预算限制为不超过 128 GiB |

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
