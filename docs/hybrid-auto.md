# 混合推理自动配置

使用 `--auto` 可以一次选择完整的 CUDA 运行配置：

```sh
ftllm server /path/to/model.gguf --auto
```

支持 Linux NVIDIA CUDA 上 Qwen4、GLM-5.3、DeepSeek-V4.1 的现有混推模型。未启用 `--auto` 时，原有数值和默认行为保持不变。

## 单参数自动选择

默认采用不超过模型原生窗口的 **32K 上下文、单请求**，按代表性的 4K 输入、512 输出估计交互延迟。读取可见 CUDA 设备的可用显存、SM 数、内存带宽参数、PCIe 条件和 peer access 能力，比较单卡、PP、双卡等比例 TP，并联合选择专家驻留、显存缓存、主存缓存和 chunk。权重可全驻留且预算允许时，也可选择全部专家放 GPU。CPU 线程依据进程可用的物理核数选择，并预留启动和 IO 线程。

主存优先使用 NUMA（构建支持时），否则使用 CPU；超大 PLE/Engram 表优先按需读取，减少启动搬运和主存占用，主存仍放不下时再考虑磁盘专家缓存。CUDA Graph 默认开启，原有 `FASTLLM_CUDA_GRAPH` 显式设置仍优先。原有运行时检查继续决定 Graph 能否实际捕获。

这些是快速估计规则，不是实测吞吐或完整调优：不会为了挑选配置反复加载模型。缓存局部性、PCIe 和算子效率的估计有误差；估计延迟相差不超过 10% 时，优先选择较少设备和集合通信的方案。实际请求也可能更适合另一种模式。自动 TP 搜索目前限双卡、相近规格；多卡 PP 和显式等比例 TP 继续沿用现有预算校验。

显式参数优先，包括关闭缓存的 `0`、全部主存专家的 `--moe_device_layers -1`、`--tp off`、线程数、精度、MTP 和阈值。未指定 MTP 时仍关闭。若指定并发而未指定总 token 容量，会按上下文乘以并发预留。不会为了通过预算缩小用户指定的上下文或并发。JSON 配置中的设置也作为显式值处理；`--auto` 下同时给出的命令行设置优先于 JSON。

已有线程环境变量仍遵循原生 NUMA 运行时的优先级：`FT_THREADS` 优先于 `FASTLLM_NUMA_THREADS`，后者按有效 NUMA 节点数换算总线程数。未显式指定线程时，自动规划采用同样的顺序；这些环境变量也可能覆盖命令行线程数。

```sh
# 自动选择其余参数，固定 16K 上下文和双请求容量
ftllm server /path/to/model.gguf --auto --max_context_length 16384 --max_batch 2

# 固定 PP 模式，其余自动选择
ftllm server /path/to/model.gguf --auto --device cudapp=2
```

启动打印 `[auto]`，给出最终模式、设备、线程、chunk、驻留、缓存、上下文及内存预算；`model.hybrid_auto_plan` 保存同一结果。权重精度、路由数学和采样算法保持原有实现。

## 逐项自动预算

Qwen4、GLM-5.3 和 DeepSeek-V4.1 的混合推理命令支持以下可选值：

| 参数 | `auto` 的含义 |
| --- | --- |
| `--moe_cuda_cache auto` | 扣除权重、KV、固定状态、工作区和预留空间，计算每卡专家缓存上限 |
| `--moe_cpu_cache auto` | `moe_device=disk` 时按有效主存余量计算专家缓存上限；CPU/NUMA 常驻专家不另分配此缓存 |
| `--moe_device_layers auto` | 加载前规划 GPU 驻留前缀，将后面的专家层留在指定的主机设备 |
| `--chunked_prefill_size auto` | 在 4096、2048、1024、512、256 中依次选择满足内存预算的切片，且不超过上下文长度 |

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

规划器读取 GGUF 全部分片的张量目录，或 HF safetensors 的索引和头部，不读取权重正文。GPU 预算采用 CUDA 的逻辑设备编号，因此遵循 `CUDA_VISIBLE_DEVICES`；主存预算同时检查 Linux `MemAvailable`、当前及各级可见父 cgroup 的内存限制与占用，不将 swap 计为可用内存。

上下文、KV 精度、MTP、阈值和用户指定的并发数不参与自动降级。`--tokens` 是整体 KV token 容量；未指定时，按 `--max_context_length` 或模型原生窗口预留。自动规划下未指定 `--max_batch` 时取 1，并打印结果。大窗口放不下会在加载前报错，需要用户明确调整参数。

联合规划优先选择可容纳的 chunk，再比较 GPU 上能存放的专家总量；容量接近时优先减少主存专家。PP 按普通层和驻留专家各自的分配规则核算，TP 对专家分片核算、对公共权重和状态保守预留。Qwen 的常驻 GPU 草稿专家及 GLM 的主机草稿专家分别计入。双卡仍使用同一个每卡缓存上限，因此取显存较紧张的卡的预算，不会自动设置环境变量。

启动时的 `[hybrid auto]` 输出包括具体参数、每卡预算及估计占用、主存预算及估计占用。Python 返回的模型还可通过 `model.hybrid_auto_plan` 查看结果。估计包含加载中间缓冲和运行工作区，不能视作实测峰值；其他进程在规划后占用内存仍可能导致分配失败。原有运行时容量检查继续生效。

这是一套内存容量规划规则，不是吞吐最优配置搜索。它复用现有缓存、动态 CPU/GPU 分流、Graph 和 TP/PP 执行路径，不改变算子、路由数学或采样策略。改变缓存或驻留配置可能选用现有的不同计算后端，其数值特性与手动使用相同参数一致。

当前支持 Linux、等比例 TP、单卡及加权 CUDA PP 设备映射。CUDA 自动缓存按模型支持现有的 GGUF、FP8、packed FP4 专家格式；GLM/DeepSeek 的主存专家 CUDA 缓存要求 NUMA 和 GGUF/native packed FP4，普通 CPU 专家仍使用显式缓存参数。HF 精度转换按保守上界核算。KVMem、自定义 KV 字节上限、异比例 TP、视觉、自定义模型/权重转换、LoRA、RoPE 扩展及非 MTP 推测算法暂不参与自动规划，仍可使用原有显式参数。自动规划也不能与每卡缓存环境变量覆盖同时使用。逐项自动预算要求混合设备配置；统一 `--auto` 可以在预算允许时选择专家全驻留 GPU，但不选择纯 CPU 执行。
