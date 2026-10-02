# Qwen3.8-Flash-Next 用户手册

[English](README_EN.md) · [项目首页](../../README.md)

本手册介绍 Qwen4-Exp / Qwen3.8-Flash-Next 的文本服务部署，包括单卡混合推理、双卡串行与并行、专家缓存、固定 GPU 专家层、MTP 和 API 调用。以下双卡示例使用 NVFP4 模型。

## 准备模型

先安装支持该模型的 FastLLM，并准备完整的本地模型目录，其中应包含配置、分词器和 Safetensors 权重。将命令中的 `/data/models/Qwen3.8-Flash-Next-NVFP4` 替换为实际路径。

混合推理需要足够的主机内存；使用磁盘 PLE 时，将模型存放在 SSD 上。

## 快速启动

下面以一张 GPU 运行主干、CPU 的 NUMA 后端运行专家为例：

```bash
ftllm server /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --model_name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8080 \
  --device cuda --moe_device numa \
  --atype float16 --ngram_device disk --threads 28 \
  --chunked_prefill_size 4096 --gpu_mem_ratio 0.9 \
  --prefix_cache true --enable_thinking false
```

等待模型加载完成后，客户端使用 `http://服务器地址:8080/v1`，模型名填写 `qwen3.8-flash-next`。线程数应根据 CPU 物理核心数调整。

<a id="deployment"></a>

## 双卡混合部署

以下配置适用于两张 **22 GiB 显存版 RTX 2080 Ti**、AMD EPYC 7452 和约 126 GiB 主机内存的机器。标准 11 GiB 2080 Ti 需要降低专家缓存或 GPU 驻留层数。

先在 Bash 中设置公共参数，再选择一种部署命令：

```bash
MODEL=/data/models/Qwen3.8-Flash-Next-NVFP4
unset FASTLLM_TP
export CUDA_VISIBLE_DEVICES=0,1
export FASTLLM_CUDA_GRAPH=0

COMMON=(
  --model_name qwen3.8-flash-next
  --host 0.0.0.0 --port 8080
  --atype float16 --ngram_device disk --threads 28
  --mtp 3 --temperature 0 --top_k 1 --repeat_penalty 1
  --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95
  --chunked_prefill_size 4096
  --prefix_cache true --enable_thinking false
)
```

这组示例关闭 CUDA Graph，并开启适合多轮文本请求的前缀缓存。模型不含 MTP 权重或需要关闭推测解码时，将 `--mtp 3` 改为 `--mtp 0`。

### 双卡串行：专家缓存开关

双卡串行将主干按层放到两张显卡上。`cudapp=2` 为两卡均分主干层。

```bash
# 关闭专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 0

# 每卡使用 8 GiB 专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 8g
```

### 双卡并行：专家缓存开关

`--tp 2` 让两张显卡共同计算每一层。选择该模式时，不再同时指定 `cudapp`。

```bash
# 关闭专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 0

# 每卡使用 8 GiB 专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 8g
```

`--moe_cuda_cache 8g` 是**每卡**预算，两卡合计约 16 GiB。显存还需要容纳主干权重、KV 缓存和计算工作区，不能全部分配给专家缓存。

### 固定部分专家层在 GPU：双卡并行

关闭动态专家缓存，将前 12 层的全部专家固定在 GPU，剩余 36 层使用 NUMA：

```bash
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_device_layers 36 --moe_cuda_cache 0
```

`--moe_device_layers 36` 指**最后 36 层**使用 `--moe_device numa`；前 12 层专家由两张 GPU 分片存放。减小这个数字会增加 GPU 驻留层数和显存占用。

### 固定部分专家层在 GPU：双卡串行

使用下列配套的主干和专家映射：

```bash
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=1:7 \
  --moe_device "{'cuda:0':6,'cuda:1':6,'numa':36}" \
  --moe_cuda_cache 0
```

主干第 0～5 层放在 GPU0，第 6～47 层放在 GPU1；专家第 0～5 层放在 GPU0，第 6～11 层放在 GPU1，后 36 层和 MTP 专家使用 NUMA。调整层数时，应同时调整主干与专家映射，确保 GPU 专家所在层与对应主干使用同一张卡。

固定层会为该层全部专家分配显存，动态缓存则按访问情况保留专家。两种方式的速度和显存需求不同。

<a id="parameters"></a>

## 常用参数

| 参数 | 用途 |
| --- | --- |
| `--device cuda` | 单卡主干 |
| `--device cudapp=2` | 双卡主干按层串行 |
| `--tp 2` | 双卡张量并行 |
| `--moe_device numa` | 使用 CPU 的 NUMA 后端运行专家 |
| `--moe_cuda_cache 0` / `8g` | 关闭专家缓存 / 每卡设置 8 GiB 预算 |
| `--moe_device_layers 36` | 最后 36 层专家使用指定的 MoE 设备 |
| `--ngram_device disk` | 按需从磁盘读取 PLE 表，降低主机常驻内存 |
| `--ngram_device cpu` | PLE 表驻留主机内存 |
| `--threads 28` | CPU 计算线程数 |
| `--mtp 0` / `3` | 关闭 MTP / 每步最多提出 3 个草稿 token |
| `--chunked_prefill_size 4096` | 每块最多处理 4096 个输入 token |
| `--kv_cache_limit 1g` | KV 缓存预算，按上下文与并发需求调整 |
| `--gpu_mem_ratio 0.95` | GPU 可用内存的使用比例 |
| `--prefix_cache true` | 复用重复文本前缀，适合多轮对话 |
| `--max_context_length` | 上下文长度上限，仍受模型和缓存容量限制 |
| `--enable_thinking true` | 开启模型思考模式 |

### MTP

MTP 需要模型目录中包含匹配的 MTP 权重，主干运行在 CUDA，专家可使用 CUDA 或 NUMA。`--mtp` 最大为 8；可以先使用 3，实际收益取决于任务和草稿接受情况。双卡 TP 示例使用贪婪采样，即 `--temperature 0 --top_k 1 --repeat_penalty 1`。

### 内存与上下文

主机内存紧张时使用 `--ngram_device disk`。显存不足时，先减小专家缓存或 GPU 驻留层数，再根据请求长度调整 KV 预算、分块大小和并发数。

增大 `--chunked_prefill_size` 可减少长输入的分块次数，同时增加临时显存需求。前缀缓存与专家缓存相互独立，分别复用文本前缀和专家权重。

<a id="api"></a>

## 调用服务

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "请用三句话介绍你自己。"}],
    "max_tokens": 512,
    "temperature": 0,
    "stream": false
  }'
```

远程调用时，将 `127.0.0.1` 替换为服务器地址。若启动时设置了 `--api_key`，请求需增加 `Authorization: Bearer 你的密钥` 请求头。

也可以在浏览器中对话：

```bash
ftllm webui --api_base http://127.0.0.1:8080/v1
```

WebUI 连接已经启动的 API 服务。需要工具调用时，可在服务启动命令中加入 `--tool_call_parser auto`；API 请求使用标准 `tools` 字段。

<a id="performance"></a>

## 性能参考

以下为 2026-09-29 在上述双 22 GiB RTX 2080 Ti、EPYC 7452 机器上的 NVFP4 单请求实测，使用 28 线程和磁盘 PLE。Prefill 为 4096 输入、4096 分块；解码为 512 输入、512 输出，不含首 token 等待。测量时关闭前缀复用、CUDA Graph 和思考，各配置测一轮。

| 部署方式 | Prefill token/s | 普通解码 token/s | MTP=3 token/s |
| --- | ---: | ---: | ---: |
| 串行，缓存关闭 | 919.43 | 22.36 | 30.08 |
| 串行，每卡缓存 8 GiB | 902.92 | 26.13 | 39.84 |
| 串行，固定 12 层专家 | 960.99 | 23.99 | 32.92 |
| TP2，缓存关闭 | 1240.71 | 20.60 | 32.63 |
| TP2，每卡缓存 8 GiB | 1254.39 | 27.71 | 46.29 |
| TP2，固定 12 层专家 | 1321.08 | 22.57 | 37.82 |

这组数据中，TP2 + 每卡 8 GiB 专家缓存的解码较快，TP2 固定 12 层专家的长输入处理较快。实际速度随输入、输出内容和硬件而变化。

<a id="gguf-resident-performance"></a>

### GGUF IQ2_XS：双卡专家常驻 GPU（2026-10-01）

使用清理后的 GGUF 实现，在同一台双 22 GiB RTX 2080 Ti / EPYC 7452 机器复测 `Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS`。主干和专家使用 TP2，FP16 激活，CUDA Graph 开启，MTP 关闭且不加载外部 MTP 权重。专家缓存预算为 0；所有专家权重以原生 GGUF 格式驻留 GPU。embedding 仍在 CPU（`--low_gpu_mem`），PLE 使用磁盘，因此这里的纯 GPU 指 Dense/MoE 计算和专家驻留。

核心启动参数如下。两个 GGUF 分片需放在同一目录，模型路径指向第一分片；每组测试按表格调整 `--chunked_prefill_size`。

```bash
CUDA_VISIBLE_DEVICES=0,1 FASTLLM_CUDA_GRAPH=1 FT_NUMAS=1 \
ftllm server /data/models/Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf \
  --tp 2 --device cuda:0,1 --moe_device cuda:0,1 \
  --atype float16 --low_gpu_mem --ngram_device disk --threads 28 \
  --moe_cuda_cache 0 --mtp 0 --max_batch 1 --tokens 8192 \
  --kv_cache_limit 1g --gpu_mem_ratio 0.95 --chunked_prefill_size 1024 \
  --prefix_cache false --cache_history false --enable_thinking false \
  --temperature 0 --top_k 1 --repeat_penalty 1
```

测速直接调用原生生成接口，无 HTTP 开销或 profiler；greedy、batch=1，每组先预热，再正式测三轮并取中位数。解码为 512 输入、512 输出，速度按 `511 / (末 token 时间 - 首 token 时间)` 计算；prefill 为 4096 输入、1 输出，速度按 `4096 / TTFT` 计算，包含首 token 延迟。

| 测试 | Chunk | 三轮速度（token/s） | 中位数（token/s） | TTFT 中位数（秒） |
| --- | ---: | --- | ---: | ---: |
| 解码 | 32 | 71.11 / 70.87 / 70.61 | **70.87** | 1.23 |
| 解码 | 512 | 69.99 / 69.77 / 69.56 | **69.77** | 0.49 |
| 4096 prefill | 1024 | 1206.47 / 1205.13 / 1200.43 | **1205.13** | 3.40 |

同一 chunk 的三轮输出一致；不同 chunk 的输出从第 200 个 token 开始不同，不能将约 1.6% 的解码差异单独归因于 chunk。这组模型、放置和 Graph 配置也不同于上方 NVFP4 混推记录，不构成量化格式之间的受控对比。

| 显存（GiB） | GPU0 | GPU1 |
| --- | ---: | ---: |
| 逻辑权重 | 19.3554 | 19.3554 |
| 加载及初始化预热完成后已用 | 20.2088 | 20.2049 |
| 全程采样峰值 | 21.0586 | 20.8379 |

逻辑权重来自 TP 初始化清单，加载后显存来自 `cudaMemGetInfo`，峰值由 `nvidia-smi` 每 2 秒采样，可能遗漏短暂峰值。CPU embedding 为 2.368 GiB；主机进程 HWM 为 41.905 GiB，swap 为 0。每卡 49,152 个专家 gate/up、down 张量均为 GPU 原生 GGUF；所有请求的 CPU/混合 MoE 路径计数、专家缓存 hits/misses/payload 及 MTP verifier 调用均为 0。算术和 JSON 校验通过，chunk=1024 的 4096 prefill 全部完成，无 OOM。

[完整三轮数据、配置及校验信息](../benchmarks/qwen38_flash_next_iq2xs_2080ti_20261001.json)。此次测速使用带诊断计数的隔离构建，原生库 SHA256 为 `0e9aa1b66b2d502e150395a40bfca36f2b43c0ea937329f43f89d60861add82b`；没有覆盖已安装版本。

### GGUF Q2_0：专家 MMQ 与 Dense 解码优化（2026-10-01）

同机、相同 TP2/Graph/MTP=0 配置对 `Qwen3.8-Flash-Next-GSQ-RCO-Q2_0` 做优化前后复测。模型、输入 token、解码长度、预热和三轮中位数口径保持一致。改动包括 Q2 专家 MMQ 的 16-bit 读取与向量解包、专家输入按 token 量化后再按路由分发，以及 1–8 行 Q2 Dense 的 Q8/DP4A 点积。实现按格式和张量尺寸选择路径，不依赖模型名称，也没有新增环境变量。

| 测试 | Chunk | 优化前 token/s | 优化后三轮 token/s | 优化后中位数 token/s | 变化 |
| --- | ---: | ---: | --- | ---: | ---: |
| 解码，512 输入 / 512 输出 | 32 | 70.14 | 71.12 / 70.74 / 70.44 | **70.74** | +0.85% |
| 解码，512 输入 / 512 输出 | 512 | 68.91 | 69.76 / 69.58 / 69.49 | **69.58** | +0.97% |
| 4096 prefill，1 输出 | 1024 | 1130.22 | 1357.01 / 1354.38 / 1352.08 | **1354.38** | **+19.83%** |

Nsight Systems 的两次 4096 prefill 平均值中，GPU0 专家 gate/up 累计耗时由 1041.96 降到 706.54 ms，down 由 638.04 降到 496.87 ms，路由/量化/归并由 232.44 降到 124.61 ms；GPU1 的关键专家调用数也通过校验。解码每步只有 7 个 Q2 Dense 投影，整体 Dense GEMV 累计耗时由 6.131 降到 6.003 ms，其他量化类型的 Dense 和输出头仍占较多时间。

GPU0 冷缓存 NCU 重放中，两个 Q2 Dense 投影的 DRAM 带宽利用率由 16.01% / 17.57% 提升到 56.92% / 61.73%；新路径每次另有约 2.08 μs 的输入量化。Q2 专家 gate/up 的 MMQ 耗时由 4.431 降到 3.005 ms，DRAM 利用率由 3.32% 到 4.90%；专家输入量化由 676.86 μs 降为一次 38.24 μs 量化加 116.13 μs 分发。MMQ 仍没有跑满显存带宽。NCU 使用真实专家快照、真实 Dense 权重及非零合成 Dense 输入，时钟未锁定；这些计数器不能直接视作完整请求的带宽。

专家输入量化复用已有 product 缓冲区，权重存储和持久 workspace 容量不变。每卡逻辑权重为 18.6634 GiB；加载及初始化预热后 GPU0/GPU1 为 19.8436 / 19.8416 GiB，前后相同。两次完整测速的显存采样峰值也同为 20.6855 / 20.4648 GiB。CPU embedding、磁盘 PLE、专家全驻 GPU 和缓存预算 0 的设置与上节一致。

真实专家快照的 16 份 gate/output 张量与原实现逐位一致；分组专家、320 列及更短 K 尾部、双卡切分、CPU 专家的 GPU 辅助 prefill、通用缓存及 CUDA Graph 测试通过。新增 Dense 测试用独立 CPU Q8 参考覆盖 FP32/FP16/BF16、1/3/8 行和部分输出块。Dense 从浮点激活点积改为 Q8 激活点积，会改变数值和生成结果：相同 chunk 的三轮输出一致，但优化前后 chunk 32 / 512 分别从第 17 / 250 个输出 token 开始不同。因此约 1% 的整模型解码差异包含生成路由变化；算子收益应结合独立重放判断。4096 prefill 首 token、算术结果和 JSON 结果保持一致，这些校验不代替完整模型质量评估。

[优化前后三轮数据、数值校验及 profiler 记录](../benchmarks/qwen38_flash_next_q2_0_2080ti_20261001.json)。测速使用隔离构建，优化后原生库 SHA256 为 `cdacb884eef2ec3ce0b70386a69f4ec6211c6cd4e42de091e15af9eab8ec38ee`。

### GGUF Q2_0：继续优化 Q3_K Dense 解码（2026-10-02）

这份 Q2_0 模型的 Dense 权重混用了 Q3_K 等格式。本轮在上一节优化版的基础上，减少 Q3_K 子块 scale 的重复读取和解包指令；单 token、大投影使用每个 CUDA block 计算八个输出行并共享 Q8 激活。输出行较少或 K 较长时保留每行四个 warp 的分工，避免尺寸扫描中发现的性能退化。路径按格式、尺寸和对齐选择，不绑定模型名称，不增加环境变量、权重转换或持久 workspace，浮点累加顺序保持一致。

同一双 22 GiB RTX 2080 Ti、TP2、CUDA Graph 开启、MTP 关闭、专家全驻 GPU 配置，重新运行上一版作为对照。每组预热后正式测三轮，取中位数：

| 测试 | Chunk | 上一版 token/s | 本轮 token/s | 变化 |
| --- | ---: | ---: | ---: | ---: |
| 解码，512 输入 / 512 输出 | 32 | 70.69 | **71.61** | +1.30% |
| 解码，512 输入 / 512 输出 | 512 | 69.51 | **70.63** | +1.61% |
| 4096 prefill，1 输出 | 1024 | 1355.21 | **1358.02** | +0.21% |

14 次请求（含预热、三轮解码、prefill、算术和 JSON）的输出 token 均与对照一致。4096 prefill 的微小差异视为波动，不作为本轮优化收益。

Nsight Systems 中，GPU0 每步 53 次 Q3_K Dense 调用累计由 0.958 降到 0.787 ms。GPU0 冷缓存 NCU 重放的 5120×2560 Q3_K 投影由 23.136 降到 19.744 μs，DRAM 带宽利用率由 48.73% 提升至 56.99%，ALU 活跃比例由 63.53% 降至 35.36%，每线程寄存器从 66 降到 45。计数器使用实际权重、非零合成输入，时钟未锁定；整模型速度来自无 profiler 的独立测速。其他 Dense、HC 投影和跨卡归约仍占较多时间，因此整模型收益约为 1%–2%。

FP32/FP16/BF16 独立 CPU 参考、原四 warp 内核逐位对照、尾行、尺寸边界及 CUDA Graph 重放测试通过；72 组实际 Linear 调用前后逐位一致。Q2 回归、Compute Sanitizer memcheck/synccheck 均通过，后两项零错误。显存前后相同：逻辑权重每卡 18.6634 GiB，加载及预热后 GPU0/GPU1 为 19.8436 / 19.8416 GiB，采样峰值为 20.6855 / 20.4648 GiB。

[完整测量与校验记录](../benchmarks/qwen38_flash_next_q3_decode_2080ti_20261002.json)。本轮隔离构建 SHA256：`8c3c12485b0031d127aa2353b4e3cb40d0f99d97a0cd7a05f5900af8a67d74db`；已安装库未覆盖。当前性能及 sanitizer 验证范围为 SM75。

### GGUF Q2/Q3 代码整理与回归（2026-10-02）

在上述优化版基础上，合并 Dense 与分组专家的 Q8 输入量化、Q2 八值解包，以及 Q3 两种线程分工共享的 K 累加逻辑。保留此前验证过的尺寸分派和备用路径，去除重复 include 和融合 gate/up 中不可达的 IQ1 初始化。量化权重布局、计算精度和持久 workspace 容量保持一致。

统一入口按权重格式自身的量化块大小检查列数；融合 gate/up 只接受实际实现的 IQ2_XXS / IQ2_XS / IQ2_S。不支持的格式或非整权重块长度在分配和读取设备内存前返回 false，供调用方选择备用路径。新增回归覆盖这些边界，36 组有效 IQ2 融合调用与整理前逐位一致。

审计 17 个 GGUF 相关源文件，未发现遗留的运行时实验环境变量、测速钩子或模型路径硬编码。保留缓存容量、MTP、TP、Graph 等有效运行配置，本轮不增加开关。另补齐缓存测试入口成功分支的显式返回值，使重命名入口的动态加载测试也能正常结束。

12 组算子/缓存/双卡切分/Graph/内存与同步检查通过；16 份真实专家中间结果、72 组 Dense 输出和 14 次整模型请求的生成 token 均保持一致。同配置三轮中位数：chunk 32 解码 **71.61 → 71.59 token/s**，chunk 512 解码 **70.63 → 70.67 token/s**，4096 prefill **1358.02 → 1359.00 token/s**。时钟未锁定；小幅变化不作为新的优化收益。加载及预热后的显存占用与整理前相同。

[整理记录与完整回归结果](../benchmarks/qwen38_flash_next_gguf_cleanup_2080ti_20261002.json)。隔离构建 SHA256：`2c60b9f0661321f32447f8d3bd1de75998e6fd9029eda3a95a79aad210241f91`；验证硬件为 SM75，已安装库未覆盖。
