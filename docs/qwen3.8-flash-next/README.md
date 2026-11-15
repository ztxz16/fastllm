# Qwen3.8-Flash-Next 用户手册

[English](README_EN.md) · [项目首页](../../README.md)

本手册介绍 Qwen4-Exp / Qwen3.8-Flash-Next 的文本服务部署，包括单卡混合推理、双卡串行与并行、专家缓存、固定 GPU 专家层、MTP 和 API 调用。自动化部署建议先把本页和 [模型部署速查](../model-deployment-agent.md) 交给 agent，并优先使用 `--auto`；下面的手动命令用于固定实验和故障定位。以下双卡示例使用 NVFP4 模型。

## 准备模型

先安装支持该模型的 FastLLM，并准备完整的本地模型目录，其中应包含配置、分词器和 Safetensors 权重。将命令中的 `/data/models/Qwen3.8-Flash-Next-NVFP4` 替换为实际路径。

混合推理需要足够的主机内存；使用磁盘 PLE 时，将模型存放在 SSD 上。

## 自动配置（推荐）

Linux NVIDIA CUDA 环境下，本地 HF/NVFP4 目录直接交给自动规划器即可：

```bash
ftllm server /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --auto --model_name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8080
```

GSQ GGUF 传入第一个分片，加载器会读取同目录的其余分片；下载目录和分片名称以本地文件为准：

```bash
ftllm server /data/models/Qwen3.8-Flash-Next-GSQ-RCO/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --auto --model_name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8080
```

显存或主机内存需要人为限额时，附加 `--auto_gpu_memory_limit 20g` 或
`--auto_host_memory_limit 128g`；省略时自动读取机器配置。启动日志会给出最终的设备、专家缓存和 chunk 设置。

资源上限是规划预算，不是运行时内存硬限制。自动模式只填写需要覆盖的参数：

| 需要调整 | 追加参数 |
| --- | --- |
| 缩小会话窗口 | `--max_context_length 32768` |
| 调整并发 | `--max_batch 2` |
| 固定每卡专家缓存 | `--moe_cuda_cache 7g`，`0` 关闭 |
| 固定线程或输入分块 | `--threads 32 --chunked_prefill_size 4096` |
| 内置 MTP 草稿数 | `--mtp 3`，需要对应权重；`0` 关闭 |
| 加载匹配的外置 MTP | `--draft /path/to/mtp.gguf` |
| 关闭全部推测解码 | `--speculative_algorithm off` |

`--auto` 本身不自动启用 Flash-Next MTP。指定受支持的外置 MTP GGUF 后，默认使用 3 个草稿 token；可用 `--draft_tokens` 覆盖。自动模式默认保留前缀缓存和历史复用。完整参数影响与启动器操作见 [`--auto` 调节指南](../auto.md)。

## 跨请求解码合批

设置 `--max_batch 4` 后，调度器会将已就绪请求的普通解码或 MTP 验证行合并，跨请求共享注意力输入/输出投影、HyperConnection、路由和专家计算。MTP 每一步的草稿投影与专家计算也会合并；已接受的输出连续发布，随后各请求一起进入下一轮验证。非贪心采样保留每个请求自己的温度、分布和拒绝采样状态。

KV、QSA、GDN、PLE 历史及接受前缀回滚仍独立保存。不同上下文长度、不同接受长度、请求退出和批量重排均可合批；没有按 2/4 并发或某个验证行数写死分派。混推的合批专家入口按实际形状使用公共 GPU 缓存、动态分流和 NUMA 批量调度，超过旧单请求验证宽度也不会因此绕过缓存。

单卡、稀疏 QSA 上下文的普通合批解码和合批 MTP 均支持完整 CUDA 图：无推测每步回放一次完整主干图，MTP 每轮验证回放一次完整主干图，每一步草稿回放一次完整草稿图。NVFP4 和 GGUF 共用图入口；普通解码不保存推测验证的回滚输出。公共混推专家图保留 GPU 专家缓存、动态分流和 NUMA 批量入口。第一步草稿按配置的草稿长度预留空间，设备端有效行数决定 QSA 状态提交和末行选择；不同接受长度不需要缓存各自的图组合。输入准备、独立采样及接受前缀回滚仍在模型图之外。

单卡 `--auto` 为 NVFP4 与 GGUF 混推启用整图准备，由后端检查设备及算子能力，不限于 SM75；无需额外设置环境变量。

完整图按数据依赖并行调度 GDN 的独立投影和 QSA 的索引、Q/K/V 准备分支，普通解码、验证与草稿共用该机制。预热记录并行分支的临时内存需求，汇合后继续复用；共享库工作区冲突或临时空间不足时保留串行依赖。QSA 的精确 RoPE 逆频率按旋转维度和基数预计算，位置仍在每次回放时读取，不需要按上下文长度建立旋转表。

文本请求可使用由 `max_batch` 限定数量的持久服务槽，支持非 greedy 抽样及多并发。请求完成时清除逻辑历史和 KV 页占用，保留捕获使用的存储地址；后续请求复用同一槽的完整图。合批按稳定槽顺序组织，普通 decode、验证和各草稿阶段分别缓存常用组合，不因客户端句柄或调度顺序变化而重建。

图缓存按最近使用情况淘汰：每阶段的组合数量不超过 `max_batch`，累计行工作区不超过已见最大形状的两倍，避免所有并发形状累计造成平方级显存增长。这是缓存保留预算，不限制可执行的行数。新形状、存储地址/容量变化或被淘汰的组合仍会预热和捕获；失败或取消的槽在同步并失效关联图后释放。日志中的 `Qwen cross-request decode/verify/draft full graph ready` 表示新图已完成拼接，复用时不重复打印。

串行图段仅共享捕获期间释放的临时工作区；持久输出保持固定地址，专家图的并行工作区独立保留。普通执行和 Prefill 不借用已被图固定的临时地址，因此保留图后仍可安全处理下一轮输入。

短上下文、Prefill、TP、多模态、KVMem、前缀/历史恢复及不支持捕获的情况保留原路径。本轮跨请求复用实测单卡 IQ3_S；算子分派和权重格式接口不变，其余模型及设备组合需分别验证。

## 手动配置参考

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

### 并发与批量验证

混合推理中的 NVFP4（16 元素块、E4M3 缩放）把一次 MTP 验证的多行专家投影合并启动；受支持的 GGUF IQ 专家还可按专家分组，跨验证行复用权重解码。驻留与动态分流分支使用独立工作区，保持原有量化、归约和 NUMA 计算语义；不需要额外的调优环境变量。物理 tile 大小不限制逻辑验证行数。

GGUF 可用 `--kernel-tune quick` 在捕图前按格式、形状和设备测速并校验；默认 `off` 保留原核。算子族 `moe.gguf.gate_swiglu_down` 提供 `q8`、`grouped`、`grouped_shared`、`q8_tiles_8`、`q8_tiles_16`（统一前缀 `cuda.moe.gguf.`），对应原 Q8 路径、分组复用、分组并暂存激活，以及两种保持原归约顺序的输出分块。启动日志输出选择结果；策略参数及保存、禁用、强制选项见[算子自动调优](../kernel-policy.md)。较小批量可能继续选用原核，较大批量是否获益由实测决定。预热使用当前层已驻留的专家记录及固定输入，不消耗请求采样状态。

`--max_batch` 控制同时接纳的请求数，`--max_context_length` 控制单请求上限，`--tokens` 控制共享 KV 池的总 token 预算。例如 `--max_batch 4 --max_context_length 8192 --tokens 32768` 为四个请求规划总容量，实际仍受可用显存限制。需要固定为 48 层 NUMA 专家混推时，可显式指定 `--moe_device numa --moe_device_layers 48`。

提高并发不保证吞吐同比增长：评测时应分别记录同时 decode 的吞吐、单轮验证、各草稿步以及 prefill 等待。

CUDA 完整图中的 GDN 使用 A/B、QKV＋卷积、Z 三个独立分支，在递推前汇合；嵌套分支保留完整依赖，临时空间不足或工作区冲突时回退串行。GGUF 的 GDN QKV/Z 和 QSA Q/K/V 分别共享激活量化；使用不同量化布局时，一次量化同时生成 packed 和 planar 输入。跨请求验证、草稿仍分别维护各请求的卷积与递推历史；大批量继续使用已有 MMQ 分块调度，不增加逻辑行数上限。

`--kernel-tune quick` 也用于共用的 FP16 投影、QSA 和小 TopK 采样；`off` 使用各族的默认实现。新增算子族：

| 算子族 | 候选实现 | 选择依据 |
|---|---|---|
| `linear.multi.fp16` | `reference`、`cublas`、`tile4`、`tile8`（前缀 `cuda.linear.multi.fp16.`） | 同输入的多个投影按实际行数、输入/输出宽度和设备测速 |
| `linear.hyper_project.fp16` | `reference`、`tile4`、`tile8`、`gemm_epilogue`（前缀 `cuda.hyper_project.`） | HC Down/Injection；GEMM 版本合并 FP16→FP32 转换与激活 |
| `attention.indexed.fp16` | `gather`、`split_pv`、`cublas_qk`（前缀 `cuda.attention.indexed.`） | 保留原 gather 路径，直接索引 KV，或保留 cuBLAS QK 并直接索引 V；直接索引候选要求 CUDA SM75+ 及支持的 FP16 布局 |
| `qwen4.ple_gate` | `serial`、`block256`（前缀 `cuda.ple_gate.`） | PLE 门控点积的原串行归约或 FP32 并行归约；支持 FP32/FP16/BF16 输入，默认并行，无 SM 或逻辑行数限制 |
| `sampling.logits_topk` | `tile1024`、`tile2048`、`tile4096`（前缀 `cuda.sampling.topk.`） | GPU 分块选择候选，按实际词表、批量和设备测速；候选分数、token 与平分顺序必须逐位一致 |

候选必须先通过每个输出张量的 L2 检查，再按实测决定；数值超限或设备/布局不支持时继续使用参考实现。图捕获内不测速，已捕获图的选择保持固定。tile 数字描述物理分块，不限制逻辑行数。QSA 直接索引候选支持 head dimension ≤256 且为 16 的倍数、每 KV head 1–32 个 query heads、稀疏宽度 128–3072；其他形状走原路径。验证与草稿各自的实际形状需要在捕获前预热，不能仅用 `max_batch` 推导选择。选择、候选耗时和误差记录在 `--kernel-profile FILE` 中，可用已有 `--kernel-policy` 的 `force`/`deny` 调整，不新增调优环境变量。

PLE 并行归约会改变浮点累加顺序。需要原有逐项累加行为时，可在 `--kernel-tune quick` 下用策略 `{"force":{"qwen4.ple_gate":"cuda.ple_gate.serial"}}` 选择参考核；CPU 和不支持的输入仍使用原路径。QSA 输出门控完整覆盖输出缓冲区，回放时不再预先清零。

GGUF 专家在首次测速前等待所需权重就绪，正式推理保留计算与上传重叠。IQ3_S 已在实际 32 GiB 可用显存限制下通过 `--kernel-tune quick` 预热及无推测的 1/2/4 请求回归。

普通采样在 CUDA FP32 logits、无 token mask、无有效重复惩罚且有效 `top_k ≤ 64` 时，仅回传 GPU 筛出的候选，继续使用原 CPU top-p 和随机数流程。其他配置保留原采样路径；采样仍在模型图外。32 行工作区分块只限制临时内存，不限制逻辑批量。

抓波形请记录 `nsys --version`。本机 SM120 上，相同 NVFP4 二进制在 Nsight Systems 2025.3.1 节点跟踪中切换双请求→四请求曾出现 CUDA 719/Xid 31；2026.5.1（CUPTI 13.4，硬件跟踪）通过了该切换。旧版抓图配置触发错误的具体原因仍未确定。建议使用已验证的 2026.5.1，参数为 `--trace=cuda,nvtx --cuda-graph-trace=node`，并检查 CUDA API 错误及验证、草稿的实际图节点。

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

测量配置、速度、内存和验证范围见[Benchmark](../benchmarks/qwen4_exp.md)。

<a id="gguf-resident-performance"></a>

### GGUF 专家常驻 GPU

两个分片放在同一目录，路径指向第一分片。此配置使用 TP2，全部专家在 GPU；embedding 在 CPU，PLE 在磁盘。显存不足时请使用前文的混合部署。


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


### HC 融合与算子策略

可选算子族 `qwen4.hyper_read` 合并上一块的残差更新、分组 RMSNorm、
Down/Injection 和 Up/Mix。CUDA 内核移植自
[Strata 的 9f7d977d 提交](https://github.com/Niko1221/Strata/tree/9f7d977d220e9fdf2ec53cf2620cdadac1cfb24e)，
Copyright (c) 2026 Niko1221 and the Strata contributors，遵循 MIT 许可证。
[来源与改动说明](../../third_party/strata/README.md)及[许可证全文](../../third_party/strata/LICENSE)
随代码保留，不依赖 Strata 运行时。

当前特化支持 FP32 残差/归一化权重、FP16 投影权重、4 个流、每流 2560 维、
低秩维度 320。其它形状或存储格式走原路径；物理行分块不限制逻辑批量行数。
普通解码、验证、草稿图共用入口，CPU 路径保持原实现。

使用 `--kernel-tune quick --kernel-policy hc-policy.json`，策略文件为：

```json
{"families": ["qwen4.hyper_read"]}
```

`families` 只让指定算子族参与测速，其余算子保留后端默认选择。
候选为 `cuda.hc.reference`、`cuda.hc.strata.cols4`、`cuda.hc.strata.cols8`、
`cuda.hc.planar.cols4`、`cuda.hc.planar.cols8`，
可通过 `ftllm ops` 查看条件。预热按实际形状测速，并逐个检查输出的相对 L2；
日志及 `--kernel-profile FILE` 保存选择、耗时和误差。支持已有 `force`/`deny` 策略。
不支持或未经测速的图形状回退原实现；`--kernel-tune off` 保持原实现。
不按固定 SM 或逻辑行数阈值决定启用。

并发单 token GDN 解码将独立请求的递归更新与输出门控放入同一次 CUDA
启动，保持输出与状态的原始运算顺序。参数按有界物理块传递，请求数没有
32 行上限；不支持的布局、多行验证及 CPU 使用原路径，验证与草稿仍可进完整图。
HC 的 `planar` 候选仅调整共享内存布局，由同一 warmup benchmark 选择，
不按 C2/C4 或特定 SM 强制启用；没有新增环境变量。


### GDN 递推算子选择

单 token 的 FP16 GDN 递推、输出归一化和门控使用统一算子族
`gdn.decode.fp16`。`--kernel-tune quick` 会检查输出及 FP32 状态的相对 L2，
再比较原始 `cuda.gdn.reference` 与 `cuda.gdn.register.rg1` / `rg2` / `rg4`。
后者移植自 [Strata 的 GDN CUDA 算子](../../third_party/strata/README.md)，
保留 MIT 许可和来源标识，适配 FastLLM 状态布局与 FP16 输出边界。
测速仅写入临时状态，不推进请求状态；选定后可进入 CUDA Graph。

这些候选不按 SM 或请求数设置性能阈值。32 个请求的启动参数分块会循环
处理整个批次，没有逻辑并发上限。不同请求数分别测速，日志报告选中项；
多 token 验证继续使用原有序列递推及回退路径。CPU 路径保持原实现。
可通过现有 `--kernel-policy` 中的 `families`、`force`、`deny` 调整选择，
例如 `{"families":["gdn.decode.fp16"],"force":{"gdn.decode.fp16":"cuda.gdn.register.rg2"}}`。
未测速、关闭调优或不支持的布局仍保留原路径；不增加环境开关或 Strata 运行时依赖。


### 因果卷积 decode 选核

`conv.causal_depthwise.decode` 将原有通用实现与四抽头向量实现统一注册。
后者一次读取 FP32 历史与权重，在寄存器中完成状态移动和点积，并使用
FP32 SiLU；支持 FP32、FP16、BF16 输入。`--kernel-tune quick` 分别检查
输出与历史状态后，实测选择 64／128／256 线程块，不按 SM 或逻辑批量
设置性能阈值。线程块大小不限制输入批量。

新候选为 `cuda.conv.decode.vec4.block64`、`.block128`、`.block256`，
原实现为 `cuda.conv.decode.reference`。其他卷积长度或不满足向量对齐的
布局保留原实现；CPU 路径不变。支持现有 `--kernel-policy` 的 `families`、
`force`、`deny`，不增加环境开关。验证只写临时状态，测速在计时区间之外
初始化私有历史，再按实际原位更新方式回放；不会重复推进请求状态。
单 token 与并发 decode 可进入完整图，多 token 验证保留原序列卷积路径。


### GGUF 投影与 attention 输入准备

`gguf.planar.projection` 将 Q4_K、IQ4_XS 的普通 FP16 输出投影接入统一测速。
候选为 `cuda.planar.reference`、`cuda.planar.w2.r1.full`、
`cuda.planar.w4.r1.occupancy`、`cuda.planar.w8.r2.occupancy`。
它们复用现有 GGUF 解量化与点积，只改变 warp 数、输出分块和网格布局；
由实际格式、行数、K/N、输出步长和设备的预热结果选择，不按 SM 或并发数固定启用。
原核仍是默认回退；残差、融合 gate/up 与大批量 MMQ 保持原分派。

使用 `--kernel-tune quick`，可在策略的 `families` 中加入
`gguf.planar.projection`，通过既有 `force`/`deny` 控制候选。
图内不测速，独立 stream 的首次准备先等待输入就绪，并在私有输出上校验。
planar 公共入口按最多 8 行的物理 tile 循环，不再限制逻辑行数为 16；
大批量是否调用它仍由上层公共投影分派决定。

HC 融合核可以直接输出 attention 所需的 FP16/BF16，保持 FP32 残差和注入系数，
省去独立转换及同 dtype 复制。其他形状、CPU 和未选择融合核时保留原计算加转换。
GDN 的 A/B 分支可与共享 Q8 量化重叠，QKV 与 Z 仍在量化完成后分叉，
并在递推前完整汇合；普通、验证和草稿共用同一接口。

### QSA 跨请求整批处理

FP16 QSA 图路径直接读取合批投影，并按请求访问各自的 KV、位置和有效行数。
Norm/RoPE 与 KV 写入合并，TopK 与索引展开合并；分块 attention 融合
QK、Softmax、PV，归并时直接写门控输出。普通 decode、多行验证和草稿共用
同一入口，保留验证原始键、因果边界和回滚状态，没有固定并发数或验证行数上限。
CUDA 参数包和 attention 分块只是物理调度单位。

`attention.qsa.requests.fp16` 提供优化候选 `cuda.qsa.requests.chunk64`、
`cuda.qsa.requests.chunk128`，以及基础融合实现 `cuda.qsa.requests.reference64`。
使用 `--kernel-tune quick`，并在已有策略的
`families` 中加入该名称；日志与 `--kernel-profile FILE` 记录选择和数值误差。
例如 `{"force":{"attention.qsa.requests.fp16":"cuda.qsa.requests.chunk64"}}`
可固定候选。测速只重放无状态 attention，不重复写入实时 KV 或压缩尾部。
关闭调优时使用 64-token 分块，捕获中不测速。

该路径使用通用 CUDA SIMT，支持 128/256 的 attention 头维度、FP16 连续或分页 KV
及不同请求长度；其他 dtype、布局、KVMem、CPU 和不支持的几何保留原路径。
输入投影仍经过公共权重格式分派。分块 attention/归并参考并改编自
[Strata 的 CUDA 算子](../../third_party/strata/README.md)，MIT 许可证和来源版本
随代码保留；不依赖 Strata 或 Maya 的推理引擎。

优化候选在 KV 单元之间复用 Query，使用向量化 KV 读取、合并的头归约和
按 KV 单元排列的概率；未对齐的 KV 地址保留标量读取。索引投影与 QKV 投影
并行，索引预处理/选择与 Q/K 预处理分支也独立提交，汇合后再运行 attention。
这些优化共用于单请求、合批验证和草稿，不按特定行数或 SM 启用。
