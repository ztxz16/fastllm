# Qwen4-Exp / Qwen3.8-Flash-Next 部署指南

[English architecture notes](qwen4_exp.md) · [返回 README](../README.md) · [Benchmark](benchmarks/qwen4_exp.md)

FastLLM 支持 Qwen4-Exp / Qwen3.8-Flash-Next 的 FP8、NVFP4 和 `qwen4exp` 架构 GGUF 文本生成模型，覆盖四路超连接、Gated DeltaNet、QSA 稀疏注意力、PLE n-gram 和 MoE。复合 checkpoint 中的视觉权重不会由文本模型加载；Qwen3.8-Flash-Next 的 MTP 权重仅在 `--mtp` 大于 0 时按需加载。

## API Server 快速启动

~~~bash
ftllm server /data/models/qwen4-exp \
  --model_name qwen4-exp \
  --host 0.0.0.0 --port 8080
~~~

PLE 表默认驻留在 CPU，模型会按需读取选中的行，不会把整张表展开成 FP32。

## 按设备选择启动命令

### CUDA + NUMA MoE

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

### CUDA + CPU MoE

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device cpu \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

### 纯 CPU / NUMA

~~~bash
ftllm server /data/models/qwen4-exp \
  --device numa --moe_device numa \
  -t 64
~~~

Qwen4-Exp 的模型和 PLE 表都很大，纯 CPU/NUMA 命令主要用于容量验证与调试。线程数需要结合物理核心数和内存带宽调整。

## GGUF：Unsloth UD-Q2_K_XL

支持直接读取 `unsloth/Qwen3.8-Flash-Next-GGUF` 的 `UD-Q2_K_XL` 三分片。把三个文件放在同一目录，保留原文件名，启动时传入 `00001-of-00003.gguf`；首分片只有元数据也是正常的，不需要合并文件或另行提供 HF 配置。

下面使用双卡串行主干、NUMA 专家和磁盘 PLE，适用于本次测试的双卡 22 GiB 2080 Ti 机器：

~~~bash
GGUF=/ssd/models/Qwen3.8-Flash-Next-GGUF/UD-Q2_K_XL/Qwen3.8-Flash-Next-UD-Q2_K_XL-00001-of-00003.gguf
unset FASTLLM_TP
CUDA_VISIBLE_DEVICES=0,1 FT_NUMAS=1 FASTLLM_CUDA_GRAPH=0 \
ftllm server "$GGUF" \
  --model_name qwen3.8-flash-next \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 0 \
  --atype float16 --ngram_device disk --threads 28 \
  --mtp 0 --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95 \
  --chunked_prefill_size 4096 \
  --prefix_cache false --cache_history false --enable_thinking false
~~~

`UD-Q2_K_XL` 是混合量化：专家包含 IQ2_XS、IQ3_XXS 和 IQ4_NL，PLE 表为 IQ4_NL。专家保留低比特格式，PLE 只解码当前需要的行；主干线性投影加载为 FP16，归一化等参数保留 FP32。加载器会还原 GDN 头部顺序、合并 QSA 的 Q/K 投影，并读取 GGUF 中的 PLE 哈希元数据。三个文件合计约 **73.45 GiB**，这是磁盘占用。

这组三分片不含 MTP 权重，使用 `--mtp 0`；按文本模型部署。下方 NVFP4 的专家缓存、GPU 固定专家层和性能数据不代表此 GGUF 配置。

本机 2026-09-29 单轮实测（每项预热一次，Graph 关闭、28 线程、PLE 磁盘模式）：

| 配置 | Prefill（4096 输入，token/s） | Decode（512 输入 / 512 输出，token/s） | Decode 请求首 token | 主机 VmHWM | 两卡峰值显存 |
| --- | ---: | ---: | ---: | ---: | --- |
| 双卡串行、NUMA 专家、缓存 0、MTP 0 | 801.56 | 20.50 | 1.29 秒 | 55.58 GiB | 8192 / 8754 MiB |

Prefill 使用 `chunked_prefill_size=4096`，取 native Forward 时间；Decode 取首、末输出 token 之间的间隔。中文算术、JSON 去重排序任务均通过并自然结束，测速请求完整生成 512 token；进程退出正常，无 swap。此结果验证上述配置，不代表最大上下文或完整 Agent 任务的能力评估。

本轮已启用 IQ4_NL CPU 多行 down 内核，权重格式和部署参数不变。相较最初 GGUF 候选的 19.07 token/s，单轮 decode 提升约 7.5%；独立短剖析中，48 层 down 合计耗时从 12.97 降到 9.82 ms/token。长回复前 96 个 token 相同、后续生成轨迹不同，因此这些单轮结果不代表所有任务的固定加速比。


## 双卡混合部署：Qwen3.8-Flash-Next NVFP4

以下配置对应两张 **22 GiB 显存版 RTX 2080 Ti**、AMD EPYC 7452（32 核）、约 126 GiB 系统内存的测试机，不能直接套用到标准 11 GiB 的 2080 Ti。模型为 `Qwen3.8-Flash-Next-NVFP4`，共有 48 个主干 MoE 层，每层 512 个路由专家；PLE 使用磁盘模式。

双卡串行按层分配主干，一条请求依次经过两张卡；TP2 则让两张卡共同计算每一层。专家放置另行配置，可以全部使用 NUMA、在 NUMA 基础上开启动态 GPU 专家缓存，或把部分层的全部专家固定放在 GPU。

### 公共参数

在 Bash 中先设置模型路径和公共参数，再选择下面六种命令之一：

~~~bash
MODEL=/data/models/Qwen3.8-Flash-Next-NVFP4
unset FASTLLM_TP
export CUDA_VISIBLE_DEVICES=0,1
export FT_NUMAS=1 FASTLLM_CUDA_GRAPH=0

COMMON=(
  --model_name qwen3.8-flash-next
  --host 0.0.0.0 --port 8080
  --atype float16 --ngram_device disk --threads 28
  --mtp 3 --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95
  --chunked_prefill_size 4096
  --prefix_cache false --cache_history false
  --enable_thinking false --temperature 0 --top_k 1
)
~~~

公共参数与下方性能测试对齐，关闭前缀复用以测量实际预填充。多轮 Agent 部署可将 `--prefix_cache` 改为 `true`；**前缀缓存与专家缓存是两个独立功能**。普通解码将公共参数中的 `--mtp 3` 改为 `--mtp 0`。本机只有一个 NUMA 节点，因此设置 `FT_NUMAS=1`；其他机器应按实际拓扑调整。

### 双卡串行：关闭或开启专家缓存

`cudapp=2` 将 48 层主干按 24/24 分配到两张卡，不添加 `--tp`。

~~~bash
# 主干分卡串行，全部路由专家走 NUMA
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 0

# 同一主干布局，每卡增加 8 GiB 动态专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 8g
~~~

### 双卡并行：关闭或开启专家缓存

`--tp 2` 启用两卡张量并行，不再使用 `cudapp`。

~~~bash
# 每层主干由两卡共同计算，路由专家走 NUMA
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 0

# TP2 + 每卡 8 GiB 动态专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 8g
~~~

`--moe_cuda_cache 8g` 是**每卡**预算，两卡合计约 16 GiB；不包含主干权重、KV 和临时工作区。缓存自动保留活跃专家，混合调度可以将未驻留的路由交给 NUMA。关闭缓存不需要额外设置环境变量，也不再依赖 GPU 专家缓存来规避此前 NUMA 按需注册造成的主机内存膨胀。

### 固定部分 MoE 层在 GPU：双卡并行

~~~bash
# 前 12 层专家固定在 GPU，后 36 层专家使用 NUMA
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_device_layers 36 --moe_cuda_cache 0
~~~

`--moe_device_layers 36` 表示**最后 36 层**采用 `--moe_device numa`，不是把 36 层放到 GPU。前 12 层的全部专家由两张卡分片驻留，每层仍做 TP。改为 `30` 则是前 18 层驻留 GPU，但会进一步挤占长输入、KV 和临时工作区的显存；本页的大块 prefill 对照使用 12 层。

### 固定部分 MoE 层在 GPU：双卡串行

串行使用显式专家映射，并让主干与专家的分卡边界对齐：

~~~bash
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=1:7 \
  --moe_device "{'cuda:0':6,'cuda:1':6,'numa':36}" \
  --moe_cuda_cache 0
~~~

此配置的主干第 0～5 层在 GPU0，第 6～47 层在 GPU1；专家第 0～5 层在 GPU0，第 6～11 层在 GPU1，后 36 层在 NUMA。MTP 专家也使用 NUMA。这里不用 `--moe_device_layers` 简写，以避免 `cudapp` 展开后的设备映射被当作单个设备名；也不要只把主干改回 24/24 而保留这份专家字典，历史测试中该跨卡组合出现过非法访存。

固定驻留会为选中层的全部 512 个专家预留显存；动态缓存则跨层保留活跃专家。两者的内存分配方式不同，固定层数不保证解码更快。固定驻留能降低推理阶段主机 RSS，但加载峰值仍应单独考虑。

### 本机单轮实测

2026-09-29，当前提交 `d6ef5c784`，上面的共同配置。Prefill 为 4096 输入、4096 分块；decode 另用 512 输入、固定 512 输出，各预热一次、测一次。

| 部署方式 | Prefill token/s | Prefill TTFT 秒 | 普通 decode token/s | MTP3 token/s |
| --- | ---: | ---: | ---: | ---: |
| 串行，缓存关闭 | 919.43 | 4.456 | 22.36 | 30.08 |
| 串行，每卡缓存 8 GiB | 902.92 | 4.538 | 26.13 | 39.84 |
| 串行，固定 12 层专家 | 960.99 | 4.273 | 23.99 | 32.92 |
| TP2，缓存关闭 | 1240.71 | 3.303 | 20.60 | 32.63 |
| TP2，每卡缓存 8 GiB | 1254.39 | 3.267 | 27.71 | 46.29 |
| TP2，固定 12 层专家 | 1321.08 | 3.102 | 22.57 | 37.82 |

本轮 TP2 + 每卡 8 GiB 专家缓存的普通/MTP 解码最快；TP2 固定 12 层专家的 prefill 最快，并降低了解码阶段的主机 RSS。串行固定层配置的主机内存占用最低，但 GPU1 峰值达到 20.30 GiB，增大上下文时要重新评估显存余量。

Prefill 取原生前向计时，decode 排除首 token 等待；表中 TTFT 是 4096-token prefill 请求的首 token 延迟。跨部署生成内容可能不同，单轮数据不代表所有任务。显存、主机 RSS/HWM、复测脚本和详细口径见 [Flash-Next 双 2080 Ti 实测](benchmarks/qwen4_exp.md)。

## TP Decode 的 CUDA Graph

~~~bash
FASTLLM_CUDA_GRAPH=1 ftllm server /data/models/qwen3.8-flash-next \
  --tp 4 --atype float16 --chunked_prefill_size 1024
~~~

单 token TP decode 在支持的 CUDA 路径上分别捕获第 0 层和 PLE 后的主干。第 0 层的小图随请求 KV 缓存保留；缓存地址变化时，各 rank 一起重新捕获。捕获失败时统一回退普通算子提交。

TP 调度保留当前 host token，PLE 查表无需等待 token 从 GPU 读回；查出的行使用请求独立的 pinned buffer，沿当前 worker stream 异步搬运并执行投影。历史 token 和卷积状态仍在正常 PLE 执行位置更新。

`FASTLLM_CUDA_GRAPH` 统一控制是否启用图，PLE 搬运复用自动应用于单 token TP 路径。CPU/NUMA 混合推理和专家缓存的调度策略保持原有行为。

TP 的 FP16 `lm_head` 自动按词表行分片，prefill 和 decode 共用；分片起点按 256 行对齐，保持 CUDA top1 的同分选择顺序。简单贪婪采样按最小输出长度要求在分片内屏蔽 EOS/stop token，再汇总各卡候选；其他采样、返回 logits 和调试 dump 在第 0 卡汇总完整 logits，再执行原逻辑。不满足分片条件的 head 保留复制方式。此优化无需额外环境变量，也不依赖 CUDA Graph。

比较性能时固定实际输入/输出 token 数、提示词和采样配置，先预热再测量；Nsight Systems 波形用于解释等待来源，吞吐以未开启 profiler 的结果为准。

## PLE 磁盘模式

主机内存不足时：

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

`--ngram_device disk` 会从 checkpoint 的 Safetensors 文件按行读取 PLE 表，显著降低常驻内存，但增加小块随机读取。建议使用高速 SSD，并单独测量 Decode 抖动和操作系统页缓存占用。

## 长上下文与 Triton

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --triton
~~~

`--triton` 只在当前 Python 环境能导入 Triton 时启用，否则自动回退到内置 CUDA。实际上下文仍受模型原生上限和 KV Cache 容量限制。

SM75（如 RTX 2080 Ti）的 chunk GDN prefill 需要支持 SM75 Tensor Core 的
Triton（已验证 3.2.0）。由于 FastLLM 默认依赖 Triton 3.6 以上，建议单独
安装编译器，保留以下两个必要的环境变量：

~~~bash
python3 -m venv ~/.venvs/fastllm-triton-sm75
~/.venvs/fastllm-triton-sm75/bin/pip install 'triton==3.2.0' setuptools

FASTLLM_CUDA_TRITON=1 \
FASTLLM_CUDA_TRITON_PYTHON="$HOME/.venvs/fastllm-triton-sm75/bin/python" \
ftllm server /data/models/qwen3.8-flash-next \
  --device cuda --moe_device numa
~~~

此命令不要再加 `--triton`，该参数会改用运行 FastLLM 的 Python 环境。
端口、缓存目录和日志无需额外配置。若编译器仅生成标量乘加，FastLLM 会
回退原生 CUDA，同一进程内失败的形状不会逐层重试。更换编译器后，应停止
旧编译服务并重启 FastLLM。

该路径要求 FP16 激活、GDN 算子内部 chunk 大小 64、K/V head dimension 128，至少两个内部 chunk；这里的 64 不是 `--chunked_prefill_size`。
Qwen4 的 recurrent state 保持 FP32，但 prefill 激活由原路径的 FP32 改为 FP16，
因此输出不保证逐位一致。
单 token 解码、MTP 验证和 SM80 及更新 GPU 的原有选择规则不变。

## MTP 推测解码

~~~bash
ftllm server /data/models/qwen3.8-flash-next \
  --device cuda --moe_device numa \
  --mtp 4
~~~

`--mtp` 设置每轮 draft token 数，当前最大为 8。Qwen3.8-Flash-Next MTP 目前要求简单贪婪采样，目标网络运行在 CUDA，MoE 可放在 CUDA 或 NUMA；条件不满足时会自动回退普通解码。

Qwen3.8-Flash-Next 的跨请求前缀快照保存目标网络和可用的 MTP 状态。开启 MTP 后遇到不含草稿状态的旧快照，会先重算并生成兼容快照，后续请求可继续使用 MTP。

## TP 混合推理的前缀缓存

`--tp 2 --moe_device numa --prefix_cache true` 支持跨请求复用前缀。所有 rank 必须持有同一 token 前缀的完整快照，才会恢复各自的 KV、线性注意力和 QSA 状态，以及第 0 卡的 PLE 历史；缺失或恢复失败时重新计算。图片、视频请求不使用仅按 token 匹配的跨请求快照。

`FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_MB` 是整个 TP 实例的快照预算（默认 4096 MiB），按 rank 均分。默认快照间隔为 16 页，默认页长 128，即至少累计 2048 个 token 才记录；较短请求不一定产生快照。命中后仍至少计算一个未缓存 token，重复完整提示词时可复用更早的分块快照。

## 思考与工具调用

~~~bash
ftllm server /data/models/qwen4-exp \
  --enable_thinking true \
  --tool_call_parser auto
~~~

当前服务支持思考内容分离和 Qwen 工具协议。API 请求可通过 `chat_template_kwargs.enable_thinking` 按请求关闭思考。

## Benchmark

- [Qwen4 / Flash-Next 混合部署性能与历史冒烟](benchmarks/qwen4_exp.md)
- [架构、精度对齐和验证详情](qwen4_exp.md)
- [Benchmark 索引](benchmark.md)

已提供本机 NVFP4 双卡混推的 4096-token prefill 和单并发 decode 实测；旧 FP8 短输入冒烟单独保留。多并发吞吐、超长上下文和 Agent 自然结束需要另行验证。
