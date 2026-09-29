# Qwen4-Exp / Qwen3.8-Flash-Next Benchmark

[English](qwen4_exp_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../qwen4.md) · [架构详情](../qwen4_exp.md)

## 2026-09-29：双 RTX 2080 Ti，Flash-Next NVFP4

本轮推理代码为提交 `d6ef5c784`，统一重测六种混合部署方式。完整启动命令见[部署指南](../qwen4.md)，数值与实际设备映射见[机器可读结果](qwen38_flash_next_2080ti_20260929.json)。

- 硬件：两张 **22 GiB 显存版** RTX 2080 Ti（每卡 22528 MiB），AMD EPYC 7452，32 核 / 64 线程，单 NUMA 节点，125.7 GiB RAM；Ubuntu 24.04.4，驱动 595.84。
- 模型：`Qwen3.8-Flash-Next-NVFP4` 文本推理，48 个主干 MoE 层。FP16 激活，PLE disk，28 计算线程，batch=1，KV 预算 1 GiB，`gpu_mem_ratio=0.95`。
- 统一关闭 CUDA Graph、Triton、前缀缓存、历史缓存和思考。普通与 MTP 模式共用已加载的 MTP 权重。
- 每项预热一次、正式测一次；decode 预热生成 128 token。输入为统一的代码生成提示词，填充到指定长度，所有配置使用相同输入 token。没有把之前其他版本、512-token 小块 prefill 或多轮平均结果混入表格。

### Prefill 与 decode

Prefill 使用 **4096 输入 token、`chunked_prefill_size=4096`、MTP=0**，完整请求只生成 1 个 token。吞吐取原生 `[Prompt]` 前向计时，包含得到首 token 所需的 logits/采样，不是单一 GPU kernel 时间；TTFT 则从原生请求提交计到首 token。

Decode 另用 512 输入、固定 512 输出，分别测普通解码和 MTP=3；吞吐为 `511 / (末 token 到达时间 − 首 token 到达时间)`，不含模型加载、预热或 TTFT。两个阶段都没有 HTTP/SSE 网络开销。

| 部署方式 | Prefill token/s | Prefill TTFT 秒 | 普通 decode token/s | MTP3 token/s |
| --- | ---: | ---: | ---: | ---: |
| 串行，缓存关闭 | 919.43 | 4.456 | 22.36 | 30.08 |
| 串行，每卡缓存 8 GiB | 902.92 | 4.538 | 26.13 | 39.84 |
| 串行，固定 12 层专家 | 960.99 | 4.273 | 23.99 | 32.92 |
| TP2，缓存关闭 | 1240.71 | 3.303 | 20.60 | 32.63 |
| TP2，每卡缓存 8 GiB | 1254.39 | 3.267 | 27.71 | 46.29 |
| TP2，固定 12 层专家 | 1321.08 | 3.102 | 22.57 | 37.82 |

不同部署的生成内容和 MTP 接受情况可能不同，以上是单轮部署效果比较，不是强制相同输出的算子微基准。固定 512 输出也不代表完整编程任务或 Agent 自然结束验收。

### 内存与显存

| 部署方式 | GPU0 峰值 GiB | GPU1 峰值 GiB | 解码 RSS GiB | 主机 HWM GiB |
| --- | ---: | ---: | ---: | ---: |
| 串行，缓存关闭 | 7.81 | 10.25 | 72.60 | 72.86 |
| 串行，每卡缓存 8 GiB | 15.83 | 18.26 | 79.64 | 80.42 |
| 串行，固定 12 层专家 | 12.46 | 20.30 | 56.56 | 65.05 |
| TP2，缓存关闭 | 10.76 | 10.84 | 77.05 | 80.79 |
| TP2，每卡缓存 8 GiB | 18.78 | 18.86 | 84.09 | 88.37 |
| TP2，固定 12 层专家 | 18.33 | 18.41 | 61.05 | 80.62 |

显存为每 2 秒采样的全程峰值，包含大块 prefill；解码 RSS 为正式解码阶段采样中位数。HWM 包含加载峰值，不能拿较低的稳态 RSS 作为启动所需内存。所有组进程 swap 为零，正常完成后退出，未出现 OOM 或 CUDA 错误。

固定 12 层时，TP2 使用 `--moe_device_layers 36`，前 12 层专家跨两卡分片。串行则使用 `cudapp=1:7` 主干和 `{'cuda:0':6,'cuda:1':6,'numa':36}` 专家映射；它与其他串行组的 24/24 主干分界不同。固定层会占用冷门专家的显存，动态缓存可跨层保留活跃专家，不能只按相同容量推断速度。

### 复测

使用仓库中的[同口径测量脚本](../../test/benchmark/qwen38_hybrid.py)，在已安装或构建的 `ftllm` Python 环境运行。脚本固定测试一轮，输出目录必须不存在：

~~~bash
CUDA_VISIBLE_DEVICES=0,1 FT_NUMAS=1 FASTLLM_CUDA_GRAPH=0 \
python test/benchmark/qwen38_hybrid.py /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --output-dir /tmp/qwen38-tp-cache8 \
  --tp 2 --moe_device numa --moe_cuda_cache 8g \
  --atype float16 --ngram_device disk --threads 28 --mtp 3 \
  --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95 \
  --chunked_prefill_size 4096 --prefix_cache false --cache_history false \
  --enable_thinking false
~~~

把布局参数替换为部署指南中的其他组合，串行时清除 `FASTLLM_TP`。脚本自动做 4096-token prefill 和两种解码模式；`results.json` 保存 TTFT、完整输出 token 和到达时间，终端 `[Prompt]` 行保存原生 prefill 吞吐。原生计时与内置 `ftllm benchmark` 使用 `输入 token / TTFT` 得到的 Prefill 指标口径略有不同。

## 历史 FP8 / Blackwell 冒烟测试

以下使用另一模型量化格式和硬件，单独保留作为功能验证，不能与上面的 NVFP4 混推数据直接比较。

### 测试范围

- 模型：Qwen4-Exp / Qwen3.8-Flash-Next FP8 文本模型。
- 硬件：RTX PRO 6000 Blackwell 96GB，72 CPU 核，双 NUMA 节点。
- 输入/输出：2 个输入 token、最多 1 个输出 token。
- 目的：验证 CPU、CUDA + CPU、CUDA + NUMA 三条路径可以完成前向。
- 限制：这是极短输入的冒烟数据，不是持续 Prefill 或 Decode Benchmark。

### 建议命令

#### CPU + CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cpu --moe_device cpu \
  --atype float32 --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

#### CUDA + CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device cpu \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

#### CUDA + NUMA MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

#### 主机内存不足：磁盘 PLE

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

这组 FP8 冒烟没有测量磁盘 PLE 速度。它降低常驻内存，但会增加随机 I/O，建议使用高速 SSD。

### 实测结果

| 执行路径 | TTFT | 短输入 Prefill |
| --- | ---: | ---: |
| CPU + CPU MoE | 274.44 ms | 7.29 token/s |
| CUDA + CPU MoE | 78.03 ms | 25.63 token/s |
| CUDA + NUMA MoE | 69.68 ms | 28.70 token/s |

这些值只说明相同冒烟输入下三条路径的相对开销。由于输出最多为 1 token，不能从中得到稳定 Decode token/s。

### 正式部署起点

~~~bash
ftllm server /data/models/qwen4-exp \
  --model_name qwen4-exp \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

PLE 表默认驻留 CPU。内存不足时追加 `--ngram_device disk`。

### 待补数据

- 单并发持续 Decode。
- 不同输入长度的 Prefill 与 TTFT。
- 多 batch 服务吞吐。
- PLE CPU 与磁盘模式的 RSS、I/O 和 Decode 对比。
- 多 GPU 与不同量化 checkpoint。

原始验证和精度对齐信息见 [Qwen4-Exp 文档](../qwen4_exp.md)。
