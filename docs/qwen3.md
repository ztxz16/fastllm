# Qwen3.5 / Qwen3.6 / Qwen3.8 部署指南

[English](qwen3_en.md) · [返回 README](../README.md) · [Qwen4-Exp](qwen4.md) · [Benchmark](benchmarks/qwen3.md)

本文面向当前 Qwen3.5、Qwen3.6 和 Qwen3.8 文本模型。Qwen4-Exp / Qwen3.8-Flash-Next 使用独立架构，请阅读 [Qwen4-Exp 部署说明](qwen4.md)。

不同 checkpoint 可能是稠密、MoE、FP8、NVFP4 或其他量化格式。首次启动建议保留 `--dtype auto`，再根据显存和实测结果调整。

## API Server 快速启动

~~~bash
ftllm server /data/models/qwen \
  --model_name qwen \
  --host 0.0.0.0 --port 8080
~~~

FastLLM 会根据模型配置选择默认精度和设备。生产部署建议显式设置 `--model_name`、`--max_batch`、上下文上限和显存比例。

## 按设备选择启动命令

### 单张 NVIDIA GPU

适合能完整放入显存的稠密或量化 checkpoint：

~~~bash
ftllm server /data/models/qwen \
  --device cuda \
  --max_batch 8 \
  --gpu_mem_ratio 0.9
~~~

### 多张 NVIDIA GPU 张量并行

~~~bash
ftllm server /data/models/qwen \
  --device cuda --tp 0,1 \
  --cuda_embedding \
  --max_batch 16 \
  --gpu_mem_ratio 0.9
~~~

`--tp 0,1` 显式选择两张卡；也可以用 `--tp 2` 表示前两张可见 GPU。具体模型支持的并行路径和显存容量需要在目标机器上验证。

### GPU + NUMA 混合 MoE

适合专家权重无法全部放入显存的 MoE checkpoint：

~~~bash
ftllm server /data/models/qwen-moe \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

如果机器只有单路 CPU，可将 `--moe_device` 改为 `cpu`。NUMA 模式下线程数默认自动选择，也可以通过 `-t` 调整。

### CPU / NUMA

~~~bash
# 单路 CPU
ftllm server /data/models/qwen \
  --device cpu -t 32

# 多路 NUMA
ftllm server /data/models/qwen \
  --device numa -t 64
~~~

线程数应结合物理核心数和内存带宽实测，不建议直接照搬示例值。

## 长上下文与缓存

~~~bash
ftllm server /data/models/qwen \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

`--max_context_length` 是输入与输出合计上限。超出模型声明值时须指定有效的 `--rope_scaling`，容量不足会在启动时失败。Qwen3 的 YaRN 需要明确原始长度（常见为 32768）；具体配置和验证范围见[上下文扩展](context-length-extension-design.md)。实际窗口可从 `/v1/models` 查询。

## MTP

对于包含兼容 MTP 权重的模型：

~~~bash
ftllm server /data/models/qwen-with-mtp \
  --device cuda --mtp 4
~~~

`--mtp` 设置每轮 draft token 数，`0` 表示关闭，当前最大为 8。MTP 并不只属于某一个 Qwen 版本，是否可用取决于模型结构和 checkpoint。

Qwen3.5 GGUF 会直接使用文件中内置的 NextN/MTP 层。如果 GGUF 不含该层，可以挂载同结构模型中抽出的独立 `mtp.safetensors`：

~~~bash
ftllm server /data/models/qwen3.5-target.gguf \
  --device cuda --cuda_embedding \
  --draft /data/models/qwen3.5-fp8/mtp.safetensors \
  --draft_tokens 5
~~~

`mtp.safetensors` 的同目录必须有匹配的 `config.json`。也可以把 `--draft` 指向包含这两个文件的目录。加载时会校验 Qwen3.5 架构、层数、hidden size、head 数、词表和全部 MTP 张量形状；独立 MTP 当前只支持挂载到 GGUF target。MTP 会频繁访问词表投影，显存允许时应保留 `--cuda_embedding`，否则速度会明显下降。

## Qwen3.8 DFlash2

DFlash2 需要独立的 draft checkpoint：

~~~bash
ftllm server /data/models/qwen3.8-27b-fp8 \
  --model_name qwen3.8 \
  --tp 2 --cuda_embedding \
  --max_batch 1 \
  --speculative_algorithm dflash \
  --speculative_draft_model_path /data/models/qwen3.8-27b-dflash2 \
  --speculative_num_draft_tokens 8
~~~

当前推荐使用 draft checkpoint 的原生 block 大小。完整复现方式与实测速度见 [Qwen3.8 DFlash2 Benchmark](benchmarks/qwen38_27b_dflash2.md)。

## 思考与工具调用

~~~bash
# 关闭模型的 thinking 模板
ftllm server /data/models/qwen --enable_thinking false

# 自动选择工具调用解析器
ftllm server /data/models/qwen --tool_call_parser auto
~~~

API 请求也可以通过 `chat_template_kwargs.enable_thinking` 按请求控制思考模式。工具调用能力取决于具体 Instruct checkpoint 和 chat template。

## Benchmark

- [Qwen3.6-27B-FP8 双 RTX 5090 TP2](benchmarks/qwen36_27b_fp8.md)
- [Qwen3.8-27B DFlash2 / MTP 双 RTX 5090](benchmarks/qwen38_27b_dflash2.md)
- [通用 Benchmark 入口](benchmark.md)

速度只对文档中记录的模型、精度、硬件、batch 和上下文口径有效。
