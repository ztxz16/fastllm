# Qwen4-Exp / Qwen3.8-Flash-Next Benchmark

[English](qwen4_exp_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../qwen4.md) · [架构详情](../qwen4_exp.md)

## 测试范围

- 模型：Qwen4-Exp / Qwen3.8-Flash-Next FP8 文本模型。
- 硬件：RTX PRO 6000 Blackwell 96GB，72 CPU 核，双 NUMA 节点。
- 输入/输出：2 个输入 token、最多 1 个输出 token。
- 目的：验证 CPU、CUDA + CPU、CUDA + NUMA 三条路径可以完成前向。
- 限制：这是极短输入的冒烟数据，不是持续 Prefill 或 Decode Benchmark。

## 建议命令

### CPU + CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cpu --moe_device cpu \
  --atype float32 --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

### CUDA + CPU MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device cpu \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

### CUDA + NUMA MoE

~~~bash
ftllm benchmark /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --atype float16 --moe_atype float32 \
  --threads 64 \
  --input_tokens 2 --output_tokens 1 \
  --batch 1 --warmup 0 --temperature 0
~~~

### 主机内存不足：磁盘 PLE

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

磁盘 PLE 没有正式速度记录。它降低常驻内存，但会增加随机 I/O，建议使用高速 SSD。

## 实测结果

| 执行路径 | TTFT | 短输入 Prefill |
| --- | ---: | ---: |
| CPU + CPU MoE | 274.44 ms | 7.29 token/s |
| CUDA + CPU MoE | 78.03 ms | 25.63 token/s |
| CUDA + NUMA MoE | 69.68 ms | 28.70 token/s |

这些值只说明相同冒烟输入下三条路径的相对开销。由于输出最多为 1 token，不能从中得到稳定 Decode token/s。

## 正式部署起点

~~~bash
ftllm server /data/models/qwen4-exp \
  --model_name qwen4-exp \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

PLE 表默认驻留 CPU。内存不足时追加 `--ngram_device disk`。

## 待补数据

- 单并发持续 Decode。
- 不同输入长度的 Prefill 与 TTFT。
- 多 batch 服务吞吐。
- PLE CPU 与磁盘模式的 RSS、I/O 和 Decode 对比。
- 多 GPU 与不同量化 checkpoint。

原始验证和精度对齐信息见 [Qwen4-Exp 文档](../qwen4_exp.md)。
