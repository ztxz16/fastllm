# Qwen3.6-27B-FP8 Benchmark

[Benchmark 索引](../benchmark.md) · [完整性能分析](../qwen36_27b_fp8_tp2_benchmark_20260809.md) · [部署指南](../qwen3.md)

## 测试配置

- 日期：2026-08-09。
- 模型：Qwen3.6-27B-FP8。
- 硬件：2 × RTX 5090，每卡 32607 MiB，PCIe PIX，无 NVLink。
- 并行：Tensor Parallel 2。
- FastLLM：0.1.7.1，commit `6e01b27663c0`。
- 上下文上限：1024。
- 每个请求固定输出 256 token，关闭 thinking、Prefix Cache 和 MTP。
- 结果是客户端可见的聚合 Decode 吞吐，正式测试 3 次取中位数。

## 推荐 TP2 启动命令

~~~bash
FASTLLM_CUDA_GRAPH=1 \
ftllm server /data/models/Qwen3.6-27B-FP8 \
  --model_name qwen3.6-27b \
  --dtype auto \
  --tp 2 --cuda_embedding \
  --max_batch 64 \
  --max_context_length 1024 \
  --gpu_mem_ratio 0.98 \
  --prefix_cache false \
  --enable_thinking false
~~~

这是复现实测环境的起点。生产环境可以重新开启 Prefix Cache，并根据目标上下文降低 `--max_batch`。

## 实测 Decode 吞吐

| Batch / 并发 | FastLLM token/s |
| ---: | ---: |
| 1 | 94.8087 |
| 2 | 183.0459 |
| 4 | 338.1496 |
| 8 | 630.5965 |
| 16 | 1163.9353 |
| 32 | 1854.3000 |
| 64 | 2930.4490 |
| 128 | 不可用：预热 OOM |
| 256 | 不可用：容量/OOM |

## 其他设备布局

| 布局 | 建议命令 | 当前速度 |
| --- | --- | --- |
| 单张 CUDA | `ftllm server /data/models/Qwen3.6-27B-FP8 --device cuda --max_batch 1` | 待实测 |
| 双卡 TP2 | 使用上方推荐命令 | B1 94.81；B64 2930.45 token/s |
| CPU / NUMA | `ftllm server /data/models/Qwen3.6-27B-FP8 --device numa -t 64` | 待实测 |

单卡是否能容纳模型和所需 KV Cache 取决于 checkpoint 的实际权重布局与上下文设置。

## 如何复测

~~~bash
python3 test/benchmark/decode.py /data/models/Qwen3.6-27B-FP8 \
  --tp 2 \
  --batch-size 16 \
  --max_batch 16 \
  --prefill-length 512 \
  --max-tokens 256
~~~

当前脚本和历史报告的请求构造、采样与计时窗口可能不同。发布新数据时必须保留完整 JSON，并标明是否严格复现历史口径。

三框架对比、Nsight 波形和逐算子归因见[完整性能分析](../qwen36_27b_fp8_tp2_benchmark_20260809.md)。
