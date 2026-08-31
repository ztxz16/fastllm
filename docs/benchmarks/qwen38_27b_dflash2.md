# Qwen3.8-27B-FP8 DFlash2 / MTP Benchmark

[Benchmark 索引](../benchmark.md) · [完整 DFlash2 报告](../dflash2_qwen38_27b_tp2_20260819.md) · [部署指南](../qwen3.md)

## 测试配置

- 日期：2026-08-19。
- 目标模型：Qwen3.8-27B-FP8。
- Draft：独立 Qwen3.8-27B-DFlash2 checkpoint。
- 硬件：2 × RTX 5090，每卡 32607 MiB，PCIe，无 NVLink。
- DFlash2 推荐 block：8（包含 anchor，即 7 个实际 draft token）。

## 推荐 DFlash2 启动命令

~~~bash
ftllm server /data/models/Qwen3.8-27B-FP8 \
  --model_name qwen3.8-dflash2 \
  --dtype auto \
  --tp 2 --cuda_embedding \
  --max_batch 1 \
  --tokens 8192 \
  --gpu_mem_ratio 0.98 \
  --speculative_algorithm dflash \
  --speculative_draft_model_path /data/models/Qwen3.8-27B-DFlash2 \
  --speculative_num_draft_tokens 8
~~~

Draft checkpoint 的 `architectures` 必须包含 `DFlash2DraftModel`。省略 block 参数时使用 checkpoint 中的原生值。

## 固定 token 流 Decode

固定 64-token 输入，在 TTFT 后强制生成固定长度，用于排除 EOS 和输出长度差异。

| 路径 | 输出长度 | 运行次数 | 中位吞吐 |
| --- | ---: | ---: | ---: |
| Target only | 512 | 历史 sweep | 81.45 token/s |
| MTP5 | 1024 | 2 | 273.65 token/s |
| DFlash B6 | 1024 | 2 | 287.29 token/s |
| DFlash B8 | 1024 | 3 | 381.00 token/s |
| DFlash B8 | 4096 | 2 | 377.52 token/s |

固定流结果表明当前 checkpoint 应优先使用原生 B8，而不是旧的 B6 配置。

## GSM8K 端到端

128 题、非 greedy、单并发：

| 版本 | 输出 token | 总耗时 | 吞吐 | 正确 |
| --- | ---: | ---: | ---: | ---: |
| 初始 FastLLM B6 | 72584 | 587.664 s | 123.51 token/s | 120/128 |
| 最终 FastLLM B8 | 70878 | 342.308 s | 207.06 token/s | 120/128 |

这是包含采样、不同 EOS 和不同输出长度的服务端到端吞吐，不能与固定流的 381.00 token/s 混为同一指标。

## 其他设备布局

| 布局 | 建议命令 | 当前速度 |
| --- | --- | --- |
| 双卡 TP2 target only | 去掉 `--speculative_*` 参数 | 81.45 token/s，固定流 |
| 双卡 TP2 MTP | 目标命令追加 `--mtp 5` | 273.65 token/s，固定流 |
| 双卡 TP2 DFlash B8 | 使用上方推荐命令 | 381.00 token/s，固定流 |
| 单张 CUDA | `ftllm server /data/models/Qwen3.8-27B-FP8 --device cuda` | 待实测 |
| CPU / NUMA | `ftllm server /data/models/Qwen3.8-27B-FP8 --device numa -t 64` | 待实测 |

DFlash 和 `--mtp` 不能同时启用。

性能优化过程、SGLang 对比、接受率和准确性分析见[完整报告](../dflash2_qwen38_27b_tp2_20260819.md)。
