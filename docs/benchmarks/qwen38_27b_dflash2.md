# Qwen3.8-27B-FP8 DFlash2 / MTP Benchmark

[Benchmark 索引](../benchmark.md) · [部署指南](../qwen3.md)

## 测试配置

- 日期：2026-08-19，FastLLM commit `161962de`。
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
| MTP5 | 1024 | 2 | 273.65 token/s |
| DFlash B8 | 1024 | 3 | 381.00 token/s |
| DFlash B8 | 4096 | 2 | 377.52 token/s |

固定流结果表明该 checkpoint 的推荐配置为原生 B8。

## GSM8K 端到端

128 题、单并发，temperature=1、top_p=0.95、top_k=20、最多输出 4096 token：

| 配置 | 输出 token | 总耗时 | 吞吐 | 正确 |
| --- | ---: | ---: | ---: | ---: |
| DFlash B8 | 70878 | 342.308 s | 207.06 token/s | 120/128 |

这是包含采样、不同 EOS 和不同输出长度的服务端到端吞吐，不能与固定流的 381.00 token/s 混为同一指标。

MTP5 配置保留相同目标模型、TP2、embedding 和容量参数，移除上述三个 `--speculative_*` 参数，改用 `--mtp 5`。DFlash 和 `--mtp` 不能同时启用。本页没有单卡或 CPU / NUMA 的实测速率。

## 接受率与采样语义

上述 GSM8K 测量中，七个候选位置的累计接受率分别为 86.68%、71.18%、58.33%、48.29%、39.90%、33.37%、27.69%；每轮平均提交 4.654 个 token。
接受率取决于输入和采样参数，不能视为固定配置的常数。

随机采样使用标准拒绝采样：候选分布为 q，目标分布为 p，以 `min(1, p/q)` 接受；首次拒绝后从归一化的 `max(p-q, 0)` 采样，全部接受时额外采样目标 token。
这保证的是给定目标条件分布下的采样语义；不同批量前向仍可能有浮点舍入差异，不保证相同随机种子产生逐 token 相同的输出。
