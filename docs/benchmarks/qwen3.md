# Qwen3.5 / Qwen3.6 / Qwen3.8 Benchmark

[English](qwen3_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../qwen3.md)

Qwen3 系列按具体 checkpoint、精度和推理路径分别记录，不能把不同版本的数据合并比较。

## 已有实测

| 模型与路径 | 测试设备 | 代表结果 | 完整记录 |
| --- | --- | --- | --- |
| Qwen3.6-27B-FP8 TP2 | 2 × RTX 5090 32GB，PCIe，无 NVLink | Decode：B1 94.81、B16 1163.94、B64 2930.45 token/s | [启动命令与完整结果](qwen36_27b_fp8.md) |
| Qwen3.8-27B-FP8 DFlash2 TP2 | 2 × RTX 5090 32GB，PCIe，无 NVLink | 固定流：target 81.45、MTP5 273.65、DFlash B8 381.00 token/s | [启动命令与完整结果](qwen38_27b_dflash2.md) |

## Qwen3.5

仓库目前没有可发布的 Qwen3.5 标准吞吐数据。以下命令仅作为复测起点，速度均待实测：

| 设备布局 | 建议启动命令 | 速度 |
| --- | --- | --- |
| 单张 CUDA | `ftllm server /data/models/qwen3.5 --device cuda --max_batch 1` | 待实测 |
| 双卡 TP2 | `ftllm server /data/models/qwen3.5 --tp 2 --max_batch 8` | 待实测 |
| CPU / NUMA | `ftllm server /data/models/qwen3.5 --device numa -t 64` | 待实测 |

发布新结果时，请记录 checkpoint、精度、FastLLM commit、设备与互联、输入输出长度、batch、完整命令和统计口径。测试脚本说明见 [Benchmark 工具](../../test/benchmark/README.md)。
