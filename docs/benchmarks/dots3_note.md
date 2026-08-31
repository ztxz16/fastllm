# Dots3-Note Benchmark

[English](dots3_note_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../dots3_note.md)

仓库目前记录了算子、长缓存和 API 回归，但没有完整的 Dots3-Note 设备吞吐数据。下列配置是建议的复测起点。

| 设备布局 | 建议启动命令 | 速度 |
| --- | --- | --- |
| GPU + NUMA MoE | `ftllm server /data/models/dots3-note --device cuda --moe_device numa --chunked_prefill_size 8192` | 待实测 |
| GPU + CPU MoE | `ftllm server /data/models/dots3-note --device cuda --moe_device cpu --chunked_prefill_size 8192` | 待实测 |
| 长上下文 | `ftllm server /data/models/dots3-note --device cuda --moe_device numa --max_context_length 131072 --chunked_prefill_size 8192` | 待实测 |

长上下文测试应分别记录不同输入长度的 TTFT、Prefill token/s、显存占用和持续 Decode，避免只报告短输入结果。测试脚本说明见 [Benchmark 工具](../../test/benchmark/README.md)。
