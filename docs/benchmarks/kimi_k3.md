# Kimi-K3 Benchmark

[English](kimi_k3_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../kimi_k3.md)

仓库目前只有功能与协议验证，没有可发布的 Kimi-K3 标准吞吐数据。下列配置是建议的复测起点，不代表实测性能。

| 设备布局 | 建议启动命令 | 速度 |
| --- | --- | --- |
| GPU + NUMA MoE | `ftllm server /data/models/kimi-k3 --device cuda --moe_device numa --chunked_prefill_size 8192` | 待实测 |
| GPU + CPU MoE | `ftllm server /data/models/kimi-k3 --device cuda --moe_device cpu --chunked_prefill_size 8192` | 待实测 |
| GPU + NUMA + 磁盘专家 | `ftllm server /data/models/kimi-k3 --device cuda --moe_device "{'cuda':1,'numa':8,'disk':1}"` | 待实测 |

Kimi-K3 还支持外部 DSpark。对比时应分别记录普通 Decode 与 DSpark 的接受长度、TTFT 和持续吞吐，不能把功能回归结果作为速度数据。测试脚本说明见 [Benchmark 工具](../../test/benchmark/README.md)。
