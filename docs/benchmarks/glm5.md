# GLM-5 / GLM-5.3-Flash Benchmark

[English](glm5_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../glm5.md)

仓库目前没有可对外发布的 GLM-5 / GLM-5.3-Flash 完整吞吐表。下列配置是建议的复测起点。

| 设备布局 | 建议启动命令 | 速度 |
| --- | --- | --- |
| GPU + NUMA MoE | `ftllm server /data/models/glm5 --device cuda --moe_device numa --chunked_prefill_size 8192` | 待实测 |
| GPU + CPU MoE | `ftllm server /data/models/glm5 --device cuda --moe_device cpu --chunked_prefill_size 8192` | 待实测 |
| 量化 KV-B CPU / NUMA | `ftllm server /data/models/glm5.2-quantized-kvb --device numa --moe_device numa -t 64` | 待实测 |

结果必须注明具体模型版本、量化格式以及实际启用的 DSA/KDA 和分页缓存路径。测试脚本说明见 [Benchmark 工具](../../test/benchmark/README.md)。
