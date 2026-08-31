# Laguna Benchmark

[English](laguna_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../laguna.md)

仓库目前只有 Laguna 多卡、CUDA Graph、混合 MoE 和量化路径的功能验证，没有统一硬件上的正式速度表。下列配置是建议的复测起点。

| 设备布局 | 建议启动命令 | 速度 |
| --- | --- | --- |
| 四卡 TP4 | `ftllm server /data/models/laguna --tp 4 --max_batch 8 --gpu_mem_ratio 0.9` | 待实测 |
| 八卡 TP8 | `ftllm server /data/models/laguna --tp 0,1,2,3,4,5,6,7 --max_batch 16 --gpu_mem_ratio 0.9` | 待实测 |
| GPU + NUMA MoE | `ftllm server /data/models/laguna --device cuda --moe_device numa --chunked_prefill_size 8192` | 待实测 |

发布结果时应注明 GPU 互联、checkpoint 精度、CUDA Graph 状态、上下文长度和 batch。NVFP4 与 INT4_GROUP32 必须分开记录。测试脚本说明见 [Benchmark 工具](../../test/benchmark/README.md)。
