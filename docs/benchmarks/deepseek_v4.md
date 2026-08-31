# DeepSeek-V4 Sparse Attention Benchmark

[English](deepseek_v4_en.md) · [Benchmark 索引](../benchmark.md) · [完整稀疏注意力分析](../deepseek_v4_sparse_attention.md) · [部署指南](../deepseek.md)

## 测试配置

- 模型：DeepSeek-V4 / DeepSeek-V4-Flash。
- 并行：TP8。
- CUDA 架构：SM120/SM121。
- 输入/输出：128 / 256 token。
- 固定开启 CUDA Graph 和自定义 AllReduce。
- 原始报告没有记录 GPU 具体 SKU、显存和互联，因此不能将结果标注为某一款消费卡或数据中心卡。

## 推荐 TP8 启动命令

~~~bash
FASTLLM_CUDA_GRAPH=1 \
FASTLLM_CUDA_CUSTOM_ALLREDUCE=1 \
ftllm server /data/models/deepseek-v4 \
  --model_name deepseek-v4 \
  --tp 8 --triton \
  --max_batch 1
~~~

`--triton` 会检测当前 Python 环境中的 Triton；不可用时回退到内置 CUDA。SM120 稀疏路径还可以通过 `FASTLLM_CUDA_TRITON_DSV4_SPARSE_SM120` 单独控制。

## 全模型结果

| 指标 | 优化前 | SM120 fast path | 变化 |
| --- | ---: | ---: | ---: |
| TTFT | 415.27 ms | 405.47 ms | -2.36% |
| TPOP | 13.51 ms/token | 10.01 ms/token | -25.91% |
| Decode throughput | 74.03 token/s | 99.93 token/s | +34.99% |
| Total throughput | 66.32 token/s | 86.57 token/s | +30.53% |
| Total time | 3.8600 s | 2.9572 s | -23.39% |

## 稀疏 Attention Core

相同机器、TP8、CUDA Graph node trace：

| 路径 | 每层平均 GPU 时间 |
| --- | ---: |
| FastLLM 原串行 Triton | 81.859 µs |
| vLLM FlashInfer SM120 主核 + merge | 13.660 µs |
| FastLLM SM120 split + merge | 5.796 µs |

这是短上下文口径，不能外推到 top-k 512/1024 的长上下文。

## 其他设备布局

| 布局 | 建议命令 | 当前速度 |
| --- | --- | --- |
| TP8 SM120/SM121 | 使用上方推荐命令 | Decode 99.93 token/s |
| 单 GPU + NUMA | `ftllm server /data/models/deepseek-v4 --device cuda --moe_device numa` | 待实测 |
| GPU + CPU | `ftllm server /data/models/deepseek-v4 --device cuda --moe_device cpu` | 待实测 |
| GPU + NUMA + 磁盘 | `ftllm server /data/models/deepseek-v4 --device cuda --moe_device "{'cuda':1,'numa':8,'disk':1}"` | 待实测 |

不同布局的瓶颈完全不同，不能用 TP8 数据估算混合 MoE 或磁盘专家速度。

数值回归、kernel ABI、环境变量和后续限制见[完整分析](../deepseek_v4_sparse_attention.md)。
