# DeepSeek-V4 Sparse Attention Benchmark

[English](deepseek_v4_en.md) · [Benchmark 索引](../benchmark.md) · [完整稀疏注意力分析](../deepseek_v4_sparse_attention.md) · [部署指南](../deepseek.md)

## 测试配置

- 模型：DeepSeek-V4 / DeepSeek-V4-Flash。
- 并行：TP8。
- 测试 CUDA 架构：SM120；计算路径支持 SM120/SM121。
- 输入/输出：128 / 256 token。
- 固定开启 CUDA Graph 和自定义 AllReduce。
- 原始报告没有记录 GPU 具体 SKU、显存和互联，因此不能将结果标注为某一款消费卡或数据中心卡。
- 未记录测量日期与源码版本；这些数据不能视为当前 HEAD 的性能。

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

## 实测结果

| 指标 | 实测值 |
| --- | ---: |
| TTFT | 405.47 ms |
| TPOP | 10.01 ms/token |
| Decode throughput | 99.93 token/s |
| Total throughput | 86.57 token/s |
| Total time | 2.9572 s |

CUDA Graph trace 中，SM120 split + merge 稀疏注意力每层平均为 5.796 µs。这是短上下文测量，不能外推到长上下文、其他硬件或混合专家放置。

算子接口、数值限制和回退行为见[稀疏注意力说明](../deepseek_v4_sparse_attention.md)。
