# DeepSeek-V4 / DeepSeek-V4-Flash 部署指南

[English](deepseek_en.md) · [返回 README](../README.md) · [混合推理](mixforward.md) · [Benchmark](benchmarks/deepseek_v4.md)

本文面向 DeepSeek-V4 与 DeepSeek-V4-Flash。早期 DeepSeek 模型仍可兼容运行，但不再作为本指南的主要示例。

## API Server 快速启动

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --model_name DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 8080
~~~

对于大型 MoE checkpoint，FastLLM 默认会优先使用 CUDA 运行非专家部分，并根据构建能力将专家层放到 CPU 或 NUMA。正式部署时建议显式指定设备布局。

## 按设备选择启动命令

### 单 GPU + NUMA

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

单路 CPU 机器可以使用 `--moe_device cpu`。

### 多 GPU 张量并行

当 GPU 总显存足以容纳目标 checkpoint 时：

~~~bash
ftllm server /data/models/deepseek-v4 \
  --tp 0,1 \
  --max_batch 8 \
  --gpu_mem_ratio 0.9
~~~

DeepSeek-V4 的 `--tp` 会选择 MultiCUDA 张量并行路径。更多 GPU 可以显式写成 `--tp 0,1,2,3` 或使用对应的裸数字数量。

### GPU、NUMA 与磁盘混合专家

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --device cuda \
  --moe_device "{'cuda':1,'numa':8,'disk':1}" \
  --chunked_prefill_size 8192
~~~

比例表示 MoE 层在不同设备上的相对分配。磁盘只适合容量补充，并强依赖 SSD 随机读取性能。更多布局见[混合推理指南](mixforward.md)。

### 多 GPU + NUMA 混合 MoE

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --device cuda \
  --moe_device "{'multicuda:0,1':15,'numa':85}"
~~~

该方式让部分专家层使用两卡张量并行，其余专家层运行在 NUMA 内存中。比例应根据显存、内存带宽和实际吞吐调整。

## 内置 DSpark

包含内置 DSpark 权重和配置的 DeepSeek-V4 checkpoint 可以使用：

~~~bash
ftllm server /data/models/deepseek-v4 \
  --dspark 7
~~~

`--dspark` 不能小于 checkpoint 训练时的 block 大小。服务启动时会校验模型架构和 DSpark 配置。

## 长上下文

~~~bash
ftllm server /data/models/deepseek-v4 \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

实际上下文上限取模型原生上限、命令行上限和 KV Cache 容量的较小值。

## SM120 TP8 稀疏注意力路径

在支持的 SM120/SM121 CUDA 设备上，TP8 稀疏注意力测试使用：

~~~bash
FASTLLM_CUDA_GRAPH=1 \
FASTLLM_CUDA_CUSTOM_ALLREDUCE=1 \
ftllm server /data/models/deepseek-v4 \
  --tp 8 --triton \
  --max_batch 1
~~~

该命令是已记录 TP8 Benchmark 的推荐起点，不代表其他 GPU 架构也应使用相同设置。实测速度与限制见 [DeepSeek-V4 Benchmark](benchmarks/deepseek_v4.md) 和[稀疏注意力分析](deepseek_v4_sparse_attention.md)。

## 思考与工具调用

~~~bash
ftllm server /data/models/deepseek-v4 \
  --enable_thinking true \
  --tool_call_parser auto
~~~

DeepSeek-V4 使用模型对应的编码和工具协议。API 端建议保留模型原始模板与 parser 自动选择。

## Benchmark

- [DeepSeek-V4 TP8 SM120 性能摘要](benchmarks/deepseek_v4.md)
- [DeepSeek-V4 稀疏注意力详细分析](deepseek_v4_sparse_attention.md)
- [通用 Benchmark 入口](benchmark.md)

尚未收录单 GPU + NUMA 和磁盘专家布局的正式吞吐；这些命令是部署起点，不能用 TP8 数据外推速度。
