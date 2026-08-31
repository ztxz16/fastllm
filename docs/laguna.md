# Laguna 部署指南

[English](laguna_en.md) · [返回 README](../README.md) · [Benchmark](benchmarks/laguna.md)

FastLLM 的 Laguna 路径支持长上下文缓存、CUDA Graph、多卡张量并行、混合 MoE、NVFP4 和 INT4_GROUP32。

## API Server 快速启动

~~~bash
ftllm server /data/models/laguna \
  --model_name laguna \
  --host 0.0.0.0 --port 8080
~~~

## 多 GPU 张量并行

当多卡总显存能够容纳目标 checkpoint 时：

~~~bash
ftllm server /data/models/laguna \
  --tp 4 \
  --max_batch 8 \
  --gpu_mem_ratio 0.9
~~~

Laguna 会根据 `--tp` 选择主设备和专家张量并行布局。四卡场景下 CUDA slab 默认值也会按模型布局自动调整。

八卡示例：

~~~bash
ftllm server /data/models/laguna \
  --tp 0,1,2,3,4,5,6,7 \
  --max_batch 16 \
  --gpu_mem_ratio 0.9
~~~

### GPU + NUMA 混合 MoE

~~~bash
ftllm server /data/models/laguna \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192
~~~

## 长上下文

~~~bash
ftllm server /data/models/laguna \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

## 精度

首次加载建议使用 `--dtype auto`。NVFP4 和 INT4_GROUP32 是否可用取决于 checkpoint 格式、GPU 架构和当前 kernel 支持，不应仅根据文件名强制覆盖精度。

## Benchmark 状态

仓库目前只记录了 Laguna 多卡、CUDA Graph、混合 MoE 和量化路径的功能验证，没有统一硬件上的正式速度表。建议设备命令和数据状态见 [Laguna Benchmark](benchmarks/laguna.md)。
