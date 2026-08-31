# Laguna Deployment

[中文](laguna.md) · [Back to README](../README_EN.md) · [Benchmark](benchmarks/laguna_en.md)

FastLLM supports Laguna long-context caching, CUDA Graph, multi-GPU tensor parallelism, hybrid MoE, NVFP4, and INT4_GROUP32.

## API server quick start

~~~bash
ftllm server /data/models/laguna \
  --model_name laguna \
  --host 0.0.0.0 --port 8080
~~~

## Multi-GPU tensor parallelism

When aggregate GPU memory can hold the checkpoint:

~~~bash
ftllm server /data/models/laguna \
  --tp 4 \
  --max_batch 8 \
  --gpu_mem_ratio 0.9
~~~

Laguna derives the main-device and expert tensor-parallel layout from `--tp`. The CUDA slab default is also adjusted for the four-GPU model layout.

Eight-GPU example:

~~~bash
ftllm server /data/models/laguna \
  --tp 0,1,2,3,4,5,6,7 \
  --max_batch 16 \
  --gpu_mem_ratio 0.9
~~~

## GPU + NUMA hybrid MoE

~~~bash
ftllm server /data/models/laguna \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192
~~~

## Long context

~~~bash
ftllm server /data/models/laguna \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

## Precision

Use `--dtype auto` for an initial load. NVFP4 and INT4_GROUP32 support depends on checkpoint format, GPU architecture, and available kernels; do not force a precision solely from the filename.

## Benchmark status

The repository contains functional validation for Laguna multi-GPU, CUDA Graph, hybrid MoE, and quantization paths, but no standardized speed table. See the [Laguna benchmark page](benchmarks/laguna_en.md) for suggested device commands and the current data status.
