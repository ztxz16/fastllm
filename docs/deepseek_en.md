# DeepSeek-V4 / DeepSeek-V4-Flash Deployment

[中文](deepseek.md) · [Back to README](../README_EN.md) · [Hybrid inference](mixforward.md) · [Benchmark](benchmarks/deepseek_v4_en.md)

This guide focuses on DeepSeek-V4 and DeepSeek-V4-Flash. Earlier DeepSeek models remain compatible but are no longer the primary examples.

## API server quick start

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --model_name DeepSeek-V4-Flash \
  --host 0.0.0.0 --port 8080
~~~

Large MoE checkpoints normally use CUDA for non-expert layers and CPU or NUMA for experts. Set the layout explicitly for production.

## Commands by device

### Single GPU + NUMA

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

Use `--moe_device cpu` on a single-socket host.

### Multi-GPU tensor parallelism

When aggregate GPU memory can hold the checkpoint:

~~~bash
ftllm server /data/models/deepseek-v4 \
  --tp 0,1 \
  --max_batch 8 \
  --gpu_mem_ratio 0.9
~~~

DeepSeek-V4 selects the MultiCUDA path when `--tp` uses multiple GPUs.

### GPU, NUMA, and disk experts

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --device cuda \
  --moe_device "{'cuda':1,'numa':8,'disk':1}" \
  --chunked_prefill_size 8192
~~~

Disk experts are a capacity fallback and depend heavily on SSD random-read performance.

### Multi-GPU + NUMA hybrid MoE

~~~bash
ftllm server fastllm/DeepSeek-V4-Flash \
  --device cuda \
  --moe_device "{'multicuda:0,1':15,'numa':85}"
~~~

Tune the ratio against GPU memory and host-memory bandwidth. See the [hybrid inference guide](mixforward.md).

## Embedded DSpark

For a DeepSeek-V4 checkpoint containing embedded DSpark weights:

~~~bash
ftllm server /data/models/deepseek-v4 --dspark 7
~~~

The requested value cannot be smaller than the block size used to train the checkpoint.

## Long context

~~~bash
ftllm server /data/models/deepseek-v4 \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

## SM120 TP8 sparse-attention path

~~~bash
FASTLLM_CUDA_GRAPH=1 \
FASTLLM_CUDA_CUSTOM_ALLREDUCE=1 \
ftllm server /data/models/deepseek-v4 \
  --tp 8 --triton \
  --max_batch 1
~~~

This is the starting point for the recorded TP8 result, not a universal recommendation for other GPU architectures. See the [DeepSeek-V4 benchmark](benchmarks/deepseek_v4.md) and [sparse-attention analysis](deepseek_v4_sparse_attention.md).

## Thinking and tool calling

~~~bash
ftllm server /data/models/deepseek-v4 \
  --enable_thinking true \
  --tool_call_parser auto
~~~

## Benchmark status

The repository has a published TP8 SM120 result. Single-GPU + NUMA and disk-expert throughput have not been measured under a publishable common methodology.
