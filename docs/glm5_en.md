# GLM-5 / GLM-5.3-Flash Deployment

[中文](glm5.md) · [Back to README](../README_EN.md) · [Benchmark](benchmarks/glm5_en.md)

Current support covers GLM-5 DSA, GLM-5.3-Flash KDA and paged caching, and pure-CPU inference for selected GLM-5.2 checkpoints with quantized KV-B weights.

## API server quick start

~~~bash
ftllm server /data/models/glm5 \
  --model_name glm5 \
  --host 0.0.0.0 --port 8080
~~~

## GPU + NUMA hybrid MoE

~~~bash
ftllm server /data/models/glm5 \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

## GPU + CPU

~~~bash
ftllm server /data/models/glm5 \
  --device cuda --moe_device cpu \
  --chunked_prefill_size 8192
~~~

## Quantized KV-B GLM-5.2 on CPU

This path requires a validated quantized KV-B checkpoint:

~~~bash
ftllm server /data/models/glm5.2-quantized-kvb \
  --device numa --moe_device numa \
  -t 64
~~~

The thread count is an example for a multi-socket server and must be tuned for physical cores and memory bandwidth.

## Long context, thinking, and tools

~~~bash
ftllm server /data/models/glm5 \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --enable_thinking true \
  --tool_call_parser auto
~~~

GLM-5.3-Flash KDA, paged history caches, and NUMA decode pipelining are selected from the model architecture.

## Benchmark status

No publishable complete GLM-5 / GLM-5.3-Flash throughput table is stored in the repository. See the [GLM-5 benchmark page](benchmarks/glm5_en.md) for suggested device commands and the current data status.
