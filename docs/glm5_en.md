# GLM-5 / GLM-5.3-Flash Deployment

[中文](glm5.md) · [Back to README](../README_EN.md) · [Benchmark](benchmarks/glm5_en.md)

Current support covers GLM-5 DSA, GLM-5.3-Flash KDA and paged caching, and pure-CPU inference for selected GLM-5.2 checkpoints with quantized KV-B weights.

## Automatic configuration (GLM-5.3-Flash)

Use `--auto` for supported GLM-5.3-Flash checkpoints on Linux with NVIDIA CUDA. GLM-5, GLM-5.2, and CPU-only deployment still use the manual commands below.

```sh
ftllm server /data/models/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  --auto --model_name glm-5.3-flash --host 0.0.0.0 --port 8080
```

Pass the first GGUF shard and keep the remaining shards in the same directory. Supported `glm5next` GGUF files include configuration and tokenizer metadata and do not require `--ori`. For HF/NVFP4, pass the complete model directory instead and keep the checkpoint's default weight precision.

| Adjustment | Append to the command |
| --- | --- |
| 32K session window, up to 2 concurrent requests | `--max_context_length 32768 --max_batch 2` |
| Limit planning resources | `--auto_gpu_memory_limit 20g --auto_host_memory_limit 128g` |
| Set the per-GPU expert cache | `--moe_cuda_cache 7g`; `0` disables it |
| Set CPU threads and prefill chunk size | `--threads 32 --chunked_prefill_size 4096` |
| Enable embedded MTP | `--mtp 3`; requires NextN draft weights |
| Disable all speculative decoding | `--speculative_algorithm off` |

GLM MTP is not enabled merely because NextN weights are present. Leave options unset to use automatic planning; see the [`--auto` tuning guide](auto_en.md) for their effects and Launcher controls.

Supported GGUF expert combinations use compressed GPU caches and dynamic CPU/GPU routing. The exact format, grouped-MMQ, TP, and MTP coverage is documented in the [detailed GLM guide (Chinese)](glm5.md#glm-53-flash-gguf).

## Manual configuration reference

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

## Benchmark

Scoped GLM-5.3-Flash prefill measurements and their limits are available on the [benchmark page](benchmarks/glm5_en.md).
