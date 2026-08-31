# GLM-5 / GLM-5.3-Flash Benchmark

[中文](glm5.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../glm5_en.md)

The repository does not yet contain a complete, publishable throughput table for GLM-5 or GLM-5.3-Flash. The configurations below are suggested measurement starting points.

| Device layout | Suggested launch command | Speed |
| --- | --- | --- |
| GPU with NUMA MoE | `ftllm server /data/models/glm5 --device cuda --moe_device numa --chunked_prefill_size 8192` | Not measured |
| GPU with CPU MoE | `ftllm server /data/models/glm5 --device cuda --moe_device cpu --chunked_prefill_size 8192` | Not measured |
| Quantized KV-B on CPU / NUMA | `ftllm server /data/models/glm5.2-quantized-kvb --device numa --moe_device numa -t 64` | Not measured |

Results must name the exact model revision, quantization, and the DSA/KDA and paged-cache path that was active. See the [benchmark tools](../../test/benchmark/README.md).
