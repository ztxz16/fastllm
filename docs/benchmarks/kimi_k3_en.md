# Kimi-K3 Benchmark

[中文](kimi_k3.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../kimi_k3_en.md)

The repository currently has functional and protocol validation for Kimi-K3, but no standardized throughput result suitable for publication. The configurations below are starting points for measurement, not performance claims.

| Device layout | Suggested launch command | Speed |
| --- | --- | --- |
| GPU with NUMA MoE | `ftllm server /data/models/kimi-k3 --device cuda --moe_device numa --chunked_prefill_size 8192` | Not measured |
| GPU with CPU MoE | `ftllm server /data/models/kimi-k3 --device cuda --moe_device cpu --chunked_prefill_size 8192` | Not measured |
| GPU, NUMA, and disk experts | `ftllm server /data/models/kimi-k3 --device cuda --moe_device "{'cuda':1,'numa':8,'disk':1}"` | Not measured |

Kimi-K3 also supports an external DSpark draft model. A comparison should report standard decode and DSpark separately, including accepted length, TTFT, and sustained throughput. See the [benchmark tools](../../test/benchmark/README.md).
