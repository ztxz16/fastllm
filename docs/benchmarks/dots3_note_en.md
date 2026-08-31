# Dots3-Note Benchmark

[中文](dots3_note.md) · [Benchmark index](../benchmark_en.md) · [Deployment guide](../dots3_note_en.md)

Operator, long-cache, and API regressions exist, but the repository does not yet contain a complete Dots3-Note device-throughput report. The configurations below are suggested measurement starting points.

| Device layout | Suggested launch command | Speed |
| --- | --- | --- |
| GPU with NUMA MoE | `ftllm server /data/models/dots3-note --device cuda --moe_device numa --chunked_prefill_size 8192` | Not measured |
| GPU with CPU MoE | `ftllm server /data/models/dots3-note --device cuda --moe_device cpu --chunked_prefill_size 8192` | Not measured |
| Long context | `ftllm server /data/models/dots3-note --device cuda --moe_device numa --max_context_length 131072 --chunked_prefill_size 8192` | Not measured |

Long-context tests should report TTFT, prefill token/s, memory use, and sustained decode at multiple input lengths instead of publishing only a short-input result. See the [benchmark tools](../../test/benchmark/README.md).
