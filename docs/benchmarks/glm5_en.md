# GLM-5 / GLM-5.3-Flash Benchmark

[Deployment guide](../glm5_en.md) · [Benchmark index](../benchmark_en.md)

## GLM-5.3-Flash prefill

Measured 2026-10-03 on eight RTX 5090 GPUs: `cudapp=8`, NVFP4 routed experts, BF16 activations, chunk 1024, batch 1, MTP off, context budget 65536. The default `auto` DSA backend uses FlashInfer sparse prefill when eligible. Prefix/history caching is disabled; loading and initial weight rearrangement are excluded. Median of three runs per length after warmup.

| Input tokens | TTFT seconds | Input tokens / TTFT |
| ---: | ---: | ---: |
| 16384 | 6.106 | 2683.4 token/s |
| 32768 | 12.402 | 2642.1 token/s |

TTFT includes scheduling and the first token. The 32K input repeats a 16K sequence and is a performance/regression sample, not a long-context quality evaluation. No complete decode measurement for this final configuration is recorded here.
FP8 computation need not match BF16 bit for bit; measured first-step logits relative L2 differences were 18.9%/24.6% at 16K/32K. See the [backend and numerical contract](../glm5.md#glm-53-flash-flashinfer-sparse-prefill).

Measured native library SHA256: `9f1988e01cae1892083d21f0f9f92c99e564ef9e9d135eea2125fa8372477df7`.
