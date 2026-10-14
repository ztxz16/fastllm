# GLM-5 / GLM-5.3-Flash Benchmark

[部署指南](../glm5.md) · [Benchmark 索引](../benchmark.md)

## GLM-5.3-Flash prefill

2026-10-03，8 × RTX 5090，`cudapp=8`、NVFP4 routed experts、BF16 激活、chunk 1024、单请求、MTP=0、上下文预算 65536。默认 `auto` DSA 在满足条件时使用 FlashInfer sparse prefill；关闭 prefix/history cache，排除加载和首次权重重排。各长度预热后测三次取中位数。

| 输入 token | TTFT 秒 | 输入 token / TTFT |
| ---: | ---: | ---: |
| 16384 | 6.106 | 2683.4 token/s |
| 32768 | 12.402 | 2642.1 token/s |

TTFT 包含调度和首 token。32K 输入由 16K 序列重复构成，只用于性能与回归；该页没有相同最终配置的完整 decode 数据。
FP8 计算不保证与 BF16 后端逐位一致，已记录的 16K/32K 首步 logits 相对 L2 差异为 18.9%/24.6%。详细能力、精度限制及回退见[部署指南](../glm5.md#glm-53-flash-flashinfer-sparse-prefill)。

被测原生库 SHA256：`9f1988e01cae1892083d21f0f9f92c99e564ef9e9d135eea2125fa8372477df7`。
