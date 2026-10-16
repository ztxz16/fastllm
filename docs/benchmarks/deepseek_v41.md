# DeepSeek-V4.1-Flash Benchmark

[Benchmark 索引](../benchmark.md) · [部署与验证](../deepseek_v41.md)

以下为 2026-10-02 记录的最终配置结果；量化格式、设备放置和采样口径分别注明，不能混合解释成同一配置的性能。

## Q2_K/Q4_K NUMA 专家 prefill

RTX 4090 + RTX 4090 D 按层 1:1 串行，单 NUMA、28 线程、每卡 8 GiB 专家缓存、chunk 4096、fastprefill 开启、FP16 加载、BF16 KV、专家和 Engram 常驻主存。输入 8600 token、输出 1 token，关闭推测解码和前缀复用，预热后三次正式请求。

| 指标 | 结果 |
| --- | ---: |
| TTFT 中位数 | 5.893 秒 |
| 输入 token / TTFT | 1459.5 token/s |
| CPU RSS 峰值 | 425.06 GiB |
| GPU0 / GPU1 显存峰值 | 23.69 / 23.01 GiB |

被测原生库 SHA256：`438a2a404ac7459eb9557edb873f15be74f2775b7cd0547f4b3f842645fa8b6c`。该配置完成 grouped MMQ，无回退。长文本三编码检索正确；数值回归与内存检查通过，不能代替完整困惑度评测。

## Q2_K/Q4_K 磁盘专家

单 RTX 4090、单 NUMA、28 线程、64 GiB CPU 专家缓存、Engram 磁盘、chunk 4096、fastprefill 开启、FP16 加载、BF16 KV。41-token 代码输入、192-token 输出，history/prefix cache 关闭。两种模式均加载草稿权重，分别预热后各测三次，取中位数。

| 模式 | Decode token/s | 说明 |
| --- | ---: | --- |
| 普通解码 | 5.72 | 该任务推荐配置 |
| DSpark 5，阈值 0.5 | 5.44 | 接受率 61.8%，TTFT 10.81 秒 |

DSpark 正式请求峰值 RSS 为 70.93 GiB，活跃专家缓存不超过 64 GiB。每请求物理 SSD 读取约 117.98 GiB，普通解码约 86.02 GiB；高接受率仍可能因专家读取成本而无法提高吞吐。
192 个输出 token 的一致性、缓存容量及独立算子参考检查通过；该测量未覆盖长上下文或完整模型质量。按此任务部署可使用 `--dspark 0 --mtp 0`。

磁盘配置被测原生库 SHA256：`b141c7fdc951aa1c02d80071b80bbf6b60437992f59e5a963c17a40f960c2a98`。
