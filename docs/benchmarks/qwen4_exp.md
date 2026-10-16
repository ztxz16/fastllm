# Qwen4-Exp / Qwen3.8-Flash-Next Benchmark

[English](qwen4_exp_en.md) · [Benchmark 索引](../benchmark.md) · [部署指南](../qwen4.md) · [架构详情](../qwen4_exp.md)

## 2026-09-29：双 RTX 2080 Ti，Flash-Next NVFP4

测试版本为提交 `d6ef5c784`，覆盖六种混合部署方式。完整启动命令见[部署指南](../qwen4.md)。

- 硬件：两张 **22 GiB 显存版** RTX 2080 Ti（每卡 22528 MiB），AMD EPYC 7452，32 核 / 64 线程，单 NUMA 节点，125.7 GiB RAM；Ubuntu 24.04.4，驱动 595.84。
- 模型：`Qwen3.8-Flash-Next-NVFP4` 文本推理，48 个主干 MoE 层。FP16 激活，PLE disk，28 计算线程，batch=1，KV 预算 1 GiB，`gpu_mem_ratio=0.95`。
- 统一关闭 CUDA Graph、Triton、前缀缓存、历史缓存和思考。普通与 MTP 模式共用已加载的 MTP 权重。
- 每项预热一次、正式测一次；decode 预热生成 128 token。输入为统一的代码生成提示词，填充到指定长度，所有配置使用相同输入 token。

### Prefill 与 decode

Prefill 使用 **4096 输入 token、`chunked_prefill_size=4096`、MTP=0**，完整请求只生成 1 个 token。吞吐取原生 `[Prompt]` 前向计时，包含得到首 token 所需的 logits/采样，不是单一 GPU kernel 时间；TTFT 则从原生请求提交计到首 token。

Decode 另用 512 输入、固定 512 输出，分别测普通解码和 MTP=3；吞吐为 `511 / (末 token 到达时间 − 首 token 到达时间)`，不含模型加载、预热或 TTFT。两个阶段都没有 HTTP/SSE 网络开销。

| 部署方式 | Prefill token/s | Prefill TTFT 秒 | 普通 decode token/s | MTP3 token/s |
| --- | ---: | ---: | ---: | ---: |
| 串行，缓存关闭 | 919.43 | 4.456 | 22.36 | 30.08 |
| 串行，每卡缓存 8 GiB | 902.92 | 4.538 | 26.13 | 39.84 |
| 串行，固定 12 层专家 | 960.99 | 4.273 | 23.99 | 32.92 |
| TP2，缓存关闭 | 1240.71 | 3.303 | 20.60 | 32.63 |
| TP2，每卡缓存 8 GiB | 1254.39 | 3.267 | 27.71 | 46.29 |
| TP2，固定 12 层专家 | 1321.08 | 3.102 | 22.57 | 37.82 |

不同部署的生成内容和 MTP 接受情况可能不同，以上是单轮部署效果比较，不是强制相同输出的算子微基准。固定 512 输出也不代表完整编程任务或 Agent 自然结束验收。

### 内存与显存

| 部署方式 | GPU0 峰值 GiB | GPU1 峰值 GiB | 解码 RSS GiB | 主机 HWM GiB |
| --- | ---: | ---: | ---: | ---: |
| 串行，缓存关闭 | 7.81 | 10.25 | 72.60 | 72.86 |
| 串行，每卡缓存 8 GiB | 15.83 | 18.26 | 79.64 | 80.42 |
| 串行，固定 12 层专家 | 12.46 | 20.30 | 56.56 | 65.05 |
| TP2，缓存关闭 | 10.76 | 10.84 | 77.05 | 80.79 |
| TP2，每卡缓存 8 GiB | 18.78 | 18.86 | 84.09 | 88.37 |
| TP2，固定 12 层专家 | 18.33 | 18.41 | 61.05 | 80.62 |

显存为每 2 秒采样的全程峰值，包含大块 prefill；解码 RSS 为正式解码阶段采样中位数。HWM 包含加载峰值，不能拿较低的稳态 RSS 作为启动所需内存。所有组进程 swap 为零，正常完成后退出，未出现 OOM 或 CUDA 错误。

固定 12 层时，TP2 使用 `--moe_device_layers 36`，前 12 层专家跨两卡分片。串行则使用 `cudapp=1:7` 主干和 `{'cuda:0':6,'cuda:1':6,'numa':36}` 专家映射；它与其他串行组的 24/24 主干分界不同。固定层会占用冷门专家的显存，动态缓存可跨层保留活跃专家，不能只按相同容量推断速度。

### 复测

使用仓库中的[同口径测量脚本](../../test/benchmark/qwen38_hybrid.py)，在已安装或构建的 `ftllm` Python 环境运行。脚本固定测试一轮，输出目录必须不存在：

~~~bash
CUDA_VISIBLE_DEVICES=0,1 FT_NUMAS=1 FASTLLM_CUDA_GRAPH=0 \
python test/benchmark/qwen38_hybrid.py /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --output-dir /tmp/qwen38-tp-cache8 \
  --tp 2 --moe_device numa --moe_cuda_cache 8g \
  --atype float16 --ngram_device disk --threads 28 --mtp 3 \
  --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95 \
  --chunked_prefill_size 4096 --prefix_cache false --cache_history false \
  --enable_thinking false
~~~

把布局参数替换为部署指南中的其他组合，串行时清除 `FASTLLM_TP`。脚本自动做 4096-token prefill 和两种解码模式；`results.json` 保存 TTFT、完整输出 token 和到达时间，终端 `[Prompt]` 行保存原生 prefill 吞吐。原生计时与内置 `ftllm benchmark` 使用 `输入 token / TTFT` 得到的 Prefill 指标口径略有不同。


## GGUF 专家常驻 GPU

硬件为双 22 GiB RTX 2080 Ti、EPYC 7452、单 NUMA；TP2、FP16 激活、CUDA Graph 开启、MTP 关闭、专家缓存预算为 0。全部专家保留原生 GGUF 格式驻留 GPU，CPU embedding，磁盘 PLE，28 线程、1 GiB KV 预算、GPU memory ratio 0.95。
模型路径指向同目录中第一份 GGUF 分片，启动参数见[部署指南](../qwen3.8-flash-next/README.md#gguf-resident-performance)。

原生生成接口计时，batch=1、greedy，无 HTTP 或 profiler；各组预热后测三轮，取中位数。解码为 512 输入、512 输出，按 `511 / (末 token 时间 - 首 token 时间)` 计；prefill 为 4096 输入、1 输出，按 `4096 / TTFT` 计。

| 权重 | 测量日期 | 解码 chunk 32 | 解码 chunk 512 | Prefill chunk 1024 |
| --- | --- | ---: | ---: | ---: |
| GSQ-RCO-IQ2_XS | 2026-10-01 | 70.87 | 69.77 | 1205.13 |
| GSQ-RCO-Q2_0（含 Q3_K Dense） | 2026-10-02 | 71.59 | 70.67 | 1359.00 |

单位为 token/s。两种权重分别测量，不是量化格式的受控对比。不同 chunk 也可能改变生成轨迹。

| 权重 | 每卡逻辑权重 GiB | GPU0/1 加载及预热后 GiB | GPU0/1 采样峰值 GiB |
| --- | ---: | --- | --- |
| IQ2_XS | 19.3554 | 20.2088 / 20.2049 | 21.0586 / 20.8379 |
| Q2_0 | 18.6634 | 19.8436 / 19.8416 | 20.6855 / 20.4648 |

显存峰值每 2 秒采样，可能漏掉短暂峰值。IQ2_XS 主机 HWM 为 41.905 GiB、swap 为 0，CPU embedding 为 2.368 GiB。算术、JSON 和 4096-token prefill 检查通过，不能替代完整质量评测。Q8 激活点积存在量化误差，不保证与浮点激活计算逐位相同。

被测隔离原生库 SHA256：IQ2_XS 为 `0e9aa1b66b2d502e150395a40bfca36f2b43c0ea937329f43f89d60861add82b`；Q2_0 为 `2c60b9f0661321f32447f8d3bd1de75998e6fd9029eda3a95a79aad210241f91`。结果仅代表上述构建和 SM75 硬件。
