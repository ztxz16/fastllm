# Qwen4-Exp / Qwen3.8-Flash-Next 部署指南

[English architecture notes](qwen4_exp.md) · [返回 README](../README.md) · [Benchmark](benchmarks/qwen4_exp.md)

FastLLM 当前加载 Qwen4-Exp / Qwen3.8-Flash-Next 的 FP8 文本生成模型，覆盖四路超连接、Gated DeltaNet、QSA 稀疏注意力、PLE n-gram 和 MoE。复合 checkpoint 中的视觉权重不会由文本模型加载；Qwen3.8-Flash-Next 的 MTP 权重仅在 `--mtp` 大于 0 时按需加载。

## API Server 快速启动

~~~bash
ftllm server /data/models/qwen4-exp \
  --model_name qwen4-exp \
  --host 0.0.0.0 --port 8080
~~~

PLE 表默认驻留在 CPU，模型会按需读取选中的行，不会把整张表展开成 FP32。

## 按设备选择启动命令

### CUDA + NUMA MoE

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

### CUDA + CPU MoE

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device cpu \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

### 纯 CPU / NUMA

~~~bash
ftllm server /data/models/qwen4-exp \
  --device numa --moe_device numa \
  -t 64
~~~

Qwen4-Exp 的模型和 PLE 表都很大，纯 CPU/NUMA 命令主要用于容量验证与调试。线程数需要结合物理核心数和内存带宽调整。

## TP Decode 的 CUDA Graph

~~~bash
FASTLLM_CUDA_GRAPH=1 ftllm server /data/models/qwen3.8-flash-next \
  --tp 4 --atype float16 --chunked_prefill_size 1024
~~~

单 token TP decode 在支持的 CUDA 路径上分别捕获第 0 层和 PLE 后的主干。第 0 层的小图随请求 KV 缓存保留；缓存地址变化时，各 rank 一起重新捕获。捕获失败时统一回退普通算子提交。

TP 调度保留当前 host token，PLE 查表无需等待 token 从 GPU 读回；查出的行使用请求独立的 pinned buffer，沿当前 worker stream 异步搬运并执行投影。历史 token 和卷积状态仍在正常 PLE 执行位置更新。

`FASTLLM_CUDA_GRAPH` 统一控制是否启用图，PLE 搬运复用自动应用于单 token TP 路径。CPU/NUMA 混合推理和专家缓存的调度策略保持原有行为。

比较性能时固定实际输入/输出 token 数、提示词和采样配置，先预热再测量；Nsight Systems 波形用于解释等待来源，吞吐以未开启 profiler 的结果为准。

## PLE 磁盘模式

主机内存不足时：

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

`--ngram_device disk` 会从 checkpoint 的 Safetensors 文件按行读取 PLE 表，显著降低常驻内存，但增加小块随机读取。建议使用高速 SSD，并单独测量 Decode 抖动和操作系统页缓存占用。

## 长上下文与 Triton

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --triton
~~~

`--triton` 只在当前 Python 环境能导入 Triton 时启用，否则自动回退到内置 CUDA。实际上下文仍受模型原生上限和 KV Cache 容量限制。

## MTP 推测解码

~~~bash
ftllm server /data/models/qwen3.8-flash-next \
  --device cuda --moe_device numa \
  --mtp 4
~~~

`--mtp` 设置每轮 draft token 数，当前最大为 8。Qwen3.8-Flash-Next MTP 目前要求简单贪婪采样，目标网络运行在 CUDA，MoE 可放在 CUDA 或 NUMA；条件不满足时会自动回退普通解码。

Qwen3.8-Flash-Next 复用跨请求前缀快照时，该请求会回退普通解码；采用 Qwen3.5 架构的 Qwen3.8 不受此限制。

## 思考与工具调用

~~~bash
ftllm server /data/models/qwen4-exp \
  --enable_thinking true \
  --tool_call_parser auto
~~~

当前服务支持思考内容分离和 Qwen 工具协议。API 请求可通过 `chat_template_kwargs.enable_thinking` 按请求关闭思考。

## Benchmark

- [Qwen4 不同设备路径命令与冒烟速度](benchmarks/qwen4_exp.md)
- [架构、精度对齐和验证详情](qwen4_exp.md)
- [Benchmark 索引](benchmark.md)

现有数据只有极短输入冒烟结果，没有持续 Decode 或多并发吞吐，不能用于估算生产速度。
