# Kimi-K3 部署指南

[English](kimi_k3_en.md) · [返回 README](../README.md) · [Benchmark](benchmarks/kimi_k3.md)

FastLLM 的 Kimi-K3 路径覆盖 KDA/MLA、分块 Prefill、CUDA/NUMA、CPU/GPU 专家、磁盘专家、外部 DSpark、XTML 服务协议和工具调用。

## API Server 快速启动

~~~bash
ftllm server /data/models/kimi-k3 \
  --model_name kimi-k3 \
  --host 0.0.0.0 --port 8080
~~~

## GPU + NUMA 混合 MoE

~~~bash
ftllm server /data/models/kimi-k3 \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

单路 CPU 机器可将 `--moe_device` 改为 `cpu`。

## 磁盘专家

主机内存不足时，可把少量 MoE 层放到高速 SSD：

~~~bash
ftllm server /data/models/kimi-k3 \
  --device cuda \
  --moe_device "{'cuda':1,'numa':8,'disk':1}" \
  --chunked_prefill_size 8192
~~~

磁盘比例越高，解码越容易受随机 I/O 限制。建议先从很小比例开始。

## 外部 DSpark

外部 draft 目录必须是与目标模型匹配的 `DSparkDraftModel` checkpoint：

~~~bash
ftllm server /data/models/kimi-k3 \
  --speculative_algorithm dspark \
  --speculative_draft_model_path /data/models/kimi-k3-dspark \
  --speculative_dspark_confidence_threshold 0.5
~~~

block size 默认读取 draft 配置。FastLLM 会在启动时校验 draft 架构和 block 大小。

## 长上下文、思考与工具调用

~~~bash
ftllm server /data/models/kimi-k3 \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --enable_thinking true \
  --tool_call_parser auto
~~~

Kimi-K3 使用 XTML 对话协议；通常应保留模型自带 tokenizer 和模板。

## Benchmark 状态

仓库目前没有可发布的 Kimi-K3 完整吞吐记录。建议设备命令和数据状态见 [Kimi-K3 Benchmark](benchmarks/kimi_k3.md)。
