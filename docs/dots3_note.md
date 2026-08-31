# Dots3-Note 部署指南

[English](dots3_note_en.md) · [返回 README](../README.md) · [Benchmark](benchmarks/dots3_note.md)

FastLLM 的 Dots3-Note 路径支持 DSA 索引、稀疏注意力、长上下文缓存、分块 Prefill、思考内容和工具调用。

## API Server 快速启动

~~~bash
ftllm server /data/models/dots3-note \
  --model_name dots3-note \
  --host 0.0.0.0 --port 8080
~~~

## GPU + NUMA 混合 MoE

~~~bash
ftllm server /data/models/dots3-note \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

## GPU + CPU

单路 CPU 或不使用 NUMA 后端时：

~~~bash
ftllm server /data/models/dots3-note \
  --device cuda --moe_device cpu \
  --chunked_prefill_size 8192
~~~

对于 FP8 checkpoint，FastLLM 会按模型和设备选择适合的 MoE 激活类型与 CPU 快速路径。建议先使用 `--dtype auto`。

## 长上下文与缓存

~~~bash
ftllm server /data/models/dots3-note \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --gpu_mem_ratio 0.9
~~~

## 思考与工具调用

~~~bash
ftllm server /data/models/dots3-note \
  --enable_thinking true \
  --tool_call_parser auto
~~~

API 支持流式和非流式 `reasoning_content`，工具调用 parser 默认自动选择。

## Benchmark 状态

仓库目前记录了算子、长缓存和 API 回归，但没有完整的 Dots3-Note 设备吞吐表。建议设备命令和数据状态见 [Dots3-Note Benchmark](benchmarks/dots3_note.md)。
