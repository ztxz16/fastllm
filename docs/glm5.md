# GLM-5 / GLM-5.3-Flash 部署指南

[English](glm5_en.md) · [返回 README](../README.md) · [Benchmark](benchmarks/glm5.md)

当前路径覆盖 GLM-5 DSA、GLM-5.3-Flash KDA 与分页缓存，以及部分 GLM-5.2 量化 KV-B checkpoint 的纯 CPU 推理。

## API Server 快速启动

~~~bash
ftllm server /data/models/glm5 \
  --model_name glm5 \
  --host 0.0.0.0 --port 8080
~~~

## GPU + NUMA 混合 MoE

~~~bash
ftllm server /data/models/glm5 \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192 \
  --gpu_mem_ratio 0.9
~~~

该布局适合模型主体和热点路径放在 GPU、专家权重放在多路 NUMA 内存的机器。

## GPU + CPU

~~~bash
ftllm server /data/models/glm5 \
  --device cuda --moe_device cpu \
  --chunked_prefill_size 8192
~~~

## GLM-5.2 量化 KV-B 纯 CPU

仅适用于已验证的量化 KV-B checkpoint：

~~~bash
ftllm server /data/models/glm5.2-quantized-kvb \
  --device numa --moe_device numa \
  -t 64
~~~

`-t 64` 只是多路服务器示例，应根据物理核心数和内存带宽重新测试。

## 长上下文、思考与工具调用

~~~bash
ftllm server /data/models/glm5 \
  --max_context_length 131072 \
  --chunked_prefill_size 8192 \
  --prefix_cache true \
  --enable_thinking true \
  --tool_call_parser auto
~~~

GLM-5.3-Flash 的 KDA、分页历史缓存和 NUMA 解码流水会根据模型结构自动选择。

## Benchmark 状态

仓库目前没有可对外发布的 GLM-5 / GLM-5.3-Flash 完整吞吐表。建议设备命令和数据状态见 [GLM-5 Benchmark](benchmarks/glm5.md)。
