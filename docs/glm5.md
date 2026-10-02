# GLM-5 / GLM-5.3-Flash 部署指南

[English](glm5_en.md) · [返回 README](../README.md) · [Benchmark](benchmarks/glm5.md)

当前路径覆盖 GLM-5 DSA、GLM-5.3-Flash KDA 与分页缓存，以及部分 GLM-5.2 量化 KV-B checkpoint 的纯 CPU 推理。

## API Server 快速启动

~~~bash
ftllm server /data/models/glm5 \
  --model_name glm5 \
  --host 0.0.0.0 --port 8080
~~~

## GLM-5.3-Flash NVFP4 显存

GLM-5.3-Flash 的 ModelOpt NVFP4 路由专家在单设备 CUDA 后端（包括 `cudapp` 按层串行）默认使用紧凑存储。每 16 个权重保留 8 字节 FP4 数据和 1 字节 E4M3 块缩放，每行另存 4 字节全局缩放。合并 gate/up 权重时保留各自的全局缩放；这是存储布局转换，不重新量化权重。

## GLM-5.3-Flash NVFP4 grouped Marlin

兼容的 CUDA 紧凑 NVFP4 路由专家默认使用 grouped Marlin，支持 BF16 激活、独立 gate/up 全局缩放和 GPU 路由。prefill 与 decode 共用一次重排后的权重布局；成功准备后释放原 GPU 布局。首次执行包含重排开销，测速需先预热。

GLM 的 `swiglu_limit` 会传入 Marlin 激活核和普通 CUDA 回退路径：gate 只限制上界，up 限制正负两侧，再计算 `SiLU(gate) * up`。非零限幅不会进入只支持普通 SwiGLU 的快捷路径。形状或缩放不满足 Marlin 条件时保留源权重并回退。

存储格式与计算路径自动选择。Marlin 的 Tensor Core 累加和 BF16 舍入与普通路径不同，整模 logits 和 greedy 文本不保证逐位一致。

启用 `UNIT_TEST` 构建后，可运行：
~~~bash
ctest --test-dir build-fastllm -R 'cuda_nvfp4_(marlin|compact)' --output-on-failure
~~~
回归覆盖零限幅行为、BF16/FP16/FP32 激活边界、带限幅的 CUDA 调度、1024-token 路由、4096/2048 top-8、CUDA Graph 重放，以及不支持形状和内存分配方式的回退。

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

2026-10-03 的 GLM-5.3-Flash NVFP4 MoE 专项对照：8 × RTX 5090，`cudapp=8`，BF16，分块 1024，上下文预算 32768，关闭 prefix/history cache，MTP=0，单请求。完整 16k 预热一次后测三次，以下为中位数；两版都启用限幅。数据来自清理临时对照开关前的构建，具体动态库 SHA256 记录在结果 JSON 中。

| 路径 | 16384-token TTFT | 16384 / TTFT |
| --- | ---: | ---: |
| 普通 CUDA MoE | 21.933 s | 747.0 token/s |
| grouped Marlin | 14.116 s | 1160.7 token/s |

有效 prefill 提升 1.554 倍，TTFT 降低 35.64%。该指标包含请求调度和首 token 开销。Marlin 的 512/2048-token 输入、256-token 输出测试，decode 分别为 49.53/49.49 token/s。GPU 仍按层串行执行。

算子 FP64 对照通过；整模 16 个相同 teacher-forced 前缀的 top-1 有 14 个一致，平均 KL=0.03174、最大 KL=0.16215，未做广泛质量评测。[参数与数值结果](benchmarks/glm53_marlin_20261003.json)。
