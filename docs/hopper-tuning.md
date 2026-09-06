# H 系列显卡（SM90）调优指南

[返回 README](../README.md) · [Qwen 部署](qwen3.md) · [Benchmark](benchmark.md)

本指南面向 Hopper / SM90 的 H 系列 CUDA GPU，以单张 **H800 PCIe 80GB** 上的
Qwen3.8-27B-FP8 为实测案例。该 checkpoint 内部使用 Qwen3.5 架构，包含 GDN 线性注意力。
其他 H 系列型号、模型、量化格式和多卡配置需要分别复测；下文的 GDN 参数只适用于相应算子路径。

## 先确认构建和 Python 环境

检查 GPU 型号和 compute capability，SM90 对应 `9.0`：

```bash
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

源码安装时指定 SM90：

```bash
bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=90
```

构建环境要求见 [源码安装](../README.md#源码安装)。满足布局和 shape 要求的 block-scaled
FP8 Linear 会自动尝试 SM90 专用实现，无需额外设置 FP8 开关；`--triton` 主要用于启用
包括 GDN recompute 在内的可用 Triton 算子。

在 **启动 ftllm 的同一个 Python 环境** 中安装 Triton。下面使用实测版本，`python` 应替换为
ftllm 脚本 shebang 指向的解释器；如果在虚拟环境中安装 ftllm，则先激活该环境。

```bash
python -m pip install triton==3.7.1
python -c "import sys, triton; print(sys.executable, triton.__version__)"
```

使用 `--triton` 时，启动日志应包含 `Triton enabled with the current Python environment`。
若提示当前环境没有 Triton，CLI 会关闭该开关并回退 CUDA。CLI 也会覆盖
`FASTLLM_CUDA_TRITON_PYTHON` 为当前解释器，所以只在另一个 vLLM 虚拟环境中安装 Triton，
或只设置该环境变量，都不能替代选择正确的启动环境。
普通 Triton GDN 方案不要求安装 vLLM、FlashInfer 或 CuTe。

## 少参数启动

对于单请求长 prompt、前缀复用较少的场景，先使用：

```bash
MODEL=~/autodl-tmp/Qwen3.8-27B-FP8
ftllm server "$MODEL" --triton --prefix_cache false
```

这条命令保留默认的 prefill 分块和 GDN 上限。H800 实测 4K/8K 输入约为
**8797/8559 tok/s**，decode 约 55 tok/s。

关闭前缀缓存适合测量完整 prefill，也能避免本案例的前缀快照开销。
如果业务有大量重复系统提示词或多轮对话，前缀缓存命中可能明显改善请求延迟，
应同时测试开启和关闭时的真实业务 TTFT，按命中率选择。

## 复现 9000+ 的单请求配置

使用原 CUDA/Triton GDN 路径，不启用 FlashInfer：

```bash
MODEL=~/autodl-tmp/Qwen3.8-27B-FP8
FASTLLM_CUDA_TRITON_CHUNK_GDN_PREFILL_MAX_CHUNKS=128 \
FASTLLM_CUDA_TRITON_CHUNK_GDN_RECOMPUTE_MAX_CHUNKS=128 \
ftllm server "$MODEL" --triton \
  --prefix_cache false --chunked_prefill_size 8192 \
  --max_batch 1 --max_context_length 16384 --gpu_mem_ratio 0.85
```

运行时保持 `FASTLLM_CUDA_TRITON_FLASHINFER_GDN` 未设置或为 `0`。
这组配置在最终提交上实测 4K/8K 为 **9458/9251 tok/s**；此前独立复测为 9574/9238 tok/s。

| 参数 | 作用与选择依据 |
| --- | --- |
| `--triton` | 启用完整的可用 Triton 优化；仅安装 Triton 不等于启用所有 GDN 路径。 |
| `--prefix_cache false` | 避开本模型前缀快照对单请求分块的限制，并测量无前缀命中的 prefill。 |
| `--chunked_prefill_size 8192` | 允许单个请求每段处理最多 8192 tokens；会增加临时显存需求。 |
| `...PREFILL_MAX_CHUNKS=128` | 原 Triton GDN H/O 路径每个内部 chunk 为 64 tokens，8192 tokens 对应 128 个 chunks。 |
| `...RECOMPUTE_MAX_CHUNKS=128` | 同时放宽 Triton recompute 路径的上限，避免大分块超出默认限制后回退。 |
| `--max_batch 1` | 固定单请求延迟基线。高并发服务需另测 batch、总吞吐和 TTFT。 |
| `--max_context_length 16384 --gpu_mem_ratio 0.85` | 固定本次测试的上下文和显存预算；业务部署按实际上下文、并发及显存调整。 |

两个 max-chunks 默认均为 64，分别控制不同阶段；它们是准入上限，不会自行增大服务端的
prefill 分块。2048 tokens 只有 32 个内部 chunks，此时把上限从 64 增加到 128 通常不会改变
这两项限制的判断。8192 tokens 才需要 128 的上限。也不要把内部 64-token GDN chunk
与 `--chunked_prefill_size` 的服务端分段混为一谈。

### 为什么只加环境变量或只改 chunk 不一定更快

Qwen3.5 GDN 的前缀缓存开启时，实际单请求分块取配置分块与前缀快照间隔的较小值。
本次测试的默认间隔为 **16 页 × 128 tokens = 2048 tokens**，因此只写
`--chunked_prefill_size 8192`，实际仍可能按 2048 处理。

在保留前缀缓存、未加 `--triton` 的对照中，只提高两个 max-chunks，或再加 8192 分块，
都没有观察到明确提速。原 GDN 的部分 H/O 路径在条件满足时可自动使用 Triton，
而 recompute 路径还要求全局 Triton 开关；不能通过设置 max-chunks 来开启它。

排查分块是否生效时，关注启动日志中的 `single-request recompute` 及实际请求行为。
日志中多请求聚合预热的预算达到 8192，不代表单请求也使用了 8192 分块。
上述完整配置还固定了 batch、上下文和显存预算，不能把相对少参数命令的全部差异归因于某一个参数。

## 可选：FlashInfer GDN

**SM90 默认仍使用原 CUDA/Triton 路径。** 在上面的完整命令前再加
`FASTLLM_CUDA_TRITON_FLASHINFER_GDN=1`，才会尝试 FlashInfer 的 SM90 CuTe CP kernel。
单独 `--triton` 不会自动开启 FlashInfer。

该路径需要额外的编译依赖，适合愿意维护独立实验环境的部署。启动 ftllm 的 Python 环境应同时具备
Triton、FlashInfer 和 CuTe；已导出的共享库仍需要对应 CUDA/CuTe runtime。
当前只支持 SM90、单请求、FP16、K/V 维度为 128 等限定条件，缺依赖或不兼容输入会回退。
完整约束、版本和缓存说明见 [FlashInfer GDN](flashinfer-gdn.md)。

H800、相同二进制和相同完整配置下，仅改变 FlashInfer 开关：

| GDN 路径 | 4K prefill | 8K prefill | 4K / 8K decode |
| --- | ---: | ---: | ---: |
| 原 CUDA/Triton，FlashInfer 未设置 | 9458 tok/s | 9251 tok/s | 55.77 / 55.22 tok/s |
| 显式开启 FlashInfer | 10283 tok/s | 10013 tok/s | 55.75 / 55.24 tok/s |

普通部署可以停留在原 Triton 方案，无需为本指南安装 FlashInfer。

## 实测条件和测速口径

测试日期为 2026-09-06。模型为 Qwen3.8-27B-FP8，FastLLM 使用 FP16 激活；
GPU 为单张 H800 PCIe 80GB（114 SM）。少参数结果来自提交 `04b75cec`，最终完整配置
及 FlashInfer 对照来自提交 `e29de387`，跨提交数据用于复现参考。
普通路线验证了 Python 3.10 + Triton 3.7.1；FlashInfer 实验环境使用
FlashInfer 0.6.16.post3、nvidia-cutlass-dsl 4.6.2、CUDA toolkit 12.8 和 CUDA 13 系列 CuTe runtime。

- 使用 OpenAI 兼容流式接口，输入包含聊天模板后严格为 4096/8192 tokens，实际输出 128 tokens，并发为 1。
- 每个长度预热一次，正式测三次取中位数；排除首次编译、模型加载和启动预热。
- 请求前缀各不相同，确认返回 usage 的 `cached_tokens=0`；测速时无 profiler 和其他 GPU 任务。
- 本页 prefill = 输入 token 数 / 首个非空内容 token 到达时间，包含分词、HTTP 和首 token 开销。
  它与服务端 `[Prompt] Speed` 或纯 GPU kernel 吞吐的口径不同。
- Decode = 127 /（最后一个内容 token 时间 − 第一个内容 token 时间）；比较时保持输出长度一致。

更换模型、并发、缓存策略或参数后，记录完整命令与环境，再按相同口径复测。
若需要定位瓶颈，可在预热后另开一轮 nsys 捕获，分开比较 FP8 Linear、GDN、full attention
和前后处理；不要把 profiler 下的请求速度混入上表。不要用本页单请求结果预测高并发吞吐。

## 常见排查

| 现象 | 优先检查 |
| --- | --- |
| 加了 `--triton` 仍回退 | 当前 ftllm 解释器能否导入 Triton，以及启动日志是否明确启用。 |
| 写了 8192，速度仍像默认配置 | 前缀缓存是否限制到快照间隔；区分单请求分块和聚合预算。 |
| 4K 正常、8K 大段明显变慢 | 是否同时提高两个 GDN max-chunks，上限是否覆盖实际段长。 |
| FlashInfer 没有生效 | 是否显式开启、当前 Python 是否有可选依赖、输入是否符合 SM90/FP16/单请求约束。 |
| 增大 chunk 后显存不足或排队延迟增加 | 降低 chunk，结合实际并发重新设置 batch、上下文和显存预算。 |
| 首次很慢、后续正常 | 将首次编译和预热排除，再测稳定状态；检查是否反复清空了编译缓存。 |

相关实现：[前缀快照与分块](../src/models/qwen3_5.cpp)、
[Triton GDN 分发](../src/devices/cuda/cudadevice.cpp)、
[CLI 解释器选择](../tools/fastllm_pytools/util.py)。
