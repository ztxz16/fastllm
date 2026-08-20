# Qwen3.6-27B-FP8 双卡 TP2 推理性能对比

- 测试日期：2026-08-09
- 硬件：2 × NVIDIA GeForce RTX 5090（每卡 32607 MiB，PCIe PIX，无 NVLink）
- 模型：`/root/hfmodels/Qwen3.6-27B-FP8`
- 并行方式：Tensor Parallel 2

## 测试口径

- 框架版本：FastLLM 0.1.7.1（commit `6e01b27663c0`）、vLLM 0.26.0、SGLang 0.5.16。
- 使用相同的短固定提示词，关闭 thinking、prefix/radix cache 和 MTP。
- 采样参数：temperature 0.7、top-p 0.9、top-k 20。
- 每个请求强制生成 256 token，忽略 EOS。
- 每个 batch 先生成 16 token 预热，再正式测试 3 次，表中为 3 次中位数。
- 指标为客户端可见的 decode 聚合吞吐：从所有请求中首个非空流式输出到最后一个请求结束；单位为 token/s。正常情况下它排除共同 prefill/TTFT，但如果并发请求没有同时进入 decode，后进入请求残留的 prefill/调度时间仍会计入；batch 2 的 vLLM 正好存在这种情况，见下文。
- 最大上下文长度为 1024。

## 结果

| Batch / 并发 | FastLLM (token/s) | vLLM (token/s) | SGLang (token/s) |
| ---: | ---: | ---: | ---: |
| 1 | 94.8087 | 84.9428 | 79.6452 |
| 2 | 183.0459 | 143.2340 | 146.5741 |
| 4 | 338.1496 | 277.9786 | 293.6020 |
| 8 | 630.5965 | 528.5985 | 552.4771 |
| 16 | 1163.9353 | 1058.3315 | 1033.1407 |
| 32 | 1854.3000 | 1835.7461 | 1744.7290 |
| 64 | 2930.4490 | 2667.5960 | 2667.0634 |
| 128 | 不可用（预热 OOM） | 3157.9444 | 3020.8107 |
| 256 | 不可用（容量/OOM） | 3024.4135 | 不可用（KV 容量不足） |

在共同可运行的 batch 1–64 范围内，FastLLM 均为最高吞吐。batch 2 时，FastLLM 相比 vLLM 快 27.79%，相比 SGLang 快 24.88%。

## Batch 2 波形与逐算子分析

波形抓取日期：2026-08-10。

### 抓取方法与可比性

- 三个框架均保持上表 batch 1–64 时的原始启动配置，先完成模型加载和一轮 batch 2 × 256 token 预热，再开始抓取恰好一轮 batch 2 × 256 token 请求。
- 使用 Nsight Systems 的 CUDA software trace、NVTX、NCCL 和 CUDA Graph node trace。node trace 会展开图内 kernel，便于逐算子归类，但会引入额外开销，尤其会放大节点最多的 SGLang 的间隙。因此，上表无 profiler 的吞吐是最终性能结果，波形只用于解释差异。
- 每个请求生成 256 token，所以稳定区间包含 255 个 batch 2 decode step；每个 step 同时生成 2 token。
- 下文的逐算子时间是关键路径 GPU 上、每个稳定 decode step 的 GPU kernel 累计时间。不同 stream 上的 kernel 可以重叠，因此原始时间不能简单逐行相加；另给出了时间并集。

波形本身没有改变三者的性能排序：

| 框架 | 无 profiler (token/s) | node trace (token/s) | trace 扰动 |
| --- | ---: | ---: | ---: |
| FastLLM | 183.0459 | 180.1350 | -1.59% |
| vLLM | 143.2340 | 140.3874 | -1.99% |
| SGLang | 146.5741 | 138.4553 | -5.54% |

原始 `.nsys-rep`、导出的 SQLite 以及 Nsight 汇总 CSV 保存在：

```text
/root/nsys-profiles/qwen36-27b-fp8-tp2-b2-20260810/
├── fastllm_b2_node.{nsys-rep,sqlite}
├── vllm_b2_node.{nsys-rep,sqlite}
├── sglang_b2_node.{nsys-rep,sqlite}
├── *_kernel_summary_cuda_gpu_kern_sum.csv
└── *_cuda_api_summary_cuda_api_sum.csv
```

### 先拆开“固定错位”和“稳定 decode”

用同一次启动下的 16-token 预热与 256-token 正式请求，按
`duration(N) = fixed_skew + (N - 1) × steady_step` 做二点诊断分解：

| 框架 | B2 × 16 时长 (ms) | B2 × 256 时长 (ms) | 稳定 pair-step (ms) | 固定错位 (ms) |
| --- | ---: | ---: | ---: | ---: |
| FastLLM | 168.159 | 2797.112 | 10.954 | 3.849 |
| vLLM | 362.405 | 3574.570 | 13.384 | 161.645 |
| SGLang | 201.623 | 3493.114 | 13.715 | 约 0（拟合值 -4.095，属于测量噪声） |

FastLLM 与 vLLM 的 B2 总时长相差 777.458 ms，其中：

- 619.662 ms（79.7%）来自稳定 decode step：`(13.384 - 10.954) × 255`；
- 157.796 ms（20.3%）来自并发请求进入 decode 的固定错位。

Nsight 波形与这个分解一致：vLLM 在 255 个稳定 decode step 之前有两个分开的 prefill/入图段，客户端计时窗口比纯稳定 kernel 窗口多 175.399 ms。由此推断，这次 B2 测试中两个请求没有像 FastLLM 那样从一开始就完整合批进入 decode。FastLLM 与 SGLang 的 696.002 ms 总时长差则几乎全部来自稳定路径。

这个二点分解用于定位时间来源，不应被理解为所有输入长度下都恒定不变的调度模型。

### 逐算子对比

模型配置为 64 层，其中 48 层是 GDN/linear-attention、16 层是 full-attention；每层有 4 个主要 FP8 linear。这与每 step 约 256 个 FP8 linear、48 组 GDN 算子和 16 组 full-attention 完全对应。

表内格式为“每 step 调用数 / 每 step GPU 累计时间”；边界事件会使平均调用数有小数，已按模型结构取近似整数。SGLang 的多 stream 重叠最明显，带 `*` 的原始累计时间尤其不能直接相加。

| 算子族 | FastLLM（调用 / ms） | vLLM（调用 / ms） | SGLang（调用 / ms） |
| --- | ---: | ---: | ---: |
| 主 FP8 linear | 256 / 7.966 | 256 / 8.478 | 256 / 8.482* |
| 独立 activation FP8 quant | 0 / 0 | 192 / 0.310 | 256 / 0.286 |
| GDN 小 BF16/FP16 projection | 48 / 0.121 | 48 / 1.417 | 48 / 2.219* |
| GDN recurrent state update | 48 / 0.208 | 48 / 0.264 | 48 / 0.264 |
| GDN causal conv1d | 48 / 0.096 | 48 / 0.093 | 48 / 0.110 |
| Full-attention core + merge | 32 / 0.147 | 32 / 0.183 | 32 / 0.156 |
| TP all-reduce | 128 / 0.731 | 129 / 0.589 | 129 / 0.900 |
| TP all-gather | 0 / 0 | 1 / 0.037 | 1 / 0.037 |
| LM head | 1 / 0.748 | 1 / 0.774 | 1 / 0.773 |
| Sampling | 3 / 0.082 | 14 / 0.447 | 2 / 0.067 |

关键差异如下。

1. **GDN 小投影是 FastLLM 最大的单项优势。** B2 会命中 `FastllmGemvFp16Fp16Kernel2MultiRow<PART=2>` 专用路径，48 个 projection 平均每个约 2.51 µs；vLLM 的通用 WMMA 路径约 29.65 µs，是 11.8 倍；SGLang 原始累计约 46.2 µs，是 18.4 倍。把 state update 和 conv1d 也计入后，FastLLM 相比 vLLM 每 step 节省约 1.349 ms，占两者稳定 GPU 路径差异的约 54.1%。FastLLM 对 `n == 2` 的明确 dispatch 位于 `src/devices/cuda/linear/fastllm-linear-fp16.cu:687`。

2. **FastLLM 的 B2 FP8 路径不需要单独量化 activation。** 它直接以 half activation、FP8 weight 和 scale 进入 `FastllmGemvHalfFP8E4M3KernelWarpMultiRowBlock128<PART=2>`；主 linear 为 7.966 ms/step。vLLM 是 8.478 ms 主 GEMM 加 0.310 ms 独立量化，合计约 8.788 ms；因此 FastLLM 在这一段再省约 0.822 ms/step，占与 vLLM 稳定 GPU 路径差异的约 32.9%。对应 kernel 和 B2 dispatch 分别位于 `src/devices/cuda/linear/fastllm-linear-fp8.cu:431`、`:605`。

3. **vLLM 的 sampling 在本次配置下明显偏重。** 波形中每 step 约 14 个 kernel，包括 softmax、tensor scan、one-sweep radix 和 sort 后处理，共 0.447 ms；FastLLM 和 SGLang 分别只有 0.082、0.067 ms。仅这一项 FastLLM 相比 vLLM 节省约 0.365 ms/step。原测试配置显式设置了 `VLLM_USE_FLASHINFER_SAMPLER=0`，所以这是当前启动配置的结论；0.36 ms/step 是更换 sampling 路径理论上可收回的时间上限，实际收益需要重新测试，但即使全部收回也不足以抹平主要差距。

4. **通信不是 FastLLM 击败 vLLM 的原因。** vLLM 的 all-reduce 反而比 FastLLM 每 step 快约 0.141 ms。SGLang 的 all-reduce 为 0.900 ms，部分原因是本次原始启动配置使用了 `--disable-custom-all-reduce`，走 NCCL；这同样是配置相关结论。

5. **Full-attention、LM head 都不是主因。** 模型只有 16 个 full-attention 层，三者 attention core 相差最多约 0.036 ms/step；LM head 相差约 0.027 ms/step。

FastLLM 与 vLLM 的稳定 GPU 路径差异可近似分解为：

| 来源 | vLLM 相对 FastLLM 增加 (ms/step) | 占稳定 GPU 路径差异 |
| --- | ---: | ---: |
| 主 FP8 linear + activation quant | 0.821 | 32.9% |
| GDN projection/state/conv | 1.349 | 54.1% |
| Sampling | 0.365 | 14.6% |
| Attention + all-gather | 0.073 | 2.9% |
| LM head | 0.027 | 1.1% |
| All-reduce | -0.141 | -5.7% |
| **合计** | **2.493** | **100.0%** |

### 重叠、launch 数与空隙

| 关键卡每 step 指标 | FastLLM | vLLM | SGLang |
| --- | ---: | ---: | ---: |
| 主 FP8 + GDN 小投影时间并集 (ms) | 8.086 | 9.893 | 9.268 |
| 已归类主要算子时间并集 (ms) | 10.092 | 12.587 | 11.859 |
| 全部 GPU kernel 时间并集 (ms) | 10.825 | 13.324 | 12.956 |
| 波形稳定 step 窗口 (ms) | 11.118 | 13.614 | 14.482 |
| 窗口内无 kernel 间隙 (ms) | 0.293 | 0.291 | 1.526 |
| 图内展开后的 GPU kernel launch / step | 约 971 | 约 1322 | 约 1841 |
| 图外 eager `cudaLaunchKernel` / step（双卡合计） | 7.95 | 105.7 | 47.9 |

- vLLM 的关键卡也接近连续忙碌（kernel 并集占窗口 97.9%），所以其主要问题不是 GPU 饿死，而是每一步做的 kernel 本身更慢、更多。
- SGLang 把 GDN 小投影放到另一条 stream 与主 FP8 分支重叠：其 2.219 ms 原始 GDN projection 中约 68.1% 与其他 kernel 重叠，主 FP8 与小投影的交叠约 1.45 ms。因此，它虽然单个 GDN kernel 慢，关键路径损失没有原始累计时间看起来那么大；FastLLM 在这两个分支的时间并集上仍节省约 1.182 ms/step。
- SGLang 每 step 还能看到约 0.607 ms 的 float/FP8 fill、copy 和 cat 原始核时间，其中一部分彼此重叠。B2 的行数 2 需要补齐到 CUTLASS 要求的 4 行；当前 `sglang_per_token_group_quant_fp8_row_padded` 仍会对补齐尾行做两次 zero-fill。对应实现位于 `/root/sglang_bench/lib/python3.10/site-packages/sglang/kernels/ops/quantization/fp8_kernel.py:669` 和 `:733`。
- SGLang 的 node trace 扰动达到 5.54%，远高于另两者，因此上表 1.526 ms 的空隙不能全部视为无 profiler 时的调度损失；更多图节点和填充/拷贝 kernel 是确定事实，空隙的绝对值则是上界式观察。

### 结论

FastLLM 在 batch 2 快很多，不是由某一个“大算子”或 TP 通信造成，而是两个适合小 batch 的专用计算路径叠加：

- 48 个 GDN 小投影使用 B2 专用 multi-row GEMV，避免通用矩阵乘内核在极小 M 上的高固定开销；
- 256 个主 FP8 linear 直接执行 half × FP8 GEMV，省掉 activation quant，并且主 kernel 本身也稍快；
- vLLM 当前 sampling 配置额外损失约 0.365 ms/step；
- 在算子稳态差异之外，vLLM 的两个请求还有约 158 ms 相对固定错位，贡献了客户端可见总差距的约 20%。

所以，对 vLLM 而言，优先级应是 GDN 小投影专用 kernel、B2 请求合批/入图和兼容的融合 sampling；对 SGLang 而言，优先级是 B2 FP8 padding/fill/copy、较慢的小投影路径，以及在环境允许时恢复 custom all-reduce。FastLLM 的 all-reduce 并不优于 vLLM，继续只优化通信不会解决这次观察到的主要差距。

### 启动配置限制

- 三者均开启 CUDA Graph，最大上下文 1024，batch 2 波形与上表 batch 1--64 使用相同的最大并发 64 配置。
- FastLLM GPU memory fraction 为 0.98；vLLM 为 0.95，关闭 prefix cache、DeepGEMM 和 FlashInfer sampler；SGLang 为 0.84，关闭 radix cache 和 custom all-reduce，attention/sampling backend 为 FlashInfer。
- 因此，以上结论准确描述的是这组可复现配置，而不是声称每一种 vLLM/SGLang 配置都会有完全相同的差距。
