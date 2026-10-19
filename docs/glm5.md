# GLM-5 / GLM-5.3-Flash 部署指南

[English](glm5_en.md) · [返回 README](../README.md) · [Agent 部署速查](model-deployment-agent.md) · [Benchmark](benchmarks/glm5.md)

当前路径覆盖 GLM-5 DSA、GLM-5.3-Flash KDA 与分页缓存，以及部分 GLM-5.2 量化 KV-B checkpoint 的纯 CPU 推理。

自动化部署建议先把本页和 [Agent 部署速查](model-deployment-agent.md) 交给 agent，并优先使用 `--auto`。下面的手动命令用于固定设备、缓存和量化路径。

## 自动配置（推荐）

~~~bash
ftllm server /data/models/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  --auto --model_name glm-5.3-flash \
  --host 0.0.0.0 --port 8080
~~~

GGUF 只需指定第一个分片，其他分片保留在同一目录。需要限制规划预算时，增加
`--auto_gpu_memory_limit 20g` 或 `--auto_host_memory_limit 128g`。

## 手动配置参考

~~~bash
ftllm server /data/models/glm5 \
  --model_name glm5 \
  --host 0.0.0.0 --port 8080
~~~

## GLM-5.3-Flash NVFP4 紧凑存储

GLM-5.3-Flash 的 ModelOpt NVFP4 路由专家在单设备 CUDA 后端（包括 `cudapp` 按层串行）及 NUMA 后端默认使用紧凑存储。`--moe_device numa` 的 CPU 专家计算和 CUDA 混合 prefill 共用该紧凑布局。

每 16 个权重保留 8 字节 FP4 数据和 1 字节 E4M3 块缩放，每行另存 4 字节全局缩放。合并 gate/up 权重时保留各自的全局缩放；这是存储布局转换，不重新量化权重。

## GLM-5.3-Flash GGUF

支持 `general.architecture=glm5next` 的 GLM-5.3-Flash GGUF，包括 Unsloth 的四分片 `UD-IQ2_XXS`。指定第一个分片即可加载其余分片，无需 `--ori`。当前路径支持文本推理及内置 NextN 草稿层的 MTP；`--mtp 0` 时不加载草稿权重。

路由专家保留 GGUF 混合量化格式；加载时还原 KDA 衰减参数和拆分、转置的 MLA KV-B 权重。支持 AVX512 BF16 的 CPU 在 IQ4_XS 单行计算时直接读取压缩权重，在寄存器中解码并完成 BF16 点积，省去临时 FP32/BF16 权重缓冲，保留原有舍入和累加顺序；多行计算和其他 CPU 继续使用分块 BF16 回退，权重不整体展开。KDA 的 128 维 Q8 投影使用 CUDA 反量化 GEMM，避免进入要求 K 维度按 256 对齐的 MMQ 内核。

CPU 词嵌入保留 GGUF 量化表，按 token 解码所需行，避免将完整词表展开为 FP32；显式 CUDA 词嵌入仍使用浮点导入。混合 TP 模式按层加载、恢复并上传非专家权重，随后释放 CPU 源数据，初始化 rank 时直接接管已上传的分片。NUMA 专家仍由各 rank 共享。这些优化自动启用，无需额外环境变量；分组上传目前针对 GGUF TP 路径。

双卡按层串行、单 NUMA 的启动示例（按机器调整线程数）：

~~~bash
FT_NUMAS=1 numactl --cpunodebind=0 --membind=0 \
  ftllm chat /data/models/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  --device cudapp=2 --moe_device numa --threads 20 \
  --dtype bfloat16 --atype bfloat16 --moe_atype bfloat16 \
  --kv_cache_dtype bfloat16 --chunked_prefill_size 8192 \
  --moe_cuda_cache 0 --moe_cpu_cache 0 --mtp 0
~~~

GGUF 的 BF16 混合推理可通过 `--moe_cuda_cache 12G` 为每张显卡设置 12 GiB 专家缓存，支持 IQ2_XXS/IQ2_S gate/up 与 IQ3_XXS/IQ4_XS down。GPU 保留压缩权重，并遵循 GGUF 的 Q8_K/BF16 激活、限幅和路由权重顺序。decode 与最多 9 行的小批验证使用缓存和动态分流，更大批次保留 NUMA 路径。支持的固定页 NUMA 布局直接复用专家权重，避免保存第二份完整主机快照；无法共享的布局保留快照回退。

上述 GGUF 类型配合 `--tp 0,1 --moe_device numa --moe_cuda_cache 16G` 时，默认按完整专家分卡缓存：专家 `e` 存在 `devices[e % devices.size()]`，遵循 TP 参数中的设备顺序。每卡独立维护频率策略和容量，只换入分配给本卡的专家，实际容量为运行时工作区预留显存后确定，允许各卡不同。缓存直接使用原生 NUMA/R4 权重布局，gate/up 和 down 在所属卡上完成；每条路由的 FP32 结果汇集后按原有 BF16 顺序归约。未命中专家继续使用 NUMA／多卡动态分流。prefill 在两卡上传时填充缓存，后续 chunk 和 decode 复用；decode/verify 继续更新频率和驻留专家。该模式自动启用，无需额外参数；单卡、零缓存及不支持的格式保留既有路径。

需要降低主存占用时，可将部分 MoE 层固定放到 CUDA，其余层使用 NUMA 动态分流。例如 45 层、前三层为 dense 的 GLM-5.3-Flash：

~~~bash
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  ftllm chat /data/models/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  --device cuda:0 --tp 0,1 \
  --moe_device '{"cuda:0":7,"cuda:1":4,"numa":34}' --threads 20 \
  --dtype bfloat16 --atype bfloat16 --moe_atype bfloat16 \
  --moe_cuda_cache 0 --moe_cpu_cache 0 --mtp 0
~~~

该映射将第 3–6 层的全部路由专家放到 GPU 0，第 7–10 层放到 GPU 1（层号从 0 开始），其余 34 个 MoE 层保留在 NUMA。GPU 层在加载时上传并释放 CPU 权重，不占用动态专家缓存，不再参与 CPU/GPU 分流。TP 的注意力与共享专家仍由两卡协同计算；每个驻留 MoE 层由映射指定的一张卡计算路由专家。设备映射的数值按全部模型层分配，因此 GPU 0 的 7 层包含前三个 dense 层。

进一步降低主存可使用 `--moe_device '{"cuda:0":9,"cuda:1":6,"numa":30}'`，将第 3–8 层放到 GPU 0、第 9–14 层放到 GPU 1，每卡驻留 6 个 MoE 层。上述 UD-IQ2_XXS 权重比每卡 4 层配置多移出约 9.04 GiB 主机权重。双 24 GiB 显卡、512-token 输入和 640-token 输出的短测可运行，但显存峰值已达约 23.48 / 23.22 GiB；更长上下文需重新预留 KV 与工作区空间。这些驻留层包含 IQ4_XS down，prefill 会使用下面说明的 GEMV 路径，增加驻留层数不保证 prefill 提速。

要让纯 GPU MoE 层也参与张量并行，使用 `--tp 0,1 --moe_device numa --moe_device_layers 30`。`moe_device_layers` 指最后 30 个模型层使用 NUMA；前 15 层中包含 3 个 dense 层和 12 个 MoE 层。每个 GPU MoE 层按中间维度等分 gate/up 的行与 down 的列，各卡独立计算路由和专家分片，随后复用 FFN AllReduce 合并路由专家与共享专家的部分结果。其余 NUMA 层保持原有多卡动态分流。GGUF 分片按层加载并直接上传，保留压缩格式，不保存完整 CPU 副本；GLM GGUF 的 GPU 专家 TP 默认启用 64 MiB 权重 slab，减少大量小分配的显存浪费，可用已有的 `--cuda_slab` 参数覆盖。

在上述 TP 配置中使用 `--mtp 3 --speculative_algorithm mtp` 可开启最多 3 个草稿 token 的自适应深度 MTP，`--mtp_min_p` 沿用既有置信度阈值，0 表示关闭。目标模型多行验证仍使用 TP 和 NUMA 动态分流；草稿层在协调 rank 上运行 attention/共享专家，路由专家沿用最后一个目标层的 MoE 设备设置，词表输出头复用 rank 0 的存储。GGUF 草稿权重保留自身量化格式；不支持 GLM GPU 分流的格式（如此 IQ2_XXS 文件内草稿层的 Q2_K/Q3_K）走通用 NUMA 路径，不会关闭目标层的分流。MTP 不复用 history/prefix 快照。

多行验证保留单行推理的归约与舍入顺序。支持的 KDA 使用寄存器扫描处理任意正序列长度；MLA 的两个吸收投影直接读写原张量中的各行，复用单行 cuBLAS 调用，省去逐行拆分与拼接。无法使用直接行访问的布局保留原路径。TP 提交线程已绑定独立核心时，验证沿用 decode 的有限自旋等待，仍保留 NCCL 提交前后的跨 rank 同步。可用 `cuda_kda_prefill` 和 `cuda_matmul_single_rows` 回归检查递归状态及投影的逐位一致性。

GGUF 混推的小批量验证复用专家注册时的 NUMA 适用性检查结果。2–32 行验证中，持久协调线程推进 CPU 的 gate/up、激活和 down 阶段，GPU 上传与计算提交保留在原调用线程，避免 CPU 后续行等待 GPU 提交完成；CPU 数学计算仍使用配置的 NUMA 工作线程。无需额外开关，单行、其他权重格式和更大批次保留各自路径。`cuda_glm5_gguf_cache` 回归覆盖多行的逐位一致性、CPU/GPU 路由归属以及 GPU 提交异常后的 CPU 任务收尾。

驻留 GGUF 路径支持 BF16 激活、IQ2_XXS/IQ2_S gate/up 与 IQ3_XXS/IQ4_XS down，复用 GLM 缓存路径的限幅、score-before-down、量化边界和有序归约。支持 INT8 MMA 的 NVIDIA GPU（SM75+）上，超过 32 行且 down 为 IQ3_XXS 时自动按专家聚合 token，使用 grouped MMQ；保留每 256 个值一组的正 FP32 Q8_K scale 和最近偶数舍入，不引入 V4.1 的 FP8 量化边界。长输入按最多 1024 行分块，限制临时显存，无需额外开关。decode/小批量、IQ4_XS down 和不支持 MMQ 的设备保留原来的驻留 GEMV 路径；IQ4_XS 的 BF16 down 不会被改为 Q8。可运行 `cuda_glm5_gguf_resident` 验证两卡上的量化组合、32/33 行分派边界、1025 行分块尾部及 CPU 权重释放。

真实命中率应使用 `get_moe_cuda_cache_route_stats()` 在请求前后的差值计算 `resident_routes / routes`，覆盖分给 CPU 的专家，并排除预取查询；`get_moe_cuda_cache_stats()` 的 hits/misses 是查询计数，口径不同。

完整专家分卡缓存沿用 `--moe_cache_*` 参数；`--moe_cache_max_replacements` 和 `--moe_cache_max_bytes` 分别限制每卡每次更新的替换数量和上传字节数。命中路由归属实际计算卡，两卡的逻辑路由计数可以相加。C 接口 `fastllm_moe_cuda_cache_ep_stats()` 额外返回参与卡数、本卡槽位数、已填槽位数、权重字节数、计算的命中路由数及累计上传字节数。`cuda_glm5_gguf_ep_cache` 回归覆盖设备倒序、按专家编号分卡、单／双 NUMA 源布局、冷热替换和多种验证宽度，全命中结果与未分片 GPU 路径逐位对照。`cuda_glm5_gguf_ep_prefill` 和 `cuda_glm5_gguf_ep_numa_prefill` 检查缓存填充与复用、原生权重字节及 NUMA／双卡 prefill 数值。

测速时先用同长度输入完成预热，并关闭 prefix/history cache。启用 `UNIT_TEST` 后可运行 `glm5_next_gguf`、`numas_gguf_fallback`、`cuda_gguf_mmq_alignment` 和 `cuda_glm5_gguf_cache` 回归，分别覆盖分片映射与布局恢复、混合量化专家回退、窄投影的 CUDA 数值与边界、缓存的 CPU/GPU 分配与数值。

## GLM-5.3-Flash NVFP4 grouped Marlin

兼容的 CUDA 紧凑 NVFP4 路由专家默认使用 grouped Marlin，支持 BF16 激活、独立 gate/up 全局缩放和 GPU 路由。prefill 与 decode 共用一次重排后的权重布局；成功准备后释放原 GPU 布局。首次执行包含重排开销，测速需先预热。

GLM 的 `swiglu_limit` 会传入 Marlin 激活核和普通 CUDA 回退路径：gate 只限制上界，up 限制正负两侧，再计算 `SiLU(gate) * up`。非零限幅不会进入只支持普通 SwiGLU 的快捷路径。形状或缩放不满足 Marlin 条件时保留源权重并回退。

存储格式与计算路径自动选择。Marlin 的 Tensor Core 累加和 BF16 舍入与普通路径不同，整模 logits 和 greedy 文本不保证逐位一致。

启用 `UNIT_TEST` 构建后，可运行：
~~~bash
ctest --test-dir build-fastllm -R 'cuda_nvfp4_(marlin|compact)' --output-on-failure
~~~
回归覆盖零限幅行为、BF16/FP16/FP32 激活边界、带限幅的 CUDA 调度、1024-token 路由、4096/2048 top-8、CUDA Graph 重放，以及不支持形状和内存分配方式的回退。

## GLM-5.3-Flash KDA prefill

BF16 KDA prefill 在 head dimension=128、序列长度至少 64、FP32 QK 归一化、beta 舍入为 BF16、每 head 一个 `a_log` 时自动使用 CUDA 寄存器递推核。预处理并行计算 QK 归一化和 gate；每个 warp 负责 16 个 value 列，将 FP32 状态列保存在寄存器中，减少逐 token 的全局显存读写和同步。

实现保持原递推的累加顺序、独立的 decay 舍入及 gate 倒数后乘法，不使用三角分块求解重排。正常编译即可启用，无额外环境开关或 `LD_PRELOAD`。decode、辅助输出、state-only 续算、不匹配的形状或参数，以及 CUDA Graph capture 保留原 CUDA 路径。

临时空间通过现有 CUDA workspace 管理器复用，需求为 `batch × sequence × heads × (3 × 128 + 1) × 4` 字节；batch=1、sequence=1024、heads=64 时为 96.25 MiB。该空间在设备上复用，不按层重复常驻分配。

启用 `UNIT_TEST` 后运行：
~~~bash
ctest --test-dir build-fastllm -R '^cuda_kda_prefill$' --output-on-failure
~~~
回归逐位比较逐 token 递推与寄存器路径的 BF16 输出和 FP32 最终状态，覆盖边界长度、多 batch、长记忆 gate、零值与极小值 QK、初始状态、混合长度续算和 CUDA Graph 重放。

## GLM-5.3-Flash KDA causal convolution

CUDA causal convolution 沿 token 维度增加 block 并行，在 1024 token、8192 channels 时使用 4096 个 block。每个线程仍按原顺序累加卷积 tap，历史缓存由后续同一 stream 的更新核处理。该改动适用于共用此算子的模型，无需额外环境开关、临时显存或预加载库；单 token decode 仍使用原来的 block 数。

## GLM-5.3-Flash MLA prefill

CUDA BF16 prefill 在 query 长度至少 64、QK/V 维度为 256、压缩 KV rank 为 512 时，默认按 attention head 分组临时展开完整历史 K/V，复用现有 `MatMulTransB` 和 FlashInfer `AttentionPaged`。长期缓存仍保存压缩 KV，decode 使用原来的 absorbed MLA。不需要额外运行开关或预加载库；不改变原有 dense attention 语义，也未引入 DSA indexer。

每组展开 K/V 的活动空间上限为 256 MiB，另需 latent、输出和 attention workspace。该上限不包含分配器缓存，不能视为整个设备的额外显存上限。短 query、已有精确小 batch 模式、CUDA Graph capture、不支持的类型或布局，以及展开工作区容量不足时保留原路径。

展开 K/V 改变 BF16 舍入顺序，不保证与 absorbed MLA 的 logits 或 greedy 文本逐位一致。7 组 512/2048/16384-token 输入各生成 8 token，其中 3 个短输入样本分叉，37/56 个 token 位置相同；16K 样本的 8 个 token 相同。仅比较相同前缀的 40 个位置时，logits 最大绝对差为 4.125、最大 RMSE 为 0.5238，概率分布最大 TV 为 0.2246。该抽样不替代完整模型质量评测。

启用 `UNIT_TEST` 后运行 `ctest --test-dir build-fastllm -R '^glm5_next_mla_prefill$' --output-on-failure`。测试以独立 CPU 双精度计算为参考，覆盖不同 head 分组、碎片化页、尾页、带历史的 causal mask、压缩缓存内容不变及回退；CUDA memcheck 为 0 errors。当前实测硬件为 RTX 5090。

## GLM-5.3-Flash learned DSA

CUDA BF16 的 11 个 DSA 层加载 checkpoint 中的 Indexer 权重，执行 32-head、128-dim 的学习式索引。KPool-4 使用逐维 gate + APE 的 softmax 汇聚，经过 BF16、归一化 Hadamard 和 UE8M0 scale 的 E4M3 舍入后，对可见完整组评分，选择最多 512 组，展开成 2048 个 token，再附加当前未完整组的 0–3 个 token。2048 token 以内仍全选，但同时维护索引缓存。

接入复用 DeepSeek-V4 的压缩汇聚、DeepSeek-V4.1 的评分/Top-K/512 维稀疏注意力、Qwen4 的组索引展开和 Naive 的 FP8 量化。复用现有 LayerNorm、量化器及 host 接口。为复用 BF16 Tensor Core 评分，FP8 舍入后的值按 2 的幂反量化后保存在 BF16 中。分页 latent KV 保持原格式，稀疏注意力临时收集连续历史；KPool 缓存和尾组随 chunk、decode 和历史快照保持同步。

当前支持 compressed MLA，包括 MTP 多行验证。目标层按 rank 保存 KDA 状态和 DSA KPool 尾部；拒绝草稿后按接受前缀恢复 KDA、截断 KV 页并重新生成受影响的 KPool 分组，草稿层也独立回滚其 KV 和索引缓存。expanded attention 尚未接入 learned DSA。`FASTLLM_GLM5_NEXT_DSA_BACKEND=dense` 可恢复原 dense 路径用于对照。稀疏选择会改变长输入的注意力语义，不保证与旧 dense 路径生成相同 token，不同 GEMM 后端也不保证逐位相同。

启用 `UNIT_TEST` 后可运行 `ctest --test-dir build-fastllm -R '^glm5_next_dsa$' --output-on-failure`。回归包含独立 LayerNorm、Hadamard/FP8、KPool 参考计算，非 4 倍数分块与尾组，2048 和 32K 附近的因果 Top-K，以及连续/碎片化分页 latent attention。

## GLM-5.3-Flash FlashInfer sparse prefill

显式 `CUDA_ARCH` 列表包含 SM120（例如 `-DCUDA_ARCH=120` 或 `'-DCUDA_ARCH=80;120'`），且 CUDA 编译器版本至少为 12.9、未设置 `CUDA_NO_TENSOR_CORE` 时，默认编译 FlashInfer `GLM53_NOPE` sparse prefill。运行时只在 SM120、单 batch、BF16、64 attention heads、latent rank=512、query 至少 64 token、连续输入且不在 CUDA Graph capture 内时使用。其他情况使用现有 BF16 DSA attention。DSA 的前 2048 token 仍使用 dense 全选路径。

该路径直接复用固定版本的 FlashInfer `swapAB` kernel，并用现有 Dots3 `QuantizeKKernel` 将临时收集的 BF16 latent KV 转成每 128 维一组的 E4M3 + FP32 scale。每个 token 为 528 字节；2051 宽的选择结果用 `-1` 补齐到 2112，保留完整因果尾组。softmax scale 继续使用模型的 `1/16`。Q/输出接口为 BF16，kernel 内部 Q、KV 和概率使用 FP8 运算。分页缓存和历史快照仍保存 BF16，每个 chunk 重新量化临时历史，不持久化 FP8 状态。

只需现有 CUDA/C++ 构建环境。`third_party/flashinfer_glm53` 固定上游 `7eb86aa0fdc1248fab43c89801de4ed450e35e77` 的最小头文件集合，保留逐文件 BSD-3-Clause 和项目 Apache-2.0 许可。两个 fast-div 兼容头复用现有 vendored 实现，以兼容 CUDA 12.9。新头文件只对专用 object target 可见，kernel 单独生成 `sm_120a` cubin。无需安装 FlashInfer Python 包、PyTorch或 Triton。

需要保留 BF16 DSA attention 时，在进程启动前设置：

~~~bash
FASTLLM_GLM5_NEXT_DSA_BACKEND=bf16 ftllm server /data/models/glm5.3-flash --mtp 0
~~~

后端统一由 `FASTLLM_GLM5_NEXT_DSA_BACKEND` 控制，在模型初始化时读取：

| 值 | 行为 |
| --- | --- |
| `auto`（默认） | learned DSA；满足条件时使用 FlashInfer，否则回退 BF16 |
| `bf16` | learned DSA，始终使用 BF16 sparse attention |
| `dense` | Dense attention；expanded 模式使用此路径，compressed MLA 的 MTP 可使用 DSA |

未设置或空值等同 `auto`，其他值会报错。生成 token 不保证与 BF16 后端一致。

FP8 sparse attention 与 BF16 后端的误差可能超过一般 BF16 舍入差异。已记录的 16K/32K 首步 logits 相对 L2 差异为 18.9%/24.6%，余弦相似度为 0.9822/0.9707；有限样本的 greedy token 相同不能替代完整质量评估。32K 输入为 16K 序列重复，不用于评判长文质量。

测量配置和最终吞吐见[Benchmark](benchmarks/glm5.md)。启用 `UNIT_TEST` 后运行：

~~~bash
ctest --test-dir build-fastllm -R '^glm5_next_(flashinfer.*|dsa|mla_prefill)$' --output-on-failure
~~~

回归包含独立 E4M3 打包、double softmax、全屏蔽行、单 key、选择索引补齐、非整块 query、碎片化分页历史以及 dtype/layout/Graph capture 回退。运行验证覆盖 RTX 5090；其他架构的编译检查不代表运行性能或精度已验证。

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

## Benchmark

限定配置的最终 prefill 测量及限制见[GLM Benchmark](benchmarks/glm5.md)。
