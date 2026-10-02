# DeepSeek-V4.1-Flash 支持说明

[返回 README](../README.md) · [DeepSeek-V4 部署指南](deepseek.md) · [混合推理](mixforward.md)

本文记录 FastLLM 对 DeepSeek-V4.1 系列（`model_type = deepseek_v41`，目前为 DeepSeek-V4.1-Flash）的支持范围、
架构差异、启动方式与验证方法。实现位于：

- `include/models/deepseekv41.h` / `src/models/deepseekv41.cpp`：模型（继承 `DeepSeekV4Model`，复用其 MoE / HC-post / WoA / 采样等基础设施）
- `src/models/deepseekv41_vision.cpp`：视觉编码器（ViT + aligner）与图文前向 `ForwardMultimodal`
- `src/devices/cpu/deepseekv41ops.cpp`：V4.1 专用算子的 CPU 参考实现
- `src/devices/cuda/models/deepseekv41-kernels.cu`：对应的 CUDA kernel（面向 SM86 等无 FP8 tensor core 的设备；
  稀疏注意力与 indexer 打分在 SM80+ 上走 BF16 mma，其余算子为 FP32）
- `src/models/deepseekv41_dspark.cpp`：DSpark 投机解码（草稿层 `mtp.*`、校验与回滚）
- `src/devices/cuda/models/deepseekv41-dspark-kernels.cu`：DSpark 草稿侧 markov head 整条链的融合 kernel
- `tools/fastllm_pytools/deepseek_v41_engram.py`：Engram 哈希元数据生成
- `tools/fastllm_pytools/encoding_dsv41.py`：官方 V4.1 prompt 编码（vendored）
- `tools/fastllm_pytools/deepseek_v41_multimodal.py`：图像动态分辨率预处理（移植自官方 `image_processor.py`）与图像占位符展开
- `test/basic/deepseek_v41_reference.py`：与官方 `inference/model.py` 的端到端数值对齐测试
- `test/basic/deepseek_v41_vision_reference.py`：图文输入的端到端对齐测试（含真实 ViT 权重验证）
- `test/basic/deepseek_v41_dspark.py`：DSpark 的精确性（开 / 关输出一致）与回滚测试
- `test/basic/test_deepseek_v41_cpu_fixture.py`：CPU-only 的固定 fixture 对齐回归（只依赖 numpy，可进 CI）
- `test/ops/deepseekV41OpsRegression.cpp`：11 个 V4.1 专用算子的 CPU / CUDA 一致性测试

## 相对 DeepSeek-V4 的架构变化

| 项目 | DeepSeek-V4 | DeepSeek-V4.1 |
| --- | --- | --- |
| Hyper-Connections | 每个子层自己算 pre/post/comb 并立即使用 | pre 由上一个子层计算、下一个子层使用；最后一层 FFN 的 pre 用于 lm_head 前的折叠 |
| 压缩注意力 | 每层独立 compressor（ratio 4 / 128，带 ape、overlap） | `compress_ratios` 取 0 / 1 / 2；只有 `kv_source_layer_ids` 中的层拥有 compressor 与压缩 KV cache，其后各层直接读取（跨层共享） |
| Indexer | 每个 CSA 层有独立的 128 维 compressor + Hadamard 旋转 | key 由 compressor latent 经 `wk + k_norm` 派生；只有 `index_source_layer_ids` 层运行 indexer，其它层复用最近的 top-k |
| 两级 top-k | 无 | `candidate_source_layer_id` 层先按 `candidate_block_size` 分块选出 `candidate_topk_blocks` 个块，之后的 index source 只在候选块内选 top-k |
| Engram | 无 | `engram_layer_ids` 层前对 residual stream 做 n-gram 哈希查表（每层约 100 GB 的 FP8 表）并门控写回 |
| 路由 | 前若干层 hash 路由 | 全部 noaux_tc，`gate.bias`；图像 token 使用 `gate.bias_vl` |
| q 处理 | 额外 RMS 归一 | 无 |
| 权重格式 | FP8 128x128 块 scale | 稠密与共享专家 FP8 32x32 块 UE8M0 scale；路由专家 FP4（沿 K 每 32 个一组 UE8M0 scale） |
| 附加模块 | MTP / DSpark | DSpark 草稿层（mtp.*，markov head 为 embed + head、草稿层 128 专家 top-3、目标层取自主干）、视觉编码器（vision.* / aligner.* / image_start / image_end / image_newline） |

## 当前支持范围

已实现并通过数值对齐测试：

- 文本推理（prefill + decode，含分块 prefill），CUDA 与 CPU 两套算子路径；
- 多请求批量 decode：多个请求的 token 拼成一个序列共享一次前向（Linear / MoE / Engram 查表按整批执行，
  注意力与压缩 KV 按请求分别执行），见下文"多请求与前缀缓存"；
- 前缀缓存 / 多轮对话复用（`--cache_history true`）；
- 跨层共享压缩 KV（ratio 1 / 2）、两级 indexer top-k、Engram、Hyper-Connections、sqrt-softplus 路由；
- 与官方实现一致的 FP8 / FP4 伪量化（窗口 KV、压缩 KV、indexer q/k）；
- 真实 checkpoint 的权重格式（FP8 32x32、FP4 路由专家、FP8 + UE8M0 的 Engram 表）；
- 图像输入（OpenAI 接口的 `image_url`）：ViT + aligner、图像 token 的 `gate.bias_vl` 路由与 Engram 掩码，详见下文；
- DSpark 投机解码（`mtp.*` 草稿层），详见下文"DSpark 投机解码"。
- 双卡张量并行（`--tp 2`），详见下文"张量并行"。

尚未实现：

- 视觉编码器与 DSpark 草稿层的张量并行及 CUDA Graph；主干 decode / DSpark 校验已支持分段图；
- DSpark 与批量 decode 的组合（批内不产生候选，只保持草稿缓存同步）、DSpark 与采样 / 图文请求的组合。

## Engram 元数据

Engram 的哈希基于 tokenizer 归一化（NFKC / NFD / 去重音 / 小写 / 空白折叠）后的"压缩 token id"。
该映射依赖 HuggingFace `tokenizers` 的 normalizer，因此由 Python 侧生成、C++ 侧加载：

```bash
python -m ftllm.deepseek_v41_engram /path/to/DeepSeek-V4.1-Flash
# 生成 /path/to/DeepSeek-V4.1-Flash/engram_meta.json（压缩词表大小应为 99092）
```

通过 `ftllm` 启动时自动读取模型目录的 `engram_meta.json`，缺失时自动生成；模型目录只读时写到
`~/.cache/fastllm/engram/`，C++ 侧自动从该缓存目录读取。素数桶布局由 C++ 侧按官方算法推导，
并与 `engram_num_embeddings` 做一致性校验，无需指定元数据环境变量。

Engram 表（两层，各约 100 GB）以 FP8 + UE8M0 scale 原样保存；查表在 CPU 完成，
`wkv` 投影与门控在 GPU 完成。默认 `--ngram_device cpu` 将表常驻内存；
`--ngram_device disk` 与 Qwen4 PLE 共用通用磁盘权重加载和 `EmbeddingDirect` 算子，
按行读取权重及 scale，不加载整张表。磁盘读取可能增加延迟，访问过的数据仍可能占用操作系统页缓存。

### Engram 的分段计时与几个开关

查表发生在模型代码里而不是算子里，`FASTLLM_PRINT_PROFILE` 看不到它，需要单独计时：

```bash
FASTLLM_DSV41_ENGRAM_PROFILE=1 FASTLLM_CUDA_SYNC=1 ftllm server ...
```

会按 decode / prefill 分桶打印四段的平均耗时：`hash`（算行号）、`gather`（读表 +
FP8→BF16）、`wkv`（投影）、`apply`（门控写回）。后两段在 GPU 上异步下发，
不加 `FASTLLM_CUDA_SYNC=1` 只能量到 kernel launch 的时间。`=2` 额外逐次打印，
`FASTLLM_DSV41_ENGRAM_PROFILE_EVERY=N` 控制中途汇总的频率（默认 64 次，0 表示只在退出时打印）。

下面几个开关默认关闭，打开后数值不变（`ENGRAM_WKV_FP8` 除外，见下）：

| 开关 | 作用 |
|---|---|
| `FASTLLM_DSV41_ENGRAM_POOL`（默认开，=0 关） | 查表改用 fastllm 的常驻线程池，替掉每次调用现场 create/join 最多 32 个 `std::thread` 的写法。输出逐位相同，prefill 收益最明显。 |
| `FASTLLM_DSV41_ENGRAM_PREFETCH=1` | 跨层预取。后台线程提前计算下一层的 n-gram 行号；常驻表还会在缓存预算内预读表行，磁盘模式只预计算行号。后台线程只看历史窗口的快照，不引用请求状态。 |
| `FASTLLM_DSV41_ENGRAM_MADVISE=random / hugepage / both` | 仅作用于常驻表：`random` 设置 `MADV_RANDOM`；`hugepage` 使用匿名 mmap + `MADV_HUGEPAGE` 分配，减少页表与 TLB 开销，并省掉 `std::vector` 的清零。 |
| `FASTLLM_DSV41_ENGRAM_WKV_FP8`（默认开，=0 关） | `layers.{1,14}.engram.wkv.weight` 在真实权重里本来就是 F8_E4M3 + UE8M0 块 scale（`[25600, 6144]`，block 32x32），默认会被解量化成启动 dtype（float16），每层 157 MB 变 314 MB。打开后按原样保留 FP8，**不做任何重量化**——权重数值就是 checkpoint 里的那份，比解成 float16 还少一次舍入；省下每层 157 MB 显存与同样多的每步带宽。只有伴随的 `.scale` 张量存在时才切换，权重是 BF16 的迷你模型不受影响。 |

## 启动

以 2 x 24 GB GPU + 大内存主机为例（专家与 Engram 表放在 CPU 内存）：

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device cuda --moe_device numa \
  --dtype float16 \
  --chunked_prefill_size 4096
```

- 没有 FP8 tensor core 的 GPU（如 SM86）请使用 `--dtype float16`，稠密 FP8 权重会在加载时按 32x32 块 scale 反量化；
- `--kv_cache_dtype` 控制长期 KV 缓存（压缩 KV + indexer key）的存储精度，见下面的“KV 缓存存储精度”一节。
  默认 BF16，可选 `fp8_e4m3`（1650 B/token）与 `fp4_e2m1`（890 B/token，与官方一致，且数值无损）；
- `ftllm` launcher 的自动配置按 config 计算 V4.1 的常驻内存下界（FP4 专家约 289 GB + Engram 表约 203 GB +
  稠密部分），权重文件不全时也不会低估；主机内存不足以放下专家与 Engram 表时会退到 `moe_device=disk`；
- 单路 CPU 机器可用 `--moe_device cpu`；
- 内存需求：Engram 表约 200 GB + 路由专家（FP4）约 270 GB + 加载临时空间；
- 首次启动会生成 `engram_meta.json`（约 1 分钟）并读入两张 Engram 表。

CPU / NUMA 专家使用 FastLLM 自有线程池，由 `--threads` 控制；CLI 会自动设置 `FT_THREADS`，无需重复指定。

### 导出 Q2_K/Q4_K 混合量化

`tools/deepseek_v41_export_q2.py` 按分片导出可独立加载的 FastLLM 模型目录，配置见
`example/quant/deepseekv41/Q2_K_MIXED.json`：主干路由专家的 `w1/w3` 使用 Q2_K，`w2` 使用 Q4_K；
其它线性权重使用 FP16，模型映射保护的权重保留指定精度。Engram 表和 DSpark 草稿权重逐字节复制。
混合分片会先分离这些原样保留的张量，导出不依赖 `FASTLLM_DSPARK_TOKENS` 设置。
原始 Flash checkpoint 的路由专家已经是 FP4，因此这里是从 FP4 再量化，未使用校准集或 importance matrix。

```bash
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  python tools/deepseek_v41_export_q2.py \
  --model /path/to/DeepSeek-V4.1-Flash \
  --output /path/to/DeepSeek-V4.1-Flash-Q2_K-Mixed \
  --threads 28
```

中断后使用相同命令并增加 `--resume`；脚本检查源配置、量化配置和分片信息，跳过已经完成的分片。
输出是 FastLLM 的扩展 safetensors 格式，需使用支持导出激活元数据和 V4.1 GGML NUMA 激活转换的 FastLLM：
独立 scale 会内嵌到转换后的权重，`config.json` 中的 `fastllm_activation_quantized_linears` 保留原模型的激活量化边界。

单 NUMA 混推启动示例（CPU 核号应按机器的 NUMA 拓扑调整）：

```bash
FT_NUMAS=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash-Q2_K-Mixed \
  --device cuda:0 --moe_device numa --ngram_device cpu \
  --threads 28 --dtype float16 --kv_cache_dtype bfloat16 \
  --chunked_prefill_size 4096 --max_batch 1 --dspark 5 \
  --fast_prefill --cache_history true --host 0.0.0.0 --port 8080
```

2026-10-01 在单 NUMA、28 线程、RTX 4090、chunk 4096、DSpark 5、fastprefill 开启的配置下验证：
导出权重为 419.45 GiB（源权重 475.25 GiB，缩小 11.74%），整模型可加载和生成。
8,600-token prefill 预热后 3 次 TTFT 中位数为 14.12 秒，短输入解码为 25.75 token/s；
同配置原 FP4 分别为 7.37 秒和 27.56 token/s，因此当前方案节省空间但没有加速。
5 次长上下文取值 JSON 全部正确，算术和生成代码的 17 个用例通过；这些功能用例不能替代完整精度评测。

2026-10-02 拉取 `origin/master` 的 8 个新提交（`2faae1d9` → `2a3c71c8`），保留现有本地优化后，
对更新前已安装的库和重新编译安装的库分别运行 16 个请求。配置为 Q2 混合量化、RTX 4090、
单 NUMA 28 线程、`--moe_device numa --ngram_device cpu`、chunk 4096、fastprefill 开启；
专家和 Engram 常驻内存，未开启显卡专家缓存，关闭历史前缀缓存。
每种正式测量重复 3 次，表中取中位数；普通解码逐请求关闭推测，DSpark 请求使用 `--dspark 5`。

| 指标 | 更新前 | 更新后 | 变化 |
| --- | --- | --- | --- |
| 短输入普通解码（33-token 输入、128-token 输出） | 25.13 token/s | 24.99 token/s | -0.5% |
| 同一短输入 DSpark 解码 | 24.03 token/s | 26.13 token/s | +8.7% |
| 8,600-token 输入首 token 时间（包含 prefill） | 13.93 s | 14.07 s | +1.0% |
| 长输入 DSpark 解码（30-token 输出） | 41.68 token/s | 41.62 token/s | -0.1% |

普通解码和长输入速度基本没有变化。短输入 DSpark 草稿接受率由 50.0% 变为 57.9%，
每次验证轮数由 48 变为 46；普通解码、DSpark 的短输入和中等输入自然语言输出也随版本变化。
因此短输入的 8.7% 提升不能全部归因于算子加速，也不能据此保证其它任务有同样提升。
两版算术、生成代码及全部长上下文检索输出 token 一致，各模式内三次重复输出一致；
检索 JSON 和生成代码的 17 个功能用例均通过。

两版稳定 RSS 均约 408.9 GiB，包含加载和预热的 RSS 峰值约 424.9 GiB，计算卡显存峰值约 19.91 GiB。
更新后模型加载时间从 397 秒降至 284 秒，但未清空系统页缓存，不能把加载时间差异视为代码加速。
重新编译时修正了上游 `cuda/moe/fastllm-moe-gguf-cache.cu` 对父目录头文件的相对包含路径；
`cpu_gguf_repack`、`numas_v41_q2_moe`、`disk_v41_q2_moe`、`disk_v41_q2_moe_cuda_prefill` 四项回归通过。
库哈希、逐次测量及输出一致性记录见[更新前后混推实测](benchmarks/deepseek_v41_q2_numa_latest.json)。

2026-10-02 补齐了上述 Q2_K/Q4_K 混合模型的显卡专家缓存。缓存直接借用已注册的
NUMA Q2_K_R4/Q4_K_R4 权重；普通解码和 2–8 行 DSpark 验证保留 block-32 FP8 激活量化、
GGML Q8 编码、路由权重位置、SwiGLU 限幅和逐专家 BF16 舍入。启动时增加
`--moe_cuda_cache 3g` 即可启用，其它参数沿用上面的单 NUMA 命令。

在同一进程中交替开关缓存，关闭历史前缀缓存，完成 30 个预热、测速和功能请求。
每种短输入模式和长输入 DSpark 模式重复 3 次，表中取中位数；两种长输入普通解码各测 1 次。
关闭缓存的对照保留已分配的缓存显存，模型与草稿权重均相同。

| 指标 | 关闭显卡专家缓存 | 3 GiB 显卡专家缓存 | 变化 |
| --- | --- | --- | --- |
| 短输入普通解码（33-token 输入、128-token 输出） | 24.92 token/s | 22.95 token/s | -7.9% |
| 同一短输入 DSpark 解码 | 23.59 token/s | 26.24 token/s | +11.2% |
| 长输入 DSpark 解码（8,600-token 输入、30-token 输出） | 41.43 token/s | 43.49 token/s | +5.0% |
| 长输入 DSpark 首 token 时间 | 14.066 s | 14.092 s | +0.2% |
| 长输入普通解码（单次） | 23.93 token/s | 20.60 token/s | -13.9% |

缓存容纳 224 个专家，payload 为 2.999 GiB，借用 NUMA 权重避免了额外的 205.664 GiB
完整专家快照。稳定 RSS 约 409.0 GiB，包含加载的峰值约 425.0 GiB，计算卡显存峰值约
22.92 GiB。最后一次累计路由报告中，普通解码约 33.9%、验证约 27.6% 的 route 在 GPU
计算；缓存 lookup 的 90.7% 命中率只覆盖参与 lookup 的请求，不能当作全部路由的驻留率。
模型卸载后 payload 回到 0。

长检索的输出 token 和草稿接受数量在缓存开关两侧一致，接受率均为 96%；该用例的约 5%
解码提升较稳定。短输入自然语言输出在两侧不同，开缓存后三次输出也不同，草稿接受率
从 47.4% 变为 49.7%，因此 11.2% 包含输出和路由变化，不能全部归因于算子加速。
缓存当前没有加速普通解码，也未加速常规 prefill；建议根据实际 DSpark 任务测量后决定是否开启。

7 项缓存及 Q2 NUMA/磁盘回归通过。新增 top-6、1–8 行缓存对照覆盖 CPU/GPU 分担、
淘汰、零输入、零路由权重和限幅，与 NUMA 对照的相对 RMS 误差最大约 0.000272，
并通过独立标量参考。整模型两侧的算术、生成代码 17 个用例和各 5 次长检索 JSON 均通过，
这些检查不代表完整精度评测。逐次计时、接受率、输出一致性及库哈希见
[Q2 显卡专家缓存实测](benchmarks/deepseek_v41_q2_cuda_cache.json)。

随后使用同一原生库、单 NUMA 28 线程，将主干按层 1:1 串行分到 RTX 4090 和 RTX 4090 D，
把缓存扩大为每卡 8 GiB。先尝试的每卡 12 GiB 在 8,600-token 长输入 prefill 时 OOM；
启动预热已提前分配缓存，关闭缓存 dispatch 的对照仍保留这些显存，因此不能用短请求的
显存占用推断长请求也能容纳同样的缓存。完整测速使用每卡 8 GiB，总预算 16 GiB。

```bash
FT_NUMAS=1 OPENBLAS_NUM_THREADS=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash-Q2_K-Mixed \
  --device "{'cuda:0':1,'cuda:1':1}" --moe_device numa --ngram_device cpu \
  --threads 28 --dtype float16 --kv_cache_dtype bfloat16 \
  --chunked_prefill_size 4096 --max_batch 1 --dspark 5 --mtp 0 \
  --fast_prefill --moe_cuda_cache 8g --cache_history true \
  --host 0.0.0.0 --port 8080
```

保持默认关闭 CUDA Graph。两张卡按层串行计算，各自缓存所属层的专家；不使用 TP。
测速关闭历史前缀缓存，完成 30 个请求。先预热短、长请求，再交替开关缓存；缓存开启侧
额外预热普通解码和 DSpark 各 256 个输出 token，让更大的缓存驻留和调度成本有时间稳定。
正式短请求和长输入 DSpark 各重复 3 次，下表取中位数；长输入普通解码各测 1 次。

| 指标 | 双卡串行、关闭缓存 | 双卡串行、每卡 8 GiB | 变化 |
| --- | --- | --- | --- |
| 短输入普通解码（33-token 输入、128-token 输出） | 24.58 token/s | 25.65 token/s | +4.4% |
| 同一短输入 DSpark 解码 | 23.04 token/s | 32.02 token/s | +39.0% |
| 长输入 DSpark 解码（8,600-token 输入、30-token 输出） | 39.63 token/s | 48.07 token/s | +21.3% |
| 长输入 DSpark 首 token 时间 | 13.622 s | 13.484 s | -1.0% |
| 长输入普通解码（单次） | 23.51 token/s | 22.74 token/s | -3.3% |

大缓存主要改善 DSpark 验证，普通解码收益仍有限。长输入的输出和草稿接受数量一致，接受率
均为 96%；缓存开启侧三次解码为 46.05、48.07、49.96 token/s，缓存继续学习驻留和调度成本。
短输入自然语言输出在两侧不同，缓存开启侧三次输出也不同，草稿接受率从 44.4% 增至
56.6%，每轮接受数量从 1.19 增至 1.47，因此 39.0% 不能全部归因于 GPU 计算加速。
常规 prefill 仍走配置的后端，首 token 时间基本没有变化。

与前一轮单卡 3 GiB 缓存比较，短输入 DSpark 从 26.24 增至 32.02 token/s（+22.1%），
长输入 DSpark 从 43.49 增至 48.07 token/s（+10.5%）。两轮使用相同的库，但为不同进程；
双卡关闭缓存的普通解码和 DSpark 没有比前一轮单卡关闭缓存更快，主要收益来自更大的
缓存与验证阶段的 CPU/GPU 分担。

每张卡实际 payload 为 7.994 GiB、597 个槽，共 1,194 个槽。最后一次累计路由报告中，
两张卡各有约 51–52% 的普通解码 route、45–47% 的验证 route 在 GPU 计算；验证 route
驻留率约 61%，高于单卡 3 GiB 时的约 28%。97.5% 的 lookup 命中率只描述参与 lookup
的请求。CUDA 0（RTX 4090）的显存峰值约 19.14 GiB，CUDA 1（RTX 4090 D）约 20.54 GiB；
稳定 RSS 约 409.0 GiB，加载峰值约 425.0 GiB。两侧算术、生成代码 17 个用例和各 5 次
长检索 JSON 均通过；模型卸载后两张卡的缓存 payload 均回到 0。这些用例不代表完整精度评测。
缓存容量、路由、计时、接受率、输出对照和 12 GiB OOM 记录见
[Q2 双卡串行大缓存实测](benchmarks/deepseek_v41_q2_dual_serial_cuda_cache.json)。

同一天将模型换回原始 NVFP4 checkpoint `/path/to/DeepSeek-V4.1-Flash`，其余参数、原生库、
双卡串行 1:1 分层、单 NUMA 28 线程、每卡 8 GiB 缓存、DSpark 5、fastprefill 和测速顺序
均与上述 Q2 测试一致，完成同样的 30 个请求。三次正式测量的中位数如下；长输入普通解码各测 1 次。

| 指标 | NVFP4 关闭缓存 | NVFP4 每卡 8 GiB | Q2 每卡 8 GiB |
| --- | --- | --- | --- |
| 短输入普通解码（33-token 输入、128-token 输出） | 26.36 token/s | 28.91 token/s | 25.65 token/s |
| 同一短输入 DSpark 解码 | 25.60 token/s | 34.18 token/s | 32.02 token/s |
| 长输入 DSpark 解码（8,600-token 输入、30-token 输出） | 44.53 token/s | 52.29 token/s | 48.07 token/s |
| 长输入 DSpark 首 token 时间 | 5.824 s | 5.826 s | 13.484 s |
| 长输入普通解码（单次） | 24.88 token/s | 25.97 token/s | 22.74 token/s |

同缓存预算下，NVFP4 的短普通解码比 Q2 快 12.7%，短 DSpark 快 6.7%，长 DSpark 快 8.8%；
8,600-token 首 token 耗时缩短 56.8%，按输入数除以 TTFT 计约为 2.31 倍。短输入首 token
时间约 0.33 秒，Q2 约 0.67 秒。NVFP4 常规 prefill 比 Q2 更快，显卡专家缓存本身没有
改变 NVFP4 的长输入首 token 时间。

NVFP4 内部开关缓存的短普通解码提升 9.7%、短 DSpark 提升 33.5%、长 DSpark 提升 17.4%。
长检索输出 token 和草稿接受数量在两种缓存模式及两种模型格式间一致，接受率均为 96%；
缓存开启侧三次长解码为 49.52、52.29、54.59 token/s，缓存继续学习驻留和调度成本。
短输入开关缓存后自然语言输出不同，各模式内三次输出一致；草稿接受率分别为 44.70% 和
44.68%。NVFP4 和 Q2 的短输入输出、草稿候选数量也不同，因此短解码的跨格式差异不能
全部归因于算子耗时。

NVFP4 稳定 RSS 约 471.0 GiB，比 Q2 多 62.0 GiB；含加载的峰值约 487.2 GiB。
CUDA 0（RTX 4090）显存峰值约 19.05 GiB，CUDA 1（RTX 4090 D）约 20.61 GiB。
每卡缓存实际 payload 为 7.984 GiB、456 个槽，共 912 个槽；相同字节预算容纳的 NVFP4
专家少于 Q2。两张卡的最后一次累计报告中，约 43–45% 的普通解码 route、48% 的验证
route 在 GPU 计算；验证 route 驻留率约 54–55%。97.1% 的 lookup 命中率不代表全部路由。

运行前的 `cuda_dsv41_moe_cache_test --dual` 通过，覆盖独立 FP8/BF16 标量参考、4,096 个
FP4 编码/scale 用例、409 次多行对照、1,008 次模式/分担对照和 GPU 0/1/0 切换。
整模型两侧的算术、生成代码各 17 个用例和各 5 次长检索 JSON 均通过，卸载后两卡 payload
均回到 0；这些检查不代表完整精度评测。详细记录见
[NVFP4 双卡串行大缓存实测](benchmarks/deepseek_v41_nvfp4_dual_serial_cuda_cache.json)。

以下记录 grouped MMQ 适配前 Q2 与 NVFP4 的 prefill 差距：保持上述双卡串行、单 NUMA 28 线程、每卡
8 GiB 缓存和 fastprefill 配置，输入同一份 8,600-token 文本，只生成 1 个 token，关闭
推测解码和历史前缀缓存。每个模型先预热，再测 3 次常规请求和 2 次独立的逐算子 profile。
首 token 时间取常规请求中位数，MoE 阶段时间取这 3 次的均值，单位为秒。

| 指标 | NVFP4 | Q2 混合量化 |
| --- | --- | --- |
| 首 token 时间 | 5.930 | 13.653 |
| NUMA MergeMOE 内部阶段合计 | 3.504 | 11.198 |
| CPU 阶段 | 2.745 | 9.442 |
| CPU 阶段结束后等待 GPU 的时间 | 0.674 | 1.628 |
| 4,096-row 大块的 CPU 阶段合计（42 次） | 1.701 | 7.814 |
| 128-row 裁剪块的 MoE 阶段合计（57 次） | 0.968 | 2.060 |

Q2 的 routed MergeMOE prefill 确实使用 CPU 和两张 GPU 并行计算；4,096-row 大块中，
CPU 处理 38,010 条 route，占 3.68%，GPU 0/1 分别处理 497,060 和 497,122 条 route。
但 CPU 阶段累计耗时 7.814 秒，其后等待 GPU 累计仅约 0.000076 秒，说明大块的关键路径
在 CPU 一侧。CPU 阶段包括输入就绪等待、CPU 专家准备和计算，以及 CPU partial 的拷贝
提交；GPU 工作与其重叠，不能把这个 GPU 等待数解释为 GPU 总计算耗时。

适配前的源代码检查确认了以下额外成本。Q2 CPU 专家输入先经过整批 BF16 → FP32 → Q8_K 转换，
随后 `QuantizeNumasV41Input` 又从原始 BF16 重新分配整批 FP32 缓冲，串行应用 block-32
FP8 量化，再编码为 Q8_K，覆盖前一次编码。即使 CPU 只处理少量 route，仍准备全部
4,096 × 5,120 个输入元素，然后才提取 CPU 专家需要的行。同时，
`useDeepSeekV4LargeFast` 要求 down 输入不是 GGUF 格式，因此 Q2 没有启用 NVFP4 的并行
down 输入准备、舍入和输出存储，也没有跳过随后会重算的通用 SwiGLU。

128-row 裁剪块则出现较多 GPU 等待。Q2_K_R4 / Q4_K_R4 不在 CUDA GGUF MMQ/MMVQ 的
接入列表中，多行专家计算落到反量化为 BF16 再调用 cuBLAS GEMM 的路径。NVFP4 的
直接量化 GEMV 支持少于 32 行的专家批次，更适合这里的稀疏路由。两种格式在这组
128-row 块的 CPU 阶段接近，Q2 的额外 GPU 等待约为 1.05 秒。

两次独立逐算子 profile 中，MergeMOE 均值分别为 NVFP4 3.551 秒、Q2 11.220 秒；
普通 Linear 为 0.370 / 0.357 秒，SparseAttention 为 0.334 / 0.346 秒，差距主要来自
MoE。常规 prefill 没有新增任何显卡专家缓存 lookup；这条大批量 GPU 分担路径临时上传
专家权重，与解码和小批量验证的驻留缓存路径不同。

当时确定的优化方向包括 CPU 输入准备：只量化 CPU 实际使用的行、移除被覆盖的首次 Q8_K 编码并复用
工作缓冲，再让 GGUF 后处理接入保留相同数值边界的并行实现。随后补齐 R4 小批量 CUDA
计算。默认专家分流校准没有传入 V4.1 的激活量化参数，也值得改用真实层耗时验证；
本轮没有单独测量切分策略的贡献。整模型 CPU 阶段内部尚未进一步细分，以上内部成本
依据代码检查，不应将 9.442 秒全部归于某一个量化步骤。测速进程已退出，显存已释放；
配置、路由、分块计时和算子记录见
[Q2 / NVFP4 prefill 分析](benchmarks/deepseek_v41_q2_nvfp4_prefill_analysis.json)。

2026-10-02 补齐 V4.1 grouped MMQ 后，沿用同一配置和 8,600-token 输入重新实测。
`!deepSeekV4Mode` 已替换为模型数值规则、量化格式、形状、设备和显存容量的检查：
BF16 输入、block-32 激活、Q2_K gate/up + Q4_K down（含 NUMA R4 布局）可以复用
分组、路由重排、workspace 和 MMQ 的矩阵分块/MMA；其他组合保留回退。
每张卡在每次正式长请求中执行 120 次 grouped MMQ，未发生回退。

| 指标 | 适配前 | 适配后 |
| --- | --- | --- |
| 首 token 时间，中位数 | 13.653 秒 | 9.316 秒 |
| 输入 token / 首 token 时间 | 629.9 token/s | 923.1 token/s |
| NUMA MergeMOE 内部阶段合计，均值 | 11.198 秒 | 6.806 秒 |
| CPU 阶段，均值 | 9.442 秒 | 2.729 秒 |
| CPU 完成后等待 GPU，均值 | 1.628 秒 | 3.994 秒 |

首 token 耗时降低 31.8%，上述吞吐提高 46.5%。CPU 输入准备去掉不再使用的首次 Q8
编码，将源格式转换、FP8 和 GGML 编码合并到现有线程池，并复用每行 scratch。
GPU 只量化每个输入 token 一次，然后按 route 提取；保留 V4.1 的截断、路由权重先乘、
BF16/FP8 边界和专家累加顺序。MMQ 适配保留 FP32 scale，并用整数 MMA 求激活和，
避免普通半精度 scale/sum 元数据在抵消较强的输入上放大误差。
现在更多时间花在等待 GPU 分担阶段；该时间包含剩余传输与计算，不能直接视为纯 kernel 时间。

11 项回归全部通过，CUDA memcheck 为 0 错误；包含实际 5,120 × 2,304 专家尺寸的
独立标量参考检查，最大相对 RMS 为 0.0884%，对 NUMA 参考最大为 0.0909%。
长文本检索输出的三个编码及完整 JSON 正确，生成 30 个 token；这些检查不等同于完整
困惑度或模型精度评测。矩阵累加顺序仍可能改变 BF16/FP8 舍入和后续路由。
此轮 CUDA 0/1 显存峰值约 23.08 / 23.49 GiB（仍为每卡 8 GiB 专家缓存），
CPU RSS 峰值约 425.07 GiB。显存不足会在上传前回退；测速进程已退出并释放显存。
配置、逐次计时、调用计数、路由和精度结果见
[Q2 grouped MMQ prefill 实测](benchmarks/deepseek_v41_q2_grouped_prefill.json)。

同日进一步分析适配后与 NVFP4 的剩余差距：Q2 首 token 9.316 秒，相同配置的 NVFP4
基线为 5.930 秒，耗时多 57.1%；按输入 token / TTFT 计为 923.1 / 1,450.3 token/s。
CPU 阶段为 2.729 / 2.745 秒，已经接近；CPU 完成后等待 GPU 为 3.994 / 0.674 秒。
MoE 增加的 3.302 秒占首 token 差距约 97.5%，其中 81.1% 来自 4,096-row 大块。
整模型数据沿用上述两轮记录，NVFP4 基线使用适配前的库；以下 GPU 对照则全部使用当前同一库。

固定 BF16 输入、相同路由与 score，采用模型实际 5,120 / 2,304 专家尺寸，单卡承担
44 个专家、约一半 route。使用合成权重，分别调用当前 Q2_K_R4 / Q4_K_R4 与 NVFP4
host MoE 路径，预热 3 次后测量 5 次。RTX 4090 的 Nsight Systems 分解如下；kernel
和传输取每次调用均值，总耗时取中位数，存在重叠，不能直接相加。

| 4,096-row 单卡 MoE | Q2 混合量化 | NVFP4 |
| --- | --- | --- |
| 总耗时 | 98.03 ms | 32.75 ms |
| 权重上传字节数（十进制 MB） | 632.59 | 827.23 |
| H2D 传输 | 23.72 ms | 30.98 ms |
| R4 / gate-up 布局还原 | 14.59 ms | 无独立步骤 |
| 矩阵乘法 | 56.85 ms | 7.34 ms |
| 独立权重反量化为 BF16 | 无独立步骤 | 6.18 ms |
| kernel 与 H2D 重叠时间 | 0 ms | 15.04 ms |

Q2 大块 MMQ 的 gate/up 与 down 分别为 38.03 / 18.82 ms。当前精度适配保留 FP32
scale 和 min 修正，并固定使用 16-token tile；这些是需要进一步对照的实现成本。
NVFP4 在这里反量化后走 BF16 GEMM。Q2 权重少传 23.5%，两条路径传输期间的带宽均约
26.7 GB/s，但 Q2 在同一 stream 依次上传、还原全部选中专家，再开始 grouped MMQ；
NVFP4 提前上传下一个专家，约 15 ms 计算被传输覆盖。关闭 profiler 后两者为
97.82 / 32.47 ms；RTX 4090 D 为 102.65 / 32.50 ms，趋势一致。

128-row、22 专家对照为 Q2 24.52 ms、NVFP4 15.99 ms。此时 Q2 矩阵乘法合计
4.44 ms，低于 NVFP4 GEMV 的 5.40 ms，额外的 7.57 ms 布局还原和缺少传输重叠成为
主要差距。因此后续应分别处理大小批次：大批次比较更大 MMQ tile 与保留数值边界的
反量化/BF16 GEMM；减少或融合 R4 还原，并按专家组流水上传和计算。现有驻留专家缓存
未参与这条临时上传路径，单纯增大缓存不会消除这些成本。微基准不含 CPU 专家并行和
整模型显存压力，不能直接把单算子加速比当作整模型收益。原始采样及限制见
[Q2 GPU prefill 瓶颈分析](benchmarks/deepseek_v41_q2_gpu_prefill_bottlenecks.json)。

随后完成三项 GPU 优化：Q2/Q4 R4 还原按 SM 数量扩大线程块网格；V4.1 的至少 1,024-row
批次使用 64-row MMQ 计算块，同时保留 16-row 路由填充和原有 workspace 容量，尾部读取
限制在当前专家内；down 权重在独立高优先级 stream 上传和还原，与 gate/up 计算重叠。
两个 event 分别保护描述符就绪和 down 权重就绪，只增加一个投影大小的 staging，
本模型约 7.38 MiB。高优先级可避免短还原 kernel 等待大矩阵全部执行完，进而阻塞下一次上传。
没有新增运行环境变量，仍沿用上述能力检查和显存不足时的回退。

固定路由、相同合成权重的微基准中，RTX 4090 的 4,096-row MoE 从 97.94 ms 降至
38.60 ms，128-row 从 24.48 ms 降至 15.31 ms；RTX 4090 D 分别为 40.69 / 15.69 ms。
两卡、两种尺寸的 BF16 输出均与优化前逐字节一致。Nsight 中 H2D 与 kernel 重叠从 0
增至每次约 10.92 ms；在加入并行上传前，单独的还原阶段已从 14.59 ms 降至 2.81 ms，
矩阵乘法从 56.85 ms 降至 21.36 ms。最终并行版本中的 kernel 时间含资源竞争，
不能直接与传输时间相加；端到端收益以墙钟计时为准。

相同单 NUMA、双卡串行、每卡 8 GiB 缓存、fastprefill、8,600-token 输入配置下，
预热后三次正式请求的首 token 为 5.854 / 5.904 / 5.893 秒：

| 指标 | 本轮 GPU 优化前 | 本轮优化后 | NVFP4 同配置基线 |
| --- | --- | --- | --- |
| 首 token 时间，中位数 | 9.316 秒 | 5.893 秒 | 5.930 秒 |
| 输入 token / 首 token 时间 | 923.1 token/s | 1,459.5 token/s | 1,450.3 token/s |
| NUMA MoE 内部阶段合计，均值 | 6.806 秒 | 3.452 秒 | 3.504 秒 |
| CPU 阶段，均值 | 2.729 秒 | 2.670 秒 | 2.745 秒 |
| CPU 完成后等待 GPU，均值 | 3.994 秒 | 0.710 秒 | 0.674 秒 |

首 token 耗时减少 36.8%，有效吞吐提高 58.1%，已与 NVFP4 基线基本持平；约 0.6% 的
差异不应解释为确定的领先。NVFP4 整模型仍引用之前相同配置的基线，本轮单卡 NVFP4
算子对照使用当前库，为 32.51 ms。正式请求每张卡均有 120 次 grouped MMQ、0 次回退。
CPU RSS 峰值约 425.06 GiB，CUDA 0/1 显存峰值约 23.69 / 23.01 GiB。

14 项回归全部通过，新增 1,023 / 1,024 / 1,041-row 边界覆盖；NUMA 参考最大相对 RMS
为 0.0909%，独立标量参考为 0.2461%（包含新增样例，不能直接与前一轮更小的样例集比较）。
静态测试程序在两卡完成宽分块尾部和异步上传的 CUDA memcheck，0 错误；动态微基准的
首次 sanitizer 注入未成功，未将该次运行计作内存检查通过。长文本检索再次生成正确的
完整三编码 JSON，共 30 token；这些检查不等于完整困惑度评测。新库已安装到本机 ftllm，
测速完成后释放显存。实现、逐次计时、波形分解和验证记录见
[Q2 GPU prefill 优化实测](benchmarks/deepseek_v41_q2_gpu_prefill_optimized.json)。

Q2 混合量化也可以使用磁盘专家和有容量上限的 CPU 专家缓存。Engram 同时放磁盘、关闭 DSpark 的示例：

```bash
FT_NUMAS=1 OPENBLAS_NUM_THREADS=1 numactl -C 0-31 -m 0 \
  ftllm server /path/to/DeepSeek-V4.1-Flash-Q2_K-Mixed \
  --device cuda:0 --moe_device disk --moe_cpu_cache 64g --ngram_device disk \
  --threads 28 --dtype float16 --kv_cache_dtype bfloat16 \
  --chunked_prefill_size 4096 --max_batch 1 --dspark 0 --mtp 0 \
  --fast_prefill --cache_history true --host 0.0.0.0 --port 8080
```

`64g` 表示 64 GiB 专家缓存容量，不是进程总内存上限。磁盘缓存路径为 V4.1 的 GGML 路由专家和
FP16 共享专家保留 block-32 激活量化、BF16 中间舍入及路由权重位置；这两种专家在 CPU 上计算。
专家缓存未命中时默认使用 direct I/O，避免另在系统页缓存中保留整份专家；Engram 行读取仍使用系统页缓存。
磁盘 MoE 已用独立标量计算验证，不能直接沿用 NUMA 配置的整模型精度评测结论。

2026-10-01 在上述单 NUMA、RTX 4090 配置下，不使用前缀缓存：模型加载约 35 秒；专家缓存达到
63.99 GiB 后，进程 RSS 约 70.8 GiB，运行期间 RSS 峰值约 75.5 GiB。同一 33-token 输入生成 64 token，首次 TTFT 21.48 秒、
解码 1.74 token/s，重复请求 TTFT 1.18 秒、解码 5.06 token/s，两次输出 token 完全相同。
8,600-token 输入首 token 约 330 秒，检索 JSON 正确，但从 SSD 读取约 218 GiB。
此配置降低常驻内存，长上下文的首次 prefill 仍然很慢；实际部署还需为工作区、KV 和系统留出余量。

随后优化了 Q2_K/Q4_K 的 AVX2 权重重排，并将 V4.1 CPU GGML 专家的重排放到磁盘预取线程；
小批次输入量化减少线程调度。重排保持原有 R4 字节布局，缓存容量和量化规则不变，无需重新导出模型。
单专家 Q2 gate/up 重排从 5.63 ms 降至 0.59 ms，Q4 down 从 2.38 ms 降至 0.66 ms。
同配置的代码生成任务（41-token 输入、192-token 输出，关闭前缀缓存，每种模式两次正式测量）结果如下：

| 模式 | 优化前 decode | 优化后 decode | 加速比 | 优化前 TTFT | 优化后 TTFT |
| --- | --- | --- | --- | --- | --- |
| 普通解码 | 2.25 token/s | 3.67 token/s | 1.63x | 29.69 s | 15.50 s |
| DSpark 5 | 1.60 token/s | 2.78 token/s | 1.74x | 29.30 s | 15.43 s |

数值取两次测量的中位数；优化前数据来自上一轮同任务测量。六次新请求的 192 个输出 token 与优化前一致，
DSpark 接受率仍为 61.8%。RSS 稳定值约 70.8 GiB、峰值 75.69 GiB，专家缓存仍受 64 GiB 上限约束。
此磁盘配置下普通解码仍比 DSpark 快，部署命令继续使用 `--dspark 0`。
完整配置、算子对照和验证结果见[实测记录](benchmarks/deepseek_v41_q2_disk_repack.json)。

进一步在加载时完全关闭 DSpark/MTP（`--dspark 0 --mtp 0`），同一代码生成任务复测了磁盘读取优化：
Q2_K/Q4_K 专家直接读入最终的对齐权重缓冲区，不对齐的文件偏移和分散的 gate/up 片段在缓冲区内移动并保留前一片段尾部；
预取复用一个后台线程，单个专家替换足够腾出空间时不再排序整个缓存。对齐分配的额外空间计入原有缓存预算。
两版均生成 192 token、禁用历史与前缀缓存，热缓存数值取三次重复请求的中位数：

| 指标 | 本轮优化前 | 本轮优化后 |
| --- | --- | --- |
| 热缓存 decode | 3.80 token/s | 4.90 token/s（1.29x） |
| 热缓存 TTFT | 15.36 s | 12.35 s |
| 首次请求 TTFT | 34.11 s | 27.01 s |
| 稳定 RSS | 70.80 GiB | 70.72 GiB |
| 运行 RSS 峰值 | 75.55 GiB | 71.79 GiB |

六项 CPU/CUDA 磁盘专家回归通过，包含奇数文件偏移、不同片段前缀、不连续 gate/up 和文件尾部不足一扇区的情形；
直接读取、普通读取及冷/热缓存的 MoE 输出逐字节一致。四次整模型请求的 192 个输出 token 与优化前完全一致，
没有推测验证轮次，专家缓存不超过 64 GiB。这里的首次请求发生在模型 warmup 后，未清空全机系统页缓存；
长上下文未在本轮复测。单独同步 CUDA、关闭 CUDA Graph 的算子采样覆盖 191 步解码：每步 40 层的
`MergeMOE` 合计从 236.93 ms 降到 187.44 ms；超过 1 MiB 的复制合计从 35.40 ms 降到 3.37 ms，
缓存腾出空间从 17.92 ms 降到 8.12 ms。复制和读取包含重叠的预取工作，不能与算子耗时相加。
配置、算子耗时和验证记录见[磁盘读取优化实测](benchmarks/deepseek_v41_q2_disk_io.json)。

随后用同一优化版本重测 `--dspark 5 --mtp 0`（默认置信度 0.5）：同进程先预热普通解码和 DSpark，
再按关/开、开/关、关/开交替，各测三次上述 41-token 输入、192-token 输出。草稿权重在两种模式下均已加载，
普通对照通过请求的 `output_token_least=1` 禁用推测，DSpark 请求设为 0；历史和前缀缓存均关闭。

| 指标（三次中位数） | 普通解码（已加载草稿权重） | DSpark 5 |
| --- | --- | --- |
| decode | 4.80 token/s | 3.67 token/s（0.76x） |
| TTFT | 12.30 s | 12.39 s |
| 每次请求物理 SSD 读取 | 86.02 GiB | 117.98 GiB |
| 稳定 RSS | 70.72 GiB | 70.66 GiB |

DSpark 三次速度为 3.69、3.63、3.67 token/s，草稿接受率均为 61.8%；八次请求的输出 token 与完全关闭
DSpark/MTP 的参考输出逐 token 一致。磁盘读取优化使 DSpark 相对先前的 2.78 token/s 提高约 32%，
但本任务仍比同进程普通解码慢约 24%，物理 SSD 读取多约 37%；草稿与验证的额外专家读取很可能限制了净收益。
当前磁盘专家和 64 GiB 缓存配置继续使用 `--dspark 0 --mtp 0`。本轮未修改推理代码，完整记录见
[DSpark 磁盘读取优化后复测](benchmarks/deepseek_v41_q2_dspark_disk_io.json)。

同一版本随后用两组关/开、开/关请求做逐算子采样：普通前向 382 次、验证前向 102 次，每次覆盖 40 层。
为量到 CUDA 执行时间，采样同步算子边界并关闭 CUDA Graph；下面耗时是所有层合计的每次前向均值，
普通前向处理 1 个位置，验证平均处理 5.31 个位置、提交 3.67 个位置。折算收益为
`3.67 × 普通耗时 / 验证耗时`，只包含本算子，不含草稿、普通回退和提交。

| 算子 | 普通前向 ms | 验证前向 ms | 按提交位置折算收益 |
| --- | --- | --- | --- |
| MergeMOE | 199.15 | 927.00 | 0.79x |
| attention wq_b Linear | 3.92 | 4.72 | 3.05x |
| attention wo_b Linear | 3.94 | 4.09 | 3.53x |
| WoA | 3.41 | 4.76 | 2.62x |
| vocabulary Linear | 1.40 | 1.41 | 3.62x |
| routed router Linear | 0.71 | 2.46 | 1.06x |
| sparse attention | 1.37 | 1.73 | 2.91x |
| HC PreNorm | 0.75 | 0.90 | 3.08x |
| HC Mix | 1.82 | 2.47 | 2.71x |
| HC Post | 1.41 | 3.40 | 1.52x |
| activation quantization | 1.04 | 1.32 | 2.88x |
| rotary quantization | 1.15 | 1.40 | 2.99x |
| RMSNorm | 0.79 | 0.98 | 2.96x |
| Engram row gather | 0.08 | 0.33 | 0.90x |

验证前向总耗时 969.73 ms，MoE 占 95.6%。MoE 内 CPU 路由 gate/up Linear 从 17.22 增至 88.37 ms，
down Linear（共享与路由合计）从 20.02 增至 82.51 ms；共享 gate/up Linear 则只从 7.69 增至 10.14 ms。
每次前向的路由 gate/up 调用从 240 增至 855.33，验证每个选中专家平均只接到 1.49 行，
而非整体批大小的 5.31 行。专家读取也随之增加：`pread` 累计时间从 92.42 增至 472.48 ms，
读取量从 322 MiB 增至 1,779 MiB，Q2/Q4 重排总时间从 30.61 增至 177.87 ms。
这些读取和重排包含后台预取的重叠工作，不能与 MoE 或 CPU Linear 时间相加。

按实际收到的输出 token（排除首 token）归一化，采样中的普通路径为 228.65 ms/token；
DSpark 的验证为 258.93、普通回退 10.49、草稿 12.10、提交 0.16 ms/token，合计 281.69 ms/token。
验证中延迟写入的 window KV 在提交阶段记录，验证中的零 WindowStore 时间不代表删除了这项工作。
六次请求的 192 个输出 token 均与普通参考一致，候选数、接受数分别与验证/提交的记录行数核对通过。
完整的 48 项算子及调用次数见[算子 CSV](benchmarks/deepseek_v41_q2_dspark_operators.csv)，
MoE 嵌套计时见[内部计时 CSV](benchmarks/deepseek_v41_q2_dspark_operator_details.csv)，
阶段、I/O、精确性验证和采样配置见[完整 JSON](benchmarks/deepseek_v41_q2_dspark_operators.json)。

针对这次采样中的缓存管理和内存回收开销，CPU 驻留专家改用按原有热度、最后使用时间排序的索引堆；
每 4,096 次访问发生热度衰减时重建，其余访问只更新对应位置。对齐读取缓冲区按分配大小精确复用，
最多四个、单个不超过 16 MiB、合计不超过 32 MiB；它们是 64 GiB 活跃专家缓存以外的工作空间，
每次读取都覆盖旧内容并重新重排，关闭 CPU 缓存或卸载最后一个缓存条目时释放。未新增环境变量。

同一模型、单 NUMA、28 线程、磁盘专家、64 GiB CPU 缓存和 DSpark 5 配置，继续用两次预热加
关/开、开/关、关/开交替测量。两种模式都加载草稿权重，每种模式三次正式请求的中位数如下：

| 指标 | 优化前 | 优化后 | 变化 |
| --- | --- | --- | --- |
| 普通 decode | 4.80 token/s | 5.71 token/s | +18.8% |
| DSpark 5 decode | 3.67 token/s | 4.24 token/s | +15.5% |
| 普通热请求 TTFT | 12.30 s | 11.05 s | -10.1% |
| DSpark 热请求 TTFT | 12.39 s | 10.97 s | -11.4% |

普通三次速度为 5.71、5.67、5.89 token/s，DSpark 为 4.24、4.16、4.24 token/s。
八次请求均生成与优化前完全相同的 192 个 token，DSpark 每次仍为 51 轮、220 个候选、136 个接受，
接受率 61.8%；每次请求的缓存命中、未命中、淘汰次数、读取字节数和驻留缓存字节数也全部与旧版一致。
每次物理 SSD 读取仍约为普通 86.02 GiB、DSpark 117.98 GiB。稳定 RSS 约 70.75 GiB，
正式请求峰值分别为 70.77、70.92 GiB，专家缓存始终不超过 64 GiB；卸载后专家缓存计数归零。
六项 CPU/CUDA 磁盘缓存与 Q2 MoE 回归通过，涵盖缓存热度衰减、容量缩小、卸载和非对齐读取。
这次加速来自管理和分配开销的减少；DSpark 在该任务仍只有普通解码的 0.74 倍，部署继续关闭推测。
本轮只检查了这个 41-token 提示词的速度和输出一致性，未复测长上下文或完整模型精度。
配置、逐次测量和验证见[专家缓存管理优化实测](benchmarks/deepseek_v41_q2_dspark_cache.json)。

另用相同的两组关/开、开/关请求采样，仍为普通 382 次、验证 102 次前向，每次 40 层，
同步 CUDA 边界并关闭 Graph；每次前向合计均值如下，嵌套计时不与 MoE 总时间相加：

| 项目 | 普通前向优化前 → 后 | 验证前向优化前 → 后 |
| --- | --- | --- |
| MergeMOE | 199.15 → 161.79 ms | 927.00 → 790.89 ms |
| 缓存腾出空间（嵌套） | 13.18 → 0.13 ms | 70.63 → 0.65 ms |
| malloc_trim（嵌套） | 10.71 → 0.20 ms | 50.91 → 1.29 ms |

两版采样中的读取字节数、读取次数和专家重排次数完全一致。验证仍平均每个选中专家只处理 1.49 行，
每次前向读取约 1,779 MiB，Q2/Q4 重排合计约 185 ms；这些后台工作可与计算重叠，不能相加为总耗时。
后续减少验证带来的额外专家读取和重排，才有机会让 DSpark 在这个磁盘配置下超过普通解码。

继续将 V4.1 的 2～8 行 MoE 小批次改为双专家预取：两个常驻后台线程并发读取、重排，
最多保留两个未消费的专家；消费、计算、缓存入驻和 FP32 求和仍按原来的专家顺序执行。
正常结束或异常退出都等待所有预取任务完成，再释放缓存锁和模型权重。单 token 解码与大块 prefill
保持单专家预取，原有 32 MiB 缓冲复用池和 64 GiB 活跃缓存预算不变，未新增环境变量。

同一代码提示词、相同草稿权重及配置，各版本均两次预热，再关/开、开/关、关/开交替测试：

| 指标（三次中位数） | 单专家预取 | 双专家预取 |
| --- | --- | --- |
| 普通 decode | 5.71 token/s | 5.72 token/s |
| DSpark 5 decode | 4.24 token/s | 5.44 token/s（+28.2%） |
| DSpark / 同进程普通 decode | 0.74x | 0.95x |
| DSpark 热请求 TTFT | 10.97 s | 10.81 s |
| DSpark 正式请求峰值 RSS | 70.92 GiB | 70.93 GiB |

双路 DSpark 三次为 5.45、5.28、5.44 token/s；另外完整测试三路预取，三次中位数为 5.34 token/s，
同进程普通解码为 5.77 token/s，没有得到额外收益，因此保留双路。两种候选版本的全部 16 次请求
均与单路版本的 192 个输出 token 完全一致；逐请求缓存命中、未命中、淘汰次数、读取量及驻留字节数
全部一致，DSpark 每次仍为 51 轮、220 个候选、136 个接受，接受率 61.8%。双路只让原有读取和重排
更多地重叠，未减少工作量，也未修改量化、接受规则或置信度阈值。新增的未消费专家是缓存预算外的
暂存空间，该模型一个路由专家的 gate/up 加 down 约 13.7 MiB。最终双路版本重新编译安装，
六项 CPU/CUDA 缓存与 Q2 MoE 回归通过，包含 2、3、6、7、8 行的冷/热缓存与独立数值参考。
该任务中 DSpark 仍比普通路径慢约 5%，速度结论限于当前 41-token 输入、192-token 输出。
详细参数、双路/三路测量及验证见[DSpark 并发专家预取实测](benchmarks/deepseek_v41_q2_dspark_prefetch.json)。

最终双路版本另做两组算子采样，同样同步 CUDA 边界、关闭 Graph，覆盖 382 次普通前向和
102 次验证前向。普通 `MergeMOE` 仍为 161.76 ms/前向；验证 `MergeMOE` 从 790.89 降到
648.49 ms，验证前向总时间从 835.00 降到 691.85 ms。六次采样请求的输出 token、候选数和接受数
全部核对通过，读取字节数、读取次数及重排次数也与单路采样相同。验证中的 `pread` 累计时间反而
从 394.52 增到 577.43 ms，因为多个读取互相并发并共享 SSD 带宽；这个累计值不能当作等待时间，
也不能和重排、CPU Linear 或 MoE 墙钟时间相加。部署吞吐使用上表未插桩的 5.44 token/s。

常规推理不调用 OpenMP / MKL，`OMP_NUM_THREADS`、`MKL_NUM_THREADS`、`OMP_WAIT_POLICY`、`KMP_BLOCKTIME`
可从上述命令中省略。Tokenizer 的 Python 依赖可能加载带 OpenMP / MKL 的 PyTorch，但不承担模型前向。
NumPy 会加载 OpenBLAS，建议保留 `OPENBLAS_NUM_THREADS=1` 以免建立额外的大线程池。
前缀缓存默认未禁用，无需 `FASTLLM_DSV41_DISABLE_PREFIX_CACHE=0`；`FASTLLM_DSV41_PREFIX_CACHE_DEBUG` 仅用于诊断。
DSpark 默认每 32 轮校验打印总体及逐位置接受率，无需设置 `FASTLLM_DSPARK_STATS` / `FASTLLM_DSPARK_STATS_EVERY`。
`FT_NUMAS` 按部署需要设置；Engram 表的存放方式使用 `--ngram_device cpu|disk` 控制。

### 实测（DeepSeek-V4.1-Flash 真实权重，2026-09-12）

主机：EPYC 7C13（Zen 3，无 AVX512，128 线程）+ 943 GB 内存 + 2 x RTX 3090 Ti（SM86，24 GB），
权重在共享盘上（478 GB，48 个分片）。线程数 `-t 31`，`--chunked_prefill_size 4096`。

| 指标 | 实测 |
| --- | --- |
| 加载耗时 | 7–20 分钟（取决于两张各 101.4 GB 的 Engram 表是否还在页缓存） |
| 内存 | 约 520 GB 常驻 |
| 图像 | 448x336 的图 210 个 prompt token，端到端 6 秒 |

单卡与按层分片双卡的对照（两者的 31k 与 123k"大海捞针"均命中）：

| 配置 | 每卡显存 | decode | 8k 首 token | 32k 首 token |
| --- | --- | --- | --- | --- |
| 单卡 | 15.0 GB | 78 ms/token | 13.1 s | 50.1 s |
| 按层分片（`--device "{'cuda:0':1,'cuda:1':1}"`） | 7.2 / 8.1 GB | 78–80 ms/token | 11.3 s | 44.0 s |

**decode 在 14 到 32014 token 之间完全持平**，说明 BF16 mma kernel 之后注意力与 indexer 已经不再是
长上下文的瓶颈。按层分片的 prefill 反而快 12–14%，推测是每卡显存宽裕、分配器压力小。

优化历程（短上下文 decode / 32k prefill）：

| 阶段 | decode | 32k prefill |
| --- | --- | --- |
| 初始实现 | 194 ms/token | 207 s |
| BF16 mma 的注意力与 indexer | 164 ms/token | 66 s |
| AVX2 融合 FP4 专家 kernel | 78 ms/token | 50 s |
| 按层分片双卡 | 78–80 ms/token | 44 s |

单步 decode 的算子分解（`FASTLLM_PRINT_PROFILE=1 FASTLLM_CUDA_SYNC=1`，短上下文）显示 `MergeMOE`
（CPU 上的 FP4 路由专家）占绝对多数：AVX2 kernel 上线前为 137 ms / 190 ms，之后约 43 ms / 78 ms。
所以短上下文的 decode 仍由 CPU 专家决定，多卡并行只能切 GPU 的那部分。

行为验证（贪心解码）：中英文常识、算术、代码生成、逻辑推理均正确；31k 与 123k 上下文的"大海捞针"命中
（这两个长度都会激活候选块两级 top-k）；工具调用能正确产出 `tool_calls`；图像输入能正确描述图中的形状与颜色。

## 张量并行

支持 GPU 共享专家、单 token 异步调度与分段 CUDA Graph。吞吐收益取决于 GPU、卡间通信和
CPU 专家占比，应在相同配置下分别测量 decode 与首 token 延迟。

```bash
FASTLLM_CUDA_GRAPH=1 ftllm server /path/to/DeepSeek-V4.1-Flash \
  --tp 2 --moe_device numa --cuda_shared_expert true --dtype float16
```

`--tp 2` 会把主 device 归一化成 `multicuda:0,1` 并设置 `FASTLLM_TP`（触发加载期的权重切分）。
用户显式给出的 `--moe_device`（如 `numa`）不受影响：路由专家仍在 CPU / NUMA 上算一份，
结果广播回两张卡。直接写 `--device multicuda:0,1` 也能启用，只是稠密权重改为在首次使用时切分，
显存峰值更高。

### 切什么、为什么

沿用 DeepSeek-V4 的切法，只切 **query head** 这一条线：

| 权重 | 切法 | 说明 |
| --- | --- | --- |
| `attn.wq_b` | 按行切（`linearRow`） | 输出 q 的 head 维分片，`Reshape` 会把 tpAxis 从最后一维换算到 head 维 |
| `attn.attn_sink` | 按 head 切 | 与 q 的分片区间一致 |
| `attn.wo_a` | 按 head 组切（`linearRow`） | 区间必须对齐到 `o_group`（每组 `num_attention_heads / o_groups` 个 head） |
| `attn.wo_b` | 按列切（`linearColumn`） | 输入分片，输出 all-reduce 回复制布局 |
| `ffn.shared_experts.gateup` / `w2` | 行切 / 列切 | 与其它 MoE 模型一致 |
| `head.weight` | 按行切 | 两卡各算一半词表，`LLMSamplingBlock` 已有分片 logits 的贪心 / 汇聚路径 |

其余全部**复制**（每卡一份完整副本）：`wq_a`、`wkv`、`compressor.*`、`indexer.*`、`ffn.gate`、
各 RMSNorm 权重，以及跨层共享的压缩 KV、indexer key、滑窗 KV 缓存。

为什么跨层共享的压缩 KV 与两级 indexer 选复制而不是切分 + all-gather：

- **压缩 KV 与 head 无关**。V4.1 的注意力是 MLA 式的：每个位置只有一份 `head_dim` 的 latent，
  所有 query head 共享它。沿 head 切它没有意义，沿序列切则每一层注意力都要 all-gather 整段 KV，
  通信量正比于上下文长度 × 层数。复制的代价只是一份显存（压缩后本身就比滑窗 KV 小 1~2 倍）。
- **压缩 KV 是跨层共享的**。`kv_source_layer_ids` 的层算出的 cache 会被后面若干层直接读，
  切分后每个消费层都要重新 all-gather，而不是只在生产层付一次代价。
- **两级 indexer 的候选块也是跨层共享的**。`candidate_source_layer_id` 选出的候选块被之后所有
  index source 层复用。indexer 的分数矩阵是 `[token, m]`（1M 上下文时每 token 有 50 万个候选），
  切 index head 意味着要对这个矩阵做 all-reduce 才能取 top-k——通信量比整个注意力还大。
  而且 top-k 必须在两张卡上**逐位一致**：卡 0 和卡 1 选到不同的 KV 块，
  两边算出的注意力就不是同一个函数的两个分片了。复制 indexer 既省通信又天然保证一致。
- **路由分数变换与专家选择同理**逐卡各算一份，保证两卡选到同一组专家。
- **Engram 表在 CPU**，查表结果按 token 复制到每张卡。

### 约束

TP 当前要求 `num_attention_heads / tp` 是 32 的倍数，以兼容 CUDA 稀疏注意力的回退路径，
并且要对齐到 `o_group`。真实模型 64 头、`o_groups=8`，TP=2 满足（每卡 32 头 = 4 个 o_group）；
TP=4 不满足。不满足时模型会打印一行说明并**整体退回单卡**（撤销注意力与 head 的 TP 权重注册，
把 device map 改回 `cuda:<第一张卡>`），而不是做"只切 FFN"的半张量并行。

视觉编码器（ViT + aligner）与 DSpark 草稿层仍在单卡上算。`--tp 2 --dspark 5` 可以组合使用：
目标模型按上述方式张量并行，草稿层在第一张卡运行，共享的 `head.weight` 沿用两卡词表分片，
完整 logits 汇总到第一张卡进行候选生成与校验。校验回滚同时截断两张卡上的缓存副本。

TP 主模型特征传回草稿 GPU 时显式指定设备，避免 `ToDevice` 的布尔重载跳过上传；
回归覆盖两种首卡顺序、预填充、不同接受长度及缓存副本一致性。

2026-09-17，RTX 4090D + RTX 4090、TP=2、FP16、BF16 KV、NUMA 30 线程、专家缓存关闭，
`--dspark 5`、默认置信度 0.5、CUDA Graph 开启。三种 LRU 任务各预热一轮、计时两轮，
每请求最多 1024 token，吞吐按总输出 token 数与总解码时间计算，排除首字延迟：

| LRU 任务 | 普通 TP + Graph（此前） | DSpark 优化前 | DSpark 优化后 | 优化后接受率 |
| --- | ---: | ---: | ---: | ---: |
| Python 字典 + 双链表 | 26.99 | 29.80 | 40.00 | 86.86% |
| C++ list + unordered_map | 26.98 | 30.49 | 43.01 | 91.96% |
| Python OrderedDict | 27.07 | 28.58 | 40.80 | 89.60% |
| 总体 | 27.02 | 29.53 | 41.07 | 89.21% |

单位为 token/s。优化后每轮 verify 平均 113.9 ms，KV 提交/回滚 2.91 ms；
1～6-token 图均实际重放。普通 TP 和优化前数据引用此前同配置测量。
单独优化 KV 提交的对照中，输出全文一致，提交耗时由 52.03 降至 3.22 ms。
组合优化的自由生成轨迹存在差异，真实模型端到端逐 token 一致性及标准精度评估尚未完成；
算子、主特征传输、Graph 与 KV 回滚回归通过。


### 混合推理的 decode 调度

TP 的共享专家 gate/up 按中间维度切分，down 执行 all-reduce；路由专家仍使用配置的 CPU / NUMA 后端。
输入先暂存，再发射 GPU 共享专家，使两条分支重叠。单 token TP 默认使用常驻 worker 与流事件，
保持 Graph 和段外算子使用的 GPU 地址稳定，并在采样前等待所有 logits 分片的生产完成。

CPU 专家结果上传、共享结果相加与 HC post 在一次 worker 调用中执行，保留 AddTo 的中间舍入。
普通单 token 和 DSpark verify 共用该路径。输入直接 D2H，CPU MoE 输出保持二维以复用 GPU 副本；每次上传覆盖新结果。
不支持的布局、类型或设备使用原路径。同步 H2D 保证 CPU 源在回调返回后即可复用。

DSpark 提交 KV 时复用同一套异步 TP 派发，在整轮提交结束后同步两张卡。
部分接受时，截取的 KV 行保留到同步结束，避免逐层同步以及临时缓冲提前释放。
NUMA 的多行 BF16 输入在至少 8192 个元素时按完整 block-32 分配激活量化任务，保持原有量化结果；
单行和较小输入保持串行，避免线程派发开销。
这些优化不改变 prefill 的动态专家分配。

比较 TP 与按层分卡时，应保持共享专家位置、CPU 线程、NUMA、KV 类型和 prefill 分块相同。
NUMA 按多个 LLC 分散绑核时，TP 执行线程保留启动时的 CPU 亲和性，避免把专家正在使用的核
误判为空闲核。不需要额外设置绑核环境变量。

### 早期同步调度的测量

迷你模型（`--perf-config`：4 层、64 头、`o_groups=8`，与真实模型同构）在 2 x RTX 3090 Ti 上：

| 场景 | 单卡 | TP=2 |
| --- | --- | --- |
| prefill 2048 token | 0.60 s | 0.33 s |
| prefill 8192 token | 0.80 s | 0.43 s |
| decode（4 层，每 token） | 3.3 ms | 7.7 ms |

prefill 接近减半；decode 变慢，因为 multicuda 的 eager 调度对**每个算子**都要唤醒两个 worker 线程
并同步一次，实测每个算子多约 34 us。迷你模型每层只有几十微秒的真实计算，被调度开销淹没；
真实模型每层的 GPU 计算是它的十几倍，两者大致相抵。

这条开销正是 CUDA Graph 要解决的（见下一节）：同一个 4 层迷你模型上开图之后，
TP=2 的 decode 从 7.4 ms/token 降到 4.5 ms/token。**长上下文 / prefill 为主的负载开 `--tp 2`；
纯 decode 负载开 `--tp 2` 时建议同时打开 CUDA Graph。**

## 快速 prefill（`--fast_prefill`）

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash --device cuda --moe_device numa \
  --fast_prefill
```

默认关闭，也支持 `--fast-prefill` 写法，启动时以参数为准。
开启后使用 decoder SWA bounded replay：每个文本 prefill chunk 完整计算到最后一个
KV source 层，后续层只计算各请求末尾 `sliding_window` 个 token。V4.1-Flash 对应
第 0–20 层处理完整 chunk，第 21–39 层处理末尾至多 128 token。
长度不超过窗口的片段、包含图像嵌入或图像掩码的前向保持完整计算。

压缩 KV 和 indexer key 仍完整保留，RoPE、缓存槽位和请求长度使用原始绝对位置。
支持分块 prefill、混合批次、前缀缓存恢复及 DSpark；DSpark 的 main hidden 同步取尾部，
verify 始终计算全部候选位置。单 token decode 和 CUDA Graph 的执行范围不变。

这是**近似计算**：后段层的局部 attention 在保留窗口的起点截断，可能改变 logits，
差异不局限于浮点舍入。开启后应按实际任务验证质量，并固定输入与 chunk 配置对比速度。

## decode 与 DSpark 校验的 CUDA Graph

```bash
FASTLLM_CUDA_GRAPH=1 ftllm server /path/to/DeepSeek-V4.1-Flash --device cuda --moe_device numa
```

把 decode 和 DSpark 校验里与位置无关的那部分 GPU 计算捕获成 CUDA Graph，按 token 数分别缓存，减少
kernel launch。**单卡有收益（省下每步上千次 launch），TP 下收益大得多**——multicuda 每个算子
要唤醒两个 worker 并同步一次，进图之后这笔钱一次付清。

### 开关

| 变量 | 默认 | 作用 |
| --- | --- | --- |
| `FASTLLM_CUDA_GRAPH` | 关 | 统一控制 decode 与 DSpark 校验的 CUDA Graph，`1` 开、`0` 关 |
| `FASTLLM_DSV41_CUDA_GRAPH_WARMUP` | 2 | 捕获前的预热轮数（让显存池、权重量化缓存达到稳态） |
| `FASTLLM_DSV41_CUDA_GRAPH_DEBUG` | 关 | 打印捕获 / 首次重放的 token 数，以及失效 / 关闭事件 |
| `FASTLLM_DSV41_CUDA_GRAPH_REPLAY_MASK` | 15 | 排查用：按位选择回放哪几种段（bit0 pre / bit1 post / bit2 route / bit3 sharedExpert），其余走逐算子 |
| `FASTLLM_DSV41_CUDA_GRAPH_FAIL_AT` | 关 | 排查用：让第 N 段捕获强制失败，验证回退路径 |
| `FASTLLM_DSV41_CUDA_GRAPH_INVALIDATE_EVERY` | 关 | 排查用：每 N 次回放强制失效一次，验证重捕获路径 |
| `FASTLLM_DSV41_CUDA_GRAPH_FORCE_ROUTE_CAPTURE` | 关 | 排查用：强行捕获本来进不了图的路由，验证撞上非法同步 D2H 时的回退 |

### 捕获了什么、没捕获什么

整段前向没法一次捕获：Engram 查表在 CPU 上做、压缩 KV 与 indexer key 用 `Expansion + CatDirect`
追加（写偏移是 host 状态、容量增长时会重新分配）、滑窗 KV 是环形缓冲（写位置随 token 变）、
两级 indexer 的候选数随上下文增长、路由专家还可能落在 cpu / numa 上。这些"随 token 变化"的
部分集中在每层的注意力核心与 MoE 两处。

因此按层做**分段捕获**，每层捕获四段与 token 位置完全无关的纯 GPU 计算：

| 段 | 内容 |
| --- | --- |
| pre | hc_attn 混合 -> attn_norm -> wq_a / q_norm / wq_b、wkv / kv_norm -> compressor 的 wkv / wgate 投影、indexer 的 wq_b / weights_proj 投影 |
| post | wo_a -> wo_b -> hc 残差 -> hc_ffn 混合 -> ffn_norm |
| route | 路由 gate + `SelectExpert` |
| sharedExpert | 共享专家 gateup / SwiGLU / down |

段与段之间保持逐算子执行：RoPE、压缩块追加、indexer 打分与 top-k、稀疏注意力、滑窗写入
（pre 与 post 之间），以及 `MergeMOEBlock`、共享专家相加、hc 残差（sharedExpert 之后）。

**占位段**：某一段本来就进不了图时，它会被标成占位段——捕获与回放时都逐算子执行，
只占住段号让前后段对齐。目前有两种情况：

- **路由**：`FastllmCudaDeepSeekV4RouteScoreTransform` 的 kernel 只支持 ≤ 256 个专家，
  真实模型是 384，于是路由整段走 CPU 参考实现，里面的 `logits.ToDevice(CPU)` 是同步 D2H，
  捕获期间非法。真实模型上 40 个 route 段全部是占位段。
- **共享专家**：`GetCudaSharedExpert()` 为假或权重是 disk weight 时。

因此真实模型每步是 120 次图启动 + 40 段占位 + 约 360 次逐算子调用。

这样做的好处是**图里不含任何 startPos、缓存长度或缓存指针**：图一经捕获，在权重与设备布局
不变的前提下一直有效，上下文增长既不会让它失效，也不需要为 KV 预分配上下文上限。
代价是注意力核心（约每层 1/3 的算子）留在图外。

### 适用范围

只有同时满足下面全部条件的前向才会走图，其余一律逐算子执行：

- 单请求、单片段、`startPos > 0`，且为单 token decode 或 DSpark 多 token 校验；
- 单卡，或 multicuda 张量并行。**按层切分（`--device "{'cuda:0':1,'cuda:1':1}"`）不支持**：
  一段图只能属于一张卡，而按层切分下每层跑在不同的卡上，层间的跨卡拷贝在捕获期需要
  预先建好的 NCCL 通信子，因此按层切分时不启用图；
- 纯文本（图像 token 走 CPU 参考路由，无法进图）；
- 模型主体确实跑在 CUDA / multicuda 上（`--device cpu` 时不启用）；
- 没有开 `FASTLLM_DSV41_DUMP_DIR`、`FASTLLM_CUDA_SYNC`、`FASTLLM_PRINT_PROFILE`
  （它们会在捕获中插入 host 侧拷贝或同步）。

DSpark 校验按本轮 token 数（已确定的一个 token 加候选数）各自预热、捕获和重放，
置信度截断或剩余输出长度变化时复用对应形状。目标层特征采集、KV 更新、回滚和草稿层仍在图外。
批量 decode（`batch > 1`）和普通多 token prefill 不走图。

按层切分请保持 `FASTLLM_CUDA_GRAPH=0`（默认值）。全局开关为 `1` 时还会隐式启用
CUDA embedding，即使当前布局不捕获图也会生效。DSpark 的主模型与草稿使用不同 GPU 时，
共享 embedding 权重会随两条路径的切换反复跨卡搬运，造成明显降速。

### 失效与回退

- 整个模型按 token 数共用图与常驻工作区（不碰任何请求私有的缓存），
  并发前向用 `try_lock` 抢工作区，抢不到的直接逐算子执行；
- 每次回放前核对全部边界张量的设备地址，任何一个搬了家（设备迁移、重新分配）就销毁重捕获，
  连续失效超过三次彻底关图；
- 设备布局、TP 切分方式、KV dtype、共享专家开关任一变化都会重捕获；
- 捕获或回放失败（含 TP 下某个 rank 失败）就地退回逐算子并永久关掉图，打印一行原因，不会崩。

### 实测（迷你模型，2 x RTX 3090 Ti）

每档 3 次取中位数，decode 300 token：

| 场景 | 关图 | 开图 | |
| --- | --- | --- | --- |
| 单卡，6 层 / 32 头 | 3.75 ms/token | 3.52 ms/token | -6.1% |
| 单卡，4 层 / 64 头 | 3.02 ms/token | 3.00 ms/token | 持平 |
| TP=2，4 层 / 64 头 | 7.53 ms/token | 4.70 ms/token | **-38%** |

单卡的收益取决于每层的 GPU 计算能不能把 launch 掩盖掉：32 头的模型每层计算很少，
省下 launch 有约 6% 的收益；64 头的模型每层计算已经够大，launch 基本被掩盖，开图持平。
**TP 下收益才是主要的**——multicuda 每个算子要唤醒两个 worker 并同步一次（实测每算子约 34 us），
每层约 14 个算子进图，4 层省下约 2.8 ms/token，与估算一致。按 40 层外推，TP 下每 token 约省 28 ms。

开图与关图的 decode 输出**逐 token 一致**（1500 步长上下文 decode 的 token 序列完全相同），
logits 的 `max|diff|` / `cos` 与关图逐位相同——图没有改变任何计算，只改变了提交方式。

### 真实权重上的实测（DeepSeek-V4.1-Flash，单卡 3090 Ti + `--moe_device numa` + `--kv_cache_dtype fp4_e2m1`）

以下为统一开关前的历史实测：同一棵代码树、同一份配置，只切换当时 V4.1 的分段图开关。
全局开关还会影响 CUDA embedding，因此此表不能视为当前全局开关的直接对照结果。

| | 关图 | 开图 |
| --- | --- | --- |
| decode（ctx 14 / 974 / 8014 / 32014） | 78 / 78 / 79 / 80 ms/token | 79 / 79 / 79 / 79 ms/token |
| prefill 首 token（同上四档） | 0.9 / 5.1 / 13.1 / 51.9 s | 0.8 / 5.1 / 13.1 / 51.3 s |
| 贪心输出与基线比对 | 一致 3/3 | 一致 3/3 |
| 大海捞针 31k / 123k | 命中 / 命中 | 命中 / 命中 |
| 捕获情况 | — | 160 段（其中 40 段路由为占位段） |

**真实模型单卡上开图基本持平**。原因是 decode 的 78 ms 里有 45–55 ms 是 CPU 上的 FP4 路由专家，
GPU 侧只有 25–30 ms，而逐算子的 launch 是异步下发的、正好被 CPU 那段掩盖掉，省下它并不缩短
关键路径。图的收益集中在 **multicuda 张量并行**那条路径上（每个算子要唤醒两个 worker 并同步
一次，约 34 us/算子，没有任何东西掩盖它）。

按层切分（流水线）下 decode 与单卡持平，同样是 CPU 专家为瓶颈，因此即使把图支持到那条路径上，
预期收益也接近于零——这也是目前不支持它的原因之一。


## 按层切分（推荐的多卡方案，已在真实权重上验证）

```bash
FASTLLM_CUDA_GRAPH=0 ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device "{'cuda:0':1,'cuda:1':1}" --moe_device numa
```

用普通的 device map 就能把层平均分到两张卡上（`SelectDeviceFromMap` 按权重划分层区间），
V4.1 的 `ForwardSegments` 每层都会 `ApplyDeviceMap`，隐藏状态在 stage 边界由执行器自动搬运。
每卡只保存自己那一半的层权重，**每卡显存约减半**；单请求延迟基本不变（两张卡是串行执行的，
不是真正的流水线——调度器没有 micro-batch，所以并发请求也不会形成 stage 级重叠）。

### 切分点必须落在 kv source 层的边界上

V4.1 的压缩 KV 是跨层共享的：`kv_source_layer_ids` 的层算出的 cache 会被之后若干层直接读。
如果 source 层和消费层落在不同卡上，执行器会在每次访问时把整段压缩 KV 搬到另一张卡，
下一步 source 层追加时再搬回来——每个 token 都要来回搬整段缓存。

真实模型（40 层）的 source 布局正好让**均分**成为合法切分点：

| source 层 | 服务的层 |
| --- | --- |
| 2 | 2–7 |
| 8 | 8–13 |
| 14 | 14–19 |
| 20 | 20–39（同时是 `candidate_source_layer_id`，候选块被 24/28/32/36 复用） |

`{'cuda:0':1,'cuda:1':1}` 得到的切分点正好是 20，stage 0 = 0–19、stage 1 = 20–39，
每个 (source, 消费者) 对都在同一张卡上，Engram 层 [1, 14] 也都在 stage 0。
**不要为了平衡显存把切分点挪到 21**（`head.weight` 1.3 GB 在 stage 1，会让 stage 1 略重）：
挪一层就会把 source 20 和它的 19 个消费层拆到两张卡上。

### 与张量并行的取舍

| | 张量并行 `--tp 2` | 按层切分 |
| --- | --- | --- |
| 每卡稠密显存 | 约一半（注意力 / 共享专家 / head 切开，其余复制） | 约一半（整层归属一张卡） |
| 单请求 prefill | 明显变快（注意力与稠密 GEMM 并行） | 基本不变 |
| 单请求 decode | 变慢（每个算子多一次两卡 worker 调度，实测约 34 us/算子） | 基本不变 |
| 通信 | 每层 2 次 all-reduce（NVLink，量小） | 每个 stage 边界 1 次隐藏状态搬运 |
| 约束 | `num_attention_heads / tp` 必须是 32 的倍数并对齐 o_groups | 切分点必须落在 kv source 边界 |

建议：长上下文 / prefill 为主用 `--tp 2`；要显存（给 KV cache 或专家缓存腾地方）、
或者以 decode 吞吐为主，用按层切分。两者目前是二选一。

## prefill 的多卡专家流（`FT_MOE_ASSIST_DEVICES`）

`--moe_device numa` 下，一个 prefill 分块要把**全部**路由专家的权重过一遍，专家权重是从主机内存
流式送到 GPU 的，不需要张量并行分片。所以瓶颈是「主机内存 -> GPU」这条链路的带宽，而不是算力：
多一张卡就多一条链路。但 `GetNumasMoeCudaAssistDevices()` 原本只返回承载稠密层的那张卡，
模型放得下单卡的机器上第二张卡在 prefill 期间完全闲置。

```bash
FT_MOE_ASSIST_DEVICES=0,1 ftllm server ... --device cuda --moe_device numa
```

把额外的 CUDA 设备加进专家流。默认为空，行为不变。每张卡拿到一份输入激活的副本、一组不相交的
专家，各自算出一份 partial，最后在 root 卡（产出这一层激活的那张）上相加。

### 输入搬运与归约重叠

只把第二张卡加进来是不够的：每层会多出两段**只在主线程上串行**的搬运，正好把算子级省下来的时间
还回去。

输入搬运与 partial 归约始终使用重叠路径，无需设置环境变量。
没有 peer 通路时自动使用 pinned host 中转。下面两项调度策略仍默认关闭。

| 变量 | 作用 |
| --- | --- |
| `FT_MOE_ASSIST_BALANCE=1` | 按各卡实测的「每专家毫秒」分配 GPU 专家，而不是按 route 数均分 |
| `FT_EXPERT_LIMIT_AUTO=1` | 用真实层反馈出的 CPU / GPU 速度算 expertLimit，取代单专家合成 benchmark |

重叠路径包括两处：

- **输入 staging**：原来是「`waitForCpuInput()` 等输入的 D2H 落到 pinned host」+「一次阻塞的 H2D
  把整块激活推上第二张卡」，两步都压在主线程上，既不与 root 卡的专家计算重叠、也不与 CPU 专家重叠。
  现在主线程只准备副本缓冲，搬运挪进该卡的 worker 线程、排在它自己的 per-thread stream 上：
  优先 `cudaMemcpyPeerAsync` 直接从产出激活的那张卡拉（这台机器上两张 3090 Ti 之间是 NVLink），
  拉不动再退回「等 `inputCopyStream` 上的 D2H 完成事件 + pinned H2D」。后续 compute 走同一条 stream，
  顺序天然成立，主线程不必等待这次搬运。
- **partial 归约**：原来是所有 worker join 之后才开始跨卡搬运，每搬一块 `AddTo` 一次、再
  `cudaStreamSynchronize` 一次。现在跨卡搬运同样放进 worker 线程，落到每卡独立的 root 侧缓冲，
  与 root 卡剩余的专家、以及主线程的 CPU 专家重叠；主线程只在 root stream 上等事件、做 `AddTo`，
  中间的逐块同步全部去掉，末尾统一同步一次再释放 partial。

事件、归约缓冲、pinned 中转缓冲都按设备缓存在每层的 MoE manager 上，跨层复用。
设置 `FASTLLM_PROFILE_NUMAS_MOE=1 FASTLLM_PROFILE_DETAIL=1` 可统一查看每层的
CPU/GPU 专家划分、各卡 route 数，以及 stage / limit / prep / cpu / join / reduce 耗时。

### expertLimit 的选择

`expertLimit` 是「一个专家至少要有多少 route 才值得送上 GPU」的阈值：低于它的专家留在 CPU 上算，
和 GPU 并行，理想情况下两边同时结束。默认由 `MoeExpertSpeedEstimator` 在第一次 prefill 时跑一个
合成 benchmark 推出来——**每轮只跑一个专家、每轮 sync**。这样量到的 GPU 每专家耗时里含着
无法摊掉的固定开销（workspace 准备、stream 创建、启动延迟），也拿不到真实层里跨专家的
「copy(i+1) 与 compute(i) 重叠」，会系统性高估 GPU，把过多 route 留在 CPU 上。

`FT_EXPERT_LIMIT_AUTO=1` 改成用真实层反馈的两个系数直接算 makespan 最优点：

- 每张卡的「每专家毫秒」= worker 线程墙钟 ÷ 分到的专家数（EMA）。因为是两张卡并发跑时量的，
  跨专家流水重叠、两卡争抢主存带宽的影响都已经算进去。
- CPU 的「每 route 毫秒」= CPU 专家段墙钟 ÷ 落在 CPU 上的 route 数（EMA）。

然后枚举阈值 t，用与实际分配一致的贪心把 GPU 专家摊到各卡上，取 `max(cpuMs, gpuMs)` 最小的 t。
样本不足（前几层）时先将 route 最少的两个专家分给 CPU，其余交给 GPU，以收集两侧耗时。
`FT_EXPERT_LIMIT=<n>` 的显式覆盖优先级最高，
两种自动估计都不会执行。

动态 prefill 分配可能改变专家归约和舍入路径。数值比较应使用相同输入及生成历史，结合 logits
误差与任务结果判断；不要仅为逐字节复现而关闭动态分配。生成历史分歧之后的 logits 不能直接比较。

### 实测（2 x RTX 3090 Ti，NVLink，6 层真实 MoE 尺寸的模型）

以下为默认开启前、逐项启用各优化的历史测试结果。

模型：hidden 5120 / moe_intermediate_size 2304 / top-6 / 64 个路由专家 + 1 个共享专家，
路由专家 NVFP4 block-32（每专家约 18.8 MB），16384 token prefill、4096 分块（共 4 个 chunk x 6 层）。
「ms/层」是后两个 chunk 共 12 次调用的均值（前两层要首触 CPU scratch，不计入），3 次运行汇总；
e2e 取 3 次的中位数。

| 配置 | stage | reduce | cpu | join | ms/层 | e2e |
| --- | --- | --- | --- | --- | --- | --- |
| 单卡（现状） | 0.01 | 0.78 | 44.88 | 8.64 | **55.00** | 5.76 s |
| + `FT_MOE_ASSIST_DEVICES=0,1` | 1.75 | 5.44 | 35.09 | 6.89 | **49.95** | 5.42 s |
| + 搬运重叠（现为固定路径） | 0.02 | 1.60 | 36.14 | 1.95 | **40.49** | 5.36 s |
| + `FT_EXPERT_LIMIT_AUTO=1` | 0.01 | 0.33 | 10.58 | 25.72 | **37.18** | 3.06 s |

三步合计 **55.00 -> 37.18 ms/层（1.48x）**，端到端 **5.76 -> 3.06 s（1.88x）**。

分开看每一步：

- **只加第二张卡**，算子级确实变快（55.00 -> 49.95），但每层多出 1.75 ms 的输入 staging
  与 5.44 ms 的归约，全部串在主线程上，把省下来的吃掉大半，端到端只从 5.76 降到 5.42 s。
- **加上重叠**后这两项变成 0.02 ms 与 1.60 ms（剩下的 1.60 ms 是 CPU partial 那 42 MB 的
  pinned H2D——它只能在 CPU 专家算完之后才发得出去，属于固有开销）。把 expertLimit
  固定成 115（去掉合成 benchmark 这个变量）单独对比这一项：42.66 -> 40.55 ms/层，
  主线程上的串行搬运 7.49 -> 1.22 ms/层。
- **最大的一笔其实是 expertLimit 的合成 benchmark**：`MoeExpertSpeedEstimator` 在第一次
  prefill 里要 2.2-2.6 s（而且后面某层 maxTaskSize 变大时会重建一次，一次 prefill 付两遍），
  相比之下稳态每层才 40 ms。`FT_EXPERT_LIMIT_AUTO=1` 不跑它，用探针 + 实测反馈，
  在第二个 chunk 内收敛到 `expertLimit=1`（predict_cpu=0、predict_gpu=29 ms），
  与手工试出来的最优值一致；手工 `FT_EXPERT_LIMIT=1` 是 39.66 ms/层 / 2.88 s，
  自动档 37.18 ms/层 / 3.06 s（e2e 多出的 0.2 s 是探针那两个 CPU 专家触发的一次性 scratch 首触）。

`FT_MOE_ASSIST_BALANCE=1` 在这台机器上是中性的（37.18 vs 38.31 ms/层）：两张卡型号相同、
挂在同一个 root complex 上，实测每专家耗时几乎一致，没有可纠正的不对称。它是给异构
或链路不对称的机器准备的。

## KV 缓存存储精度（`--kv_cache_dtype`）

长期 KV 缓存只有两部分：**压缩 KV**（`kv_source_layer_ids` 各层，每 `compress_ratio` 个 token 一行，
`head_dim` = 512）与 **indexer key**（同样的层，`index_head_dim` = 128）。滑窗缓存长度固定为
`sliding_window`，不随上下文增长，不计入每 token 开销。

写进缓存之前，这两者都已经过与官方一致的伪量化（`DeepSeekV41RotaryQuant` 的 `quantMode`）：

| 张量 | 伪量化网格 | 分组 | scale 编码 |
| --- | --- | --- | --- |
| 压缩 KV | FP4 E2M1 | 每 16 个通道 | E4M3 |
| indexer key | FP4 E2M1 | 每 32 个通道 | UE8M0（2 的幂） |
| 滑窗 KV | FP8 E4M3 | 每 32 个通道 | UE8M0 |

也就是说值本来就落在 FP4 / FP8 网格上，**按 FP4 存储是无损的**。三档存储的行布局与开销：

| `--kv_cache_dtype` | 压缩 KV 行 | indexer key 行 | 每 token | 相对 BF16 | 精度 |
| --- | --- | --- | --- | --- | --- |
| 默认（BF16） | 1024 B | 256 B | **3200 B** | 1.00x | 基准 |
| `fp8_e4m3` | 528 B（512 E4M3 + 16 UE8M0） | 132 B | **1650 B** | 0.52x | 压缩 KV 有不超过 2^-4 的相对舍入 |
| `fp4_e2m1` | 288 B（256 打包 E2M1 + 32 E4M3） | 68 B（64 打包 E2M1 + 4 UE8M0） | **890 B** | 0.28x | **逐 bit 无损** |

890 B/token 与官方实现一致。每 token 2.5 行压缩 KV（层 2/8/14 的 `compress_ratio` 是 2，层 20 是 1）：
`2.5 x 288 + 2.5 x 68 = 890`。

- **FP4 无损的原因**：`DeepSeekV41QuantizeKV` 的块 scale 推导（`DeepSeekV41BlockScale`）与伪量化
  (`DeepSeekV41FakeQuantRow`) 是同一份代码，分组大小与 scale 编码也完全对齐，所以对已经伪量化过的行
  是幂等的；而 `q * scale`（q 至多 3 个有效位、scale 是 E4M3 或 2 的幂）在 BF16 上也是精确的。
  实测：开启伪量化时 FP4 存储与 BF16 存储的 logits **逐 bit 相同**（`test/basic/deepseek_v41_kv_cache_dtype.py`）。
- **FP8 为什么有损**：压缩 KV 的网格是「FP4 值 x E4M3 scale」，而 FP8 存储用的是 2 的幂块 scale，
  两者的网格对不上，会引入一次真实的舍入。它的用途是在不支持 FP4 打包读取的场合省一半显存。
- 滑窗 KV 在 `fp8_e4m3` 与 `fp4_e2m1` 两档下**都按 FP8 存储**（它本来就在 FP8 网格上，改 FP4 会真的损失精度），
  三种行布局可以在同一次前向里混用，读路径按行宽自动识别。
- 缓存行统一放在 `INT8` 的 `Data` 里，形状 `[b, rows, rowBytes]`；FP4 打包为一个字节两个 code，
  低 4 位是偶数下标。CUDA 侧在把候选装进共享内存时就地解成 BF16 片段，再进 `mma.sync`，
  `FASTLLM_DSV41_LEGACY_ATTN` / `FASTLLM_DSV41_LEGACY_INDEXER` 的标量回退路径同样支持。

### 怎么选

- **长上下文 / 高并发**：用 `fp4_e2m1`。同样显存能装的上下文约为 BF16 的 3.6 倍，且数值与 BF16 完全一致，
  没有任何精度代价；长上下文下稀疏注意力每步读取的字节数同比下降，decode 也有正收益。
- **默认（BF16）**：短上下文、显存不紧张时省掉打包/解包的一点开销。
- `fp8_e4m3`：介于两者之间，但既比 FP4 大又比 FP4 差，现在没有明显适用场景，保留是为了兼容既有部署。

### 实测（迷你模型，KV 几何与真实 config 相同：`kv_source_layer_ids` [2, 8, 14, 20]、ratio 2/2/2/1，单卡 3090 Ti）

`test/basic/deepseek_v41_kv_cache_dtype.py` 用同一段提示词跑三档，实测的长期 KV 占用与逐 bit 比较：

| | 每 token | 32k 上下文的实际缓存 | 1 GiB 能装的上下文 | 与 BF16 的 logits |
| --- | --- | --- | --- | --- |
| BF16 | 3200 B | 102.6 MB | 33.5 万 token | 基准 |
| `fp8_e4m3` | 1650 B | 52.9 MB | 65.1 万 token | 有差（压缩 KV 有舍入） |
| `fp4_e2m1` | 890 B | 28.5 MB | **120.6 万 token** | **逐 bit 相同** |

速度（同一台机器，prefill / decode）：

| 上下文 | BF16 | `fp4_e2m1` |
| --- | --- | --- |
| 1000 | 0.32 s / 9.35 ms per token | 0.32 s / 9.64 ms |
| 8000 | 0.56 s / 10.12 ms | 0.64 s / 9.41 ms |
| 32000 | 1.53 s / 12.32 ms | 2.03 s / 10.09 ms |

上下文越长，decode 越占便宜（32k 时快 18%），因为稀疏注意力与 indexer 每步读取的字节数减半以上。
prefill 反过来会慢一些（32k 时慢约 30%）：迷你模型里 MoE 很小，FP4 的逐元素解包（相对 BF16 的
`float4` 整块搬运）成了可见开销；真实模型的 prefill 由 MoE 主导，这部分占比会小得多。
把解包改成按 4 字节一组的向量化 nibble 展开可以再优化，尚未做。

`FASTLLM_DSV41_KV_STATS=1` 会在每次前向后打印实际占用的长期 KV 字节数与每 token 均值，便于核对。

## 多请求与前缀缓存

### 批量 decode

`DeepSeekV41Model::ForwardSegments` 把若干"片段"（请求状态 + 起始位置 + token 数）拼成一个 token 流：
embedding、Hyper-Connections、各 Linear、路由、MoE 与 Engram 查表按整批执行，RoPE、压缩 KV 追加、
indexer top-k、稀疏注意力与滑窗写入按片段分别执行。通用调度器的多请求 `ForwardBatch` 直接走这条路径，
单请求 `Forward` 是它的单片段特例；含长 prefill 片段的混合批次退回为逐个前向。

每个请求的缓存（`DeepSeekV41RequestState`）与 `ResponseContext` 绑定，请求结束或 abort 时释放。
在迷你模型上，8 并发 decode 与逐个请求相比，token 序列的差异仅来自 GEMM 按 batch 选核带来的
BF16 舍入（同一请求换不同的批次伙伴 / 批内位置，logits 逐 bit 一致）；单卡 3090Ti 上迷你模型的
decode 吞吐从 209 tok/s（1 并发）提高到 573 tok/s（8 并发）。

### 前缀缓存

OpenAI 接口在模型实例内保存工具轮次的原始输出（含思考和 DSML，最多 128 轮、8 MiB，支持流式）。
续轮的调用 ID、顺序、函数名、参数值和正文匹配时恢复原文，避免 DSML 拼写、JSON 格式等变化破坏 token 前缀。
匹配忽略 JSON 空格、对象键顺序和纯空白正文；客户端可省略思考内容，但修改思考、切换思考模式或
模板要求丢弃思考时不复用。不接受客户端原始模板覆盖；未命中仍按普通模板编码和 prefill。

启动时加 `--cache_history true`。请求结束时把每层 `windowKV`、`compressedKV`、`indexK`、`rawTail`
与 Engram 历史快照到 CPU 内存（LRU，默认保留 8 条），新请求按最长公共前缀查找并恢复，只对新增 token
做 prefill。多轮对话中只要客户端原样回传上一轮的回复，通常就是精确命中。

DSML 块前的两个模板分隔换行不作为 assistant 正文返回，避免续轮重复添加。
流式解析暂存这两个换行；普通正文末尾的换行在流结束时照常输出。

恢复长度受模型结构约束：滑窗缓存是只保留最后 `window_size` 个位置的环形缓冲，因此只能恢复到记录
长度 T 或 T-1（记录不超过 `window_size` 时可以任意截断）；ratio-2 压缩层凑不满一组的原始尾块只在
精确命中时可用，其它情况要求恢复长度为偶数。不满足约束的候选会被跳过（退回完整 prefill）。

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSV41_DISABLE_PREFIX_CACHE` | 关闭前缀缓存 |
| `FASTLLM_DSV41_PREFIX_CACHE_DEBUG` | 打印命中 / 记录日志 |
| `FASTLLM_DSV41_PREFIX_CACHE_MIN_TOKENS` | 最短命中长度（默认 16） |
| `FASTLLM_DSV41_PREFIX_CACHE_MAX_RECORDS` | 最多保留的记录数（默认 8，也可用 `FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS`） |

迷你模型上精确命中与 T-1 截断命中后的贪心输出与不中断的原请求逐 bit 一致；800 token 提示词的
首 token 延迟从 39–54 ms 降到 6 ms。

## DSpark 投机解码

`config.json` 的 `text_config` 里 `dspark_block_size > 0` 且 checkpoint 带 `mtp.*` 权重时可以开启。
启动加 `--speculative_algorithm dspark --dspark 5`。`--dspark N` 指每轮最多校验 N 个候选，
运行时草稿长度取 `max(dspark_block_size, N)`：小于训练块时只校验前缀，
大于训练块时扩展整个草稿前向，包含双向注意力和 Markov 链。
本 checkpoint 训练块为 5，已验证 `--dspark 7`；全部接受时一轮最多输出 8 个 token。
更长草稿的速度取决于接受率与校验成本，训练配置本身不需要修改。

```bash
ftllm server /path/to/DeepSeek-V4.1-Flash \
  --device cuda --moe_device numa --dtype float16 \
  --speculative_algorithm dspark --dspark 5
```

`ftllm` 的自动配置（launcher）在 `enable_speculative_decoding` 时会识别 V4.1 的内置 DSpark，
按 checkpoint 的训练 block size 填 `--draft_tokens`。不加 `--dspark` / `--draft_tokens` 时
不加载 `mtp.*`。本次量化 checkpoint 的草稿原始张量（含 scale）约 7.39 GiB；
实际占用还受加载 dtype、权重转换和运行缓冲影响。

### 结构

`mtp.0/1/2` 是三个与主干同构的草稿层：`compress_ratios` 在这三层是 0（纯滑窗注意力），
MoE 是 `dspark_n_routed_experts` = 128 专家 top-3，embedding 与 lm_head 与主干共享。
另外 `mtp.0` 有 `main_proj` / `main_norm`，`mtp.2` 有 `norm`、`markov_head`（`embed` + `head`，
秩 256）与 `confidence_head`（输入 `dim + 256`）。

草稿层的滑窗 KV 不来自草稿 token 自身，而来自目标模型：`dspark_target_layer_ids`（[37, 38, 39]）
各层 attention 的输入（对 4 份 Hyper-Connections 取均值）拼成 `3 * dim` 后经 `main_proj` / `main_norm`
得到 `main_x`，每个草稿层再用自己的 `wkv` / `kv_norm` 把它写进一行滑窗 KV。也就是说每个已提交
位置在草稿侧只有一行 KV，草稿模型不需要重跑主干。

一次 proposal：把 `block_size` 个位置的输入 token 置为 `dspark_noise_token_id`（第 0 个位置放锚点
token，即目标模型刚产出、还没进 KV 缓存的那个 token），一次前向产出 `block_size` 组 logits；
再用 markov head 逐位置做 bigram 修正，得到 `block_size` 个候选 token；
`confidence_head` 对每个位置给出一个 sigmoid 后的置信度。

贪心请求取 argmax；采样请求在 GPU 保存完整草稿分布 q，通过共用的 MTP 采样与拒绝采样内核校验，
不再限制草稿支持集为 64 项。Markov 修正使用上一步实际抽到的 token；草稿与目标均使用请求的
`temperature`、`top_k`、`top_p`，无工具掩码时与普通 CUDA 采样一致。
工具名、参数名约束按位置更新：在允许集合内取 top-k 并重新归一化后取 top-p，草稿、校验和 bonus
使用各自前缀；拒绝和待发队列不会错误推进约束状态。强制候选测试使用 one-hot q，无需额外开关。

### 校验与回滚

候选与锚点拼成一个 `1 + N` 长度的片段一次喂给目标模型（`ForwardSegments` 天然支持一次多 token），
贪心请求逐位置比较 argmax，第一个不匹配处截断。采样请求复用 `qwen3_5.cpp` 的链式拒绝采样：
实际草稿概率为 q，目标条件概率为 p，以 `min(1, p(token)/q(token))` 接受候选；首次拒绝时从
归一化的 `max(p-q, 0)` 抽取替代 token，全部接受时从额外目标行抽取 bonus token。
接受 n 个候选时这一轮提交 n + 1 个 token、产出 n + 1 个输出
token：第一个立刻返回，其余进入请求的待发队列，调度器之后每轮直接出队，不再前向。

校验前向按完整 block 更新缓存，接受长度确定后要把多算的部分退回：

- 滑窗环形缓冲：写入**延后**到接受长度确定之后。片段内的位置在稀疏注意力里一律从 `chunkKV` 读取，
  推迟写入不改变本次前向的任何结果，因此也不需要为回滚保存被覆盖的旧行；
- 压缩 KV 与 indexer key：按接受后的长度重新截断行数，并用保存下来的 compressor 原始输入流
  （旧 `rawTail` + 本次新行）重建 `rawTail`；
- Engram 历史与各层 `totalLen`：截断到接受后的长度。

拒绝采样在目标条件概率相同的前提下保持目标输出分布，不要求草稿分布等于目标分布。
这不代表开启与关闭 DSpark 的原始 logits 逐 bit 一样，也不保证相同随机种子给出相同序列。
一次多 token 与逐 token 前向会选择不同的 GEMM / MoE 路径，浮点舍入可能改变 logits，
也可能翻转接近并列的 argmax；分布测试与浮点前向误差需要分别验证。

### 限制

- 支持简单贪心及 CUDA 上的 temperature / top-k / top-p 采样，也支持默认工具名、参数名约束。
  重复惩罚、工具内容采样、没有前缀状态的独立 token 白名单、`output_logits`、正的
  `output_token_least`、非有限采样参数会退回普通解码；CPU 采样也走普通路径。
  与普通解码一样，`do_sample=true`、正温度且 `top_k<=1` 时将 top-k 规范化为 5；
- 只在**单请求**前向里产生候选。批量 decode 的那一轮不投机，但仍然采集 main hidden，
  让草稿滑窗跟上目标缓存；已经校验通过的 token 在批量路径里也能正常出队；
- 图文请求不投机；
- 前缀缓存命中恢复出来的前缀没有草稿侧的滑窗（`main_x` 无法从目标缓存反推），
  草稿注意力只看得到恢复之后新增的位置，接受率会在最初的 `window_size` 个 token 内偏低；
- 校验提交在首个 EOS / stop token 处截断，避免结束时缓存超出真实输出。
  请求若因长度上限在待发队列还没取完时结束，这一轮多算的 token 仍可能让缓存长度超过
  `allTokens`，该请求的前缀缓存记录会被跳过；
- 支持与主干张量并行组合（`--tp 2`），草稿层仍在第一张卡运行；目标校验支持分段 CUDA Graph，草稿层仍逐算子执行。

### 实测

单卡 3090 Ti、迷你模型（6 层 + 3 个草稿层、2 专家、随机权重）：

| 场景 | 结果 |
| --- | --- |
| 开 / 关 DSpark 的贪心输出 | 一致（30 个 token 中 3 处 BF16 并列翻转，重新锚定后完全一致） |
| 注入候选构造接受 0 / 1 / 2 / 3 / 5 个 | 接受长度与构造完全一致，回滚后的续写与基准一致 |
| 随机权重下的接受率 | 0%（草稿模型是随机初始化的，只验证正确性，不代表真实接受率） |
| 吞吐 | 190 → 100 tok/s（草稿层占迷你模型的一半，且接受率为 0，属最坏情况） |

真实 checkpoint 的 `mtp.*`（3 个 stage × 128 专家，共 2401 个张量）已核对：加载器需要的 1221 个
张量全部存在，量化格式与主干一致（稠密 FP8 32x32 + UE8M0 scale、路由专家 FP4 沿 K 每 32 个一组），
`markov_head.embed/head` 为 BF16 `[129280, 256]`、`confidence_head.proj` 为 BF16 `[1, 5376]`。
真实权重下已验证双卡 NUMA 混合缓存与随机采样；吞吐和接受率随任务及采样参数变化。

### 接受率与分段计时

默认开启总体及逐位置接受率统计，每累计 32 轮校验打印一次（仍跳过前 3 轮预热），等同于
`FASTLLM_DSPARK_STATS=1 FASTLLM_DSPARK_STATS_EVERY=32`：

```text
[DeepSeek-V4.1 DSpark] accept_rate=70.00% (350/500), pos_accept_rate=[90.00%, 80.00%, 70.00%, 60.00%, 50.00%].
```

`accept_rate` 为累计接受的候选数除以实际送检的候选数，不包含被置信度阈值提前筛掉的候选。

服务还会在每次 prefill 完成时打印 `[Prompt]` 日志，包含实际计算的 token 数、耗时和 tokens/s；
分块 prefill 额外打印每块的进度和速度。历史缓存命中的 token 不计入 prefill 吞吐，prefill 耗时也不计入后续的 `[Decode]` 速度。

设置 `FASTLLM_DSPARK_STATS=0` 可关闭统计；需要排查耗时时，设为 `2` 打印逐轮和分段统计：

```
[DSpark 进行中] 前向 640 次（校验 612 + 普通 28），出队 918，共产出 1558 个 token；每次前向 2.434 个 token（已跳过 3 轮预热）
[DSpark 进行中] 候选 3060 接受 918（接受率 30.0%），平均每轮接受 1.50 个；草稿产出 3060 个候选，置信度筛掉 0.0%
[DSpark 进行中] 接受长度分布 0:180(29%) 1:150(25%) 2:120(20%) 3:80(13%) 4:50(8%) 5:32(5%)
[DSpark 进行中] 送检候选数分布 0:0(0%) 1:0(0%) 2:0(0%) 3:0(0%) 4:0(0%) 5:612(100%)
[DSpark 进行中] 每次前向：校验 96.100 ms（612 次）/ 普通 82.400 ms（28 次），差 13.700 ms；回滚 0.180 ms
[DSpark 进行中] 每次草稿 11.300 ms = main 0.900 + 三层 7.100 + head 1.500 + markov 1.700 + confidence 0.100（640 次）
[DSpark 进行中] 其中 markov 1.700 ms = 投影 0.600 + argmax/同步 1.100（每次草稿 5 个位置，逐位置串行）
[DSpark 进行中] 每个输出 token 合计 47.300 ms（目标前向 42.100 + 草稿 5.100 + 回滚 0.100）
```

各项的含义与用法：

- **每次前向 N 个 token**：这就是加速比的上限。等于 1 说明一个候选都没接受，DSpark 只在做无用功。
- **接受长度分布**：`0:` 那一档占比高说明草稿质量差或者目标层输入取错；分布均匀说明草稿是有效的。
- **送检候选数分布**：置信度阈值把每轮的候选截断到几个。如果集中在 `0:` / `1:`，说明
  `--speculative_dspark_confidence_threshold`（默认 0.5）太保守，把大部分候选筛掉了，
  这时**既付了草稿的代价又没拿到收益**，先把阈值调到 0 再看。
- **校验 vs 普通**：路由专家跑在 CPU / NUMA 上时代价与"选中专家的权重字节数"成正比而不是与
  token 数成正比，所以校验 N 个候选的前向应当只比单 token 前向贵一点（贵的是这 N 个 token
  选中专家的**并集**，不是 N 倍）。这个差值要靠 `FASTLLM_DSPARK_PROBE_EVERY=8` 拿到同等缓存
  状态下的对照基线；差值接近 N 倍说明专家并集几乎没有重叠，块开大了不划算。
- **草稿分段**：三层是草稿骨干（128 专家 top-3，跟着 `--moe_device` 走）。
  草稿总耗时接近甚至超过"校验与普通前向之差 × 平均接受长度"时，收益就被草稿吃掉了。

#### markov head 的融合 kernel

markov 链是串行的：每一步都要拿上一步的 token 去查嵌入、算 `[vocab, rank]` 的偏置、再取 argmax。
用通用算子拼出来的话每一步都得把 token 取回主机（查嵌入与切片都需要主机侧的下标），
于是每步一次完整的设备同步。迷你模型上实测这 5 步要 2.78 ms，比整个目标模型的一次前向（3.24 ms）
还贵；其中光 `[1, rank] x [vocab, rank]` 这个 GEMV 就占 0.39 ms／步——按带宽只该 20 us，
通用 Linear 在"一行输入、又高又窄的权重"这种形状上效率很低。

CUDA 上因此走一个把整条链留在设备上的融合 kernel（token 一直在显存里，主机只在最后取回一次），
输出与通用实现逐 bit 一致，可以用 `FASTLLM_DSPARK_DISABLE_FUSED_MARKOV=1` 对拍。
迷你模型（6 层 + 3 草稿层、`--dtype float16`、单卡 3090 Ti）上的效果：

| | 通用算子 | 融合 kernel |
| --- | --- | --- |
| markov | 2.779 ms | 0.746 ms |
| 草稿阶段合计 | 4.276 ms | 2.229 ms |
| 每个输出 token | 8.066 ms | 5.976 ms |
| decode 吞吐 | 125 tok/s | 168 tok/s |

其它设备、或者 dtype / 张量布局不满足前提（权重被量化、多卡分片等）时自动退回通用实现。
为此 `markov_head.embed/head` 两张表都按 checkpoint 原样保持 BF16（各 66 MB，不做重量化）。

注意力等 GPU 上的部分是异步下发的，要拿到真实耗时需要同时设 `FASTLLM_CUDA_SYNC=1`，
否则那些项只反映 kernel launch 的时间；路由专家在 cpu / numa 上时本来就是同步的，不受影响。

### 调试环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSPARK_TOKENS` | 每轮校验的候选数（由 `--dspark` / `--draft_tokens` 设置，不要直接设） |
| `FASTLLM_DSPARK_CONFIDENCE_THRESHOLD` | 置信度低于该值的候选之后不再校验；0 表示总是用满 block |
| `FASTLLM_DSPARK_STATS` | 默认 `1`，打印总体及逐位置接受率；`0` 关闭，`2` 打印详细分段耗时与逐轮记录。见上文"接受率与分段计时" |
| `FASTLLM_DSPARK_STATS_EVERY` | 每累计 N 轮校验打印一次（默认 32，0 表示只在退出时打印） |
| `FASTLLM_DSPARK_STATS_WARMUP` | 统计前跳过的轮数（默认 3）。第一次 decode 含 CUDA context / 显存池 / 权重量化缓存的一次性开销，会把均值拉偏 |
| `FASTLLM_DSPARK_DISABLE_FUSED_MARKOV` | 关掉 markov head 的融合 kernel，退回通用算子（对拍 / 排查用；两条路径输出逐 bit 一致） |
| `FASTLLM_DSPARK_PROBE_EVERY` | 每 N 轮故意不带候选走一次普通单 token 前向，给"校验 N 个候选"提供同等缓存状态下的对照基线。默认 0（关闭），只在诊断时打开 |
| `FASTLLM_DSPARK_STATS_FILE` | 每轮校验追加一行 `轮次 候选数 接受数`（测试用） |
| `FASTLLM_DSPARK_FORCE_DRAFTS` | 用文件里的候选替换模型的候选，构造指定的接受长度（测试用） |
| `FASTLLM_DSV41_DEBUG_STATE` | 每次前向后导出各层缓存长度（对比回滚是否正确） |

## 图像输入

`config.json` 顶层的 `vision_n_layers > 0` 时加载视觉编码器：32 层 ViT（1024 维、16 头、patch 14、2D RoPE、SwiGLU MLP）
+ aligner（3x3 下采样后 `w1 -> GELU -> w2` 投影到 5120 维）+ 三个学习到的分隔符嵌入。权重保持 float16（不参与低比特量化），
以 float32 激活在执行器所在设备上计算，长序列注意力按 1024 个 query 分块。

处理流程（与官方 `image_processor.py` / `model.py` 一致）：

1. Python 侧（`deepseek_v41_multimodal.py`）：`encode_messages(..., return_multi_modal_data=True)` 渲染带
   `<｜deepseek_image｜>` 占位符的 prompt；每张图按动态分辨率规则缩放 / 灰色补边（`vision_min_pixels` 295936、
   `vision_max_n_token` 1024），切成 `n_vit_h x n_vit_w` 个 patch，占位符展开为
   `[image_start] + ([IMAGE] * n_llm_w + [image_newline]) * n_llm_h + [image_end]`（每个位置都是 `image_token_id`），
   patch 与每张图的 `(起始位置, n_vit_h, n_vit_w)` 作为 payload 传给 C++；
2. C++ 侧：请求创建时把多模态输入记到请求状态（`OnResponseContextCreated` / `ForwardMultimodal`），
   第一个 prefill 块对每张图做 ViT + aligner，把结果与分隔符嵌入组成 span 嵌入缓存起来；之后每个 prefill 块
   （无论由调度器还是模型自己按 `chunked_prefill_size` 切分，span 可以跨块）把与 span 重叠的位置换成图像嵌入，
   图像 span 内的 token 路由使用 `gate.bias_vl`、Engram 对其不做查表也不参与 n-gram；decode 步骤与纯文本相同。

用法：与其它多模态模型相同，OpenAI 接口的 `image_url` 支持 http(s)、`data:image/...;base64` 与 `file://`：

```bash
curl http://127.0.0.1:8080/v1/chat/completions -H "Content-Type: application/json" -d '{
  "model": "v41", "messages": [{"role": "user", "content": [
    {"type": "text", "text": "描述这张图片"},
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,...."}}]}]}'
```

限制：

- 只支持图像，不支持视频；一张图最多 1024 个 token（约 9216 个 ViT patch），多张图按出现顺序对应；
- 带图请求不与其它请求合并 prefill；图像 span 必须落在 prompt 内（Python 侧展开时保证）；
- 前缀缓存 / 多轮复用尚未实现，每个请求都会重新编码图像；
- 21 环境下服务模式若加载不到 HF tokenizer，会退回 fastllm 原生 tokenizer 编码 prompt（两者对占位符的 id 相同）。

## 数值验证

`--fast_prefill` 的 CPU 回归复用现有微型模型，覆盖窗口边界、分块 / 混合批次、前缀缓存恢复、
FP8 / FP4 KV，以及 DSpark 的特征采集和完整 verify：

```bash
cmake --build build --target deepseekV41BoundedReplayRegression -j
python test/basic/test_deepseek_v41_bounded_replay.py --binary build/deepseekV41BoundedReplayRegression
```

`test/basic/deepseek_v41_reference.py` 用官方 `inference/model.py` 的模块（把 tilelang kernel 换成纯 torch 实现）
构造随机初始化的迷你 V4.1 模型，与 FastLLM 逐步比较 logits，并可逐层比较中间张量：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_reference.py \
  --work-dir /tmp/v41-tiny --tokenizer-dir /path/with/tokenizer.json \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference \
  --experts 2 --activated 2 --index-topk 128 --candidate-topk-blocks 64 --no-fake-quant --regenerate
```

- `--experts 2 --activated 2` 与 `--index-topk 128` 消除随机权重下专家选择 / top-k 选择对微小数值差的敏感性，
  这时两侧 13 个贪心 token 完全一致、每步 logits 余弦相似度 >= 0.9998；
- 去掉 `--no-fake-quant` 可以验证与官方一致的伪量化路径（量化边界翻转会带来可见但有限的差异）；
- `--quant-format real` 会按真实 checkpoint 的格式（FP8 32x32、FP4 专家）保存迷你模型，用于验证加载器；
- `--dump-dir` 逐层比较隐藏状态、注意力输出、压缩 KV、indexer 分数与 top-k，并用 numpy 复算候选块 / top-k 选择。

图文输入用 `test/basic/deepseek_v41_vision_reference.py`（迷你文本模型 + 小规模视觉塔，图片由程序合成）：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_vision_reference.py \
  --work-dir /tmp/v41-tiny-vision --tokenizer-dir /path/with/tokenizer.json \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference --no-fake-quant --regenerate --dump-dir /tmp/v41-vision-dump
```

- 先比较 Python 预处理与官方 `image_processor.prepare_vl_inputs` 的 token 序列与 patch（要求完全一致），再逐步比较 logits；
  `--dump-dir` 时另外比较 ViT 各阶段、aligner 输出与合并后的输入嵌入（`--ref-vision-fp32` 让参考侧视觉塔用 float32，
  排除官方 bf16 路径的舍入噪声，此时 ViT / aligner 输出 cos = 1.000000）；
- `--experts 8 --bias-vl-boost 5` 给 `gate.bias_vl` 的最后两个专家加偏置，检查图像 token 的路由确实使用 `bias_vl`
  （随机权重下 8 专家的贪心 token 会因路由并列翻转而不同，属已知敏感性，正确性以 2 专家配置为准）；
DSpark 用 `test/basic/deepseek_v41_dspark.py`（迷你文本模型 + 3 个随机初始化的草稿层）：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_dspark.py \
  --work-dir /tmp/v41-tiny-dspark --tokenizer-dir /path/with/tokenizer.json \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference --regenerate
```

- 先跑一遍不开 DSpark 的基准（带 logits），再跑一遍开 DSpark 的，逐 token 比较并统计接受率；
- 用 `FASTLLM_DSPARK_FORCE_DRAFTS` 注入候选构造"接受 0 个 / 接受一部分 / 全部接受 / 混合"四种场景，
  校验每轮的实际接受长度与构造的一致、回滚后的续写与基准一致；
- 迷你模型的 logits 是 BF16（分辨率约 1/32），几乎并列的 argmax 会因 GEMM 选核不同而翻转，
  脚本把"差距在 3 个 BF16 ulp 以内"的分歧判为并列，以 DSpark 的输出为新前缀重新跑基准继续比较；
- 默认用 `--dtype float32` 与 2 专家 top-2、`index_topk` 大于压缩块数，减少随机权重下的并列。

采样内核与 V4.1 接入分别用 `test/basic/test_cuda_mtp_rejection.cpp` 和
`test/ops/deepseekV41SamplingRegression.cpp` 验证。前者检查实际保存的 q、普通采样分布、
temperature/top-k/top-p 与拒绝校正；后者检查跨卡概率传递、截断候选，并对 3/4/5/7 个候选检查前三个输出的条件联合分布，
覆盖固定 q、不同 p/q、相同 p/q，并检查请求回退和滑窗、压缩 KV、raw tail、Engram 的回滚。
统计测试使用已知目标概率，不将随机序列逐 token 相同作为通过标准。

- `--real-vision /path/to/DeepSeek-V4.1-Flash --image-size 640x480,1600x1200` 用真实 ViT 权重
  （aligner 维度依赖文本侧 dim，仍为随机）验证 32 层 ViT，包括接近 1024 token 上限的大图；
- `--chunked-prefill 16` 验证带图 prompt 的分块 prefill。

## 性能

稀疏注意力与 indexer 打分是 prefill 的两个主要开销，SM80 及以上的设备走 BF16 tensor core 路径：

- **稀疏注意力**（`V41SparseAttentionMmaKernel`）：一个 block 负责一个 token 的 32 个 head，
  候选（滑窗 + 压缩 top-k）按 32 个一组进共享内存，QK^T 与 PV 都用 `mma.sync.m16n8k16`
  （BF16 输入 / FP32 累加），片段用 `ldmatrix(.trans)` 装载，Q 常驻共享内存。
  在线 softmax 与 attention sink 的语义、候选顺序都与 CPU 参考实现一致，只有 FP32 累加顺序不同。
  decode 时 token 数少，按候选维再切成若干份（split-K），部分和由合并 kernel 按在线 softmax 的
  合并公式汇总，把 block 数补到约 4 倍 SM 数。
- **indexer 打分**（`V41IndexerScoreMmaKernel`）：一个 block 负责 64 个 token x 64 个候选；
  indexer 的 key 没有 head 维，K tile 只装载一次、循环 head 时只换 Q tile。
  整块落在因果可见范围外时直接写 `-inf` 跳过计算（这些位置本来就会被候选块打分与 top-k 忽略）。
- **indexer 分数矩阵的显存**：`[token, m]` 随上下文线性增长（1M 上下文的 ratio-1 层，
  4096 token 的分块要 16 GB）。现在按 token 维分块调用「打分 -> 候选块 -> top-k」，
  峰值由 `FASTLLM_DSV41_INDEX_SCORE_MB`（默认 128 MB）控制，与上下文长度解耦。
- **Hyper-Connections 混合系数**（`V41HcMixKernelMulti`）：`hcMult` 作为模板参数，
  累加器进寄存器；一个 block 处理 4 个相邻 token，复用同一份混合系数矩阵
  （真实模型是 24 x 20480 的 FP32，约 2 MB，原来每个 token 都要重读一遍）。
  结果与旧 kernel 逐 bit 相同。

3090 Ti（SM86）上用 `test/basic/deepseek_v41_reference.py --perf-config`
（4 层、64 头、head_dim 512、窗口 128、index_topk 512、32 个 indexer head）实测：

| 4096 token prefill + decode | 优化前 | 优化后 |
| --- | --- | --- |
| 稀疏注意力（4 层合计 / 每层 prefill） | 574 ms / 165 ms | 34 ms / 10.6 ms |
| indexer 打分（3 层合计） | 797 ms | 5.7 ms |
| HcMix（合计 / 每次 prefill 调用） | 23.3 ms / 1.92 ms | 6.9 ms / 254 us |
| 端到端 prefill | 1.56 s | 0.32 s |
| 端到端 decode | 102 tok/s | 271 tok/s |
| 单 token decode 的注意力 kernel | 88 us | 12 us（+ 6 us 合并） |

长上下文（同一配置，`--chunked-prefill 4096`）：

| prefill 长度 | 旧 kernel | 新 kernel | decode（旧 / 新） |
| --- | --- | --- | --- |
| 4096 | 1.56 s | 0.32 s | 102 / 271 tok/s |
| 8192 | 3.77 s | 0.39 s | 86 / 172 tok/s |
| 32768 | 33.60 s | 1.08 s | 58 / 99 tok/s |
| 65536 | — | 2.87 s | — |

indexer 分数矩阵的分块效果（65536 token prefill，扣掉同卡其它进程的 848 MiB 底噪）：
按 token 分块后峰值显存 1832 MiB，不分块（`FASTLLM_DSV41_INDEX_SCORE_MB` 设得很大）是 3770 MiB，
两者输出逐 token 相同，prefill 耗时 2.87 s vs 2.69 s（分块多约 7%）。

### 长上下文 prefill 的第二轮 kernel 优化

上面的 mma kernel 上线后，32k / 64k prefill 的次要热点变成了 indexer top-k、旋转/伪量化，
以及 FP4 KV 存储引入的解包开销。四处改动（都有环境变量可以逐个退回旧实现）：

- **量化 KV 行的向量化解包**：读路径原来是逐元素解 nibble（一次一个 4 bit），
  现在一个 lane 一次解 16 个连续值——`head_dim` 是 512，一个 warp 正好每人 16 个，
  而 16 个值一定落在同一个 scale 块内（`blockSize` 16 / 32），所以整行每个 lane 只做
  一次数据读（FP4 8 字节 / FP8 16 字节）+ 一次 scale 读，解完直接按 `float4` 写进
  BF16 mma 片段。E2M1 解码也从查表改成纯位运算。
- **indexer top-k**（`V41TopKKernel`）：原来每个 token 对全部 visible 候选做 8 轮 4-bit
  radix select，加上「统计可用数」「统计严格大于阈值的个数」和写出，共 11 趟全量扫描。
  现在降到 3 趟：候选块下标先升序压缩进共享内存（两级 top-k 下只有候选块内的才合格）；
  radix select 改成 8 bit 一位共 4 趟，且只有第 1 趟是全量的，之后只扫压缩进共享内存的
  几百个 key；「严格大于阈值的个数」直接取 radix select 结束时的 `remaining`；
  写出阶段把两次 `BlockScan` 合成一次。输出与 CPU 参考实现逐字节相同（含并列规则）。
- **旋转 / 伪量化**（`V41RotaryQuant`）：原来一行一个 block、`blockDim = dim`，整行 load 进
  共享内存再原样写回。`quantMode == 0`（q 与注意力输出的逆旋转，`rows = seqlen x heads`）
  拆成只碰末尾 `ropeDim` 个元素的 kernel；`quantMode > 0` 的旋转对改用相邻 lane 的
  `__shfl_xor` 交换，去掉共享内存与两次 `__syncthreads`，一个 block 处理多行。
  四种 `quantMode` 的块内 amax 语义不变。
- **HcApplyPre + RMSNorm 融合**：每个子层入口的中间张量只被紧接着的 RMSNorm 读一次，
  融合后省掉它的一次写 + 一次读和一次 kernel 启动。折叠的 BF16 舍入与 RMSNorm 的
  归约树都与原来两步逐 bit 一致（`THREAD_PER_BLOCK` 按 `LaunchFastllmRMSNormBFloat16`
  的同一条规则选；`channels == 3072` 走的是另一个专用 kernel，不接管）。

3090 Ti、`--perf-config`、32768 token prefill（`--chunked-prefill 4096`）的算子耗时
（`FASTLLM_PRINT_PROFILE=1 FASTLLM_CUDA_SYNC=1`，单位 ms）：

| 算子 | BF16 KV 旧 | BF16 KV 新 | `fp4_e2m1` 旧 | `fp4_e2m1` 新 |
| --- | --- | --- | --- | --- |
| `DeepSeekV41SparseAttention` | 270 | 268 | 839 | 341 |
| `DeepSeekV41IndexerTopK` | 81 | 35 | 80 | 35 |
| `DeepSeekV41RotaryQuant` | 74 | 17 | 72 | 16 |
| `DeepSeekV41HcApplyPre` + `RMSNorm` | 2.3 + 3.8 | 0.5 + 2.3 | 2.4 + 3.7 | 0.5 + 2.1 |
| 全部算子合计 | 1079 | 966 | 1639 | 1040 |

端到端（同一配置，取两次运行的稳定值）：

| 上下文 | KV | prefill 旧 | prefill 新 | decode 旧 | decode 新 |
| --- | --- | --- | --- | --- | --- |
| 32768 | BF16 | 1.08 s | 0.98 s | 177 tok/s | 186 tok/s |
| 32768 | `fp4_e2m1` | 1.64 s | 1.04 s | 212 tok/s | 229 tok/s |
| 65536 | BF16 | 2.49 s | 2.15 s | 119 tok/s | 129 tok/s |
| 65536 | `fp4_e2m1` | 3.66 s | 2.30 s | 157 tok/s | 172 tok/s |

也就是说 **FP4 KV 相对 BF16 KV 的 prefill 代价从 +52% / +47% 降到 +6% / +7%**，
而 logits 仍与 BF16 存储逐 bit 相同。剩下的差距全部在稀疏注意力里
（341 ms vs 268 ms）：解包本身已经不是 ALU 瓶颈（把 FP32 乘 + 转换换成 BF16 上的
`__hmul2` 没有任何变化），要再往下压得靠 KV tile 的双缓冲，那要重排整个 mma kernel。

### CPU-only 固定 fixture 回归（无需 torch）

`test/basic/deepseek_v41_reference.py` 需要 torch、transformers 与官方 `inference/` 代码，进不了 CI。
`test/basic/test_deepseek_v41_cpu_fixture.py` 把「一个微型 V4.1 checkpoint + 官方实现算出的参考 logits」
固化在 `test/basic/deepseek_v41_fixture.npz`（约 2.2 MB）里，只依赖 numpy 与 fastllm 的 Python 包：

```bash
PYTHONPATH=build/tools python test/basic/test_deepseek_v41_cpu_fixture.py
```

追加 `--ngram-device disk` 可用同一份 fixture 校验磁盘 Engram 加载与按行读取。

退出码 0 表示通过，1 表示失败；fixture 缺失时打印重新生成的命令并以 0 退出（跳过）。
fixture 里的模型是 5 层、dim 128、词表 128、2 专家，覆盖 V4.1 的全部结构特性：
`compress_ratios = (0, 2, 2, 1, 1)`（三种压缩层）、`kv_source_layer_ids = (1, 3)`（层 2 / 4 跨层复用压缩 KV）、
`index_source_layer_ids = (1, 3, 4)` 且 `candidate_source_layer_id = 3`（两级 top-k）、
`engram_layer_ids = (1, 4)`、`hc_mult = 2`、`window_size = 8`（48 token 的 prefill 让滑窗环形缓冲绕多圈）。

比较方式：逐步（prefill + 4 步 decode）比较 logits 的余弦相似度（>= 0.9975）与
`max|diff|` 占 logits 值域的比例（<= 4%），并要求贪心 token 与参考一致；
参考侧是官方实现的 bf16 前向，噪声底约为值域的 2–3%（fastllm 自己的 CPU 与 CUDA 路径之间也有约 1.5%），
所以容差按值域折算而不是给绝对值。fixture 生成时会在多个提示词种子里挑一个每步 top1/top2 间距都
远大于该噪声的，避免 argmax 因舍入翻转；万一仍然翻转，测试会按"参考侧两者间距落在容差内"判为并列而不算失败。

重新生成 fixture（需要 torch + 官方 inference 代码 + 一张 GPU）：

```bash
PYTHONPATH=build/tools python test/basic/deepseek_v41_fixture_gen.py \
  --reference-dir /path/to/DeepSeek-V4.1-Flash/inference \
  --work-dir /tmp/v41-micro --out test/basic/deepseek_v41_fixture.npz
```

生成脚本自带一个 128 词的迷你 tokenizer，不依赖真实 checkpoint 的 tokenizer.json。

### 算子级 CPU / CUDA 一致性

`test/ops/deepseekV41OpsRegression.cpp`（`cmake -DUNIT_TEST=ON`）用同一份输入分别在 CPU 与 CUDA 上跑
11 个 V4.1 专用算子并比较输出，覆盖随机输入与边界情况：

```bash
./build/deepseekV41OpsRegression            # 全部算子
./build/deepseekV41OpsRegression --list     # 列出用例
./build/deepseekV41OpsRegression IndexerTopK CandidateBlocks   # 只跑指定算子
ctest -R deepseekV41Ops                     # 无 CUDA 设备时以 77 跳过
```

## Decode 算子与兼容性

单 token indexer 使用 head 作为 MMA 矩阵行。注意力将输出维度分成四份；候选较多且临时空间
充足时，先并行计算 QK，再保持原有的 64 槽在线 softmax、BF16 概率舍入和 PV 顺序。
BF16 5120 维、1–8 行 RMSNorm 与四路 HC Finish 使用保持归约顺序的专用内核。

MMA 路径检查设备及实际加载内核的架构、线程数和共享内存限制；不满足时使用原 FP32 标量路径。
非 decode 形状使用通用 MMA，注意力临时内存不足时退回无需该空间的融合 MMA；执行错误正常上报。
RMSNorm 其他类型、维度和批量保留原路径，不满足向量读取对齐时使用标量读取。
CUDA 构建仍需包含目标显卡支持的代码镜像，并使用支持该架构的工具链。

```bash
./build/deepseekV41OpsRegression
ctest --test-dir build -R 'deepseekV41(Precision|Tp|DecodeOptimization)' --output-on-failure
python test/basic/test_deepseek_v41_tp_graph.py --binary ./build/deepseekV41TpGraphRegression
```

Graph fixture 需要 PyTorch、safetensors、numpy 和两张 CUDA GPU；覆盖不同长度的连续请求、
eager / Graph 切换和共享专家重叠，以及 DSpark 的 1～6 token 图重放、目标层特征与部分接受后的 KV 回滚。
`--hc-mult 2` 可补测两路 HC，默认四路。
算子回归检查通用路径与 decode 对齐、独立 CPU RMSNorm 参考、量化 KV、远端 logits 同步、
缓冲复用、并发 Graph 及捕获期间临时空间不足时的回退。

## 调试环境变量

| 变量 | 作用 |
| --- | --- |
| `FASTLLM_DSV41_LEGACY_ATTN` | 稀疏注意力退回 FP32 标量 kernel（对比 / 排查用） |
| `FASTLLM_DSV41_LEGACY_INDEXER` | indexer 打分退回 FP32 标量 kernel |
| `FASTLLM_DSV41_LEGACY_HCMIX` | HC 混合系数退回旧 kernel |
| `FASTLLM_DSV41_LEGACY_FP4_UNPACK` | 量化 KV 缓存行退回逐元素解包（不再按 4 字节一组批量展开） |
| `FASTLLM_DSV41_LEGACY_TOPK` | indexer top-k 退回旧的「全 visible 扫描 + 8 轮 4-bit radix」kernel |
| `FASTLLM_DSV41_LEGACY_ROTARY` | 旋转 / 伪量化退回旧的「一行一个 block + 共享内存」kernel |
| `FASTLLM_DSV41_DISABLE_HCPRENORM` | 不融合 HcApplyPre 与 RMSNorm，退回两个算子分开做 |
| `FASTLLM_DSV41_INDEX_SCORE_MB` | indexer 分数矩阵的显存预算（MB，默认 128），决定 token 维分块大小 |
| `FASTLLM_DSV41_INDEX_CHUNK` | 直接指定 indexer 的 token 分块大小（覆盖上面的预算推算） |
| `FASTLLM_DSV41_ENGRAM_PROFILE` | Engram 查表 / 转换 / 投影分段计时（见 "Engram 元数据"） |
| `FASTLLM_DSV41_ENGRAM_POOL` | Engram 查表改用常驻线程池 |
| `FASTLLM_DSV41_ENGRAM_PREFETCH` | 跨层预取下一个 Engram 层的行号与表行 |
| `FASTLLM_DSV41_ENGRAM_MADVISE` | Engram 表的内存访问提示（random / hugepage / both） |
| `FASTLLM_DSV41_ENGRAM_WKV_FP8` | `engram.wkv` 保留 checkpoint 里的 FP8 精度，不解量化 |
| `FASTLLM_DSV41_DISABLE_FAKE_QUANT` | 关闭 FP8 / FP4 伪量化（仅用于对齐调试） |
| `FASTLLM_DSV41_DISABLE_CUDA_ROUTE` | 路由退回 CPU 参考实现 |
| `FASTLLM_DSV41_DUMP_DIR` | 把每层中间张量写到该目录（对齐调试） |
| `FASTLLM_DSV41_DISABLE_TP_ATTENTION` | 张量并行时不切分注意力 head（排查用，注意力改为每卡各算一份） |
| `FASTLLM_DSV41_DISABLE_TP_SHARED_EXPERT` | 张量并行时不切分共享专家（排查用） |
| `FASTLLM_CUDA_GRAPH` 等 | decode 与 DSpark 校验的 CUDA Graph，见对应章节 |
| `FASTLLM_TRACE_OPS` | 逐算子打印"算子名 / 落在哪个设备 / 权重名"（排查 TP 落点用） |
| `FASTLLM_DSV41_DISABLE_PREFIX_CACHE` 等 | 前缀缓存相关，见"多请求与前缀缓存" |
| `FASTLLM_DSPARK_*` | DSpark 投机解码相关，见"DSpark 投机解码" |
| `FT_MOE_ASSIST_DEVICES` / `FT_MOE_ASSIST_BALANCE` / `FT_EXPERT_LIMIT_AUTO` | NUMA MoE 的多卡专家流，见"prefill 的多卡专家流" |
| `FASTLLM_PROFILE_NUMAS_MOE=1 FASTLLM_PROFILE_DETAIL=1` | NUMA MoE 分阶段耗时、CPU/GPU 专家划分及各卡 route 数 |

## DSpark 与专家缓存验证

CUDA + NUMA 混合专家缓存会自动处理 2–8 行 verify，包含 TP + CUDA Graph 路径，配置和命中率口径见
[CUDA 专家缓存](cuda-expert-cache.md#deepseek-v41)。HC mix 支持 1–8 行小批量；
WoA 对小于 16 行的批次复用权重，保持各输出的累加顺序，其他形状走已有路径。
草稿的稠密层、路由专家、共享专家与 HC post 复用主干的量化和 BF16 舍入语义。

`deepseekV41SamplingRegression` 检查实际 CUDA 拒绝采样器的输出分布、部分拒绝、
bonus token 和缓存回滚；`deepseekV41OpsRegression` 检查小批量算子与原归约路径；
`cuda_dsv41_moe_cache_test --dual` 检查独立数值参考和跨卡缓存切换。
`test/basic/test_deepseek_v41_tp_graph.py --binary build-fastllm/deepseekV41TpGraphRegression --expert-cache`
使用 NVFP4 专家检查 TP verify 缓存、1–6 行 Graph 重放、主模型特征与 KV 回滚。
编程模板在 top-p 截断后可能只剩一个候选，因此高接受率本身不能证明使用了贪心验证；
评估时应记录采样参数、拒绝轮数和代码功能结果，分别检查采样校正与前向浮点误差。
