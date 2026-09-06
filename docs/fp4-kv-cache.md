# FP4 KV cache

Qwen3.5 架构的 CUDA 分页注意力支持 NVFP4 KV 存储，包括 Qwen3.8-27B-FP8。使用 `--kv_cache_dtype fp4` 开启；也接受 `nvfp4` 和 `fp4_e2m1`。默认仍为 `auto`。

双卡示例：

```bash
numactl -C 0-31 -m 0 ftllm server /path/to/Qwen3.8-27B-FP8 \
    --tp 2 --dtype auto --kv_cache_dtype fp4 --tokens 32768
```

Launcher 和终端向导的 KV Cache 类型选项也可选择 `fp4`。独立工作室连接已有 API server，缓存类型需要在模型服务的启动参数中设置。

## 范围与格式

- 当前支持 Qwen3.5 架构的 CUDA KV、128/256 维注意力头，CPU KV 不支持此格式。FlashInfer FP4 路径需要 CUDA 12.8 及以上、SM80 及以上 GPU；不满足条件、未编译 FlashInfer 或设置 `FASTLLM_FORCE_NATIVE_ATTN=1` 时自动使用原生注意力。
- 模型权重保持 `--dtype` 指定的类型；`--dtype auto` 保留源 FP8 权重。Q 和注意力计算仍使用 FP16/BF16，线性注意力的卷积缓存与 recurrent state 保留原有精度。
- 每 16 个 KV 值保存 8 字节 E2M1 数据和 1 字节 E4M3 缩放因子。包含缩放因子后，存储为 FP16 KV 的 **28.125%**、FP8 KV 的 **56.25%**。这些比例仅针对普通注意力 KV，不代表整个模型的显存占用比例。
- 每个物理页保存 `[packed KV][block scales]`，整页复制、回收和 MTP 验证页克隆会同时处理数据及缩放因子。缓存预算和实际分配都包含缩放因子。
- prefill、普通 append、融合 decode、批量追加及图捕获中的追加均写入相同格式。FlashInfer 读取打包数据和对应的 scale strides，不在显存中保留完整 FP16 KV 副本。
- 原生 fallback 使用软件解包，直接读取相同的 E2M1 数据和 E4M3 缩放因子，不依赖 FP4 硬件指令。prefill 复用分块 cuBLAS 工作区，decode 复用原生 split-KV 与在线 softmax；批量图重放从设备读取分页元数据。该路径用于兼容回退，性能可能低于 FlashInfer；原生分页注意力仍不支持滑动窗口。

FP4 是有损量化，可能改变 logits 和生成结果。内存节省不保证提高吞吐；在 4090 上解包与缩放也有计算成本。

## 验证

参数及 Launcher 透传测试：

```bash
python3 test/api/test_fp4_kv_cache.py
```

开启 CMake 的 `UNIT_TEST` 和 CUDA 后：

```bash
cmake --build build --target cuda_fp4_kv_test -j8
ctest --test-dir build -R '^cuda_fp4_kv(_native.*)?$' --output-on-failure
```

CUDA 测试检查空缓存不分配存储、物理字节数与预算一致、FP4 打包结果与独立 CPU 量化参考一致，以及注意力输出与同一量化数据反量化后的结果一致。覆盖非连续页、部分末页、128/256 维头、FP16/BF16 查询、短查询和最长 4097 token 的 KV；另检查融合 decode 与独立量化一致，以及融合写入的 CUDA Graph replay。

原生回退测试覆盖单序列入口、批量 prefill、split-KV decode 和注意力图重放。不同长度的双请求另与 CPU softmax 参考对照，图重放期间修改设备上的页数、物理页索引、末页长度和查询边界，检查分页元数据不会被固定在捕获时的值。覆盖 GQA 1–8 组、1/7/16/128 token 页、带 padding 的 Q head stride，以及最长 32769 token 的 KV。向量解包还覆盖全部有限非负 E4M3 scale 和 E2M1 打包值对。

`cuda_fp4_kv_native` 使用已有的 `FASTLLM_FORCE_NATIVE_ATTN=1` 检查公共接口的自动回退；另有关闭 GQA 共享、关闭 split 的测试，检查通用回退分支。FlashInfer 不可用时这些测试仍会运行。

## 原生 FP4 注意力优化

FP4 复用了原生 FP8 的以下优化：

- GQA 2–8 组共用 KV tile：每个 warp 负责一个 Q head，KV 读取和解包后通过共享内存供整组复用。共享内存按维度重排，避免连续读取落入少量 bank。
- 每个线程一次读取 4/8 个 FP4 值，复用对应的 block16 scale；共享 tile 还复用 E4M3 查找表。普通 per-head 回退与 prefill gather 也使用向量读取。
- 页游标同时跟踪打包数据和 scale 的物理地址，仅在跨页时重新查表，支持非连续页及短页。
- 在线 softmax 使用 log2 域，在最大值未改变时跳过重新缩放；split 结果使用同域的并行归约合并。工作区地址保持稳定，支持 CUDA Graph 重放。

这些优化不改变 FP4 量化格式，也不增加运行时环境变量。已有的 `FASTLLM_PAGED_GQA_DECODE=0` 和 `FASTLLM_PAGED_NO_SPLIT=1` 可用于对照。

RTX 4090D 上与初版原生 FP4 回退的 decode 算子对照：FP16 Q、head_dim=256、2 个 KV head、GQA=6、batch=1、128 token 页、物理页逆序。使用相同输入和 CUDA Graph 测量，包含 split 与合并耗时：

| KV 长度 | 初版耗时 | 优化后耗时 | 加速比 |
| --- | ---: | ---: | ---: |
| 128 | 16.00 μs | 16.62 μs | 0.96× |
| 1024 | 28.15 μs | 16.83 μs | 1.67× |
| 4096 | 73.57 μs | 22.58 μs | 3.26× |
| 32768 | 527.45 μs | 79.96 μs | 6.60× |

短上下文收益有限。80 组不同头维度、GQA、batch 和 KV 长度的算子基准共对照 322560 个输出，优化前后最大绝对差为 0.00012207；这些速度是 attention 算子结果，整模型收益还取决于其他层。

优化版通过了 CPU softmax 对照、CUDA Graph 元数据变化及 Compute Sanitizer memcheck（0 errors）。在 SM89 上强制进入原有 FP8 SM7x 优化分发的回归中，61440 个输出与改动前逐字节一致；SM70 目标另通过编译检查。

四种分页写入入口共用类型分发和写入逻辑；测试还覆盖 FP32/FP16/BF16 输入到 FP32/FP16/BF16/FP8 存储的 48 种组合，检查已有格式的结果和页内偏移保持一致。

模型级验证应分别运行 FP16 KV 和 FP4 KV，固定模型、TP 配置、上下文容量与生成参数，记录显存、首字延迟、吞吐和 logits 差异，并检查长文检索与多轮输出。MTP、CUDA Graph 和多模态需要分别验证，不应仅以模型成功加载作为通过标准。

### 本机双卡试跑

Qwen3.8-27B-FP8，RTX 4090D + RTX 4090，TP=2、32K KV 容量、CPU/内存绑定 NUMA 0。使用正常 CLI 的自动优化配置，开启 CUDA Graph、GPU token handoff 和前缀缓存，MTP 关闭：

| 指标 | FP16 KV | FP4 KV |
| --- | ---: | ---: |
| 测试结束时两卡显存合计 | 40042 MiB | 38634 MiB |
| 代码生成解码速度 | 57.38 token/s | 57.11 token/s |
| 4133 token 长文首次请求首字延迟 | 0.985 s | 1.065 s |
| 重复长文首字延迟（命中 4096 token 前缀） | 0.069 s | 0.070 s |

两卡合计节省 1408 MiB（约 1.38 GiB）。算术、代码生成和长文密码检索均通过；这是少量请求的实测，不能代替完整精度评测。关闭图执行和前缀缓存的 FP16/FP4 对照也通过。

FP4 和 FP16 API 对照测试还覆盖了图片首轮、携带原图的后续追问、图片 embedding 缓存命中及两条并发文本请求。测试期间修复了双卡多模态入口读取父级 embedding 张量的问题，改为使用 GPU 上的完整副本；原问题在 FP16 KV 下同样能复现。

对已启动的服务可运行图片追问回归检查（建议设置 `--max_batch 2`）：

```bash
python3 test/multimodal/fastllm_openai_image_followup_check.py \
    --base-url http://127.0.0.1:8080
```

使用正常 CLI 配置时，FP4 的 MTP 加 CUDA Graph 短请求检查通过，日志确认捕获了双卡 `verify=2` 的验证图。尚未进行 MTP 的完整吞吐或精度评测。

同一双卡模型设置 `FASTLLM_FORCE_NATIVE_ATTN=1` 后，8K KV 容量下通过了图片首轮与追问、双请求并发、4133 token 密码检索、4096 token 前缀复用和 MTP 短请求；日志确认 batch 1/2 解码图及双卡 `verify=2` 验证图均捕获成功。原生内核另通过 CUDA 12.9 的 SM70 编译检查，尚未在 SM70 实卡上验证 FP4。

格式及接口参考：[FlashInfer 注意力](https://docs.flashinfer.ai/api/attention.html)、[NVFP4 KV 分页量化](https://docs.flashinfer.ai/api/page.html)。本实现使用仓库已有的 FlashInfer 头文件，无需额外安装 Python FlashInfer。
