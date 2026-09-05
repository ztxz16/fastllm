# FP4 KV cache

Qwen3.5 架构的 CUDA 分页注意力支持 NVFP4 KV 存储，包括 Qwen3.8-27B-FP8。使用 `--kv_cache_dtype fp4` 开启；也接受 `nvfp4` 和 `fp4_e2m1`。默认仍为 `auto`。

双卡示例：

```bash
numactl -C 0-31 -m 0 ftllm server /path/to/Qwen3.8-27B-FP8 \
    --tp 2 --dtype auto --kv_cache_dtype fp4 --tokens 32768
```

Launcher 和终端向导的 KV Cache 类型选项也可选择 `fp4`。独立工作室连接已有 API server，缓存类型需要在模型服务的启动参数中设置。

## 范围与格式

- 当前支持 Qwen3.5 架构、CUDA 12.8 及以上、SM80 及以上 GPU、128/256 维注意力头。CPU KV 和原生注意力 fallback 不支持此格式。
- 模型权重保持 `--dtype` 指定的类型；`--dtype auto` 保留源 FP8 权重。Q 和注意力计算仍使用 FP16/BF16，线性注意力的卷积缓存与 recurrent state 保留原有精度。
- 每 16 个 KV 值保存 8 字节 E2M1 数据和 1 字节 E4M3 缩放因子。包含缩放因子后，存储为 FP16 KV 的 **28.125%**、FP8 KV 的 **56.25%**。这些比例仅针对普通注意力 KV，不代表整个模型的显存占用比例。
- 每个物理页保存 `[packed KV][block scales]`，整页复制、回收和 MTP 验证页克隆会同时处理数据及缩放因子。缓存预算和实际分配都包含缩放因子。
- prefill、普通 append、融合 decode、批量追加及图捕获中的追加均写入相同格式。FlashInfer 读取打包数据和对应的 scale strides，不在显存中保留完整 FP16 KV 副本。

FP4 是有损量化，可能改变 logits 和生成结果。内存节省不保证提高吞吐；在 4090 上解包与缩放也有计算成本。

## 验证

参数及 Launcher 透传测试：

```bash
python3 test/api/test_fp4_kv_cache.py
```

开启 CMake 的 `UNIT_TEST` 和 CUDA 后：

```bash
cmake --build build --target cuda_fp4_kv_test -j8
ctest --test-dir build -R '^cuda_fp4_kv$' --output-on-failure
```

CUDA 测试检查空缓存不分配存储、物理字节数与预算一致、FP4 打包结果与独立 CPU 量化参考一致，以及注意力输出与同一量化数据反量化后的结果一致。覆盖非连续页、部分末页、128/256 维头、FP16/BF16 查询、短查询和最长 4097 token 的 KV；另检查融合 decode 与独立量化一致，以及融合写入的 CUDA Graph replay。

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

格式及接口参考：[FlashInfer 注意力](https://docs.flashinfer.ai/api/attention.html)、[NVFP4 KV 分页量化](https://docs.flashinfer.ai/api/page.html)。本实现使用仓库已有的 FlashInfer 头文件，无需额外安装 Python FlashInfer。
