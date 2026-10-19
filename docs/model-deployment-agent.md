# Agent 部署速查：Qwen3.8、Flash-Next、DeepSeek V4.1 与 GLM-5.3

这份说明给自动化 agent 使用。建议用户先把本文和目标模型的部署指南交给 agent，再让 agent 根据机器的 GPU、显存、主机内存、模型路径和端口执行。下载页中的模型簇和下面的命令使用相同的 ModelScope ID；路径、端口和模型名由 agent 按机器环境替换。

## 选择与默认策略

混合推理优先使用 `--auto`。它会在启动时读取可见 GPU、显存、主机内存和模型配置，联合选择主干设备、MoE 专家设备、专家缓存、上下文容量和分块 prefill。显式参数覆盖自动规划器，适合在第一次启动失败或需要固定实验条件时使用。

当前 `--auto` 支持 Linux NVIDIA CUDA 上的 Qwen3 dense、Qwen4/Flash-Next、GLM-5.3 和 DeepSeek-V4.1。**Qwen3.8 27B 使用 Qwen3.5 架构，不等于 Qwen3 dense，目前应使用下方手动配置**；Launcher 选择“自定义”，或使用原有的长上下文/高并发推荐。

| 模型簇 | 推荐入口 | 适合的下载项 |
| --- | --- | --- |
| Qwen3.8 27B | 手动 CUDA / TP；当前不支持 `--auto` | `Qwen/Qwen3.8-27B`、`Qwen/Qwen3.8-27B-FP8`、`QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4` |
| Qwen3.8-Flash-Next GSQ-RCO | `--auto`；GGUF 需要把首个分片作为模型路径 | `Q2_0`、`IQ2_XS`、`IQ3_XXS`、`IQ3_S`，下载时保留对应子目录和 `mmproj` |
| DeepSeek V4.1 Flash | `--auto`；主机内存不足时按日志改用磁盘专家 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| GLM-5.3-Flash | `--auto`；GGUF 需要把首个分片作为模型路径 | 本地完整 HF/NVFP4 目录或 `glm5next` GGUF 分片，见 [GLM 指南](glm5.md) |

自动化流程应等待日志出现就绪事件，再请求 `/v1/models` 检查 `id` 和上下文窗口，最后发送一个短请求验证输出。不要根据进程已经启动就判定模型可用。

## 推荐部署顺序

1. 确认模型架构属于上面的支持范围，再用下方的 `--auto` 命令启动；Qwen3.8 27B 使用手动命令。
2. 等待就绪日志，读取规划器实际选择的 `device`、`moe_device`、缓存、chunk 和上下文上限。
3. 调用 `/v1/models` 和一次短请求；如果失败，把完整启动日志交给 agent，再只增加必要的显式覆盖。
4. 需要比较性能时固定 GPU 可见性、线程、输入输出长度和 warmup，只改变一个配置项。

`--auto_gpu_memory_limit` 和 `--auto_host_memory_limit` 是规划上限，不是运行时强制的进程 cgroup。省略时自动读取当前机器配置；填写后，agent 应把限制作为可用预算并在日志中确认回退路径。

## 启动命令

```bash
# Qwen3.8-27B FP8：当前使用手动 CUDA，需足够显存
ftllm server /data/models/Qwen3.8-27B-FP8 \
  --device cuda --model_name qwen3.8-27b \
  --host 0.0.0.0 --port 8080

# Qwen3.8-Flash-Next GSQ-RCO；路径指向下载目录中的第一个 GGUF 分片
ftllm server /data/models/Qwen3.8-Flash-Next-GSQ-RCO/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --auto --model_name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8080

# DeepSeek V4.1 Flash 1017
ftllm server /data/models/DeepSeek-V4.1-Flash-1017 \
  --auto --model_name deepseek-v4.1-flash-1017 \
  --host 0.0.0.0 --port 8080

# GLM-5.3-Flash GGUF；路径指向第一个分片
ftllm server /data/models/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  --auto --model_name glm-5.3-flash \
  --host 0.0.0.0 --port 8080
```

API 检查示例：

```bash
curl http://127.0.0.1:8080/v1/models
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen3.8-27b","messages":[{"role":"user","content":"用一句话介绍自己。"}],"max_tokens":64}'
```

## 手动参数与覆盖规则

自动模式失败、需要复现实验或需要固定设备布局时，再使用下面的手动参考。参数组合要以目标模型指南和启动日志为准；同一行命令不要同时指定互相冲突的 `--tp` 与 `cudapp`。

### 设备和专家放置

| 参数 | 用途 | 典型值 |
| --- | --- | --- |
| `--device` | 主干计算设备 | `cuda`、`cudapp=2`、`numa` |
| `--tp` | CUDA 张量并行 | `2`、`0,1`、`auto` |
| `--moe_device` | MoE 专家设备 | `cuda`、`numa`、`cpu`、`disk` |
| `--moe_device_layers` | 让最后 N 层使用 MoE 设备，剩余专家留在主设备 | `36` |
| `--threads` | CPU/NUMA 线程数 | `28`、`64` |
| `--gpu_mem_ratio` | 给模型、KV 和缓存的显存比例 | `0.9` |
| `--auto_gpu_memory_limit` | 自动规划时每张 GPU 的显存上限 | `20g`；省略则读取实际可用显存 |
| `--auto_host_memory_limit` | 自动规划时主机内存总上限 | `128g`；省略则读取实际可用内存 |

例如固定双卡 PP、NUMA 专家：

```bash
ftllm server /data/models/Qwen3.8-Flash-Next \
  --device cudapp=2 --moe_device numa --threads 28 \
  --chunked_prefill_size 4096 --gpu_mem_ratio 0.9
```

### 模型手动起步配置

这些命令用于定位问题和做可重复对照，数值不是所有机器的最佳值：

```bash
# Qwen3.8 27B FP8：双卡 TP，显存足够时优先
ftllm server /data/models/Qwen3.8-27B-FP8 \
  --device cuda --tp 2 --chunked_prefill_size 4096 \
  --gpu_mem_ratio 0.9 --max_batch 1

# Qwen3.8-Flash-Next GSQ：双卡按层，专家放 NUMA
ftllm server /data/models/Qwen3.8-Flash-Next-GSQ-RCO/Q2_0/Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf \
  --device cudapp=2 --moe_device numa --ngram_device disk \
  --chunked_prefill_size 4096 --moe_cuda_cache 0 --mtp 0

# DeepSeek V4.1 Flash：大内存单卡混合推理
ftllm server /data/models/DeepSeek-V4.1-Flash-1017 \
  --device cuda --moe_device numa --ngram_device cpu \
  --fast_prefill --dspark 7 --max_context_length 1048576 \
  --chunked_prefill_size 4096

# GLM-5.3-Flash：双卡 PP，NUMA 专家，先关闭 MTP 做基线
ftllm server /data/models/GLM-5.3-Flash-UD-IQ2_XXS-00001-of-00004.gguf \
  --device cudapp=2 --moe_device numa --chunked_prefill_size 8192 \
  --dtype bfloat16 --atype bfloat16 --moe_atype bfloat16 \
  --moe_cuda_cache 8g --mtp 0
```

Qwen3.8-Flash-Next 和 GLM 的 GGUF 分片必须保留同目录的其余分片；DeepSeek 的 `--ngram_device cpu` 需要约 200 GB 级别的 Engram 常驻内存，内存不足时改为 `disk`。模型支持的参数和实际默认值会随 checkpoint 变化，agent 应以 `ftllm server --help`、模型指南和启动日志为最终依据。

### 缓存、上下文和 prefill

| 参数 | 用途 | 说明 |
| --- | --- | --- |
| `--moe_cuda_cache` | GPU 专家缓存 | `4g` 表示每张参与设备的预算；只对 CPU/NUMA/disk 专家有效 |
| `--moe_cpu_cache` | 磁盘专家的主存缓存 | `32g`、`64g`；不是进程总内存上限 |
| `--ngram_device` | Qwen PLE 或 V4.1 Engram | `cpu` 常驻内存，`disk` 按需读取 |
| `--chunked_prefill_size` | 每块 prefill 的 token 数 | 长输入可从 `4096` 开始，显存允许时增大 |
| `--max_context_length` | 单会话输入和输出合计上限 | DeepSeek 自动模式默认最多 1M，其它模型通常按 128K 规划 |
| `--tokens` | 所有请求共享的 KV token 池 | 并发或长上下文时显式设置 |
| `--kv_cache_limit` | KV Cache 总预算 | 例如 `8g`、`32g` |
| `--prefix_cache true` | 复用重复前缀 | 会占用 KV 预算；自动模式可能按模型策略选择 |

### 投机解码与快速路径

| 参数 | 用途 |
| --- | --- |
| `--dspark 7` | DeepSeek V4.1 内置 DSpark，每轮最多 7 个草稿 token |
| `--mtp 3` | 支持 MTP 的 Qwen/其它模型启用 MTP |
| `--draft` / `--speculative_draft_model_path` | 指定外部 DFlash/MTP 草稿目录 |
| `--fast_prefill` | 启用模型支持的快速 prefill；DeepSeek V4.1 使用近似注意力，可能改变 logits 和输出 |
| `--no_fast_prefill` | 关闭快速 prefill，做精度或回归对照 |
| `--speculative_dspark_confidence_threshold 0.5` | DSpark 草稿置信度门槛；提高时通常更早停止草稿生成，接受率和速度需实测 |

DeepSeek V4.1 的自动模式默认规划不超过原生窗口的 1M 上下文、DSpark 7 和 `fast_prefill`；预算不足且调整专家放置、缓存和 chunk 后仍不可行时会报错，不自动缩短窗口。显式 `--dspark 0`、`--no_fast_prefill` 或上下文参数会覆盖默认值。GSQ GGUF 的量化布局由文件决定，不要再额外指定在线量化类型。

## 低资源回退

当日志显示显存或主机内存不足时，按以下顺序修改并重试：先减少 `--moe_cuda_cache`，再把 `--moe_device numa` 改成 `--moe_device disk --moe_cpu_cache 32g`，最后降低 `--max_context_length` 或 `--chunked_prefill_size`。磁盘专家对随机读取敏感，应该把模型放在本地 NVMe；`--ngram_device disk` 只控制 PLE/Engram 表，不等于专家已经走磁盘。

## Agent 验收清单

1. 启动命令返回后，等待 `ready` 或等价的服务就绪日志。
2. `GET /v1/models` 必须返回预期 `model_name`。
3. 发送短文本请求，确认 HTTP 200、输出非空且没有 NaN/设备错误。
4. 从启动日志记录实际 `device`、`moe_device`、缓存容量、`chunked_prefill_size` 和上下文窗口；这些值可能与请求的自动配置不同。
5. 需要比较速度时固定输入长度、输出长度、线程数和 GPU 可见性，只改变一个规划参数。

更多模型专用限制见 [Qwen3.8-Flash-Next 用户手册](qwen3.8-flash-next/README.md)、[Qwen3 部署指南](qwen3.md)、[DeepSeek V4.1 说明](deepseek_v41.md)、[GLM 指南](glm5.md) 和 [自动规划器说明](hybrid-auto.md)。
