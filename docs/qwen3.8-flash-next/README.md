# Qwen3.8-Flash-Next 用户手册

[English](README_EN.md) · [项目首页](../../README.md)

本手册介绍 Qwen4-Exp / Qwen3.8-Flash-Next 的文本服务部署，包括单卡混合推理、双卡串行与并行、专家缓存、固定 GPU 专家层、MTP 和 API 调用。以下双卡示例使用 NVFP4 模型。

## 准备模型

先安装支持该模型的 FastLLM，并准备完整的本地模型目录，其中应包含配置、分词器和 Safetensors 权重。将命令中的 `/data/models/Qwen3.8-Flash-Next-NVFP4` 替换为实际路径。

混合推理需要足够的主机内存；使用磁盘 PLE 时，将模型存放在 SSD 上。

## 快速启动

下面以一张 GPU 运行主干、CPU 的 NUMA 后端运行专家为例：

```bash
ftllm server /data/models/Qwen3.8-Flash-Next-NVFP4 \
  --model_name qwen3.8-flash-next \
  --host 0.0.0.0 --port 8080 \
  --device cuda --moe_device numa \
  --atype float16 --ngram_device disk --threads 28 \
  --chunked_prefill_size 4096 --gpu_mem_ratio 0.9 \
  --prefix_cache true --enable_thinking false
```

等待模型加载完成后，客户端使用 `http://服务器地址:8080/v1`，模型名填写 `qwen3.8-flash-next`。线程数应根据 CPU 物理核心数调整。

<a id="deployment"></a>

## 双卡混合部署

以下配置适用于两张 **22 GiB 显存版 RTX 2080 Ti**、AMD EPYC 7452 和约 126 GiB 主机内存的机器。标准 11 GiB 2080 Ti 需要降低专家缓存或 GPU 驻留层数。

先在 Bash 中设置公共参数，再选择一种部署命令：

```bash
MODEL=/data/models/Qwen3.8-Flash-Next-NVFP4
unset FASTLLM_TP
export CUDA_VISIBLE_DEVICES=0,1
export FASTLLM_CUDA_GRAPH=0

COMMON=(
  --model_name qwen3.8-flash-next
  --host 0.0.0.0 --port 8080
  --atype float16 --ngram_device disk --threads 28
  --mtp 3 --temperature 0 --top_k 1 --repeat_penalty 1
  --max_batch 1 --kv_cache_limit 1g --gpu_mem_ratio 0.95
  --chunked_prefill_size 4096
  --prefix_cache true --enable_thinking false
)
```

这组示例关闭 CUDA Graph，并开启适合多轮文本请求的前缀缓存。模型不含 MTP 权重或需要关闭推测解码时，将 `--mtp 3` 改为 `--mtp 0`。

### 双卡串行：专家缓存开关

双卡串行将主干按层放到两张显卡上。`cudapp=2` 为两卡均分主干层。

```bash
# 关闭专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 0

# 每卡使用 8 GiB 专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=2 --moe_device numa --moe_cuda_cache 8g
```

### 双卡并行：专家缓存开关

`--tp 2` 让两张显卡共同计算每一层。选择该模式时，不再同时指定 `cudapp`。

```bash
# 关闭专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 0

# 每卡使用 8 GiB 专家缓存
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_cuda_cache 8g
```

`--moe_cuda_cache 8g` 是**每卡**预算，两卡合计约 16 GiB。显存还需要容纳主干权重、KV 缓存和计算工作区，不能全部分配给专家缓存。

### 固定部分专家层在 GPU：双卡并行

关闭动态专家缓存，将前 12 层的全部专家固定在 GPU，剩余 36 层使用 NUMA：

```bash
ftllm server "$MODEL" "${COMMON[@]}" \
  --tp 2 --moe_device numa --moe_device_layers 36 --moe_cuda_cache 0
```

`--moe_device_layers 36` 指**最后 36 层**使用 `--moe_device numa`；前 12 层专家由两张 GPU 分片存放。减小这个数字会增加 GPU 驻留层数和显存占用。

### 固定部分专家层在 GPU：双卡串行

使用下列配套的主干和专家映射：

```bash
ftllm server "$MODEL" "${COMMON[@]}" \
  --device cudapp=1:7 \
  --moe_device "{'cuda:0':6,'cuda:1':6,'numa':36}" \
  --moe_cuda_cache 0
```

主干第 0～5 层放在 GPU0，第 6～47 层放在 GPU1；专家第 0～5 层放在 GPU0，第 6～11 层放在 GPU1，后 36 层和 MTP 专家使用 NUMA。调整层数时，应同时调整主干与专家映射，确保 GPU 专家所在层与对应主干使用同一张卡。

固定层会为该层全部专家分配显存，动态缓存则按访问情况保留专家。两种方式的速度和显存需求不同。

<a id="parameters"></a>

## 常用参数

| 参数 | 用途 |
| --- | --- |
| `--device cuda` | 单卡主干 |
| `--device cudapp=2` | 双卡主干按层串行 |
| `--tp 2` | 双卡张量并行 |
| `--moe_device numa` | 使用 CPU 的 NUMA 后端运行专家 |
| `--moe_cuda_cache 0` / `8g` | 关闭专家缓存 / 每卡设置 8 GiB 预算 |
| `--moe_device_layers 36` | 最后 36 层专家使用指定的 MoE 设备 |
| `--ngram_device disk` | 按需从磁盘读取 PLE 表，降低主机常驻内存 |
| `--ngram_device cpu` | PLE 表驻留主机内存 |
| `--threads 28` | CPU 计算线程数 |
| `--mtp 0` / `3` | 关闭 MTP / 每步最多提出 3 个草稿 token |
| `--chunked_prefill_size 4096` | 每块最多处理 4096 个输入 token |
| `--kv_cache_limit 1g` | KV 缓存预算，按上下文与并发需求调整 |
| `--gpu_mem_ratio 0.95` | GPU 可用内存的使用比例 |
| `--prefix_cache true` | 复用重复文本前缀，适合多轮对话 |
| `--max_context_length` | 上下文长度上限，仍受模型和缓存容量限制 |
| `--enable_thinking true` | 开启模型思考模式 |

### MTP

MTP 需要模型目录中包含匹配的 MTP 权重，主干运行在 CUDA，专家可使用 CUDA 或 NUMA。`--mtp` 最大为 8；可以先使用 3，实际收益取决于任务和草稿接受情况。双卡 TP 示例使用贪婪采样，即 `--temperature 0 --top_k 1 --repeat_penalty 1`。

### 内存与上下文

主机内存紧张时使用 `--ngram_device disk`。显存不足时，先减小专家缓存或 GPU 驻留层数，再根据请求长度调整 KV 预算、分块大小和并发数。

增大 `--chunked_prefill_size` 可减少长输入的分块次数，同时增加临时显存需求。前缀缓存与专家缓存相互独立，分别复用文本前缀和专家权重。

<a id="api"></a>

## 调用服务

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-flash-next",
    "messages": [{"role": "user", "content": "请用三句话介绍你自己。"}],
    "max_tokens": 512,
    "temperature": 0,
    "stream": false
  }'
```

远程调用时，将 `127.0.0.1` 替换为服务器地址。若启动时设置了 `--api_key`，请求需增加 `Authorization: Bearer 你的密钥` 请求头。

也可以在浏览器中对话：

```bash
ftllm webui --api_base http://127.0.0.1:8080/v1
```

WebUI 连接已经启动的 API 服务。需要工具调用时，可在服务启动命令中加入 `--tool_call_parser auto`；API 请求使用标准 `tools` 字段。

<a id="performance"></a>

## 性能参考

以下为 2026-09-29 在上述双 22 GiB RTX 2080 Ti、EPYC 7452 机器上的 NVFP4 单请求实测，使用 28 线程和磁盘 PLE。Prefill 为 4096 输入、4096 分块；解码为 512 输入、512 输出，不含首 token 等待。测量时关闭前缀复用、CUDA Graph 和思考，各配置测一轮。

| 部署方式 | Prefill token/s | 普通解码 token/s | MTP=3 token/s |
| --- | ---: | ---: | ---: |
| 串行，缓存关闭 | 919.43 | 22.36 | 30.08 |
| 串行，每卡缓存 8 GiB | 902.92 | 26.13 | 39.84 |
| 串行，固定 12 层专家 | 960.99 | 23.99 | 32.92 |
| TP2，缓存关闭 | 1240.71 | 20.60 | 32.63 |
| TP2，每卡缓存 8 GiB | 1254.39 | 27.71 | 46.29 |
| TP2，固定 12 层专家 | 1321.08 | 22.57 | 37.82 |

这组数据中，TP2 + 每卡 8 GiB 专家缓存的解码较快，TP2 固定 12 层专家的长输入处理较快。实际速度随输入、输出内容和硬件而变化。
