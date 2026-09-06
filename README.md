# FastLLM

[English](README_EN.md) · [快速开始](#快速开始) · [模型部署指南](#模型部署指南) · [Benchmark](docs/benchmark.md) · [常用参数](#常用参数) · [版本日志](docs/version.md)

FastLLM 是一个面向本地运行和服务部署的高性能大模型推理引擎。核心运行时使用 C++ 实现，不依赖 PyTorch，支持稠密模型与 MoE 模型，并提供 CUDA、ROCm、CPU、NUMA、磁盘混合推理以及多卡张量并行能力。

项目同时提供命令行对话、终端部署向导、WebUI、性能测试工具，以及兼容 OpenAI Chat Completions、OpenAI Responses 和 Anthropic Messages 的 API 服务。

## 核心能力

- **新模型快速适配**：当前主线覆盖 Qwen4-Exp / Qwen3.8-Flash-Next、Qwen3.5/3.6/3.8、DeepSeek-V4、Kimi-K3、GLM-5.3-Flash、Dots3-Note 等模型。
- **大规模 MoE 混合部署**：可将普通层和专家层分别放在 CUDA、CPU、NUMA 或磁盘上，也可按比例组合多种设备，适合显存有限但主机内存或 SSD 容量充足的机器。
- **多卡与高吞吐服务**：支持张量并行、奇数卡数量、动态批处理、流式输出、Paged KV Cache、前缀缓存、分块 Prefill 和 CUDA Graph。
- **投机解码**：为匹配的模型提供 MTP、内置或外部 DSpark，以及 DFlash2 等投机解码路径。
- **多种精度与格式**：支持 Hugging Face Safetensors、FastLLM 导出格式、AWQ 和部分 GGUF；可按模型与硬件使用 FP16、BF16、FP8、NVFP4、MXFP4、INT4、K-Quant 等路径。
- **完整服务接口**：支持思考内容分离、工具调用、流式响应、缓存命中统计、服务端采样参数和启动进度事件。
- **可扩展后端**：内置 CPU/CUDA/ROCm 算子，并提供 Triton 可选算子、自定义 Python 模型图和其他加速器后端的源码接入能力。

> 不同模型、量化格式和硬件后端支持的算子并不完全相同。正式部署前请用目标模型和目标硬件验证精度、显存占用及吞吐。

## 当前模型能力

下面列出当前开发主线重点，不再把早期模型作为首页介绍内容。

| 模型系列 | 当前重点能力 |
| --- | --- |
| Qwen | Qwen4-Exp / Qwen3.8-Flash-Next 文本解码、QSA、PLE n-gram 和 CPU/CUDA/NUMA 混合推理；Qwen3.8-Flash-Next MTP；Qwen3.5/3.6/3.8 MTP 和 DFlash2 |
| DeepSeek | DeepSeek-V4 / V4-Flash、稀疏注意力、内置 DSpark、多卡 CUDA 与 CPU/NUMA 混合 MoE |
| Kimi | Kimi-K3、KDA/MLA、外部 DSpark，以及 CUDA、NUMA、CPU/GPU 专家和磁盘专家 |
| GLM | GLM-5 DSA、GLM-5.3-Flash KDA 与分页缓存、GLM-5.2 量化 KV-B CPU 推理 |
| 其他 | Dots3-Note、Laguna、HY-V3、Step3.5/3.7、MiniMax-M2、Gemma4 等 |

Qwen4-Exp / Qwen3.8-Flash-Next 当前不加载视觉权重；Qwen3.8-Flash-Next 可通过 `--mtp` 按需加载 MTP 权重并启用推测解码。早期模型的兼容信息仍可在[支持模型列表](docs/models.md)中查询；最新适配和限制以[版本日志](docs/version.md)为准。

## 快速开始

### 安装

建议在独立的 Python 虚拟环境中安装。预编译包适用于以下常见环境：

| 环境 | 安装命令 | 说明 |
| --- | --- | --- |
| Linux + NVIDIA GPU | `python -m pip install -U ftllm` | 包含 Python 接口和常用 CUDA 运行时依赖；驱动需要与 CUDA 运行时兼容 |
| Windows + NVIDIA GPU | `python -m pip install -U ftllm` | 如果首次安装提示缺少 DLL，请先安装下方的 Windows 依赖包 |
| Linux + AMD GPU | [ROCm 安装与编译](docs/rocm.md) | 按显卡架构选择构建与安装方式 |
| CPU-only、特殊架构或其他加速器 | [源码安装](#源码安装) | 可按实际平台选择 CMake 后端 |

Windows 首次安装所需的依赖包：

~~~bash
python -m pip install https://www.modelscope.cn/models/huangyuyang/fastllmdepend-windows/resolve/master/ftllmdepend-0.0.0.2-py3-none-win_amd64.whl
python -m pip install -U ftllm
~~~

如果 Conda 环境出现动态库冲突，可尝试使用 `venv` 创建干净环境。安装或加载失败时先查看 [FAQ](docs/faq.md)。

### 验证安装

下面使用体积较小的 Qwen3-0.6B 做安装冒烟测试；它只是便于快速下载的测试模型，不代表当前模型主线。

~~~bash
ftllm run Qwen/Qwen3-0.6B
~~~

最常用的部署入口是 API Server：

~~~bash
# API Server，默认监听 0.0.0.0:8080
ftllm server Qwen/Qwen3-0.6B

# 命令行对话
ftllm run Qwen/Qwen3-0.6B

# WebUI，连接上面的 API Server，默认监听 127.0.0.1:1616
ftllm webui --api_base http://127.0.0.1:8080/v1

# 浏览器部署启动器；无参数时启动并自动打开本地管理页面
ftllm
ftllm launch  # 等价写法

# 终端部署向导
ftllm tui

# 性能测试
ftllm bench Qwen/Qwen3-0.6B \
  --device cuda --input_tokens 512 --output_tokens 128 --batch 4
~~~

`ftllm`（或 `ftllm launch`）默认仅监听 `127.0.0.1:8000`，并在服务就绪后自动打开浏览器；使用 `ftllm launch --no-browser` 可以关闭自动打开。页面可以从 ModelScope 下载模型、保存启动配置、预览命令，并选择托管 `ftllm server` 或聊天 `ftllm webui`。新增启动项选择本地模型后，会根据模型结构、权重规模以及本机 GPU、内存和 NUMA 拓扑自动推荐 TP、MoE 混合推理与 N-gram 存储参数，也可以手动重新分析或清空可选推理参数。界面支持简体中文和英文，会优先使用上次选择的语言，否则跟随浏览器语言；`ftllm launch` 的终端日志固定使用英文。需要从局域网访问时使用 `ftllm launch --host 0.0.0.0`；终端和 Launcher 页面随后会列出本机、局域网以及网卡上直接配置的公网访问地址（若有）。公网访问还需要放行主机防火墙及云安全组，经过 NAT 时还需配置端口映射；Launcher 不会自动探测 NAT 的公网地址。非本机监听使用未加密 HTTP，请仅在可信网络中使用。它与终端向导共用配置文件；关闭 Launcher 时，由它托管的下载和模型进程也会停止。使用 `ftllm launch --help` 查看其他选项。

API Server 就绪后，点击「打开工作室」即可在 Launcher 内容区直接使用聊天、历史会话、Markdown、附件、思考过程和智能体功能。模型管理导航始终保留，可随时切换到启动、下载、日志和硬件页面，返回「工作室」后继续当前会话。Launcher 与独立的 `ftllm webui` 共用聊天组件和后端，界面配色、尺寸及语言会适配 Launcher。组件自动连接当前模型并使用启动配置中的 API Key，无需另开 WebUI 服务或端口。会话沿用 WebUI 的本地存储，刷新页面后仍然保留；停止或切换模型时会取消正在运行的 WebUI 任务并清理旧组件。

WebUI 不会在自身进程内加载模型，请先启动 OpenAI 兼容 API Server。WebUI 的可选 `model` 位置参数只用于推导 API 模型名；省略时会从 `/v1/models` 自动发现。

代码分析和联网搜索默认使用 Pi 智能体运行时。Linux x86-64 用户可按
[`tools/ftllm_agent_runtime/`](tools/ftllm_agent_runtime/) 中的说明构建并安装配套 wheel；
该 wheel 已包含 Pi，不需要 Node.js、npm 或 Bun。尚未安装时可通过
`--agent-runtime builtin` 使用原有单轮链路。

Launcher 会自动使用已安装的 Pi 运行时；「新建 Agent」可选择工作目录。通过 `ftllm launch --agent-workspace-root /path/to/projects` 指定可选目录的根路径，默认为用户主目录。Launcher 的目录 Agent 默认启用，本机和远程监听均可使用，例如 `ftllm launch --host 0.0.0.0 --agent-workspace-root /path/to/projects`。使用 `--disable-workspace-agent` 可关闭目录 Agent，同时禁止目录浏览、新建目录 Agent 及继续执行已保存的目录 Agent 任务；普通对话仍可使用。目录 Agent 可修改文件和执行命令，请仅对可信用户开放。运行时缺失或目录 Agent 被关闭时，界面会显示原因。

对于 `run`、`server` 和 `export`，`model` 位置参数既可以是 Hugging Face 仓库 ID，也可以是本地 Hugging Face 模型目录、FastLLM 模型文件或配置文件。例如：

~~~bash
ftllm server /data/models/my-model --device cuda
~~~

### 调用 API

启动一个带固定服务名的本地模型：

~~~bash
ftllm server /data/models/my-model \
  --model_name local-model \
  --host 0.0.0.0 --port 8080 \
  --api_key local-key
~~~

通过 OpenAI Chat Completions 接口调用：

~~~bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer local-key" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "local-model",
    "messages": [{"role": "user", "content": "你好，请介绍一下 FastLLM。"}],
    "stream": false
  }'
~~~

服务还提供 OpenAI Responses API 和 Anthropic Messages API。模型支持时，可使用思考内容分离、工具调用和图文输入等能力。

## 模型部署指南

不同模型在注意力结构、MoE 布局、投机解码和量化格式上差异很大。选择目标模型后，请先阅读对应部署指南，再使用后面的通用配置作为补充。

| 模型 | 部署指南 | 推荐配置入口 | Benchmark |
| --- | --- | --- | --- |
| Qwen4-Exp / Qwen3.8-Flash-Next | [Qwen4-Exp 指南](docs/qwen4.md) | PLE、QSA、CUDA/NUMA、`--ngram_device disk` | [Qwen4 Benchmark](docs/benchmarks/qwen4_exp.md) |
| Qwen3.5 / Qwen3.6 / Qwen3.8 | [Qwen 当前模型指南](docs/qwen3.md) | 单卡、TP、混合 MoE、MTP、DFlash2 | [Qwen3 Benchmark](docs/benchmarks/qwen3.md) |
| DeepSeek-V4 / V4-Flash | [DeepSeek-V4 指南](docs/deepseek.md) | CUDA + NUMA、磁盘专家、TP、内置 DSpark | [DeepSeek-V4 Benchmark](docs/benchmarks/deepseek_v4.md) |
| Kimi-K3 | [Kimi-K3 指南](docs/kimi_k3.md) | KDA/MLA、混合专家、磁盘专家、外部 DSpark | [Kimi-K3 Benchmark](docs/benchmarks/kimi_k3.md) |
| Dots3-Note | [Dots3-Note 指南](docs/dots3_note.md) | DSA、长上下文、CUDA + CPU/NUMA | [Dots3-Note Benchmark](docs/benchmarks/dots3_note.md) |
| GLM-5 / GLM-5.3-Flash | [GLM-5 指南](docs/glm5.md) | DSA/KDA、分页缓存、NUMA、量化 KV-B CPU | [GLM-5 Benchmark](docs/benchmarks/glm5.md) |
| Laguna | [Laguna 指南](docs/laguna.md) | 多卡 TP、混合 MoE、NVFP4、INT4_GROUP32 | [Laguna Benchmark](docs/benchmarks/laguna.md) |

[Benchmark 索引](docs/benchmark.md) 按模型分别记录测试硬件、完整启动命令、TTFT、Prefill、Decode 和并发吞吐。没有仓库实测的数据会明确标为“待实测”，不会从其他设备或模型外推。

## 典型部署配置

### 多卡张量并行

`--tp 0,1` 显式使用 0、1 号 GPU；`--tp 2` 表示使用前两张可见 GPU；`--tp auto` 自动使用检测到的 GPU。

~~~bash
ftllm server /data/models/my-model \
  --device cuda --tp 0,1 \
  --max_batch 16 --gpu_mem_ratio 0.9
~~~

### GPU + NUMA 混合 MoE

~~~bash
ftllm server /data/models/my-moe-model \
  --device cuda --moe_device numa \
  --chunked_prefill_size 8192
~~~

内存不足时还可以把少量专家层放到磁盘，例如 `--moe_device "{'cuda':1,'numa':8,'disk':1}"`。磁盘路径依赖 SSD 随机读取性能，详细配置参见[混合推理指南](docs/mixforward.md)。

### 长上下文与前缀缓存

`--max_context_length`（别名 `--max-context-length`）设置单会话输入与输出合计上限。扩大模型声明窗口时，还需要有效的 `--rope_scaling`（别名 `--rope-scaling`），接受 `yarn` 或 JSON；只缩小窗口时可省略 RoPE 参数。配置在加载时生效，不修改模型的 `config.json`。

例如，将 Qwen3-0.6B 扩展到 65536 token，并开启前缀缓存。其 YaRN 原始长度为 32768，应显式指定，不能用配置声明的 40960 代替：

~~~bash
ftllm server /data/models/Qwen3-0.6B \
  --device cuda --max_batch 1 --tokens 65536 \
  --max_context_length 65536 \
  --rope_scaling '{"rope_type":"yarn","factor":2,"original_max_position_embeddings":32768}' \
  --chunked_prefill_size 8192 \
  --prefix_cache true
~~~

Qwen3.8-27B-FP8 可以使用已知原始长度的 `yarn` 简写。下面配置双卡、FP4 KV 和 1,000,000 token 的目标窗口，解析得到 original=262144、factor=4：

~~~bash
ftllm server /data/models/Qwen3.8-27B-FP8 \
  --tp 2 --kv_cache_dtype fp4 \
  --max_context_length 1000000 --rope_scaling yarn
~~~

`--tokens` 是所有会话共享的 KV 池容量，未设置时自动预算。显式目标超过 RoPE 覆盖范围或 warmup 校准容量会启动失败；`/v1/models` 返回实际窗口、模型原声明和用户目标。上述 1M 命令需要足够显存，本机双 24GB 的测试配置无法容纳，完整 1M 输入尚未实测。

当前扩展接入 HF Qwen2、Qwen3、Qwen3.5 布局，以及基于 Qwen3.5 架构的 Qwen3.8；Launcher 高级参数中的「RoPE 扩展」使用相同配置。GGUF、FLM 和自定义 GraphLLM 仅设置长度时保留旧的只缩小行为，暂不支持新的 RoPE 扩展。更多参数、适配范围和验证结果见[上下文扩展说明](docs/context-length-extension-design.md)。

### 投机解码

以下功能只适用于结构和 checkpoint 匹配的模型：

~~~bash
# 以 Qwen3.5 为例使用内置 MTP，每轮最多配置 8 个 draft token
ftllm server /data/models/qwen3.5 --mtp 4

# 给不含 MTP 权重的 Qwen3.5 GGUF 挂载独立 MTP 模块
ftllm server /data/models/qwen3.5.gguf \
  --device cuda --cuda_embedding \
  --draft /data/models/qwen3.5-fp8/mtp.safetensors \
  --draft_tokens 5

# DeepSeek-V4 内置 DSpark
ftllm server /data/models/deepseek-v4 --dspark 7

# Qwen3.8 + 独立 DFlash2 draft checkpoint
ftllm server /data/models/qwen3.8 \
  --tp 2 \
  --draft /data/models/qwen3.8-dflash2 \
  --draft_tokens 7
~~~

`--draft` 会根据 draft checkpoint 自动识别 MTP、DFlash2 或 DSpark。独立 MTP 当前用于 Qwen3.5 GGUF，路径可指向包含 `config.json` 和 `mtp.safetensors` 的目录，也可直接指向 `mtp.safetensors`；省略 `--draft_tokens` 时默认使用 5。DFlash2 的 `--draft_tokens` 表示实际 draft token 数，不包含 anchor token。

DFlash2 的完整配置和验证结果见 [Qwen3.8 DFlash2 文档](docs/dflash2_qwen38_27b_tp2_20260819.md)。

### Qwen4 PLE 磁盘模式

Qwen4-Exp / Qwen3.8-Flash-Next 的 PLE 表较大。主机内存不足时可按需从 checkpoint 读取：

~~~bash
ftllm server /data/models/qwen4-exp \
  --device cuda --moe_device numa \
  --ngram_device disk
~~~

磁盘模式会降低常驻内存，但增加随机 I/O，建议使用高速 SSD。更多限制见 [Qwen4-Exp 文档](docs/qwen4_exp.md)。

## 常用参数

CLI 会持续演进，`ftllm <command> --help` 是当前安装版本的最终依据。下面列出部署中最常用的参数。

### 模型、设备与精度

<a id="3-如何设定运行设备"></a>

| 参数 | 说明 |
| --- | --- |
| `model` / `-p, --path` | Hugging Face 仓库 ID、本地 HF 目录、FastLLM 模型文件或配置文件 |
| `--device` | 主计算设备，常用值为 `cpu`、`cuda`、`numa` |
| `--tp` | CUDA 张量并行设备；支持 `0,1`、`2` 或 `auto` |
| `--moe_device` | MoE 专家层设备，可使用 `cpu`、`cuda`、`numa`、`disk` 或按比例组合 |
| `--moe_device_layers` | 仅让最后 N 个 MoE 层使用 `--moe_device`；`-1` 表示全部 |
| `-t, --threads` | CPU/NUMA 线程数；`-1` 表示自动选择 |
| `--dtype` | 加载 HF 权重时的权重类型；默认 `auto`，已量化模型通常不应覆盖 |
| `--moe_dtype` | 单独设置 MoE 权重类型 |
| `--atype` / `--moe_atype` | 设置普通层和 MoE 层的激活类型 |
| `--kv_cache_dtype` | KV Cache 类型：`auto`、`float16`、`bfloat16`、`fp8_e4m3` 或 `fp4`，需模型与后端支持 |
| `--dtype_config` | 动态量化配置文件，参见[动态量化说明](docs/dtype_config.md) |
| `--triton` | 启用可用的 Triton CUDA 算子 |

### 显存、上下文与调度

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--gpu_mem_ratio` | `0.9` | 可用于模型与缓存的 GPU 显存比例 |
| `--kv_cache_limit` | `auto` | KV Cache 最大使用量 |
| `--tokens` | 自动 | 用于计算 Paged KV Cache 容量的总 token 数 |
| `--page_size` | 后端决定 | Paged KV Cache 每页 token 数；多卡默认通常为 16 |
| `--max_batch` | 自动 | 每轮最多同时推理的请求数 |
| `--max_context_length` / `--max-context-length` | 自动 | 单会话输入与输出合计上限；HF 显式目标需通过 RoPE 与 KV 容量检查 |
| `--rope_scaling` / `--rope-scaling` | 沿用模型配置 | RoPE 扩展，接受 `yarn` 或 JSON；仅对已适配的 HF 模型布局生效 |
| `--chunked_prefill_size` | 关闭/模型决定 | 分块 Prefill 的切片大小，例如 `8192` |
| `--prefix_cache` | 模型/环境决定 | 是否开启前缀缓存，使用 `true` 或 `false` |
| `--cuda_slab` | `0` | CUDA 权重 slab 大小（MB）；`0` 为关闭 |
| `--ngram_device` | `cpu` | Qwen4 PLE 表放在 `cpu` 或 `disk` |

### 解码、模板与工具调用

<a id="工具调用"></a>

| 参数 | 说明 |
| --- | --- |
| `--enable_thinking` | 控制模型的思考模板开关，需要模型支持 |
| `--mtp` | 支持 MTP 的模型每轮生成的 draft token 数，`0` 关闭，当前最大为 8 |
| `--dspark` | 启用模型内置 DSpark，并设置每轮 draft token 数 |
| `--draft` / `--draft_model_path` | 外部 MTP/DSpark/DFlash draft checkpoint；根据配置自动识别算法，MTP 可直接指定 `mtp.safetensors` |
| `--draft_tokens` | 每轮最多使用的 draft token 数；未指定时读取 draft 配置 |
| `--tool_call_parser` | 工具调用解析器；默认 `auto` |
| `--chat_template` | 自定义 Jinja chat template 文件 |
| `--cache_dir` | 在线模型的本地缓存目录 |
| `--ori` | 读取部分 GGUF 时指定原模型配置和 tokenizer 目录 |

### API Server

| 参数 | 默认值 | 说明 |
| --- | ---: | --- |
| `--host` | `0.0.0.0` | 监听地址 |
| `--port` | `8080` | Server 端口；WebUI 默认端口为 1616 |
| `--model_name` | 自动 | API 中校验和返回的部署名称 |
| `--api_key` | 空 | 非空时开启 Bearer API Key 校验 |
| `--temperature` / `--top_p` / `--top_k` | 模型默认 | 覆盖服务端默认采样参数 |
| `--repeat_penalty` | 模型默认 | 覆盖重复惩罚参数，也支持 `--repetition_penalty` |
| `--hide_input` | 关闭 | 不在服务日志中显示请求内容 |
| `--startup-progress` | `off` | 设置为 `ndjson` 时向 stderr 输出模型加载与就绪事件 |

查看完整参数：

~~~bash
ftllm --help
ftllm server --help
ftllm bench --help
ftllm download --help
~~~

## 模型格式、下载与导出

### 支持的输入格式

- Hugging Face 原始 Safetensors 权重，包括模型自带的 FP16、BF16 或 FP8 权重。
- 已量化的 AWQ 模型。
- FastLLM 导出的定精度或动态量化模型。
- 部分 GGUF 格式；需要通过 `--ori` 指定原模型的配置和 tokenizer 目录。

量化格式是否可用取决于模型结构、设备和对应 kernel。首次部署建议保留 `--dtype auto`；对于已经量化的 checkpoint，不要再次指定在线量化类型。

### 下载模型

~~~bash
ftllm download <repo-id> --local-dir /data/models/model-name
~~~

只下载配置和 tokenizer 文件时可以排除权重：

~~~bash
ftllm download <repo-id> \
  --exclude "*safetensors*" \
  --local-dir /data/models/model-config
~~~

### 导出模型

在线量化会增加每次启动的加载时间。可以预先导出 FastLLM 格式：

~~~bash
ftllm export /data/models/source-model \
  -o /data/models/source-model-int4 \
  --dtype int4 -t 16
~~~

MoE 模型可以分别设置普通层和专家层精度：

~~~bash
ftllm export /data/models/source-moe \
  -o /data/models/source-moe-mixed \
  --dtype float16 --moe_dtype int4 -t 16
~~~

动态量化配置见[动态量化说明](docs/dtype_config.md)。

## 源码安装

源码构建需要 C++17 编译器、Make 和 CMake；建议 GCC/G++ 9.4+、CMake 3.23+。Linux NUMA 构建通常还需要 `libnuma-dev`。CUDA 构建请预先安装兼容的 CUDA Toolkit 和 NCCL。

~~~bash
# Ubuntu/Debian 基础依赖
sudo apt-get install -y build-essential cmake libnuma-dev

# NVIDIA CUDA
bash install.sh -DUSE_CUDA=ON \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"

# 指定 CUDA 架构，例如 Ada 使用 89
bash install.sh -DUSE_CUDA=ON -DCUDA_ARCH=89 \
  -DCMAKE_CUDA_COMPILER="$(command -v nvcc)"

# CPU-only
bash install.sh
~~~

更多平台说明：

- [ROCm 编译与 wheel 打包](docs/rocm.md)
- [TFACC 平台](docs/tfacc.md)
- [示例程序、Android 和其他平台](example/README.md)
- [编译与运行 FAQ](docs/faq.md)

## 文档导航

| 主题 | 文档 |
| --- | --- |
| 发布说明 | [稳定版日志](docs/version.md) · [Nightly 使用](docs/nightly.md) · [Nightly 日志](docs/nightly_changelog.md) |
| 模型部署 | [Qwen4-Exp](docs/qwen4.md) · [Qwen3.5/3.6/3.8](docs/qwen3.md) · [DeepSeek-V4](docs/deepseek.md) · [Kimi-K3](docs/kimi_k3.md) · [Dots3-Note](docs/dots3_note.md) · [GLM-5](docs/glm5.md) · [Laguna](docs/laguna.md) |
| 混合推理 | [GPU、NUMA 与磁盘混合部署](docs/mixforward.md) |
| 性能与验证 | [按模型查看 Benchmark](docs/benchmark.md) |
| 量化 | [动态量化配置](docs/dtype_config.md) |
| 扩展开发 | [Python 自定义模型](docs/custom.md) · [自定义算子](docs/custom_op.md) |
| 平台与排错 | [ROCm](docs/rocm.md) · [TFACC](docs/tfacc.md) · [FAQ](docs/faq.md) |

## 社区与贡献

欢迎通过 [GitHub Issues](https://github.com/ztxz16/fastllm/issues) 报告问题，通过 [Pull Requests](https://github.com/ztxz16/fastllm/pulls) 参与开发。

部署交流 QQ 群：`831641348`

用户交流群（使用、部署）：

<img src="docs/wechat_group0.jpg" width="220" alt="FastLLM 用户交流群二维码">

社区开发群（贡献、开发讨论）：

<img src="docs/develop-group.png" width="220" alt="FastLLM 社区开发群二维码">

项目采用 [Apache License 2.0](LICENSE)。

## 参考代码和文章

FastLLM 的实现参考或使用了以下开源项目与文章中的思路或代码：

- [PyTorch](https://github.com/pytorch/pytorch)：底层算子实现思路。
- [Transformers](https://github.com/huggingface/transformers)：模型结构与参考实现。
- [llama.cpp](https://github.com/ggml-org/llama.cpp) 和 [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)：GGUF 量化格式与 kernel。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：Attention、MLA 等算子。
- [TurboMind / LMDeploy 的 GEMM 内核](https://github.com/InternLM/lmdeploy/tree/main/src/turbomind/kernels/gemm)：仓内 `third_party/turbomind` 的 SM70 s884（`SM70_MMA_884` / HMMA 8x8x4）核心由其源码移植，并在其上接入 AWQ INT4、块缩放 FP8 与 NVFP4 Linear。
- [1Cat-vLLM 的 SM70 TurboMind 适配](https://github.com/1CatAI/1Cat-vLLM/tree/main/csrc/sm70_turbomind)：AWQ 接入及 FP8/NVFP4 类型、布局和小批量 tactic 的参考来源；FastLLM 侧另行实现了无 Torch 的原始指针桥接、模型权重格式转换、非对齐 padding、回退和调度。
- [KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/csrc/ktransformers_ext/cpu_backend/backend.cpp)：MoE 动态线程调度；另见[思路介绍](https://zhuanlan.zhihu.com/p/1900318746402329329)。
- [Lvllm](https://github.com/guqiong96/Lvllm/blob/main/csrc/lk/moe.cpp)：NUMA MoE 动态调度。
- [FreeToken](https://github.com/FlashML-org/FreeToken)：CUDA 专家缓存及相关混合推理优化的设计参考，包括 GPU 端路由与 LRU 缓存管理、主机专家权重按需回填，以及 CUDA Graph 兼容的执行流程。具体实现与扩展方式见 [CUDA 专家缓存说明](docs/cuda-expert-cache.md)。
- [vLLM](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers)：工具调用解析。
- [json11](https://github.com/dropbox/json11)：JSON 构造与解析。

感谢所有开源贡献者。如发现遗漏的代码来源或引用，请通过 Issue 告知。
