# Fastllm Python Demo 参数说明

## 通用参数

模型相关配置，OpenAI API Server， WebUI, 对话Demo 均可使用

- **模型路径 (`-p, --path`)**: 指定模型的路径，可以是fastllm模型文件或Hugging Face模型文件夹。例如:
```bash
--path ~/Qwen2-7B-Instruct/ # 从~/Qwen2-7B-Instruct/中读取模型，这里的模型需要是从HuggingFace或ModelScope或其他网站下载的Hugging face格式的标准模型，暂不支持AWQ，GPTQ等格式
--path ~/model.flm # 从~/model.flm中读取模型，这里的模型是Fastllm格式的模型文件
```
- **推理类型 (`--atype`)**: 设置中间计算类型，可以指定为`float16`或`float32`
- **图片 embedding 缓存 (`--image-embedding-cache`)**: Qwen3.5 native 图片路径的 CPU 缓存上限，默认 `512m`，例如 `--image-embedding-cache 1g`；设为 `0` 关闭。首次处理图片请求时才创建，按实际内容分配内存，不额外常驻 GPU 显存。命中后跳过 native 图片预处理和视觉编码；图片解码、传输及语言模型预填充仍可能执行。配置、作用范围与验证方式见[多模态缓存](multimodal-cache.md)。
- **权重类型 (`--dtype`)**: 指定模型的权重类型，适用于读取Hugging Face模型时。可以指定为`float16`, `int8`, `int4`, `int4g`(int4分组量化)，例如：
```bash
--dtype float16  # 使用float16权重（不量化）
--dtype int8     # 在线量化成int8权重
--dtype int4g128 # 在线量化成int4分组权重（128个权重一组）
--dtype int4g256 # 在线量化成int4分组权重（256个权重一组）
--dtype int4     # 在线量化成int4权重
```
- **使用的设备 (`--device`)**: 指定服务器使用的设备。可以指定为`cpu`或`cuda`或额外编译的其余device类型
- **CUDA Embedding (`--cuda_embedding`)**: 若带上此配置且device设置为`cuda`，那么会在cuda设备上进行embedding操作，这样速度会略微提升，显存占用也会提升，建议在显存非常充足的情况下使用

- **低显存模式 (`--low_gpu_mem`)**: 关闭 CUDA embedding 配置和 GPU token handoff；此开关优先于 `--cuda_embedding` 和 `FASTLLM_GPU_TOKEN_HANDOFF` 环境变量。CUDA Graph 仍按原有自动策略或 `FASTLLM_CUDA_GRAPH` 设置运行，Qwen3.5/3.6 普通 CUDA 推理会将 embedding 保留在 CPU，并可继续使用 CUDA Graph；其他模型的后端可能仍要求在 Graph 模式下使用 GPU embedding。该开关可与双卡 TP、MTP 或 DFlash2 配合使用（MTP 与 DFlash2 二选一）；当前 Qwen3.5 系列的 MTP 校验 Graph 不支持 CPU embedding，因此该校验阶段会回退到普通执行。该开关默认关闭，与 `--low` 低内存模式独立，不改变 KV 类型或上下文配额，也不保证避免 OOM。例如：`ftllm server /path/to/model --device cuda --low_gpu_mem`。
- **CUDA权重slab (`--cuda_slab`)**: 设置 CUDA 模型权重 slab 分配块大小，单位 MB，默认 `0` 表示关闭。对于将大量 MOE 专家权重放在 CUDA 上的场景，可以使用如 `--cuda_slab 1024` 减少小权重分别分配造成的显存碎片和页对齐开销
- **KV缓存类型 (`--kv_cache_dtype`)**: 默认 `auto`，可使用 `float16`、`bfloat16`、`fp8_e4m3`；Qwen3.5 架构的 CUDA 分页注意力还支持 `fp4`（别名 `nvfp4`、`fp4_e2m1`）。仅量化普通注意力 KV，不改变模型权重或线性注意力状态精度。硬件要求、内存格式与示例见[FP4 KV cache](fp4-kv-cache.md)。
- **KV缓存最大使用量 (`--kv_cache_limit`)**: 设置KV缓存的最大使用量。若不使用此参数或设置为`auto`，框架会自动处理。手动设定示例如下：
```bash
--kv_cache_limit 5G   # 设置为5G
--kv_cache_limit 100M # 设置为100M
--kv_cache_limit 168K # 设置为168K
```
- **单会话上下文 (`--max_context_length`, `--max-context-length`)**: 输入与输出合计上限；扩大模型声明窗口需要有效的 RoPE 配置，warmup 后容量不足会启动失败。
- **RoPE 扩展 (`--rope_scaling`, `--rope-scaling`)**: HF Qwen2/3/3.5 布局支持静态 YaRN；填 `yarn` 或 JSON。Qwen3 等模型须明确 `original_max_position_embeddings`。用法、支持范围和验证记录见[上下文扩展](context-length-extension-design.md)。
- **最大Batch数量 (`--max_batch`)**: 设置每次同时处理的请求数量。若不使用此参数，框架会自动处理
- **线程数量 (`-t, --threads`)**: 设置CPU线程数量，device设置为`cpu`时对速度有较大影响，设置为`cuda`时影响较小，主要影响读取模型的速度
- **自定义模型描述文件 (`--custom`)**: 指定描述自定义模型的Python文件。具体见 [自定义模型](custom.md)

## OpenAI API Server配置参数
- **模型名称 (`--model_name`)**: 指定部署的模型名称，API调用时会进行名称核验
- **API服务器主机地址 (`--host`)**: 设置API服务器的主机地址
- **API服务器端口号 (`--port`)**: 设置API服务器的端口号


## Web UI 配置参数
- **模型提示 (`model` / `-p, --path`)**: 可选；用于推导默认 API 模型名，省略时会从 `/v1/models` 自动发现
- **监听地址 (`--host`)**: 设置 WebUI 监听地址，默认为 `127.0.0.1`
- **页面端口 (`--port`)**: 设置 WebUI 监听端口，默认为 `1616`
- **页面标题 (`--title`)**: 设置 WebUI 的页面标题
- **模型 API (`--api_base`)**: 设置 OpenAI 兼容 API 的地址，默认为 `http://127.0.0.1:8080/v1`
- **API 模型名 (`--api_model`)**: 设置 API 请求使用的模型名；默认使用模型目录名，也可从 `/v1/models` 自动发现
- **API Key (`--api_key`)**: 设置 WebUI 后端访问模型 API 时使用的密钥；密钥不会下发到浏览器
- **API 请求超时 (`--api_timeout`)**: 设置单次模型 API 请求的超时时间（秒）
- **API 就绪超时 (`--api_ready_timeout`)**: 设置 WebUI 启动前等待模型 API 就绪的最长时间（秒）
- **最大输出 (`--max_token`)**: 设置最大输出 token 数；默认不设人工上限，小于等于 `0` 也表示不限制
- **会话目录 (`--history_dir`)**: 设置 SQLite 会话记录和上传附件的保存目录，默认为 `~/.fastllm/webui`
- **上传上限 (`--max_upload_mb`)**: 设置单个文档、图片或视频的最大体积（MiB）
- **数据行数上限 (`--data_max_rows`)**: 设置数据分析时每个数据表最多读取的行数，默认为 `200000`
- **代码上下文上限 (`--code_max_context_chars`)**: 设置代码项目智能体单次注入模型的最大源码字符数，默认为 `60000`
- **联网超时 (`--web_search_timeout`)**: 设置 Web Agent 单次搜索或网页读取的超时时间（秒）
- **智能体运行时 (`--agent_runtime`)**: 设置代码与联网智能体运行时，默认为 `pi`；`builtin` 使用原有单轮链路，`auto` 在 Pi 可用时优先使用 Pi
- **Pi 任务超时 (`--pi_agent_timeout`)**: 设置单次 Pi 智能体任务的总超时，默认为 `300` 秒
- **Pi 最大轮数 (`--pi_agent_max_turns`)**: 设置单次 Pi 智能体任务最多使用的模型轮数，默认为 `8`
- **Pi 上下文窗口 (`--pi_agent_context_window`)**: 设置传给 Pi 的模型上下文窗口，默认为 `40000` token

新版 WebUI 由 FastAPI 和原生前端提供，但不再在 WebUI 进程内加载模型。请先用 `ftllm server` 启动 OpenAI 兼容 API，再启动 WebUI：

```bash
ftllm server /path/to/model --host 127.0.0.1 --port 8080 --model_name my-model --device cuda
ftllm webui /path/to/model --host 127.0.0.1 --port 1616 \
    --api_base http://127.0.0.1:8080/v1 --api_model my-model
```

WebUI 启动前会等待 `/v1/models` 就绪，所有模型调用都通过 `/v1/chat/completions` 完成；代码分析和联网搜索默认由 Pi 进行多轮工具调用，其余模式保持直接调用。浏览器只访问 WebUI 后端，因此 API Key 不会暴露给页面；模型只由 API Server 加载一份。Pi 配套 wheel 的构建和安装方式见 [`tools/ftllm_agent_runtime/`](../tools/ftllm_agent_runtime/)，未安装时可传入 `--agent-runtime builtin`。WebUI 支持多会话持久化、关闭/低/中/高思考档位、图片和视频上传，以及无需搜索服务 API Key 的快速搜索和深度浏览。文件知识库可读取 PDF、DOCX、XLSX、PPTX、CSV、Markdown、文本和常见代码文件，支持会话内跨请求检索、跨文件问答及原文位置引用；扫描版 PDF 暂不包含 OCR。数据分析智能体可读取 CSV、TSV、JSON、JSONL 和 XLSX，通过受限的分组汇总、趋势、相关性及排序操作生成结论、PNG 图表和可编辑的 Excel 报告，不执行模型生成的任意代码。代码项目智能体会通过只读工具按需查看会话内的源码快照和项目配置文件，并按行号进行跨文件审查和问题定位；用户要求修改时可生成经过路径校验、可下载的 unified diff，但服务器不会执行源码、应用补丁或运行模型生成的命令。“生成 PPT”可以从主题或会话文件直接生成 4–20 页、可继续编辑的 16:9 PPTX，并支持科技蓝、商务红、自然绿和高级黑金风格。文件和联网资料都会视为不可信数据，并在回答下方保留来源。
