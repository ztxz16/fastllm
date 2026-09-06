# Fastllm Python Demo Parameter Explanation

## General Parameters

Configuration related to the model, OpenAI API Server, WebUI, and conversation demo can all use these parameters.

- **Model Path (`-p, --path`)**: Specifies the path to the model, which can be a fastllm model file or a Hugging Face model directory. For example:
  ```bash
  --path ~/Qwen2-7B-Instruct/ # Reads the model from ~/Qwen2-7B-Instruct/, where the model needs to be a standard Hugging Face format model downloaded from HuggingFace, ModelScope, or other websites. Formats like AWQ, GPTQ, etc., are currently not supported.
  --path ~/model.flm # Reads the model from ~/model.flm, where the model is a Fastllm format model file
  ```
- **Inference Type (`--atype`)**: Sets the intermediate computation type, which can be specified as `float16` or `float32`.
- **Weight Type (`--dtype`)**: Specifies the weight type of the model, applicable when reading Hugging Face models. It can be specified as `float16`, `int8`, `int4`, `int4g` (int4 grouped quantization), for example:
  ```bash
  --dtype float16  # Uses float16 weights (no quantization)
  --dtype int8     # Quantizes to int8 weights online
  --dtype int4g128 # Quantizes to int4 grouped weights online (128 weights per group)
  --dtype int4g256 # Quantizes to int4 grouped weights online (256 weights per group)
  --dtype int4     # Quantizes to int4 weights online
  ```
- **Device to Use (`--device`)**: Specifies the device used by the server. It can be specified as `cpu`, `cuda`, or other device types compiled additionally.
- **CUDA Embedding (`--cuda_embedding`)**: If this configuration is included and the device is set to `cuda`, embedding operations will be performed on the cuda device, slightly increasing speed and GPU memory usage. It is recommended to use this when there is ample GPU memory.

- **Low GPU Memory (`--low_gpu_mem`)**: Disables the CUDA embedding setting and GPU token handoff, taking precedence over `--cuda_embedding` and `FASTLLM_GPU_TOKEN_HANDOFF`. CUDA Graph still follows its existing automatic policy or `FASTLLM_CUDA_GRAPH`; ordinary Qwen3.5/3.6 CUDA inference keeps embedding on the CPU and can continue using CUDA Graph. Other model backends may still require GPU embedding in Graph mode. The flag can be combined with two-GPU TP, MTP, or DFlash2 (MTP and DFlash2 are mutually exclusive). The current Qwen3.5-family MTP verification Graph does not support CPU embedding, so that verification stage falls back to eager execution. This opt-in flag is independent of `--low`. It does not change the KV dtype or context budget, and does not guarantee that the model will avoid OOM. Example: `ftllm server /path/to/model --device cuda --low_gpu_mem`.
- **CUDA Weight Slab (`--cuda_slab`)**: Sets the CUDA model-weight slab size in MB. The default value `0` disables it. For MoE runs that place many expert weights on CUDA, a value such as `--cuda_slab 1024` can reduce fragmentation and page-alignment overhead from many small weight allocations.
- **KV Cache Maximum Usage (`--kv_cache_limit`)**: Sets the maximum usage for the KV cache. If this parameter is not used or set to `auto`, the framework will handle it automatically. Manual settings examples are as follows:
  ```bash
  --kv_cache_limit 5G   # Sets to 5G
  --kv_cache_limit 100M # Sets to 100M
  --kv_cache_limit 168K # Sets to 168K
  ```
- **Context window (`--max_context_length`, `--max-context-length`)**: Limits combined input and output tokens. Extending the declared window requires a supported RoPE configuration; insufficient calibrated KV capacity fails startup.
- **RoPE extension (`--rope_scaling`, `--rope-scaling`)**: Accepts `yarn` or JSON for supported HF Qwen2/3/3.5 layouts. Supply the original RoPE length explicitly when it is not known. See [context extension](context-length-extension-design.md) for configuration and validation limits.
- **Maximum Batch Size (`--max_batch`)**: Sets the number of requests processed simultaneously each time. If this parameter is not used, the framework will handle it automatically.
- **Number of Threads (`-t, --threads`)**: Sets the number of CPU threads, which significantly affects speed when the device is set to `cpu`, and has a smaller impact when set to `cuda`, mainly affecting the speed of model loading.
- **Custom Model Description File (`--custom`)**: Specifies the Python file describing the custom model. See [Custom Model](custom.md) for details.

## OpenAI API Server Configuration Parameters
- **Model Name (`--model_name`)**: Specifies the name of the deployed model, which will be verified during API calls.
- **API Server Host Address (`--host`)**: Sets the host address of the API server.
- **API Server Port Number (`--port`)**: Sets the port number of the API server.

## Web UI Configuration Parameters
- **Model hint (`model` / `-p, --path`)**: Optional; derives the default API model name. When omitted, the model is discovered from `/v1/models`.
- **Listening address (`--host`)**: Sets the WebUI listening address; the default is `127.0.0.1`.
- **Page port (`--port`)**: Sets the WebUI listening port; the default is `1616`.
- **Page title (`--title`)**: Sets the WebUI page title.
- **Model API (`--api_base`)**: Sets the OpenAI-compatible API URL; the default is `http://127.0.0.1:8080/v1`.
- **API model (`--api_model`)**: Sets the model name used in API requests; it defaults to the model-directory name and can also be discovered from `/v1/models`.
- **API key (`--api_key`)**: Sets the credential used by the WebUI backend; it is never sent to the browser.
- **API request timeout (`--api_timeout`)**: Sets the timeout in seconds for one model API request.
- **API readiness timeout (`--api_ready_timeout`)**: Sets how long the WebUI waits for the model API before it starts listening.
- **Maximum output (`--max_token`)**: Sets the output-token limit. By default there is no artificial limit; values less than or equal to `0` also mean unlimited.
- **History directory (`--history_dir`)**: Stores the SQLite conversation database and uploaded attachments; the default is `~/.fastllm/webui`.
- **Upload limit (`--max_upload_mb`)**: Sets the maximum size of each document, image, or video in MiB.
- **Data row limit (`--data_max_rows`)**: Sets the maximum number of rows read from each table during data analysis; the default is `200000`.
- **Code context limit (`--code_max_context_chars`)**: Sets the maximum source characters injected into one code-agent model request; the default is `60000`.
- **Web timeout (`--web_search_timeout`)**: Sets the timeout in seconds for each Web Agent search or page read.
- **Agent runtime (`--agent_runtime`)**: Selects the code and web agent runtime. The default is `pi`; `builtin` uses the original single-call paths, while `auto` prefers Pi when available.
- **Pi task timeout (`--pi_agent_timeout`)**: Sets the total timeout for one Pi agent task; the default is `300` seconds.
- **Pi maximum turns (`--pi_agent_max_turns`)**: Limits one Pi task to this many model turns; the default is `8`.
- **Pi context window (`--pi_agent_context_window`)**: Sets the model context window reported to Pi; the default is `40000` tokens.

The WebUI uses FastAPI and a native frontend, but no longer loads the model in its own process. Start an OpenAI-compatible API with `ftllm server`, then point the WebUI at it:

```bash
ftllm server /path/to/model --host 127.0.0.1 --port 8080 --model_name my-model --device cuda
ftllm webui /path/to/model --host 127.0.0.1 --port 1616 \
    --api_base http://127.0.0.1:8080/v1 --api_model my-model
```

Before listening, the WebUI waits for `/v1/models`, and every model call uses `/v1/chat/completions`. Code analysis and web search use Pi for multi-turn tool calls by default; the other modes continue to call the model directly. The browser talks only to the WebUI backend, so API credentials are not exposed and the model is loaded exactly once by the API Server. Build and install the companion wheel from [`tools/ftllm_agent_runtime/`](../tools/ftllm_agent_runtime/), or pass `--agent-runtime builtin` when it is absent. The WebUI supports persistent conversations, off/low/medium/high reasoning levels, image and video uploads, and quick search or deep browsing without a search-provider API key. Its file knowledge base reads PDF, DOCX, XLSX, PPTX, CSV, Markdown, text, and common source-code files, with cross-request retrieval, cross-file Q&A, and source locations inside each conversation; scanned PDFs do not yet include OCR. The data-analysis agent reads CSV, TSV, JSON, JSONL, and XLSX, executes only validated grouping, trend, correlation, and sorting operations, and produces findings, PNG charts, and editable Excel reports without running arbitrary model-generated code. The code-project agent uses read-only tools to inspect source snapshots and project-configuration files from the conversation for line-referenced cross-file review and issue localization. When changes are requested, it can produce a path-validated downloadable unified diff, but the server never executes source, applies patches, or runs model-generated commands. "Create PPT" generates an editable 4–20 slide, 16:9 PPTX from a topic or conversation files in tech, business, nature, or premium styling. File and web results are treated as untrusted data and retained as sources under the answer.
