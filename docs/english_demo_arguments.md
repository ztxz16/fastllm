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
- **KV Cache Maximum Usage (`--kv_cache_limit`)**: Sets the maximum usage for the KV cache. If this parameter is not used or set to `auto`, the framework will handle it automatically. Manual settings examples are as follows:
  ```bash
  --kv_cache_limit 5G   # Sets to 5G
  --kv_cache_limit 100M # Sets to 100M
  --kv_cache_limit 168K # Sets to 168K
  ```
- **Maximum Batch Size (`--max_batch`)**: Sets the number of requests processed simultaneously each time. If this parameter is not used, the framework will handle it automatically.
- **Number of Threads (`-t, --threads`)**: Sets the number of CPU threads, which significantly affects speed when the device is set to `cpu`, and has a smaller impact when set to `cuda`, mainly affecting the speed of model loading.
- **Custom Model Description File (`--custom`)**: Specifies the Python file describing the custom model. See [Custom Model](custom.md) for details.

## OpenAI API Server Configuration Parameters
- **Model Name (`--model_name`)**: Specifies the name of the deployed model, which will be verified during API calls.
- **API Server Host Address (`--host`)**: Sets the host address of the API server.
- **API Server Port Number (`--port`)**: Sets the port number of the API server.

## Web UI Configuration Parameters
- **API Server Port Number (`--port`)**: Sets the port number for the WebUI.
- **Page Title (`--title`)**: Sets the page title for the WebUI.