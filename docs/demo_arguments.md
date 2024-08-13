# Fastllm Python Demo 参数说明

## 通用参数

模型相关配置，OpenAI API Server， WebUI, 对话Demo 均可使用

- **模型路径 (`-p, --path`)**: 指定模型的路径，可以是fastllm模型文件或Hugging Face模型文件夹。例如:
```bash
--path ~/Qwen2-7B-Instruct/ # 从~/Qwen2-7B-Instruct/中读取模型，这里的模型需要是从HuggingFace或ModelScope或其他网站下载的Hugging face格式的标准模型，暂不支持AWQ，GPTQ等格式
--path ~/model.flm # 从~/model.flm中读取模型，这里的模型是Fastllm格式的模型文件
```
- **推理类型 (`--atype`)**: 设置中间计算类型，可以指定为`float16`或`float32`
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
- **KV缓存最大使用量 (`--kv_cache_limit`)**: 设置KV缓存的最大使用量。若不使用此参数或设置为`auto`，框架会自动处理。手动设定示例如下：
```bash
--kv_cache_limit 5G   # 设置为5G
--kv_cache_limit 100M # 设置为100M
--kv_cache_limit 168K # 设置为168K
```
- **最大Batch数量 (`--max_batch`)**: 设置每次同时处理的请求数量。若不使用此参数，框架会自动处理
- **线程数量 (`-t, --threads`)**: 设置CPU线程数量，device设置为`cpu`时对速度有较大影响，设置为`cuda`时影响较小，主要影响读取模型的速度
- **自定义模型描述文件 (`--custom`)**: 指定描述自定义模型的Python文件。具体见 [自定义模型](custom.md)

## OpenAI API Server配置参数
- **模型名称 (`--model_name`)**: 指定部署的模型名称，API调用时会进行名称核验
- **API服务器主机地址 (`--host`)**: 设置API服务器的主机地址
- **API服务器端口号 (`--port`)**: 设置API服务器的端口号


## Web UI 配置参数
- **API服务器端口号 (`--port`)**: 设置WebUI的端口号
- **页面标题 (`--title`)**: 设置WebUI的页面标题