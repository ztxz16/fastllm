# fastllm-nightly 使用文档

`fastllm-nightly` 是 fastllm 的开发预览版本，包含最新的功能和改进，适合希望提前体验新特性的用户。

> **注意：** 目前 nightly 版本仅支持 **Linux + Nvidia GPU** 环境，暂不支持 AMD GPU、Windows 等其他平台。

查看 [Nightly 更新日志](nightly_changelog.md) 了解每次更新的详细变更记录。

安装 nightly 版本：

```
pip install ftllm-nightly -U
```

## 与稳定版的区别

`ftllm-nightly` 和 `ftllm` 的使用方式完全一致，安装后同样通过 `ftllm` 命令使用。两者不能同时安装，安装 nightly 版本前请先卸载稳定版：

```
pip uninstall ftllm
pip install ftllm-nightly -U
```

如需切换回稳定版：

```
pip uninstall ftllm-nightly
pip install ftllm -U
```

## 参数说明

以下是 `ftllm` 命令支持的全部参数，按功能分类介绍。

如果您是第一次使用 ftllm，建议直接使用基本的启动命令，所有参数都会自动设置：

```
ftllm run Qwen/Qwen3-0.6B
ftllm server Qwen/Qwen3-0.6B
```

如果您希望进一步调优，请参照以下参数说明。

### 模型加载参数

- `model`:
  - **描述**: 位置参数，指定要加载的模型。可以是 Huggingface 模型名、本地 HF 模型文件夹、fastllm 模型文件或 JSON 配置文件。
  - **示例**: `ftllm run Qwen/Qwen3-0.6B`、`ftllm run /mnt/Qwen3-0.6B`

- `-p` 或 `--path`:
  - **描述**: 与 `model` 位置参数等价的可选写法，指定模型路径。
  - **示例**: `ftllm run -p /mnt/Qwen3-0.6B`

- `--custom`:
  - **描述**: 指定描述自定义模型结构的 python 文件。文件中需定义 `__model__` 变量作为模型图。
  - **示例**: `ftllm run Qwen/Qwen3-0.6B --custom my_model.py`

- `--lora`:
  - **描述**: 指定 LoRA 权重路径，加载模型时会同时加载 LoRA 适配器。
  - **示例**: `ftllm run Qwen/Qwen3-0.6B --lora /mnt/my_lora`

- `--cache_dir`:
  - **描述**: 指定在线 Huggingface 模型的本地缓存目录。不设定时默认缓存到系统缓存路径（Linux 下为 `~/.cache/fastllm`）。
  - **示例**: `ftllm run Qwen/Qwen3-0.6B --cache_dir /mnt/model_cache`

- `--ori`:
  - **描述**: 指定原始模型权重路径，读取 GGUF 格式文件时需要配合使用。GGUF 文件不包含完整的模型配置，需要从原始模型中读取。
  - **示例**:

``` sh
ftllm server DeepSeek-V3-0324-Q4_K_M-00001-of-00009.gguf --ori DeepSeek-V3
```

### 数据类型参数

当启动的模型为浮点精度（`BF16`、`FP16`、`FP8`）时，可以通过以下参数来设置推理精度。若不设定，ftllm 会使用模型中设定的精度进行推理。若使用的模型已经是量化好的模型（如 AWQ、Fastllm 导出的量化模型等），建议不指定这些参数。

- `--dtype`:
  - **描述**: 指定模型的权重数据类型（读取 HF 模型时有效）。
  - **可选值**: `int4g` `int4` `int8` `fp8` `float16` `bfloat16` 或其他支持的数据类型。
  - **默认值**: `auto`（自动根据模型配置选择）
  - **示例**: `--dtype int4`

- `--atype`:
  - **描述**: 指定推理时的激活类型。
  - **可选值**: `float32`、`float16`、`bfloat16`
  - **默认值**: `auto`（当 device 为 `cpu`/`numa`/`tfacc` 时自动设为 `float32`）
  - **示例**: `--atype bfloat16`

- `--dtype_config`:
  - **描述**: 指定权重类型配置文件，用于对不同层设置不同的量化精度（动态量化）。
  - **示例**: `--dtype_config my_dtype_config.json`

- `--moe_dtype`:
  - **描述**: 指定 MOE（Mixture of Experts）层的权重数据类型（读取 HF 模型时有效）。如果模型不是 MOE 结构，此参数不会生效。
  - **可选值**: `int4g` `int4` `int8` `fp8` `float16` `bfloat16` 或其他支持的数据类型。
  - **示例**: `--moe_dtype int4`

- `--moe_atype`:
  - **描述**: 指定 MOE 层的激活类型。
  - **可选值**: `float32`、`float16`、`bfloat16`
  - **示例**: `--moe_atype bfloat16`

### 设备与线程参数

可以通过以下参数来设定执行推理的设备和线程配置。

- `--device`:
  - **描述**: 指定模型运行的计算设备。
  - **示例**: `--device cpu`、`--device cuda`
  - **常用值说明**:
    - `cpu` 使用 CPU 推理
    - `cuda` 使用 GPU 推理
    - `numa` 使用多路 NUMA 节点加速推理，在多 CPU 的机器才会有提升。使用 numa 加速时，强烈建议关闭系统 numa 平衡（ubuntu 中可使用命令 `sudo sysctl -w kernel.numa_balancing=0`）
    - `multicuda` 使用多设备张量并行推理
      - **使用多显卡**: `--device multicuda:0,1`
      - **使用多显卡+CPU**: `--device multicuda:0,cpu`
      - **按比例使用多显卡+CPU**: `--device multicuda:0:4,1:5,cpu:1`（`cuda:0` 计算 4/10，`cuda:1` 计算 5/10，`cpu` 计算 1/10）
  - **串行计算**: 一些场景下可以指定不同的 device 串行执行，例如：
    - `--device "{'cuda:0':3,'cuda:1':2}"`: `3/5` 的层运行在 `cuda:0`，`2/5` 的层运行在 `cuda:1`
    - **简写**: `--device cudapp=N` 表示 N 卡均匀串行，例如 `--device cudapp=4` 等价于 `--device "{'cuda:0':1,'cuda:1':1,'cuda:2':1,'cuda:3':1}"`
    - **简写**: `--device cudapp=1:2:3` 表示三卡按 1:2:3 比例串行

- `--moe_device`:
  - **描述**: 指定 MOE 层的计算设备。一般和 `--device` 指定为不同的设备来实现混合推理。如果模型不是 MOE 结构，此参数不会生效。
  - **示例**: `--moe_device cpu`、`--moe_device numa`
  - **常用值说明**:
    - `cpu` 使用 CPU 推理
    - `numa` 使用多路 NUMA 节点加速推理
    - `cuda` 使用 GPU 推理（MOE 层需要大量显存，一般不建议指定为 `cuda`）
  - **说明**: 例如 `--device cuda --moe_device cpu` 实现 MOE 模型的单卡+CPU 混合推理；`--device cuda --moe_device numa` 实现单卡+多 NUMA 节点加速推理。

若不设定 `--device` 和 `--moe_device`，会使用默认配置：

| 模型类型 | device | moe_device |
|-------:|--------|------------:|
| 稠密模型 | cuda   |   不生效    |
| MOE模型  | cuda   |   numa    |

- `-t` 或 `--threads`:
  - **描述**: 设置使用的 CPU 线程数。不设定时自动检测
  - **示例**: `-t 30`

- `--cuda_embedding`:
  - **描述**: 开关参数，在 CUDA 上进行 embedding 计算。启用后可以减少 CPU-GPU 之间的数据传输。
  - **示例**: `ftllm server Qwen/Qwen3-0.6B --cuda_embedding`

- `--cuda_shared_expert` 或 `--cuda_se`:
  - **描述**: 指定 MOE 中的共享专家是否在 CUDA 上执行，默认为 `true`。设为 `false` 可以节省显存。
  - **示例**: `--cuda_se false`

- `--enable_amx` 或 `--amx`:
  - **描述**: 是否开启 Intel AMX 加速，默认为 `false`。需要 CPU 支持 AMX 指令集。
  - **示例**: `--amx true`


### 推理与批处理参数

- `--max_batch`:
  - **描述**: 设置每次最多同时推理的请求数量。不设定时由系统自动决定。
  - **示例**: `--max_batch 32`

- `--chunked_prefill_size`:
  - **描述**: 设置分块 prefill 的切片大小（首块与后续块相同）。启用分块 prefill 可以在长上下文场景下降低首 token 延迟，同时提高并发时的吞吐量。
  - **示例**: `--chunked_prefill_size 8192`

- `--moe_experts`:
  - **描述**: 指定 MOE 层使用的专家数。不设定则根据模型配置自动设定。减少专家数可以提高推理速度，但可能降低推理准确度。
  - **示例**: `--moe_experts 6`

### 缓存与显存参数

- `--tokens`:
  - **描述**: 设置总的 token 数量，用于计算 paged cache 的最大页数。不设定时由系统根据可用显存自动计算。
  - **示例**: `--tokens 65536`

- `--page_size`:
  - **描述**: 设置 paged cache 每页的大小（以 token 数计）。
  - **默认值**: `128`
  - **示例**: `--page_size 256`

- `--gpu_mem_ratio`:
  - **描述**: 设置 GPU 显存使用比例。例如 `0.9` 表示最多使用 90% 的显存，剩余部分留给系统和其他程序。
  - **默认值**: `0.9`
  - **示例**: `--gpu_mem_ratio 0.8`

### 对话与模板参数

- `--enable_thinking`:
  - **描述**: 是否开启硬思考（thinking）开关。启用后模型会在回答前先进行思考推理过程（需要模型支持，如 Qwen3、Glm4Moe 等）。对于支持的模型会自动启用。
  - **示例**: `--enable_thinking true`、`--enable_thinking false`

- `--tool_call_parser`:
  - **描述**: 指定工具调用（function calling）的解析器类型。
  - **默认值**: `auto`（根据模型自动选择）
  - **示例**: `--tool_call_parser auto`

### 服务部署参数

以下参数在使用 `ftllm server` 部署 API 服务时可用。

- `--model_name`:
  - **描述**: 设置部署的模型名称。调用 API 时会进行名称核验。不设定时，如果使用在线模型会自动使用模型的 repo id。
  - **示例**: `--model_name my-model`

- `--host`:
  - **描述**: 指定 API server 的监听地址。
  - **默认值**: `0.0.0.0`
  - **示例**: `--host 127.0.0.1`

- `--port`:
  - **描述**: 指定 API server 的监听端口。
  - **默认值**: `8080`
  - **示例**: `--port 8000`

- `--api_key`:
  - **描述**: 设定 API Key。设置后，客户端调用接口时需要在请求头中携带此密钥进行认证。
  - **示例**: `--api_key sk-xxxxxxxx`

- `--think`:
  - **描述**: 强制在输出中添加 `<think>` 标签。当模型输出中丢失了思考标签时可以使用此参数修复。
  - **默认值**: `false`
  - **示例**: `--think true`

- `--hide_input`:
  - **描述**: 开关参数，隐藏日志中的请求信息。在生产环境中使用可以保护用户隐私。
  - **示例**: `ftllm server Qwen/Qwen3-0.6B --hide_input`

- `--dev_mode`:
  - **描述**: 开关参数，启用开发模式。开启后可以通过 API 获取当前对话列表并主动停止推理任务，方便调试。
  - **示例**: `ftllm server Qwen/Qwen3-0.6B --dev_mode`

## 注意事项

- nightly 版本更新频率较高，可能包含未充分测试的功能
- 如遇到问题，可以尝试更新到最新 nightly 版本或切换回稳定版
- 欢迎在 [GitHub Issues](https://github.com/ztxz16/fastllm/issues) 中反馈 nightly 版本的问题

## 版本查看

```
ftllm -v
```
