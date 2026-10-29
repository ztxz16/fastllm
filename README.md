# FastLLM

[English](README_EN.md) · [快速开始](#快速开始) · [`--auto` 调节指南](docs/auto.md) · [文档](docs/README.md) · [Benchmark](docs/benchmark.md) · [版本日志](docs/version.md)

FastLLM 是用于本地运行和服务部署的大模型推理引擎，核心运行时使用 C++，不依赖 PyTorch。支持稠密与 MoE 模型、GPU/CPU/NUMA/磁盘混合推理，以及多卡部署。

- **自动部署**：使用 `--auto` 规划设备、内存、专家缓存和上下文，也可覆盖常用参数。
- **浏览器启动器**：下载模型、保存配置、查看日志，直接在工作室中对话或使用 Agent。
- **兼容 API**：提供 OpenAI Chat Completions、Responses 和 Anthropic Messages 接口。
- **多种权重格式**：支持 Safetensors、FastLLM、AWQ 和已适配的 GGUF，具体范围见[模型指南](#模型部署指南)。

## 快速开始

### 安装

Linux / Windows + NVIDIA GPU：

```sh
python -m pip install -U ftllm
```

<details>
<summary>Windows 首次安装依赖</summary>

```sh
python -m pip install https://www.modelscope.cn/models/huangyuyang/fastllmdepend-windows/resolve/master/ftllmdepend-0.0.0.2-py3-none-win_amd64.whl
```

</details>

AMD GPU 见 [ROCm 安装](docs/rocm.md)；CPU-only 和其他平台见[源码安装](docs/deployment.md#源码安装)。安装问题见 [FAQ](docs/faq.md)。

### 浏览器启动

单机多副本可使用独立的 `ftllm deploy` 组件，复用原生服务参数，提供后台进程管理和
流式请求路由；现有 `ftllm serve/server` 用法不变。配置与命令见 [多副本部署](docs/deploy.md)。

~~~bash
# API Server，默认监听 0.0.0.0:8080
ftllm server Qwen/Qwen3-0.6B

# 命令行对话
ftllm run Qwen/Qwen3-0.6B

# WebUI，连接上面的 API Server，默认监听 127.0.0.1:1616
ftllm webui --api_base http://127.0.0.1:8080/v1
~~~

浏览器部署启动器可直接运行：

```sh
ftllm
```

选择模型 → 自动配置 → 保存并启动。需要调整时展开“资源限制”或“模型参数”，其余留给自动规划。启动后可直接打开工作室对话；详细操作见[启动器指南](docs/launcher.md)。

### 命令行启动

将路径替换为本地模型目录或已支持的 GGUF 文件，也可使用 Hugging Face 仓库 ID：

```sh
ftllm server /path/to/model --auto
```

`--auto` 当前适用于 Linux NVIDIA CUDA 上已支持的模型；其他平台或未适配模型见[手动部署](docs/deployment.md)。默认服务地址为 `http://127.0.0.1:8080`。

只需追加要调整的参数，例如限制为 32K 上下文、最多 2 个并发请求：

```sh
ftllm server /path/to/model --auto --max_context_length 32768 --max_batch 2
```

显存/内存预算、专家缓存、分块、推测解码和快速 Prefill 的设置见 [`--auto` 调节指南](docs/auto.md)。命令行对话使用 `ftllm run`，终端向导使用 `ftllm tui`，性能测试使用 `ftllm bench`。

### 调用 API

启动时可以指定服务名称和 API Key：

```sh
ftllm server /path/to/model --auto --model_name local-model --api_key local-key
```

```sh
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Authorization: Bearer local-key" \
  -H "Content-Type: application/json" \
  -d '{"model":"local-model","messages":[{"role":"user","content":"你好"}],"stream":false}'
```

也可运行 `ftllm webui --api_base http://127.0.0.1:8080/v1 --api_key local-key` 打开独立聊天页面。思考、工具调用和多模态能力依模型而定。

## 模型部署指南

| 模型 | 部署文档 |
| --- | --- |
| Qwen4-Exp / Qwen3.8-Flash-Next | [Flash-Next 用户手册](docs/qwen3.8-flash-next/README.md) |
| Qwen3.5 / Qwen3.6 / Qwen3.8 | [Qwen 指南](docs/qwen3.md) |
| DeepSeek-V4 / V4-Flash | [DeepSeek-V4 指南](docs/deepseek.md) |
| DeepSeek-V4.1-Flash 1017 | [DeepSeek-V4.1 指南](docs/deepseek_v41.md) |
| Kimi-K3 | [Kimi-K3 指南](docs/kimi_k3.md) |
| GLM-5 / GLM-5.3-Flash | [GLM-5 指南](docs/glm5.md) |
| Dots3-Note / Laguna | [Dots3-Note](docs/dots3_note.md) · [Laguna](docs/laguna.md) |
| HY-Nomas | [世界模型与 HTTP Demo 部署](docs/hy_nomas.md) |

其他模型见[支持列表](docs/models.md)。让 Agent 帮忙部署时，可先提供[模型与参数速查](docs/model-deployment-agent.md)。实测速度与硬件、请求和量化格式有关，见 [Benchmark](docs/benchmark.md)。

## 常用参数

日常使用看 [`--auto` 参数与影响](docs/auto.md)；需要手动指定设备或模型专属选项时，看[完整参数参考](docs/deployment.md#常用参数)。当前安装版本以 `ftllm server --help` 为准。

<a id="3-如何设定运行设备"></a>
<a id="工具调用"></a>

[设备设置](docs/deployment.md#模型设备与精度) · [解码、模板与工具调用](docs/deployment.md#解码模板与工具调用) · [模型下载与导出](docs/deployment.md#模型格式下载与导出)

## 源码安装

C++ / CUDA / CPU 构建步骤见[源码安装说明](docs/deployment.md#源码安装)，平台排错见 [FAQ](docs/faq.md)。

## 文档导航

- [全部文档](docs/README.md) · [启动器与工作室](docs/launcher.md) · [Agent 接入](docs/launcher-agents.md)
- [混合推理](docs/mixforward.md) · [专家显存缓存](docs/cuda-expert-cache.md) · [上下文扩展](docs/context-length-extension.md)
- [开发与验证](docs/development-validation.md) · [自定义模型](docs/custom.md) · [自定义算子](docs/custom_op.md)
- [稳定版日志](docs/version.md) · [Nightly](docs/nightly.md)

## 社区与贡献

欢迎通过 [Issues](https://github.com/ztxz16/fastllm/issues) 报告问题，通过 [Pull Requests](https://github.com/ztxz16/fastllm/pulls) 参与开发。部署交流 QQ 群：`831641348`。

<details>
<summary>微信用户群与开发群</summary>

用户交流（使用、部署）：

<img src="docs/wechat_group0.jpg" width="220" alt="FastLLM 用户交流群二维码">

社区开发（贡献、开发讨论）：

<img src="docs/develop-group.png" width="220" alt="FastLLM 社区开发群二维码">

</details>

项目采用 [Apache License 2.0](LICENSE)。

## 参考代码和文章

<details>
<summary>开源项目与引用</summary>

FastLLM 的实现参考或使用了以下开源项目与文章中的思路或代码：

- [PyTorch](https://github.com/pytorch/pytorch)：底层算子实现思路。
- [Transformers](https://github.com/huggingface/transformers)：模型结构与参考实现。
- [llama.cpp](https://github.com/ggml-org/llama.cpp) 和 [ik_llama.cpp](https://github.com/ikawrakow/ik_llama.cpp)：GGUF 量化格式与 kernel。
- [Strata](https://github.com/Niko1221/Strata)：一些算子。
- [KVMem / kvmem-llama.cpp](https://github.com/kvmem/kvmem-llama.cpp)：分层 KV 缓存与内容检索的设计参考，包括固定 GPU 工作集、主机 KV 备份和 pre-RoPE mean-K 检索。FastLLM 接入说明见 [KVMem 文档](docs/kvmem.md)。
- [FlashInfer](https://github.com/flashinfer-ai/flashinfer)：Attention、MLA 等算子。
- [TurboMind / LMDeploy 的 GEMM 内核](https://github.com/InternLM/lmdeploy/tree/main/src/turbomind/kernels/gemm)：仓内 `third_party/turbomind` 的 SM70 s884（`SM70_MMA_884` / HMMA 8x8x4）核心由其源码移植，并在其上接入 AWQ INT4、块缩放 FP8 与 NVFP4 Linear。
- [1Cat-vLLM 的 SM70 TurboMind 适配](https://github.com/1CatAI/1Cat-vLLM/tree/main/csrc/sm70_turbomind)：AWQ 接入及 FP8/NVFP4 类型、布局和小批量 tactic 的参考来源；FastLLM 侧另行实现了无 Torch 的原始指针桥接、模型权重格式转换、非对齐 padding、回退和调度。
- [KTransformers](https://github.com/kvcache-ai/ktransformers/blob/main/csrc/ktransformers_ext/cpu_backend/backend.cpp)：MoE 动态线程调度；另见[思路介绍](https://zhuanlan.zhihu.com/p/1900318746402329329)。
- [Lvllm](https://github.com/guqiong96/Lvllm/blob/main/csrc/lk/moe.cpp)：NUMA MoE 动态调度。
- [FreeToken](https://github.com/FlashML-org/FreeToken)：CUDA 专家缓存及相关混合推理优化的设计参考，包括 GPU 端路由与 LRU 缓存管理、主机专家权重按需回填，以及 CUDA Graph 兼容的执行流程。具体实现与扩展方式见 [CUDA 专家缓存说明](docs/cuda-expert-cache.md)。
- [vLLM](https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai/tool_parsers)：工具调用解析。
- [json11](https://github.com/dropbox/json11)：JSON 构造与解析。

感谢所有开源贡献者。如发现遗漏的代码来源或引用，请通过 Issue 告知。

</details>
