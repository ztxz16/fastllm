# 图片与视频生成

[English](image-video_en.md) · [返回首页](../README.md) · [启动器与工作室](launcher.md)

FastLLM 提供 MiniMax-H3 / Pika H3 Flex 音视频生成、MiniMax-H3-Image 图片生成与编辑，以及 Qwen-Image-2.1 图片生成与编辑。模型计算使用原生 C++ / CUDA，不需要 PyTorch 或 Diffusers 推理环境；启动器和 CLI 使用 Python 管理进程与请求。

## 支持范围与入口

| 模型 | 能力 | 推荐入口 | 其他入口 |
| --- | --- | --- | --- |
| MiniMax-H3 | 文生音视频、首帧/尾帧/首尾帧、有序图片/视频/音频参考 | `ftllm launch` 的视频后端 | `minimax_h3_server` / `minimax_h3_generate` |
| Pika H3 Flex | H3 基座加官方适配器，固定六步音视频生成 | 同一视频后端，选择 Pika 目录 | 同一 H3 原生程序，自动识别适配器 |
| MiniMax-H3-Image | 六步文生图、单图编辑、多图参考合成 | `ftllm launch` 的图像后端 | `minimax_h3_image_server` / `minimax_h3_image_generate` |
| Qwen-Image-2.1 | 文生图、单图和多图编辑 | `ftllm qwen-image server` + `/demo` | `ftllm qwen-image generate` |

`ftllm server` 是语言模型服务入口。生成模型请使用上表对应后端；图像理解模型的 `image_url` 输入也不等于图片生成能力。Qwen-Image 当前没有独立的 Launcher 启动项，使用其 CLI 或独立 Demo。

HY-Nomas、LingBot World 的世界模型部署另见 [HY-Nomas](hy_nomas.md) 和 [LingBot World](lingbot_world.md)。

## 安装与准备

本指南面向 **Linux x86_64 + NVIDIA GPU**。发布包的媒体后端使用 SM80+ 架构；FP8、NVFP4 还有下文的硬件要求。语言模型后端支持的旧 GPU / 其他平台范围不能直接套用到媒体后端。

```sh
python -m pip install -U ftllm
ftllm launch
```

1. 使用含媒体组件的 CUDA wheel 或绿色包；1.9.2 发布包包含上述三个原生后端、Demo 和 API 资源。绿色包直接运行其启动入口即可，具体安装方式见[发布页](https://github.com/ztxz16/fastllm/releases)。
2. 模型权重需另行下载。保留完整原始目录，包括文本/视觉编码器、VAE、tokenizer、配置和必要的采样组件；量化 DiT 文件通常不能独立运行。
3. 使用支持相应 CUDA 运行时的 NVIDIA 驱动。多卡依赖 NCCL；视频编码和部分参考图解码需要 `ffmpeg` / `ffprobe`。绿色包带这些运行时组件；使用 wheel 时检查当前环境的依赖与可执行文件，不需要安装完整 PyTorch。

下文直接调用 H3 原生程序的示例面向已安装 wheel 的 Python 环境，先找到安装目录：

```sh
FTLLM_PACKAGE="$(python -c 'from pathlib import Path; import ftllm; print(Path(ftllm.__file__).resolve().parent)')"
```

源码构建分别见 [H3](../tools/minimax_h3/README.md#构建)、[H3 Image](../tools/minimax_h3_image/README.md#build-and-run)、[Qwen-Image](../tools/qwen_image/README.md#构建)。源码安装需构建对应目标，单独构建 LLM 库不会生成这些程序。

## 视频生成：MiniMax-H3 / Pika H3 Flex

### 启动器

在 **启动服务 → 增加启动项 → 模型类型 / 后端** 选择视频（MiniMax H3 / Pika H3），填写模型路径、端口、输出目录和可用 GPU，保存并启动。发布包中的可执行文件路径可留空。

- H3：选择完整 MiniMax-H3 目录。
- Pika：选择包含 `fl2va/manifest.json`、`fl2va/adapter.safetensors`、`ref2va/manifest.json`、`ref2va/adapter.safetensors` 的官方适配器目录。仍需匹配的完整 H3 基座；在启动项中指定基座路径，或通过 Pika 目录内的 `base_model` 软链接关联。基座版本与适配器要求见 [Pika 说明](../tools/minimax_h3/README.md#pika-h3-flex)。
- 就绪后打开 **工作室 → 视频生成** 或独立 **视频 Demo**，选择文字、首尾帧或参考素材模式。Pika 固定 6 步，H3 默认 50 步。

首尾帧与参考素材模式不能混用。参考素材模式需要 `transformer_ref/`；独立音频参考需搭配图片或视频。完整参数、素材数量和时长限制见 [H3 HTTP 文档](../tools/minimax_h3/README.md#http-与-demo)。

### 原生服务与 API

```sh
"$FTLLM_PACKAGE/minimax_h3_server" \
  --model /path/to/MiniMax-H3 --devices 0 \
  --host 127.0.0.1 --port 8009 --output ./h3-output \
  --demo "$FTLLM_PACKAGE/h3_assets/demo.html"
```

Pika 使用同一程序，将模型路径改为 Pika 目录，并在需要时追加 `--base-model /path/to/MiniMax-H3`。打开 `http://127.0.0.1:8009/demo`；`/health` 查看初始化状态，`/v1/models` 查看实际模型 ID。

提交 H3 视频任务：

```sh
curl http://127.0.0.1:8009/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3","prompt":"A cat plays piano. Soft piano music fills the room.","size":"832x480","seconds":"5","seed":42,"num_inference_steps":50}'
```

Pika 请求使用 `"model":"pika-h3-flex"`、`"num_inference_steps":6`。提交后返回 HTTP 202 和任务 `id`；将下例的 `VIDEO_ID` 替换为该 ID，完成后再下载：

```sh
curl http://127.0.0.1:8009/v1/videos/VIDEO_ID
curl http://127.0.0.1:8009/v1/videos/VIDEO_ID/content -o video.mp4
# 取消尚未完成的任务
curl -X POST http://127.0.0.1:8009/v1/videos/VIDEO_ID/cancel
```

首尾帧可用 multipart 上传 `input_reference` / `first_frame` 和 `last_frame`；参考素材使用重复的 `reference` 字段并指定媒体类型。不要传服务端文件路径或远程 URL。`seconds` 会向上对齐为支持的帧数，实际值以返回结果为准。

## 图片生成：MiniMax-H3-Image

### 启动器与模型路径

选择 **图像（MiniMax H3 Image）** 后端，填写模型路径、端口、输出目录和 GPU。就绪后打开 **工作室 → 图像生成** 或 **图像 Demo**，支持文生图、上传参考图、排序、编辑、进度、取消和下载。

BF16 使用完整原始目录。FP8 / INT8 / NVFP4 / GGUF 使用已支持布局的具体量化文件，并保留完整原始 H3 Image 基座；建议在量化目录内设置 `base_model` 目录或软链接，避免自动查找出现歧义。视频 H3 基座不能代替 H3 Image 基座。

```sh
"$FTLLM_PACKAGE/minimax_h3_image_server" \
  --model /path/to/MiniMax-H3-Image --devices 0 \
  --host 127.0.0.1 --port 8010 --output ./h3-image-output \
  --demo "$FTLLM_PACKAGE/h3_image_assets/demo.html"
```

### 生成与编辑 API

推荐异步任务接口，避免长时间等待同一个 HTTP 响应：

```sh
curl http://127.0.0.1:8010/v1/images/jobs \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3-image","prompt":"一只坐在窗边的橘猫，柔和的晨光","size":"2048x2048","seed":42,"n":1,"num_inference_steps":6}'

# 将 IMAGE_ID 替换为提交返回的 id
curl http://127.0.0.1:8010/v1/images/jobs/IMAGE_ID
curl http://127.0.0.1:8010/v1/images/jobs/IMAGE_ID/content -o image.png
```

同步文生图使用 `POST /v1/images/generations`。同步参考图编辑示例：

```sh
curl http://127.0.0.1:8010/v1/images/edits \
  -F 'image[]=@subject.png' -F 'image[]=@scene.png' \
  --form-string 'prompt=Place the subject from <Picture 1> into the scene from <Picture 2>.' \
  -F size=2048x2048 -F seed=42 -F response_format=url
```

返回的 `data[0].url` 是相对服务地址的下载路径。异步编辑将同一表单发到 `/v1/images/jobs`；取消使用 `POST /v1/images/jobs/{id}/cancel`。参考顺序影响 `<Picture 1>` 等标签，不能混用 `image` 和 `image[]` 字段。固定六步 PDD，不要套用其他扩散模型的步数或 CFG 参数。详见 [H3 Image 文档](../tools/minimax_h3_image/README.md)。

## 图片生成：Qwen-Image-2.1

### 启动服务或直接导出图片

```sh
ftllm qwen-image server --model /path/to/Qwen-Image-2.1 \
  --devices 0 --host 127.0.0.1 --port 8008 --output ./qwen-image-output
```

打开 `http://127.0.0.1:8008/demo`，可上传参考图、设置种子、查看进度、取消、下载和复制 API 命令。无需手动指定 wheel 内的 Demo 路径。

```sh
ftllm qwen-image generate --model /path/to/Qwen-Image-2.1 --devices 0 \
  --prompt 'A red apple on a wooden table.' \
  --width 1024 --height 1024 --steps 40 --seed 1234 --output apple.png
```

增加 `--image reference.png --reference-resolution 512` 可编辑参考图；多图重复 `--image`。完整参数见 `ftllm qwen-image server --help` / `generate --help`。

### HTTP API

```sh
curl http://127.0.0.1:8008/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","prompt":"一只坐在窗边的橘猫","size":"1024x1024","n":1,"seed":1234,"num_inference_steps":40,"response_format":"url"}'

curl http://127.0.0.1:8008/v1/images/edits \
  -F model=qwen-image-2.1 -F 'image[]=@apple.png' \
  --form-string 'prompt=Change the apple to green. Keep the background.' \
  -F size=512x512 -F reference_resolution=512 \
  -F seed=1234 -F num_inference_steps=40 -F response_format=url
```

也支持 `/v1/images/jobs` 的异步提交、查询与取消。当前 `n=1`，宽高各 64–2048 且为 32 的倍数，步数 2–100、默认 40。HTTP 参考图要求 PNG；Demo 可先将 JPEG/WebP 转为 PNG。文本/视觉上下文最多 4096 token，过多或过大的参考图需降低 `reference_resolution` 或减少图片。暂不支持 mask、负面提示 / CFG、LoRA 和远程 URL 输入。

### Qwen 量化目录

保留原 BF16 目录，另外建立量化目录，复用文本/视觉编码器、VAE、processor、scheduler、`model_index.json` 和 `transformer/config.json`：

- FP8 / INT8：在新目录的 `transformer/` 中将已支持的量化文件命名或软链接为 `diffusion_pytorch_model.safetensors`。
- GGUF：在 `transformer/` 内只放一份所选 GGUF，不同时放另一份 Safetensors DiT。
- `--model` 始终传新模型根目录，自动识别精度。FP8 / INT8 / GGUF 当前均为实验支持，部分精度回归仍超出既有误差线；没有接入 Qwen-Image NVFP4。

具体权重布局、来源和验证范围见 [Qwen-Image 文档](../tools/qwen_image/README.md)。

## 量化与多卡

| 模型 | 已接入的 DiT 权重 | 多卡方式与限制 |
| --- | --- | --- |
| H3 / Pika | BF16、已适配 FP8、INT8 ConvRot、混合 NVFP4 / INT8 / BF16、GGUF Q4_K_M | `--parallel-mode auto\|tp\|sp`；量化 auto 使用 SP；INT8 / GGUF 的 TP 支持 2/4 卡，8 卡使用 SP |
| H3 Image | BF16、已适配 FP8、INT8、NVFP4、GGUF | 同一并行参数；量化 auto 使用 SP；FP8 / NVFP4 TP 为 2/4/8，INT8 / GGUF TP 为 2/4 |
| Qwen-Image-2.1 | BF16；实验性 FP8 / INT8 / GGUF | DiT TP 为 1/2/4/8；编码器和 VAE 在首卡执行 |

上述媒体后端的多卡实机验证主要覆盖双卡，不能把支持的维度切分数当作所有硬件上的性能或精度保证。TP 切分权重，SP 主要切分序列计算并保留权重副本；需要降低每卡权重占用时，应明确选择支持的 TP 模式。

FP8 需要 SM89+ 的 FP8 Tensor Core，H3 系列 NVFP4 需要 SM120、CUDA 12.8+ 和相应构建。INT8 / GGUF 仍包含 BF16 模型阶段。量化只替换适配的投影，辅助组件保留基座精度；支持某种格式不代表任意导出布局都能使用，也不保证比 BF16 更快或输出相同。H3 Image 的低位 GGUF 尤其需要检查画质。

H3 / Pika 量化目录使用 `h3_quantization.json` 将 `base_model`、`fl2va`、`ref2va` 与可选 `pika` 明确绑定；不能把视频和图像 DiT 混用。完整清单示例见 [H3 量化视频权重](../tools/minimax_h3/README.md#量化视频权重)。

多卡示例：

```sh
# 进程内的 0、1 对应这里选出的物理 GPU 2、3
CUDA_VISIBLE_DEVICES=2,3 ftllm qwen-image server \
  --model /path/to/Qwen-Image-2.1 --devices 0,1 --port 8008
```

H3 原生程序同样接受 `--devices 0,1`，启动器会自动映射勾选的 GPU。H3 / H3 Image 支持 `--gpu-memory-limit GiB`（每卡）和 `--host-memory-limit GiB`（服务共享）进行常驻/逐层加载规划；Qwen-Image 根据可用显存与请求自动选择常驻或阶段卸载。模型预算不是整个进程显存的硬隔离，仍需为 CUDA、通信、激活与媒体编解码留出空间。

## 工作室中让 LLM 调用视频模型

同时启动一个支持工具调用的 LLM 服务和 H3 / Pika 视频服务。在工作室安装 Pi 运行时并创建**工作目录 Agent**，选择对话模型，随后可以要求：

> 生成一段小猫走过花园的 5 秒视频，保存到工作目录的 videos 文件夹。

Launcher 提供当前生成服务目录和临时调用工具，Agent 创建任务、轮询并下载结果，视频会显示为会话附件。普通聊天不会执行本地命令；当前这条 Agent 联动面向视频，图片可直接在图像工作室或独立 Demo 操作。不同服务使用不同端口；共用 GPU 时需要共同规划显存。详见[启动器指南](launcher.md)。

## 状态、输出与排错

- 先查看 `/health` 和服务日志，确认加载完成；H3 的 `dit_precision`、`dit_parallel_mode` 等字段可核对实际后端。完整请求字段以各服务的 `/openapi.json` 为准。
- 需要认证时，原生服务添加 `--api-key KEY`，请求添加 `Authorization: Bearer KEY`。Launcher 的控制令牌和模型服务 API Key 用途不同，浏览器应从 `ftllm launch` 输出的完整地址进入。
- 服务按任务队列生成；支持多卡不代表同时生成多个独立任务。当前运行加排队总数最多 4，超过返回 429。取消在阶段或采样边界生效，不会立即中断正在运行的 CUDA 算子。
- 图片返回 PNG，视频返回 MP4；同步 Images API 的 URL 可能是相对路径，需拼接服务地址。输出保存在指定目录，调用方管理磁盘空间；进程中断后未完成任务不自动续跑。
- 找不到原生程序时检查 wheel / 构建目标；缺少 `libnccl.so.2` 时检查发布包运行时或当前环境的 NCCL 安装，补齐后再启动服务。
- 量化文件提示缺少基座时，补齐匹配的原始组件；不要用视频、图片或其他模型版本的组件互相替代。

模型级数值边界和回归入口见 [H3](../tools/minimax_h3/README.md)、[H3 Image](../tools/minimax_h3_image/README.md)、[Qwen-Image](../tools/qwen_image/README.md)。
