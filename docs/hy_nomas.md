# HY-Nomas 部署指南

[返回 README](../README.md) · [全部文档](README.md)

HY-Nomas 支持文本生成视频、相机控制和连续场景探索。模型推理使用独立的 C++/CUDA 库，通过专用 HTTP 服务提供网页 Demo 和 API。

## 1. 准备环境

- Linux、NVIDIA 驱动及与目标 GPU 匹配的 CUDA 开发工具链，包含 cuBLAS/cuBLASLt。
- C++17 编译器和 CMake。下方使用 `CUDA_ARCH=native` 自动识别本机架构，需要 CMake 3.24 或更新版本。
- Python 3.10 或更新版本，以及 HTTP 服务使用的 FastAPI、Pydantic 2、uvicorn。

在已有 Python 环境中安装 HTTP 依赖：

```bash
python3 -m pip install fastapi 'pydantic>=2' uvicorn
```

推理无需安装 PyTorch、LibTorch、Triton 或视频编解码库。以下构建和启动命令均在 FastLLM 仓库根目录执行。

## 2. 准备模型权重

准备 HY-Nomas 和 Qwen2.5-VL-7B-Instruct 的原始权重，无需转换格式。目录至少包含：

```text
/path/to/Hy-Nomas/
├── DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors
└── VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors

/path/to/Qwen2.5-VL-7B-Instruct/
├── tokenizer.json
├── model.safetensors.index.json
└── model-*.safetensors
```

Qwen 权重分片需要与索引文件完整对应；服务只使用其中的文本编码部分。

## 3. 编译原生库

```bash
cmake -S . -B build-nomas \
  -DUSE_CUDA=ON -DBUILD_HY_NOMAS=ON \
  -DCUDA_ARCH=native -DUSE_NUMAS=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build-nomas --target fastllm_nomas -j2
```

输出为 `build-nomas/libfastllm_nomas.so`。`native` 用于在部署机器上识别 GPU；如果在其他机器交叉编译，将其替换为目标 GPU 的 CUDA 架构编号。

也可使用同一构建配置的快捷脚本：

```bash
CUDA_ARCH=native BUILD_JOBS=2 bash tools/hy_nomas/build.sh build-nomas
```

`BUILD_HY_NOMAS` 默认关闭，构建此目标不会改变其他模型的推理入口。目前 HY-Nomas 使用下方独立服务，不通过 `ftllm server` 启动。

## 4. 启动 HTTP 服务

将模型和输出路径替换为实际目录：

```bash
CUDA_VISIBLE_DEVICES=0 python3 tools/fastllm_pytools/nomas_server.py \
  --library build-nomas/libfastllm_nomas.so \
  --model /path/to/Hy-Nomas \
  --text-encoder /path/to/Qwen2.5-VL-7B-Instruct \
  --output /path/to/nomas-output \
  --precision bf16 --device 0 \
  --host 127.0.0.1 --port 8007
```

- `CUDA_VISIBLE_DEVICES` 选择使用的 GPU；`--device 0` 表示可见 GPU 中的第一张。
- `--output` 用于保存生成的 MP4 短片或 PNG 帧，服务进程需要有写入权限。
- 默认使用 `--precision bf16`：权重、主激活和 KV 使用 BF16，矩阵乘法累加、归一化内部计算及采样更新保留 FP32。需要精度排查时，可显式使用 `--precision fp32`。
- BF16 交互世界按官方顺序从种子生成 120 秒的初始噪声，并用同一随机流给每段的历史 KV 加噪。每步 Euler 更新通过 FP32 计算后舍入到 BF16。跨后端回放时，长序列的镜头仍可能逐段偏离。
- 从其他机器访问时，将 `--host` 改为 `0.0.0.0`，浏览器访问部署机器的 IP 和端口。服务没有内置认证，对外开放时可接入已有认证网关。

启动后访问：

| 地址 | 用途 |
| --- | --- |
| `http://127.0.0.1:8007/` | 网页 Demo |
| `http://127.0.0.1:8007/docs` | 交互式 API 文档 |
| `http://127.0.0.1:8007/openapi.json` | OpenAPI 描述 |

检查服务是否启动：

```bash
curl http://127.0.0.1:8007/health
```

正常返回 `status: "ok"`、`backend: "native_cpp_cuda"` 和 `torch_loaded: false`。

### 可选：多卡推理

编译时需要已有的 NCCL 头文件和动态库，CMake 会自动查找。非标准安装位置可在
配置命令中指定 `FASTLLM_NCCL_INCLUDE_DIR` 和 `FASTLLM_NCCL_LIBRARY`；不启用多卡时
无需 NCCL。

将启动命令中的 `CUDA_VISIBLE_DEVICES=0` 改为 `CUDA_VISIBLE_DEVICES=0,1`，并把
`--device 0` 替换为 `--devices 0,1`。设备编号均相对于 CUDA 可见设备列表。
网页和 HTTP 请求格式保持不变。

每张卡由一个推理线程执行，DiT 和 PixelUMM 共用这组线程及通信器。
DiT 按图像 token 切分计算，各卡保留完整权重和文本分支，每层聚合当前图像 K/V，
完整图像输出只在 DiT 末尾聚合；每卡不足 64 个图像 token 时保留完整计算。
PixelUMM 按注意力头和 FFN 通道切分权重。文本编码和采样器在第一张卡执行，
场景历史由各卡独立持有，取消请求不会修改已提交的历史。

BF16 模式保留 FP32 GEMM 累加；PixelUMM 使用 BF16 部分输出及归约。
需要全 FP32 对照时仍可使用 `--precision fp32`。双卡使用 `--devices 0,1`，
四卡使用 `--devices 0,1,2,3`，同时设置对应的 `CUDA_VISIBLE_DEVICES`。
单卡继续使用原来的 `--device` 配置。

### 可选：接入 Nomas-PE 场景扩写

准备 HY-Nomas 发布的 `PE/` 完整权重目录，并下载官方仓库的提示词文件（不需要安装官方 Python 推理环境）：

```bash
mkdir -p /path/to/nomas-prompts
for name in online_user_video_rewriter_system_prompt online_user_video_rewriter_first_turn online_user_video_rewriter_intermediate_turn; do
  curl -fL "https://raw.githubusercontent.com/Tencent-Hunyuan/Hy-Nomas/9d80c58616538eef721b9c489cb1ec4e12cdb2a3/apps/prompts/${name}.md" \
    -o "/path/to/nomas-prompts/${name}.md"
done
```

先使用已安装的 fastllm 启动 PE 的 Chat Completions 服务：

```bash
CUDA_VISIBLE_DEVICES=0 ftllm server /path/to/Hy-Nomas/PE \
  --model_name Nomas-PE --host 127.0.0.1 --port 8008 \
  --device cuda --tp off --dtype int8 --atype float16 \
  --low_gpu_mem --vision_device cpu --max_batch 1 --mtp 0 \
  --max_context_length 32768 --tokens 32768 --kv_cache_limit 2g \
  --chunked_prefill_size 512 --gpu_mem_ratio 0.48 --enable_thinking false
```

上述配置使用 INT8 权重、FP16 激活；原始权重无需转换。当前 INT8 Linear 路径不支持 BF16 激活。两个服务共卡时，需要为生成模型保留足够显存；`CUDA_VISIBLE_DEVICES` 与资源预算应按部署环境设置。

在 HY-Nomas HTTP 服务启动命令中追加：

```bash
  --prompt-enhancer-url http://127.0.0.1:8008/v1/chat/completions \
  --prompt-enhancer-model Nomas-PE \
  --prompt-enhancer-prompts /path/to/nomas-prompts
```

若服务分别运行在容器中，应使用容器内可达的 PE 地址，并挂载提示词目录。`/health` 的 `prompt_enhancer.enabled` 表示是否配置扩写。调用只使用 Python 标准库 HTTP 客户端，PE 推理仍由 fastllm 执行。

在启动器的 HY-Nomas 服务设置中，填写提示词增强地址，可选填增强模型名称和提示词目录。默认名称是 `Nomas-PE`，默认目录是模型下的 `prompts/`。启动服务后，Demo 默认勾选「提示词增强」，可以在创建前关闭。“观察方式”仅在启用扩写时可选；没有配置 PE 时隐藏，取消勾选扩写后禁用。场景创建后自动进入探索视图。用 WASD 或屏幕方向按钮移动，拖动或滑动画面后松开即可转向。单步探索时，当前片段生成期间最多保留一个后续动作，重新选择方向会替换它。在画面下方输入指令后按 Enter 发送，Shift+Enter 换行，按 / 可聚焦输入框。「纯画面」收起控件，点击「显示操作」恢复。「场景设置」可返回编辑器，再点「进入场景」继续同一个世界。

API 创建请求可传 `enhance_prompt: false` 关闭扩写；`perspective` 仅用于扩写，可选择 `auto`、`first-person`、`second-person`、`third-person follow`、`third-person observer`。`auto` 保留描述中明确的第一人称要求，否则使用外部观察视角；修改沿用当前场景视角。

扩写在后台任务中执行，进度阶段为 `prompt_enhancement`。创建和修改描述时调用 PE，移动、转向和重复按描述继续生成复用已完成的扩写。原始 `prompt` 保留用于编辑，`enhanced_prompt` 返回实际送入生成模型的场景 JSON；页面提供展开查看入口。扩写失败会令任务失败，不会悄悄退回原提示词。

首轮扩写记录人物和环境。后续扩写使用官方后续回合格式，把环境变化写入 `description_change`，人物动作写入 `contained_actions`。送进文本编码器前，动作起止时间换算为本段的相对时间。PE 可见首轮和上一轮的文本描述。场景文本最多接受 1200 个编码 token。

## 5. 使用网页 Demo

输入场景描述并创建世界后，使用 W/A/S/D 移动、方向键调整视角，也可点击界面按钮。“连续探索”会自动提交后续动作，点击停止可结束自动探索。

场景创建后，在“下一条指令”里输入环境变化或人物动作，点击“发送指令”即可沿着当前场景生成。展开“场景设置”可调整起始条件并重建场景。场景修改和长期探索仍可能出现语义偏差或画面漂移。

场景设置提供最高 1280×704 的预设，也可自定义横屏、竖屏或方形画面。宽高均为 64 到 1280 之间的 32 倍数，总像素不超过 901120。高分辨率需要更多显存和生成时间。切换到“独立短视频”可单独生成一段视频。浏览器支持 H.264 且服务端 FFmpeg 提供 libx264 时，网页优先播放 H.264 短片（CRF 18、24 fps）。其他环境使用 PNG 帧。

在交互世界中选取「起始参考图」并填写初始描述。浏览器按所选尺寸等比缩放并居中补黑边，把 RGB24 像素传给服务。C++/CUDA 编码器使用 PixelUMM_FT 权重中的 `encoder.*`，处理 120 个重复的像素帧，一次写入 30 帧潜变量的场景历史。页面显示解码结果的最后 12 帧。文本指令的初始帧数为 20，带参考图时为 120。可在「发送指令」旁为本条指令选择 20、40、60、80、100 或 120 帧。「按指令继续」沿用上次发送的帧数，移动和转向每次生成 20 帧。

API 创建请求可传 `reference_rgb`（恰好 `width * height * 3` 字节的 RGB24，使用标准 Base64 编码）。`text_frames` 可选 20 到 120 之间的 20 的倍数，默认 20。场景的 `generated_frames` 统计可播放帧数，`model_frames` 包含完整的参考图历史。后续 PATCH 可直接传官方结构化提示词字符串，或用 `{"kind":"environment","prompt":"the falling rain stops"}`、`{"kind":"action","prompt":"lifts the white mug and offers it once"}` 指定指令类型。PATCH 可同时传 `text_frames`。随后提交 `follow_prompt` 应用修改，已在生成的片段沿用提交时的帧数。

世界和任务索引保存在服务内存中，刷新页面可恢复当前服务中的场景，服务重启后需要重新创建。

## 6. 调用 HTTP API

创建世界，返回世界 `id` 和首段任务 `current_job`：

```bash
curl http://127.0.0.1:8007/v1/worlds \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"一间阳光明亮的木屋，中央有一张桌子。"}],"width":256,"height":160,"seed":42}'
```

将 `VIDEO_ID` 替换为 `current_job`，查询生成结果；默认返回逐帧 PNG，完成后 `output.frames` 提供按顺序排列的地址。创建场景或视频时可以传入 `"response_format": "mp4"`。读取结果时以 `output.format` 为准，`mp4` 的短片地址在 `output.url`，编码不可用时服务返回 `png_frames`。查询示例：

```bash
curl http://127.0.0.1:8007/v1/videos/VIDEO_ID
```

将 `WORLD_ID` 替换为世界 `id`，在上一段完成后提交移动动作：

```bash
curl http://127.0.0.1:8007/v1/worlds/WORLD_ID/actions \
  -H 'Content-Type: application/json' -d '{"action":"forward"}'
```

修改场景描述，然后生成一段应用修改：

```bash
curl -X PATCH http://127.0.0.1:8007/v1/worlds/WORLD_ID \
  -H 'Content-Type: application/json' -d '{"prompt":"一只猫出现在桌子上。"}'
curl http://127.0.0.1:8007/v1/worlds/WORLD_ID/actions \
  -H 'Content-Type: application/json' -d '{"action":"follow_prompt"}'
```

`PATCH` 只保存描述，不自动生成。后续移动沿用修改后的场景；`follow_prompt` 明确要求按完整描述继续生成。同一世界正在生成时，再提交动作返回 409，应等待当前任务结束。

创建请求可使用 `prompt` 或 `messages`，二选一；`messages` 目前仅支持 `user` 文本消息。接口采用 JSON/HTTP，但不完整兼容 OpenAI Chat Completions。独立视频、其他相机动作、取消和 SSE 进度接口可在服务的 `/docs` 中查看。
