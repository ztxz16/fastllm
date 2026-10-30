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
- `--output` 用于保存生成的 PNG 帧，服务进程需要有写入权限。
- 默认使用 `--precision bf16`：权重、主激活和 KV 使用 BF16，矩阵乘法累加、归一化内部计算及采样更新保留 FP32。需要精度排查时，可显式使用 `--precision fp32`。
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

## 5. 使用网页 Demo

输入场景描述并创建世界后，使用 W/A/S/D 移动、方向键调整视角，也可点击界面按钮。“连续探索”会自动提交后续动作，点击停止可结束自动探索。

“修改当前场景”保留当前世界和历史，并在下一段应用新描述。修改成功后，移动和转向会继续探索该场景；“重建场景”则清空历史并创建新世界。场景修改和长期探索仍可能出现语义偏差或画面漂移。

默认分辨率为 384×224，可选最高 512×288；需要减少生成资源占用时可降低分辨率。切换到“独立短视频”可单独生成一段视频。网页以 PNG 帧播放结果，不输出可直接渲染的三维网格。

世界和任务索引保存在服务内存中，刷新页面可恢复当前服务中的场景，服务重启后需要重新创建。

## 6. 调用 HTTP API

创建世界，返回世界 `id` 和首段任务 `current_job`：

```bash
curl http://127.0.0.1:8007/v1/worlds \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"一间阳光明亮的木屋，中央有一张桌子。"}],"width":256,"height":160,"seed":42}'
```

将 `VIDEO_ID` 替换为 `current_job`，查询生成结果；完成后 `output.frames` 提供按顺序排列的 PNG 地址：

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
