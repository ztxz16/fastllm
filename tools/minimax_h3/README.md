# MiniMax-H3 原生推理

面向 Linux + NVIDIA GPU 的纯 C++17 / CUDA 音视频推理，使用 CUDA Runtime、cuBLAS/cuBLASLt 和仓库已有的
FlashInfer、json11、cpp-httplib。推理服务不加载 Python、PyTorch、Diffusers、Triton
或 cuDNN；MP4 封装调用已有的 FFmpeg（H.264 / AAC），不参与模型计算。

当前接口支持 **T2VA：文字生成视频与立体声音轨**。使用 `transformer/` 权重；
首尾帧条件与 `transformer_ref/` 的参考图像、视频、音频条件尚未接入公开接口。
服务会拒绝未知参数，不会默默忽略参考媒体。

## 精度与单卡内存

当前仅支持 FP32 计算，以官方模型的 FP32 前向作为基线。文本编码只执行 Qwen3-VL
的前 50 层，读取 final norm 之前的残差；DiT 对已知时间点提前计算 AdaLN 表，释放投影
权重，然后按层加载主干权重。BF16 检查点直接传到 GPU，再精确扩展到 FP32，减少 PCIe
传输与主机内存复制。视频、音频解码按阶段加载，使用 FP32。

DiT 会利用空闲显存缓存部分原始 BF16 权重，在后续采样轮次直接展开为 FP32。
缓存预算不超过启用时空闲显存的一半，并预留至少 16 GiB 给后续工作张量；空闲量不足
16 GiB 时不启用。此缓存不改变计算精度，不跨生成请求保留，完成或取消时释放。

单卡运行不等于权重全部驻留显存。缓存容纳不下的权重仍按层传输，速度受 PCIe 和主机
页缓存影响。参考测试及实测记录见 [VALIDATION.md](VALIDATION.md)。

## 构建

```bash
cmake -S . -B build-h3 \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_MINIMAX_H3=ON \
  -DCUDA_ARCH=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-h3 --target minimax_h3_generate minimax_h3_server -j4
```

`CUDA_ARCH=120` 适用于本次 RTX PRO 5000 Blackwell 机器，其他 GPU 应使用其实际架构。
使用 CMake 4 时，若旧的仓库最低版本设置导致配置失败，可补充
`-DCMAKE_POLICY_VERSION_MINIMUM=3.5`。

模型目录必须使用 Diffusers 布局：

```text
MiniMax-H3/
  text_encoder/ tokenizer/
  transformer/
  vae/ audio_vae/
```

加载本地 safetensors / JSON 文件，无联网下载、转换脚本或额外权重副本。

## 命令行

```bash
CUDA_VISIBLE_DEVICES=7 build-h3/src/models/minimax_h3/minimax_h3_generate \
  --model /workspace2/hfmodels/MiniMax-H3 \
  --prompt 'A cat plays piano on a wooden stage. The audience applauds.' \
  --width 512 --height 512 --frames 124 --steps 50 --seed 42 \
  --device 0 --output cat.mp4
```

可见设备设置后，物理 7 号卡在进程中为逻辑 0 号。容器必须同时用 `--gpus device=7`
限制设备，不要直接把进程中的 `--device` 设置成 7。

- 宽高须为 32 的倍数，各自介于 64..4096，总像素不超过 1344×768。
- 24 fps，帧数为 `17*n+5`，支持 124、141、…、345 帧。
- `--steps` 指包含终点的采样网格点数，50 点通常执行 49 次 DiT 前向。
- 音频为 32 kHz 双声道；分别使用视频 shift=12、音频 shift=3 的官方调度。
- 随机数使用本地 C++ 生成器，同一实现内种子可复现；同一个 seed 不保证与 PyTorch
  得到同一初始噪声。差分测试通过 `--initial-video` / `--initial-audio` 共享 FP32 噪声。
- `--trace DIR` 仅供本地开发验证，会保存中间张量、原始 RGB 与音频；HTTP 不开放此参数。

开发调优先用 `--width 256 --height 256 --steps 4`（实际 3 次 DiT）快速比较，确认
收益后再跑完整精度回归。少步数输出只用于测速和数值检查，不用于评判最终生成质量。

## HTTP 与 Demo

```bash
CUDA_VISIBLE_DEVICES=7 build-h3/src/models/minimax_h3/minimax_h3_server \
  --model /workspace2/hfmodels/MiniMax-H3 --device 0 \
  --host 127.0.0.1 --port 8009 --output ./h3-output \
  --demo tools/minimax_h3/demo.html
```

浏览 `http://127.0.0.1:8009/demo`。需要鉴权时加 `--api-key YOUR_KEY`，API 请求携带
`Authorization: Bearer YOUR_KEY`。FFmpeg 不在 PATH 时可用 `--ffmpeg /absolute/path`。

接口采用异步视频任务的常见结构，同时接受 JSON 与 multipart 表单：

```bash
curl http://127.0.0.1:8009/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3","prompt":"A cat plays piano.","size":"512x512","seconds":"5","seed":42,"num_inference_steps":50}'

curl http://127.0.0.1:8009/v1/videos/VIDEO_ID
curl http://127.0.0.1:8009/v1/videos/VIDEO_ID/content -o video.mp4
```

| 方法与路径 | 功能 |
|---|---|
| `GET /health` | 服务初始化状态 |
| `GET /v1/models` | 模型列表 |
| `POST /v1/videos` | 提交生成，返回 HTTP 202 与任务对象 |
| `GET /v1/videos` | 本进程的任务列表 |
| `GET /v1/videos/{id}` | 状态、进度、实际帧数与错误 |
| `GET /v1/videos/{id}/content` | 完成后的 MP4 |
| `POST /v1/videos/{id}/cancel` | 请求取消，阶段或层边界生效 |
| `DELETE /v1/videos/{id}` | 删除已终止任务及视频文件 |
| `GET /openapi.json` | OpenAPI 3.1 规范 |

`seconds` 会向上对齐到支持的帧数，以返回的 `frames`、`seconds` 为准；也可直接指定
`frames`，同样向上对齐，二者不可同时传入。15 秒向上对齐后超过 345 帧，会返回 400。
每次生成一个视频，不支持负面提示词或 CFG 参数。

一条工作线程执行 GPU 推理，总待处理任务上限 4；超限返回 429。历史记录保留在内存中，
最多 1024 条，重启不会恢复任务索引；完成的视频文件仍保留在输出目录。
