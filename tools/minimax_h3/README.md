# MiniMax-H3 原生推理

面向 Linux + NVIDIA GPU 的纯 C++17 / CUDA 音视频推理，使用 CUDA Runtime、cuBLAS/cuBLASLt 和仓库已有的
FlashInfer、json11、cpp-httplib。推理服务不加载 Python、PyTorch、Diffusers、Triton
或 cuDNN；MP4 封装调用已有的 FFmpeg（H.264 / AAC），不参与模型计算。

当前接口支持 **T2VA：文字生成视频与立体声音轨**。使用 `transformer/` 权重；
首尾帧条件与 `transformer_ref/` 的参考图像、视频、音频条件尚未接入公开接口。
服务会拒绝未知参数，不会默默忽略参考媒体。

## 精度与单卡内存

默认使用 **BF16 混合精度**：文本编码器和 DiT 的权重、激活与残差使用 BF16，
GEMM 保持 FP32 累加。时间嵌入、视频/音频输入输出投影和采样状态保留 FP32；
视频 VAE 默认使用 FP16 GEMM/attention，归一化、缩放参数与残差流保留 FP32；音频 VAE
保持 FP32。CLI 与 HTTP 服务均可通过 `--precision fp32` 切回全 FP32 对比模式，
也可用 `--vae-precision fp16|bf16|fp32` 单独覆盖视频 VAE，无需环境变量。
`GET /health` 返回当前 `precision` 和 `vae_precision`。

文本编码只执行 Qwen3-VL 的前 50 层，读取 final norm 之前的残差，权重按层释放。
DiT 按每次前向的时间点数预计算 AdaLN 表，随后释放这些投影权重；BF16 主干约
37.5 GiB 权重驻留到本次去噪结束，后续采样步不再重复传输。视频和音频解码按阶段加载。
当非 FP32 主干、非 FP32 视频 VAE、总显存至少 68 GiB 且创建时空闲至少 60 GiB 时，
跨请求保留 DiT、视频/音频 VAE 权重，并复用完全相同的 AdaLN 时间表。其他配置按阶段
释放；取消或异常仍清空 GPU 权重。文本 checkpoint 字节保留在有界 CPU 缓存中，优先
使用 pinned 内存，按层上传 GPU；不缓存 prompt 的编码结果。

FP32 对比模式仍按层加载 DiT，并利用部分空闲显存缓存原始 BF16 权重，使用时精确
展开为 FP32。此缓存不改变运算精度，预算不超过启用时空闲显存的一半，预留至少
16 GiB 给工作张量。BF16 驻留模式需要比 FP32 流式模式更多显存；本次只验证 72GB 单卡。

官方 Diffusers 与 SGLang 均默认使用 BF16 主干，但混合精度、attention 内核与算子
融合方式不同，不能宣称最终输出逐位相同。当前 BF16 逐层检查通过，**完整流程尚未
通过严格精度回归**；FP32 保留严格数值对比。实测误差和限制见 [VALIDATION.md](VALIDATION.md)。

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
- `--precision bf16|fp32` 默认为 `bf16`；分别对应默认 FP16、FP32 视频 VAE。
- `--vae-precision fp16|bf16|fp32` 可单独覆盖视频 VAE 精度，优先于上述默认值。
  HTTP 服务在启动时选择精度，生成请求结构保持不变；BF16 VAE 是可选对照路径。
- 音频为 32 kHz 双声道；分别使用视频 shift=12、音频 shift=3 的官方调度。
- 随机数使用本地 C++ 生成器，同一实现内种子可复现；同一个 seed 不保证与 PyTorch
  得到同一初始噪声。差分测试通过 `--initial-video` / `--initial-audio` 共享 FP32 噪声。
- `--trace DIR` 仅供本地开发验证，会保存中间张量、原始 RGB 与音频；HTTP 不开放此参数。

开发调优先用 `--width 256 --height 256 --steps 4`（实际 3 次 DiT）快速比较，确认
收益后再跑完整精度回归。少步数输出只用于测速和数值检查，不用于评判最终生成质量。

CLI 和 HTTP 默认先解码音频，再将视频 VAE 已完成拼接的帧分段交给后台 FFmpeg，
让编码与后续解码重叠。队列有容量上限，取消和编码失败会回收子进程与临时文件；
完成后才原子发布 MP4。编码参数保持 libx264、fast、CRF 18、4 线程及 AAC 192k。
C++ 调用 `Pipeline::generateMP4` 可使用相同路径；`Pipeline::generate` 仍返回原始
RGB/音频，`writeMP4` 仍可单独编码。视频 VAE 的中间投影按最多 4 个空间 tile 合批，
各 tile 的 attention 保持独立；FP32 对照保留原分块与矩阵形状。

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
