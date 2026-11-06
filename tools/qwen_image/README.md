# Qwen-Image-2.1 原生 C++ / CUDA

支持文生图、单张/多张参考图编辑、RGBA PNG、前缀 KV cache、同步 Images API 和带进度/取消的异步任务。推理与 HTTP 服务均为 C++，不调用 Python、PyTorch、Diffusers 或 Triton。

当前仅支持 FP32，以官方 Diffusers 的 FP32 路径为精度基线。权重直接读取发布的 safetensors，无需转换。文本编码器、DiT、VAE 按阶段加载并释放，单张 RTX PRO 5000 72GB 可运行。当前优先保证精度，未做量化或 BF16 性能优化。

## 构建

仅需要 C++17、CUDA 12.8+、cuBLAS/cuBLASLt 和 pthread；json11、httplib、FlashInfer 已在仓库中。GPU 架构按实际设备设置，RTX PRO 5000 Blackwell 为 120。

```bash
cmake -S . -B build-qwen-image \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_QWEN_IMAGE=ON \
  -DCUDA_ARCH=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-qwen-image --target qwen_image_server qwen_image_generate -j4
```

CUDA 13 的 `120f` 架构需要较新的 CMake。只构建指定目标，无需构建整个 LLM 后端。

模型目录应包含 `transformer/`、`text_encoder/`、`vae/`、`processor/`、`scheduler/`，其中分片索引及所有 safetensors 必须完整。

## 启动与单卡隔离

```bash
CUDA_VISIBLE_DEVICES=7 build-qwen-image/src/models/qwen_image/qwen_image_server \
  --model /workspace/hfmodels/Qwen-Image-2.1 \
  --device 0 --host 127.0.0.1 --port 8008 \
  --output /workspace/qwen-image-output \
  --demo tools/qwen_image/demo.html
```

`CUDA_VISIBLE_DEVICES=7` 将物理 7 号卡映射为进程内的逻辑 0 号卡。容器运行时同时使用 `--gpus device=7`，不要把 `--device` 再设为 7。可用 GPU UUID 避免编号变化。服务默认仅监听 loopback；`--api-key VALUE` 可启用 Bearer 验证。

浏览器打开 `http://127.0.0.1:8008/demo`。页面支持上传参考图、设置种子、进度、取消、下载和复制 curl。浏览器将 JPEG/WebP 等输入转换为 PNG；直接 HTTP 上传需要 PNG。

命令行示例：

```bash
CUDA_VISIBLE_DEVICES=7 build-qwen-image/src/models/qwen_image/qwen_image_generate \
  --model /workspace/hfmodels/Qwen-Image-2.1 --device 0 \
  --prompt 'A red apple on a wooden table.' \
  --width 1024 --height 1024 --steps 40 --seed 1234 --output apple.png
```

增加 `--image reference.png --reference-resolution 512` 即为参考图编辑。可重复 `--image`，最多 10 张，仍受上下文长度限制。

## HTTP API

`GET /openapi.json` 提供 OpenAPI 3.0 定义；`GET /health` 返回 ready/loading/failed；`GET /v1/models` 返回 `qwen-image-2.1`。

```bash
curl http://127.0.0.1:8008/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","prompt":"一只坐在窗边的橘猫","size":"1024x1024","n":1,"seed":1234,"num_inference_steps":40,"response_format":"b64_json"}'
```

响应使用 Images API 的 `created`、`data[].b64_json` 或 `data[].url`，另返回实际 `seed`。

```bash
curl http://127.0.0.1:8008/v1/images/edits \
  -F model=qwen-image-2.1 \
  -F 'image[]=@apple.png' \
  --form-string 'prompt=Change the apple to green. Keep the background.' \
  -F size=512x512 -F reference_resolution=512 \
  -F seed=1234 -F num_inference_steps=40 -F response_format=url
```

同步调用可用 OpenAI 风格客户端；扩展参数通过客户端的 `extra_body` / 表单字段传递。长任务也可调用 `POST /v1/images/jobs`，JSON 为文生图，multipart 为参考图编辑。返回 HTTP 202 后轮询 `GET /v1/images/jobs/{id}`，取消用 `POST /v1/images/jobs/{id}/cancel`，下载用返回的 `url`。

GPU 由一个工作线程串行执行。运行中加排队总数最多 4，超出返回 429；内存保留最近约 128 个已完成任务，PNG 持久保存在 output 目录，磁盘清理由调用方管理。取消在下一个阶段或采样步边界生效，正在运行的 CUDA 算子不会被中断。

## 边界

- `n=1`；输出宽高各 64..2048，须为 32 的倍数；步数 2..100，默认 40。
- 最多 4096 个文本/视觉编码 token。多张高分辨率参考图会达到此限制，需降低 `reference_resolution` 或减少图片。
- 参考分辨率 256..1024，须为 32 的倍数，默认 1024。按原图长宽比调整到该面积；调整后须至少 65536 像素，极端比例或小面积会明确报错。
- HTTP 请求最多 20MiB，参考图总计最多 1600 万像素。PNG 接收 8-bit、非 interlace 的灰度、RGB、调色板、灰度 alpha、RGBA；其他格式可用 demo 转换。
- 暂未实现 mask inpainting、负面提示/CFG、LoRA、远程 URL 输入、视频生成。未知参数返回 400，避免静默忽略。
- 当前只开放已对齐的 FP32；未验收的 BF16 实验分支不包含在本实现中。
- 相同参数/种子在本原生后端可复现，但随机数生成器与 PyTorch 不同。跨实现精度比较必须共享初始噪声，不能只共享 seed。

## 算子复用

`fastllm_continuous` 从最新 HY-Nomas 实现抽出共用运行时：cuBLAS/cuBLASLt 线性层、FP32/BF16 attention（含现有 FlashInfer）、RMSNorm/LayerNorm、激活、逐元素运算、embedding、张量切片/拼接、RoPE、显存池、safetensors、Qwen BPE/NFC 分词。HY-Nomas 继续链接同一库。

原有算子的数学实现保持不变；共用层仅增加显式精度构造参数及 Diffusers 分片索引名。Qwen 独有的 MRoPE/视觉 RoPE、zero-centered RMSNorm、调制门、VAE im2col/采样在 `ops.cu`，卷积仍复用共用 GEMM，不引入 cuDNN。PNG 的 DEFLATE 解码、无压缩 PNG 编码和 Pillow 对齐的 Lanczos 缩放为独立 CPU C++ 实现。

## 验证

测试依赖仅用于开发验证，不参与部署。需要在仅可见指定 GPU 的独立环境中运行：

```bash
# 已安装 torch/transformers，PYTHONPATH 指向含 Qwen-Image-2.1 的官方 diffusers/src
cmake -S . -B build-qwen-image -DUNIT_TEST=ON
cmake --build build-qwen-image --target qwen_image_reference -j4
python test/qwen_image/compare_reference.py --model /path/to/model \
  --library build-qwen-image/libqwen_image_reference.so --stage all

# 原生先导出初始噪声、prompt 和每步 latent，然后官方实现读取相同噪声
build-qwen-image/src/models/qwen_image/qwen_image_generate --trace /tmp/qwen-trace \
  --model /path/to/model --prompt 'A red apple on a wooden table.' \
  --width 128 --height 128 --steps 4 --seed 1234 --output /tmp/qwen-trace/native.png
python test/qwen_image/compare_pipeline.py --model /path/to/model --trace /tmp/qwen-trace --cpu-offload

c++ -std=c++17 -O2 -Iinclude src/models/qwen_image/image.cpp test/qwen_image/image_codec.cpp -o /tmp/image_codec
python test/qwen_image/test_image_codec.py /tmp/image_codec
python test/qwen_image/test_http.py --url http://127.0.0.1:8008 --image /path/to/reference.png
```

`--trace` 只在命令行显式指定时保存验证所需的 prompt、条件和每步 latent；HTTP 服务不写调试张量，也不引入 Qwen 专属环境变量。验证记录见 [VALIDATION.md](VALIDATION.md)。
