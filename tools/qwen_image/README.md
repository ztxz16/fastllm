# Qwen-Image-2.1 原生 C++ / CUDA

支持文生图、单张/多张参考图编辑、RGBA PNG、前缀 KV cache、同步 Images API 和带进度/取消的异步任务。推理与 HTTP 服务均为 C++，不调用 Python、PyTorch、Diffusers 或 Triton。

默认使用 BF16 激活，自动识别 BF16、逐行缩放的 E4M3 FP8、ConvRot INT8 或 GGUF DiT 权重。BF16 矩阵乘在块内使用 FP32 累加，大尺寸降维投影允许 BF16 split-K 部分和；归一化归约仍为 FP32。FP8 投影在 GEMM 前动态逐行量化输入，行缩放值按 TorchAO 的 BF16 舍入边界计算，输出恢复 BF16；文本/视觉编码器与 VAE 保持 BF16。权重直接读取发布的 safetensors / GGUF，无需转换。每次请求根据空闲显存、输出尺寸和参考图数量估算空间：充足时常驻权重并复用完整 KV 缓冲，不足时按文本编码、DiT、VAE 阶段加载/释放，保留较小的前缀缓存。HTTP 和命令行自动选择，无需精度或显存策略参数。

常驻只复用模型权重；提示词、参考图和噪声仍逐请求计算。KV、RoPE 和临时张量在请求结束释放；取消或异常同时卸载权重，下一请求重新评估。临时张量在请求内通过有界显存池复用。常驻速度需要预热，不应将热启动耗时当作包含首次权重加载的耗时。

## 构建

单卡仅需要 C++17、CUDA 12.8+、cuBLAS/cuBLASLt 和 pthread；多卡还需要 NCCL 的头文件与动态库；json11、httplib、FlashInfer、CUTLASS 已在仓库中。GPU 架构按实际设备设置，RTX PRO 5000 Blackwell 为 120。

```bash
cmake -S . -B build-qwen-image \
  -DUSE_CUDA=ON -DUSE_NUMAS=OFF -DBUILD_QWEN_IMAGE=ON \
  -DCUDA_ARCH=120 -DCMAKE_BUILD_TYPE=Release
cmake --build build-qwen-image --target qwen_image_server qwen_image_generate -j4
```

CUDA 13 的 `120f` 架构需要较新的 CMake。只构建指定目标，无需构建整个 LLM 后端。

模型目录应包含 `transformer/`、`text_encoder/`、`vae/`、`processor/`、`scheduler/`，其中分片索引及所有 safetensors 必须完整。

### FP8 权重

FP8 当前为实验支持：40 步文生图、单图和双图回归通过；随机输入 DiT 分项、双图 4 步仍超出原精度检查线，详见 [验证记录](VALIDATION.md#fp8-接入2026-11-15)。未放宽阈值，也不保证任意提示词或步数的误差上界。

已接入 [Unsloth Qwen-Image-2.1-FP8](https://huggingface.co/unsloth/Qwen-Image-2.1-FP8) 的 `Qwen-Image-2.1-FP8.safetensors`（约 7.12GB，DiT）。创建独立模型目录，在 `transformer/` 保留原模型的 `config.json`，将该文件放置或软链接为 `diffusion_pytorch_model.safetensors`；其余组件目录和 `model_index.json` 可软链接至原 BF16 模型。原 BF16 目录无需修改。

`--model` 指向这个新目录即可；命令行、API、工作室共用同一加载路径，不增加量化参数。通过张量 dtype、逐行 scale 和 TorchAO metadata 判断格式，不依赖模型目录名。支持该发布文件的单文件 E4M3 `._weight_qdata` / `._weight_scale`，其 metadata 指定 `TORCH` 行量化、`hp_value_lb=1e-12`；分片 FP8、blockwise FP8、E5M2 和 FP8 文本编码器不在本次支持范围内。融合 QKV 或 gate/up 内的投影须采用一致格式；不兼容格式明确报错。

FP8 需要支持 FP8 Tensor Core 的 SM89+ GPU；SM120 使用复用自 H3 的 CUTLASS GEMM，其他支持架构使用 cuBLASLt 回退。FP8 权重同样计入自动显存策略，支持常驻、按阶段卸载及取消后释放。部署不需要 TorchAO 或 PyTorch。

### INT8 与 GGUF 权重

支持 [Comfy-Org 的 INT8 ConvRot DiT](https://huggingface.co/Comfy-Org/Qwen-Image-2.1/tree/main/diffusion_models)（`qwen_image_2.1_int8_convrot.safetensors`，约 7.26GB），以及 [Unsloth GGUF](https://huggingface.co/unsloth/Qwen-Image-2.1-GGUF)；实测文件为 `qwen-image-2.1-Q4_K_M.gguf`（约 4.20GB）。均为原版 2.1，不是 Turbo。INT8 / GGUF 当前均为实验支持：两者的 DiT 分项及部分多图用例超过既有相对 L2 检查线，详见验证记录。

与 FP8 一样创建独立模型目录，保留原 BF16 的 `transformer/config.json`、文本/视觉编码器、VAE、processor、scheduler 和 `model_index.json`。INT8 文件放置或软链接为 `transformer/diffusion_pytorch_model.safetensors`；GGUF 文件直接放在 `transformer/`，该目录只保留一个 GGUF 文件，不放另一份 safetensors DiT。`--model` 仍选择模型根目录，API 和工作室不增加量化参数，原 BF16 / FP8 目录无需更改。

- INT8 由 I8 payload、逐输出行 F32 scale 和 `comfy_quant` 的 H256 ConvRot 标记识别。原生 CUDA 旋转与动态逐行量化激活，执行 W8A8 整数 GEMM，输出 BF16；直接支持发布权重的合并 `gate_up`。要求 SM80+；未接入无旋转、非对称、分组 scale 或 INT8 文本编码器。
- GGUF 按文件魔数和张量块类型读取，支持 F32/F16/BF16、Q4_0、Q8_0、Q2_K、Q3_K、Q4_K、Q5_K、Q6_K，允许同一层混合位宽。兼容无前缀和 `model.diffusion_model.` 前缀、合并或分开的 gate/up。要求 SM80+。小 token GEMM 复用 H3 的 MMQ；大 token 或不适用的形状使用有界 CUDA 解量化与 BF16 GEMM，权重始终以原块格式保存，不做二次量化。默认 MMQ 自动范围为 8..512 个 token；SM120 上 Q4_K 的 4096→24576 合并 gate/up 投影在缓存后的去噪阶段扩展到 1024 token，使 512² 编辑的主干也使用 MMQ；该扩展路径直接写 BF16/FP32 输出并使用经相对 L2 验证的 FMA 累加，避免大块 FP32 暂存。回退 GEMM 按形状调整分块，大 token QKV 直接写最终列区间。前缀构建保留原调度以避免误差累积，更大尺寸保留实测更快的回退路径，无需开关；范围与测速见验证记录。
- 两种格式均计入常驻显存统计，支持阶段卸载、取消清理及恢复；不要求安装 Torch、GGUF Python 包或额外转换工具。Python 参考仅供开发验证。

量化质量与实现误差的实测范围见 [验证记录](VALIDATION.md)；量化文件更小不保证更快。DiT 支持 1 / 2 / 4 / 8 卡张量并行，四种权重格式使用相同启动方式。FP8 TP2 使用 BF16 部分结果归约后，当前文生图、单图和双图样例相对同格式 TP1 均通过原 5% 线；这不等同于 FP8 相对 BF16 的量化质量验收，量化仍为实验支持。

## 启动与 GPU 选择

```bash
CUDA_VISIBLE_DEVICES=7 build-qwen-image/src/models/qwen_image/qwen_image_server \
  --model /workspace/hfmodels/Qwen-Image-2.1 \
  --device 0 --host 127.0.0.1 --port 8008 \
  --output /workspace/qwen-image-output \
  --demo tools/qwen_image/demo.html
```

`CUDA_VISIBLE_DEVICES=7` 将物理 7 号卡映射为进程内的逻辑 0 号卡。容器运行时同时使用 `--gpus device=7`，不要把 `--device` 再设为 7。可用 GPU UUID 避免编号变化。服务默认仅监听 loopback；`--api-key VALUE` 可启用 Bearer 验证。

双卡使用 `CUDA_VISIBLE_DEVICES=6,7` 与 `--devices 0,1`（或 `--device 0,1`）；同样适用于生成命令行。参数始终指 CUDA 可见的逻辑编号，顺序中的第一张卡负责文本/视觉编码、参考图 VAE、采样和输出解码。部署决定设备，不增加请求级 TP 参数；文生图、单/多参考图和取消操作共用该模型实例。

QKV 按完整注意力头切分，MLP 按 gate/up 分组切分。BF16、FP8、GGUF 输出投影将各卡部分结果舍入 BF16 后归约；有 P2P 的双卡对不超过 8 MiB 的 BF16 部分结果使用直接读对端显存的归约，其余情况自动使用 NCCL。FP8/INT8 的行并行输入先同步量化范围；INT8 保留 INT32 原始整数求和，再按原顺序缩放、舍入一次。有 P2P 的双卡融合整数求和与缩放，小张量一次完成；原始累加张量达到 1 MiB、元素数可均分且输出为 16 位时，各卡计算半片后只交换缩放后的结果。无 P2P 或更多卡时保留 NCCL 整数归约，不增加部署参数。量化权重在上传前按原始行/块读取各卡切片，GGUF 保持压缩驻留。显存策略检查所有卡，只要任一卡不够常驻就按阶段释放全部卡的 DiT 权重；请求结束或取消会清理各卡 KV 和临时缓冲。TP4/TP8 的权重布局和算子边界有测试，实机性能与精度以验证记录的双卡结果为准。

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

`GET /openapi.json` 提供 OpenAPI 3.0 定义；`GET /health` 返回 ready/loading/failed 和 `tensor_parallel_size`；`GET /v1/models` 返回 `qwen-image-2.1`。

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

请求由一个工作线程串行调度，多卡使用各卡持久工作线程。运行中加排队总数最多 4，超出返回 429；内存保留最近约 128 个已完成任务，PNG 持久保存在 output 目录，磁盘清理由调用方管理。取消在下一个阶段或采样步边界生效，正在运行的 CUDA 算子不会被中断。

## 边界

- `n=1`；输出宽高各 64..2048，须为 32 的倍数；步数 2..100，默认 40。
- 最多 4096 个文本/视觉编码 token。多张高分辨率参考图会达到此限制，需降低 `reference_resolution` 或减少图片。
- 参考分辨率 256..1024，须为 32 的倍数，默认 1024。按原图长宽比调整到该面积；调整后须至少 65536 像素，极端比例或小面积会明确报错。
- HTTP 请求最多 20MiB，参考图总计最多 1600 万像素。PNG 接收 8-bit、非 interlace 的灰度、RGB、调色板、灰度 alpha、RGBA；其他格式可用 demo 转换。
- 暂未实现 mask inpainting、负面提示/CFG、LoRA、远程 URL 输入、视频生成。未知参数返回 400，避免静默忽略。
- 默认 BF16 激活，支持 BF16 / 上述 FP8 / INT8 / GGUF DiT 权重；C++ Runtime / Pipeline 的 `fp32=true` 构造参数保留给 BF16 权重的数值回归，量化权重不支持此参数。多卡仅切分 DiT，文本/视觉编码器和 VAE 在首卡执行。
- 相同参数/种子在本原生后端可复现，但随机数生成器与 PyTorch 不同。跨实现精度比较必须共享初始噪声，不能只共享 seed。

## 算子复用

`fastllm_continuous` 从最新 HY-Nomas 实现抽出共用运行时：cuBLAS/cuBLASLt 线性层、FP32/BF16 attention（含现有 FlashInfer）、RMSNorm/LayerNorm、激活、逐元素运算、embedding、张量切片/拼接、RoPE、显存池、safetensors、Qwen BPE/NFC 分词。HY-Nomas 继续链接同一库。

共用算子的默认数值策略保持不变；Qwen 单独启用大尺寸降维投影的 BF16 split-K 候选，由已有自动调优选择。权重类增加已加载显存统计，供常驻决策使用。Qwen 的 `ops.cu` 实现 LayerNorm/调制、QK RMSNorm/RoPE/KV 写入、SiLU/乘法融合与 GPU Euler 更新，保留 BF16 中间舍入点。DiT 的 QKV 和 gate/up 权重仅在加载时拼接，并释放原表示；1×1 VAE 卷积直接走 GEMM，其他 BF16 卷积优先使用 `conv.cu` 中的 CUTLASS implicit GEMM，卷积输出阶段融合 bias 并直接写 BF16；VAE 归一化与 SiLU 按实际通道形状融合。不适用时回退到有界 im2col/GEMM 和通用归一化，不引入 cuDNN。

PNG 的 DEFLATE 解码、无压缩 PNG 编码和 Pillow 对齐的 Lanczos 缩放为独立 CPU C++ 实现。LingBot 的 VAE 显式保留 FP32 基类构造，避免受 Qwen 默认精度切换影响。

## 验证

测试依赖仅用于开发验证，不参与部署。需要在仅可见指定 GPU 的独立环境中运行：

```bash
# 已安装 torch/transformers，PYTHONPATH 指向含 Qwen-Image-2.1 的官方 diffusers/src
cmake -S . -B build-qwen-image -DUNIT_TEST=ON
cmake --build build-qwen-image --target qwen_image_reference -j4
python test/qwen_image/compare_reference.py --model /path/to/model \
  --library build-qwen-image/libqwen_image_reference.so --stage all
# FP32 回归沿用原来的严格阈值；默认 BF16 按实际送入 DiT 的文本特征验收。
python test/qwen_image/compare_reference.py --model /path/to/model \
  --library build-qwen-image/libqwen_image_reference.so --stage all --precision fp32

# 原生先导出初始噪声、prompt 和每步 latent，然后官方实现读取相同噪声
build-qwen-image/src/models/qwen_image/qwen_image_generate --trace /tmp/qwen-trace \
  --model /path/to/model --prompt 'A red apple on a wooden table.' \
  --width 128 --height 128 --steps 4 --seed 1234 --output /tmp/qwen-trace/native.png
python test/qwen_image/compare_pipeline.py --model /path/to/model --trace /tmp/qwen-trace --cpu-offload

c++ -std=c++17 -O2 -Iinclude src/models/qwen_image/image.cpp test/qwen_image/image_codec.cpp -o /tmp/image_codec
python test/qwen_image/test_image_codec.py /tmp/image_codec
python test/qwen_image/test_http.py --url http://127.0.0.1:8008 --image /path/to/reference.png
```

FP8 单元测试和独立数值参考：

```bash
cmake --build build-qwen-image --target qwen_image_fp8_test qwen_image_reference -j4
ctest --test-dir build-qwen-image -R '^qwen_image_fp8$' --output-on-failure
python test/qwen_image/compare_reference.py --model /path/to/original-bf16 \
  --native-model /path/to/fp8-model \
  --fp8-checkpoint /path/to/fp8-model/transformer/diffusion_pytorch_model.safetensors \
  --library build-qwen-image/libqwen_image_reference.so --stage all --precision bf16
# 对 FP8 原生 trace 的完整流程比较也使用原 BF16 目录和同一 FP8 DiT 文件：
python test/qwen_image/compare_pipeline.py --model /path/to/original-bf16 \
  --fp8-checkpoint /path/to/fp8-model/transformer/diffusion_pytorch_model.safetensors \
  --trace /tmp/fp8-trace --precision bf16
```

FP8 参考仅在测试侧用 PyTorch `scaled_mm` 替换官方 DiT 的量化投影，其余层沿用官方实现；需要支持该算子的 PyTorch/GPU，并在参数中保持 prompt、尺寸、步数和参考图与 trace 一致。FP8 参考暂不支持 `--cpu-offload`。

`--trace` 只在命令行显式指定时保存验证所需的 prompt、条件和每步 latent；HTTP 服务不写调试张量，也不引入 Qwen 专属环境变量。验证记录见 [VALIDATION.md](VALIDATION.md)。

INT8 / GGUF 单元与数值检查：

```bash
cmake --build build-qwen-image --target qwen_image_quantized_test qwen_image_reference -j4
ctest --test-dir build-qwen-image -R '^qwen_image_(fp8|quantized)$' --output-on-failure
python test/qwen_image/compare_reference.py --model /path/to/original-bf16 \
  --native-model /path/to/quantized-model \
  --quantized-checkpoint /path/to/quantized-model/transformer/checkpoint.gguf \
  --library build-qwen-image/libqwen_image_reference.so --stage all --precision bf16
```

`--quantized-checkpoint` 也接收 Comfy INT8 safetensors，`compare_pipeline.py` 支持同一参数。INT8 参考以独立稠密 Hadamard 和 PyTorch 整数 GEMM 计算；GGUF 参考由 gguf-py 解码同一份权重并用 BF16 GEMM 计算，因此比较包含原生 MMQ 的激活量化误差。以上参数均只属于测试脚本，不属于部署 API。

张量并行回归（原生 C++，不需要 PyTorch）：

```bash
cmake --build build-qwen-image --target qwen_image_parallel_test qwen_image_pipeline_parallel_test -j4
CUDA_VISIBLE_DEVICES=6,7 ctest --test-dir build-qwen-image -R '^qwen_image_parallel$' --output-on-failure
# 必须恰好可见两张已预留的卡；MODEL 可为四种已支持的目录格式。
CUDA_VISIBLE_DEVICES=6,7 build-qwen-image/src/models/qwen_image/qwen_image_pipeline_parallel_test \
  /path/to/model /path/to/reference1.png /path/to/reference2.png
```

单元回归覆盖分组权重切分、TP2/4/8 局部头、BF16/FP32 RoPE 尾块、跨卡 scale、INT32 原始累加归约（含超出 FP32 精确整数范围的部分和）、缩放顺序与 BF16/FP16/FP32 输出；覆盖 P2P 两阶段分界、不同对齐视图、显存复用、guard 及异常恢复。真实模型回归比较单/双卡的 4/40 步双参考图，检查仅副卡显存受压时全卡卸载、常驻恢复和取消恢复。GPU 数量大于 2 的端到端精度/速度尚未实测；既有量化精度失败不因这些功能回归而视作通过。
