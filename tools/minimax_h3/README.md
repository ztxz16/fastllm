# MiniMax-H3 原生推理

面向 Linux + NVIDIA GPU 的纯 C++17 / CUDA 音视频推理，使用 CUDA Runtime、cuBLAS/cuBLASLt 和仓库已有的
FlashInfer、json11、cpp-httplib。推理服务不加载 Python、PyTorch、Diffusers、Triton
或 cuDNN；多卡通信使用仓库已有的 NCCL。媒体编解码调用已有的 FFmpeg / ffprobe，不参与模型计算。

支持文字生成、首帧、尾帧、首尾帧，以及有序的参考图片／视频／音频条件。
文字和关键帧使用 `transformer/`；参考素材使用 `transformer_ref/`，切换时卸载另一套 DiT。
两种条件不能混用；音频参考须搭配图片或视频。服务拒绝未知参数。
视觉条件经过 Qwen3-VL 视觉塔及视频 VAE；音频条件经过音频 VAE 的编码器。
视频编码器与音频编码器保持原生 FP32；视觉 posterior 按原生规则采样、舍入到 FP16 后归一化。

## 精度与单卡内存

默认使用 **BF16 混合精度**：文本编码器和 DiT 的权重、激活与残差使用 BF16，
GEMM 保持 FP32 累加。时间嵌入、视频/音频输入输出投影和采样状态保留 FP32；
视频 VAE 默认使用 FP16 GEMM/attention，归一化、缩放参数与残差流保留 FP32；音频 VAE
保持 FP32。CLI 与 HTTP 服务均可通过 `--precision fp32` 切回全 FP32 对比模式，
也可用 `--vae-precision fp16|bf16|fp32` 单独覆盖视频 VAE，无需环境变量。
`GET /health` 返回当前 `precision`、`vae_precision`、`tensor_parallel_size` 和可见设备索引 `devices`。

文本编码只执行 Qwen3-VL 的前 50 层，读取 final norm 之前的残差，权重按层释放。
DiT 按每次前向的时间点数预计算 AdaLN 表，随后释放这些投影权重；BF16 主干约
37.5 GiB 权重驻留到本次去噪结束，后续采样步不再重复传输。视频和音频解码按阶段加载。
当非 FP32 主干、非 FP32 视频 VAE、总显存至少 68 GiB 且创建时空闲至少 60 GiB 时，
跨请求保留 DiT、视频/音频 VAE 权重，并复用完全相同的 AdaLN 时间表。其他配置按阶段
释放；取消或异常仍清空 GPU 权重。文本 checkpoint 字节保留在有界 CPU 缓存中，优先
使用 pinned 内存，按层上传 GPU；不缓存 prompt 的编码结果。

FP32 对比模式仍按层加载 DiT，并利用部分空闲显存缓存原始 BF16 权重，使用时精确
展开为 FP32。此缓存不改变运算精度，预算不超过启用时空闲显存的一半，预留至少
16 GiB 给工作张量。BF16 驻留模式需要比 FP32 流式模式更多显存；单卡基线在 72GB 设备上验证，多卡见下文。

官方 Diffusers 与 SGLang 均默认使用 BF16 主干，但混合精度、attention 内核与算子
融合方式不同，不能宣称最终输出逐位相同。使用原生输入的 BF16 独立算子／模块检查通过，
**完整流程尚未通过严格精度回归**；FP32 保留严格数值对比。实测误差和限制见 [VALIDATION.md](VALIDATION.md)。

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
  transformer/ transformer_ref/  # 参考素材模式需要后者
  vae/ audio_vae/
```

加载本地 safetensors / JSON 文件，无联网下载、转换脚本或额外权重副本。

## Pika H3 Flex

Pika 复用指定版本的 H3 基座（`42ed227ee7df40d41602854ae760620d6eb651fe`）。
另下载官方 `Pika-Labs/Pika-H3-Flex` 的 `fl2va/` 与 `ref2va/`，每个目录须包含
`manifest.json` 和原生 FP32 `adapter.safetensors`。两套增量合计约 5.15 GiB。
不需要转换或复制基座；ComfyUI 格式的适配器不适用于此入口。

```bash
CUDA_VISIBLE_DEVICES=7 build-h3/src/models/minimax_h3/minimax_h3_generate \
  --model /workspace2/hfmodels/Pika-H3-Flex \
  --prompt 'A small orange cat plays a piano. Soft piano music fills the room.' \
  --width 832 --height 480 --frames 124 --seed 42 \
  --device 0 --output pika.mp4
```

CLI 和 HTTP 服务根据模型目录中的 `fl2va/manifest.json`、`ref2va/manifest.json`
自动识别 Pika；不依赖目录名。识别后启用六次 DiT 前向，显式 `--steps` 只接受 `6`。
文字／首尾帧模式加载 FL2VA，参考素材模式加载 Ref2VA。HTTP 模型名为
`pika-h3-flex`，`num_inference_steps` 默认且只允许 6，Demo 自动识别。
Pika CLI/HTTP 默认输出 832×480；参考图像短边随输出档位取 1024 或 2048，
参考视频短边取 480、544 或 768，显式参考尺寸参数仍优先。

Pika 发布的是适配器，仍需本地 MiniMax-H3 基座。默认依次查找模型目录中的
`base_model` 目录或软链接、模型目录自身的基座组件、同级 `MiniMax-H3` 目录。
若基座放在其他位置，使用 `--model PIKA --base-model BASE` 显式指定；显式路径优先，
缺失或损坏的基座会报错，不会自动下载或退回原版模型。原来的
`--model BASE --pika PIKA` 写法继续兼容，但不与 `--base-model` 混用。
C++ 可调用 `resolveModelSource`（`models/minimax_h3/model_source.h`）取得基座与适配器路径，
以基座构造 Pipeline，再执行 `pipeline.loadPika(adapterRoot)`，并设置 `GenerateOptions.steps = 6`。

实现保留独立 LoRA 分支：FP32 文件加载为 BF16，两个低秩 GEMM 和基座结果相加时
遵循 BF16 舍入；没有离线合并权重。使用 `[1000,750,500,250,100,40,0]` 网格、
视频 shift=12／音频 shift=3，每步先计算 `x0 = x + sigma * velocity`，再用新噪声
形成下一步输入；最后一步直接返回 x0。FL2VA 条件潜变量在加噪前舍入到 BF16，
Ref2VA 保留 FP32。初始噪声、重新加噪、条件增强分别使用独立的随机数流。

当前 Pika 路径支持 BF16 主干，已验证单卡和双卡；拒绝 FP32 主干配置。双卡使用
`--devices 0,1` 并需要 NCCL，设备限制与下文的 H3 张量并行相同。例如：

```bash
CUDA_VISIBLE_DEVICES=6,7 build-h3/src/models/minimax_h3/minimax_h3_generate \
  --model /workspace2/hfmodels/Pika-H3-Flex --devices 0,1 \
  --prompt 'A small orange cat plays a piano. Soft piano music fills the room.' \
  --width 832 --height 480 --frames 124 --seed 42 --output pika-tp2.mp4
```

LoRA 随基座投影切分：列并行复制 A、切分 B；行并行切分 A、复制 B。FFN up/gate
分别切分。Pika 行投影的基座分支和 LoRA A 分支都先以 FP32 跨卡求和，再舍入至 BF16，
两分支结果最后相加；该精度规则不受调用方的 H3 行归并策略影响。文本编码和视频 VAE
继续使用现有 H3 并行路径，六步采样在 rank 0 执行，FL2VA/Ref2VA 可在请求间切换。

图像尺寸、参考素材及帧数限制沿用下文的 H3 接口。种子在本实现内可复现，C++ 随机数序列与官方
PyTorch CUDA 序列不同，同一个 seed 不承诺产生相同视频。`--model` 指向原版 H3 目录时，
继续使用原采样器和默认参数。数值验证与生成实测见 [VALIDATION.md](VALIDATION.md)。

## 显存、内存预算与磁盘加载

CLI 与 HTTP 服务均支持以下可选参数，生成请求不能修改进程预算：

| 参数 | 含义 |
| --- | --- |
| `--gpu-memory-limit GIB` | 每张 GPU 的预算；`0`／省略保留原有策略，非零须大于 1 |
| `--host-memory-limit GIB` | 整个服务共享的主机内存预算；`0`／省略保留原有缓存策略，非零至少为 2 |

```bash
build-h3/src/models/minimax_h3/minimax_h3_generate \
  --model /path/to/Pika-H3-Flex --base-model /path/to/MiniMax-H3 \
  --prompt 'A small orange cat walks through a sunny garden.' \
  --width 832 --height 480 --frames 124 \
  --gpu-memory-limit 4 --host-memory-limit 2 --output cat.mp4

# 双卡：每卡各有 4 GiB 预算，所有组件和 TP rank 共享一份 CPU 缓存预算。
build-h3/src/models/minimax_h3/minimax_h3_server \
  --model /path/to/Pika-H3-Flex --base-model /path/to/MiniMax-H3 \
  --devices 0,1 --gpu-memory-limit 4 --host-memory-limit 4 --port 8009
```

`ftllm launch` 的视频启动项提供「每卡显存预算」和「主机内存预算」。`/health` 返回
`gpu_memory_limit_gib`、`host_memory_limit_gib`、`gpu_memory_reserved` 和 `layer_streaming`。
C++ 调用方在首次生成前调用 `pipeline.setGpuMemoryLimit(4)` 和 `pipeline.setHostMemoryLimit(2)`。

**默认模式不改变原来的驻留、预取、精度和计算路径。** 显式 GPU 预算较充足时也保留原有
执行方式：当前 BF16 主干／非 FP32 视频 VAE 的保守阈值为单卡 60 GiB、TP 每卡 48 GiB。
较小预算才逐层换入文本、DiT、Pika LoRA 和视频 VAE，关闭双层预取及额外的 GPU 权重缓存。
单独限制主机内存不会开启 GPU 逐层换入。精度、采样步数和分辨率不会自动改变。

显式 GPU 预算使用独立 CUDA 内存池，上限为 `预算 − 1 GiB`，包括权重、dtype 转换、
激活和注意力工作区。启动时实际申请整个池，再回收为池内可复用空间，并保留物理页；
层间切换、请求结束及同步时不把这部分显存还给其他进程。预留失败在服务就绪前报错。
另外的 1 GiB 是 CUDA 上下文、cuBLAS/NCCL、池外运行时的开销余量，并非额外可用的权重空间。
显存池需要工具链／驱动支持 `cudaMemPoolProps::maxSize`，已在 CUDA 13.0 验证；旧工具链
仍可运行默认模式。物理上限与回收阈值的区别见
[CUDA 内存池 API](https://docs.nvidia.com/cuda/cuda-runtime-api/cuda_runtime_api/group__CUDART__MEMORY__POOLS.html)。

显式主机内存预算中，`预算 − 2 GiB` 是所有组件、分片和 TP rank **合计**的权重缓存上限，
优先使用锁页内存；分配失败或配额用尽则直接读文件。`2` 表示零权重缓存。
未缓存的权重按最多 16 MiB 的缓冲区读取并上传，TP 只读本 rank 的切片，既不映射整份模型，
也不建立整层 CPU 临时副本；无需转换或另存模型。操作系统可自行缓存、回收文件页。
缓存很小时，速度依赖磁盘、文件页缓存和 PCIe 带宽。

这些是**模型内存管理预算，不是 OS/cgroup 对整个进程的硬隔离**：主机的 2 GiB 是运行时、
分块上传和媒体处理的余量，第三方库、FFmpeg、超长参考素材及 C++ 调用者保留的完整 RGB
输出不属于权重配额。GPU 的库开销也随驱动和设备变化。部署时应给系统和其他服务留下余量。
低显存文件输出只保留当前时间块，不在 GPU 上保存整段 FP32 视频。
预算不足会明确报错，提示提高预算或降低输入尺寸／时长；HTTP worker 可继续处理后续请求。

### 最低显存如何计算

以下按实际 H3／Pika 权重形状、BF16 主干、FP16 视频 VAE 计算，不包含量化：

| 部分 | 单块权重占用 | 处理方式 |
| --- | ---: | --- |
| 文本编码器词嵌入 | 1483.75 MiB | 使用后释放；这是文本阶段最大的单块 |
| 文本编码器一层 | 约 930 MiB | 逐层模式关闭双层预取和 GPU 权重缓存 |
| DiT 一层 | 约 735 MiB | AdaLN 已预计算并释放；50 层只驻留当前层 |
| Pika 一层 LoRA | 25.375 MiB | 随主干层一起释放 |
| 视频 VAE 解码一层 | 约 128 MiB | 逐层释放，包括打包的 QKV |

DiT 主矩阵按 `2 × (4HA + 3HF)` 字节计算：`H=5376`、`A=56×128=7168`、
`F=14336`，结果为 735 MiB，另有少量归一化参数。
实际峰值还必须加上活跃张量、预计算的时间调制表、加载时的 dtype 转换和内存池碎片。
默认 832×480、124 帧的潜空间有 `37×15×26=14430` 个视频 token，另加音频和文本 token。
以 128 个文本 token 估算，打包序列约 14972 行，仅 Q/K 归一化时的输入和 Q/K/V 临时量
就约 `2×N×(2H+5A)=1.30 GiB`，因此单层权重的 760 MiB 不能代表最低显存。

各阶段串行，估算应取文本、DiT、VAE 的**阶段峰值最大值**，再加 CUDA／库开销，
不必把三个完整模型相加。高分辨率、长视频、长提示词、多参考素材以及 H3 更多步数的调制表
都会提高峰值。实测值与适用范围见 [VALIDATION.md](VALIDATION.md)。

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

### 张量并行

CLI 和 HTTP 服务均接受 `--devices 0,1`。例如使用物理 6、7 号卡：

```bash
CUDA_VISIBLE_DEVICES=6,7 build-h3/src/models/minimax_h3/minimax_h3_server \
  --model /workspace2/hfmodels/MiniMax-H3 --devices 0,1 \
  --host 127.0.0.1 --port 8009 --output ./h3-output \
  --demo tools/minimax_h3/demo.html
```

容器需同时使用 `--gpus '"device=6,7"'` 限制物理设备；容器内设置
`CUDA_VISIBLE_DEVICES=0,1`。HTTP 请求和 Demo 的模式、步数等参数不变。
单卡继续使用 `--device 0`。多卡需要 NCCL；CMake 自动查找已有的 NCCL，非标准安装位置
可指定 `FASTLLM_NCCL_INCLUDE_DIR` 和 `FASTLLM_NCCL_LIBRARY`。没有 NCCL 时仍可构建和使用单卡。

每卡独立 runtime、cuBLAS 工作区和权重缓存，复用 LLM 的 `PersistentWorkerGroup`：
rank 0 在调用线程前向，其余 rank 使用常驻线程。共享后端的 `ParallelContext` 负责
列切分、行切分、打包投影分组切分及通信，模型代码只声明权重布局。

| 部分 | 双卡划分 |
| --- | --- |
| 文本编码器 | 每卡 32 个 Q 头、4 个 KV 头、12800 维 FFN；CPU 缓存也只存本卡分片 |
| DiT／文本 refiner | 每卡 28 个注意力头、7168 维 FFN；up/gate 分别切分 |
| AdaLN | 投影按输出维度切分，汇集后缓存完整调制表 |
| 时间嵌入、输入／输出投影 | 保留完整 FP32 投影副本及原 GEMM 形状 |
| Norm、残差、采样器 | 隐藏维度保持完整；采样器在 rank 0 执行 |
| 视频 VAE | 按原有 tile 批次分配到各卡，在 rank 0 拼接；VAE 权重为副本 |
| 视觉文本塔、音频 VAE | 视觉塔在文本 rank 上复制；音频 VAE 在 rank 0 执行 |

BF16 GEMM 仍为 FP32 累加；行切分保留 FP32 部分结果，大张量先做 FP32 reduce-scatter，
加一次偏置并舍入，再 all-gather BF16/FP16 结果。小张量、非整除大小和 FP32 对照保留
FP32 all-reduce。大行投影与 FFN 按行分块，在独立通信流上归并，与后续分块的 GEMM
重叠；算子返回前建立流依赖。列切分权重直接从 safetensors 的 CPU 字节切片，
无需先在各卡加载完整权重。当前模型的头数和 FFN 维度要求 TP 数整除，维度允许
1、2、4、8；目前实测单卡和双卡，4/8 卡尚未运行验证。

双卡文本权重复用已有的有界 GPU 缓存，跨请求保留 BF16 分片；为 DiT、VAE 和激活
预留显存，缓存放不下的权重继续使用 CPU 缓存与原有上传路径。取消和完整卸载释放
GPU 缓存。没有新增配置项或环境变量。

FP32 双卡已通过小场景严格对照；BF16 单层误差较小，但完整轨迹仍有累积偏差，
不能作为逐位等价替换。测试配置、误差及速度见 [VALIDATION.md](VALIDATION.md)。

- 宽高须为 32 的倍数，各自介于 64..4096，总像素不超过 1344×768。
- 24 fps，帧数为 `17*n+5`，支持 124、141、…、345 帧。
- `--steps` 指包含终点的采样网格点数，50 点通常执行 49 次 DiT 前向。
- `--precision bf16|fp32` 默认为 `bf16`；分别对应默认 FP16、FP32 视频 VAE。
- `--vae-precision fp16|bf16|fp32` 可单独覆盖视频 VAE 精度，优先于上述默认值。
  HTTP 服务在启动时选择精度，生成请求结构保持不变；BF16 VAE 是可选对照路径。
- 音频为 32 kHz 双声道；分别使用视频 shift=12、音频 shift=3 的官方调度。
- 条件模式使用与 PyTorch CPU Generator 相同顺序的 MT19937／Box-Muller 随机数；先条件、再视频、再音频。
  纯文字模式保留既有随机序列。差分测试也可通过 `--initial-video` / `--initial-audio` 共享 FP32 噪声。
- `--first-frame FILE`、`--last-frame FILE` 可单独或同时指定，支持 PNG/JPEG。
- `--reference-image FILE`、`--reference-video FILE`、`--reference-audio FILE` 可重复，参数顺序即参考顺序。
  图像短边默认 2048；`--reference-image-short-edge 256` 可显式降低到诊断尺寸（改变条件语义与质量）。
  `--reference-video-short-edge` 默认 768，可显式降低以做小尺寸验证。视频参考会解码至 24 fps、
  自身比例的原生画布，再由纯 C++ 音频重采样器转换音轨至 32 kHz。
- `--trace DIR` 仅供本地开发验证，会保存中间张量、原始 RGB 与音频；HTTP 不开放此参数。

开发调优先用 `--width 256 --height 256 --steps 4`（实际 3 次 DiT）快速比较，确认
收益后再跑完整精度回归。少步数输出只用于测速和数值检查，不用于评判最终生成质量。

CLI 和 HTTP 默认先解码音频，再将视频 VAE 已完成拼接的帧分段交给后台 FFmpeg，
让编码与后续解码重叠。队列有容量上限，取消和编码失败会回收子进程与临时文件；
完成后才原子发布 MP4。编码参数保持 libx264、fast、CRF 18、4 线程及 AAC 192k。
C++ 调用 `Pipeline::generateMP4` 可使用相同路径；`Pipeline::generate` 仍返回原始
RGB/音频，`writeMP4` 仍可单独编码。视频 VAE 的中间投影按最多 4 个空间 tile 合批，
各 tile 的 attention 保持独立；FP32 对照保留原分块与矩阵形状。
HTTP 与普通 CLI 只保留 MP4，不再额外积累完整 RGB。C++ `generateMP4` 默认仍返回
RGB；仅需文件时可将最后一个参数 `retainRGB` 设为 `false`。CLI `--trace` 保留 RGB 输出。

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

首尾帧使用 multipart（`input_reference` 与 `first_frame` 是同一参数的别名）：

```bash
curl http://127.0.0.1:8009/v1/videos \
  --form-string 'prompt=A slow camera move between the two scenes.' \
  -F 'input_reference=@first.png;type=image/png' \
  -F 'last_frame=@last.jpg;type=image/jpeg' \
  -F 'size=768x448' -F 'seconds=5' -F 'seed=42'
```

参考素材使用重复的 `reference` 文件字段；每个 part 的媒体类型必须明确，保持上传顺序：

```bash
curl http://127.0.0.1:8009/v1/videos \
  --form-string 'prompt=The subject in Picture 1 moves like Video 1.' \
  -F 'reference=@subject.png;type=image/png' \
  -F 'reference=@motion.mp4;type=video/mp4' \
  -F 'reference=@voice.wav;type=audio/wav' \
  -F 'size=768x448' -F 'seconds=5' -F 'seed=42'
```

最多 9 张图片、3 段视频、3 段音频、共 12 份。音视频每段 2..15 秒，视频总长和独立音频总长分别不超过
15 秒；视频自带音轨归属于该视频。每个上传文件最多 128 MiB，请求最多 256 MiB。
不接受服务端路径或远程 URL。任务返回 `mode` 和 `reference_count`；Demo 提供模式选择、上传和参考顺序调整。

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
最多 1024 条；已完成任务的 JSON 元数据与 MP4 一起保存，重启会恢复这些任务。未完成任务不自动续跑。
