# Duo 音视频交互与原生 Realization 后端

浏览器在文件上传或录制结束后提交完整视频。支持 WebCodecs 的浏览器通过一个持续的 HTTP 响应接收 H.264 Annex B 与 32 kHz 双声道 PCM，每段 17 帧。其他浏览器使用 H.264/AAC 分段，回合结束均可下载完整 MP4。HTTP 服务及每个会话的推理工作进程均由同一原生可执行文件启动。原生流水线还提供逐块输入接口 `Pipeline::stream`。工作进程执行语音识别与对齐、视觉编码、Intent、Realization 和 Codec。Intent、语音对齐和 SolarX 使用 C++/CUDA、cuBLAS 和 cuFFT。FFmpeg 和 ffprobe 负责容器与音视频格式。

## 构建与运行对话服务

需要 CUDA 工具链、cuBLAS、cuFFT、FFmpeg、x264 和原生 Qwen3-ASR 源码。构建需安装原生开发包 `libx264-dev`、`libavcodec-dev`、`libavutil-dev`、`libswscale-dev`。以下命令适用于本次的 SM120 GPU。Qwen3-ASR 在构建时链接进服务。 Duo 的构建、运行与测试均无需安装 PyTorch，也无需设置 Torch 环境变量。

```sh
git clone https://github.com/antirez/qwen-asr /path/to/qwen-asr
git -C /path/to/qwen-asr checkout 924694251d9e0f18e5d86bbd06aa3ab5f870002d
cmake -S . -B build-duo -DUSE_CUDA=ON -DCUDA_ARCH=120 -DBUILD_DUO=ON \
  -DDUO_QWEN_ASR_SOURCE=/path/to/qwen-asr \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-duo --target duo_live duo_pack_intent duo_solarx -j 6

duo_bin=build-duo/src/models/duo/live
duo_model=/path/to/Duo-1.0-distill-quant
"$duo_bin/duo_pack_intent" "$duo_model" /path/to/intent-nf4
"$duo_bin/duo_live" "$duo_model" tools/duo /path/to/sessions 18190 four --intent-cache /path/to/intent-nf4
```

安装了 Duo 运行包后，用以下命令在前八张可见 GPU 上启动一个会话服务。

```sh
ftllm server "$duo_model" --tp 8 --port 18190
```

`--tp` 支持 1、2、4、8 张卡。也可用 `--devices 2,3` 选择当前 CUDA 可见范围内的指定卡号。原生服务对应的配置名为 `single`、`two`、`four`、`eight`。Duo 通过流水线分工和视频空间分块使用多卡。服务监听 `127.0.0.1`，可用 `ssh -N -L 18190:127.0.0.1:18190 USER@GPU_HOST` 转发。页面支持 512×384 和 256×192，接受 30 秒、64 MiB 以内的视频。已验证默认中文角色。语音对齐按汉字与空格分词，日语及韩语专用分词尚未验证。

首次上传或录制对话视频前，可在“助手角色”中选择 PNG、JPEG 或 WebP 初始图片，并填写角色和场景说明。图片按回复分辨率居中裁切，预览就是送入编码器的画面。两项可独立覆盖启动预设，同一会话的各轮沿用同一角色。结束会话后可更换。

原生接口 `POST /sessions/custom?resolution=512x384&playback=avc` 接收 multipart/form-data。字段 `reference_rgb` 为对应分辨率的 HWC RGB24 原始字节，`description` 为最多 2048 字节的 UTF-8 文本，至少提供一项。服务按会话使用发布的 VAE 编码器、后验均值和 Duo 通道统计量生成 BF16 参考 latent，角色说明进入 Intent 的系统前缀。使用启动预设时，沿用 `POST /sessions`。

NF4 缓存包含量化投影、反量化 scale 和嵌入等权重。缓存由本地原权重生成，建包目录需有约 21 GB 空间。服务检查源分片与 adapter 文件标识。省略 `--intent-cache` 可从原权重在首次请求中直接量化。该缓存只加快模型加载，各回合仍根据实际录音、画面和聊天历史计算 Intent。

## 跨块调度与精度

- 双卡模式下，GPU 0 执行视频输入编码、Intent 和 Realization，GPU 1 负责语音识别、对齐和音视频解码。
- 四卡和八卡模式下，可见 GPU 0 执行 Realization，GPU 1 执行视频输入编码和 Intent，GPU 2 负责音频解码及视频图块融合。四卡时 GPU 2、3 各解三个 512×384 图块，GPU 3 也执行 ASR 和对齐。八卡时 ASR 和对齐在 GPU 7 上执行。对于 512×384 的 ModelOpt FP8 模型，两张支持双向 P2P 的 SM120 卡在显存充足时共同执行主干 FFN：GPU 0 和 3 分别计算不同的 token 行，GPU 2、4、5、6、7 处理六个视频图块。其他配置沿用单卡主干，GPU 3 也可参与分块解码。只有一个图块时由 GPU 2 解码。Intent 在专用显存中保留部分解包后的 BF16 权重以加速预填充，单 token 解码继续读取 NF4。每个阶段独立按序消费数据，两级队列分别最多容纳两块。单卡沿用同一调度。
- Intent 使用原始 64 层 Qwen、BF16 激活、NF4 权重、发布的 LoRA 和可训练 token；每次提交真实前向第 50 层的指定行。FP8 Realization 使用 BF16 主干、四步 DMD 和分块 KV 提交。
- SolarX 使用发布的 V3 补偿表、七槽 BF16 KV、五步窗口移位及重叠槽重算。每五个 latent 直接输出本块的 17 帧。音频 VAE 按发布实现用 FP32 解码，每块按累计采样时钟裁剪。最终 MP4 从连续 PCM 编码一次 AAC。
- `presets/default.txt` 和两个 BF16 静态参考图 latent 对应官方默认角色，来源为 Duo 提交 `897a7fefdee3b93e9c911cd9eedfa4c7a6420b5a` 的 `demo/assets/assistant-reference.jpg`，图像 SHA256 为 `aa90569031374866a5fc2e7bcac4307adb252fc43017ba3dffc1c6c6e3ba621c`。使用发布 FP32 参考图编码器的后验均值按对应分辨率生成。
- 同一会话的后续回合保留 Intent 和 Realization 历史，每轮重新开始 SolarX 输出窗口。结束会话会终止该会话的工作进程和编码子进程，并释放其 CUDA 上下文。空闲十分钟后自动结束会话。浏览器显示文本与各段媒体，已生成的最终 MP4 可下载。

SolarX 的官方来源是 [SolarX V3](https://github.com/SolarX-Team/Duo-Streaming-VAE-Decoder/tree/a7c61a6e02bef8539178a17b6e150a775b3a0ebb)，许可和转换后的常量位于 `third_party/duo_solarx/`。原生数值验证使用三块、51 帧的连续解码，完整覆盖前两次窗口移位。

## 编译

```sh
cmake -S . -B build-duo -DUSE_CUDA=ON -DCUDA_ARCH=120 \
  -DBUILD_DUO=ON -DUNIT_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build-duo --target duo_generate duo_server duo_layout_test -j 8
```

按目标 GPU 设置 `CUDA_ARCH`。目前在 CUDA 13 / SM120 上验证；FP8 要求底层 FP8 后端
支持的 GPU。编译需要完整 CUDA 开发头文件，包括 `cuda_profiler_api.h`。
两个可执行程序在 `build-duo/src/models/duo/`，共享库在构建目录根部。

## DCP 导出

BF16＋增量路径使用 Python 标准库执行一次性格式转换。FP8 实时服务直接读取合并权重。

```sh
python3 tools/duo/import_checkpoint.py /path/to/Duo-1.0-distill \
  /path/to/imported-distill
```

脚本校验 release 中的 DCP SHA256、base 配置及 index 身份、精确的 LoRA target 集合、
rank 和有限值；输出 `adapter.safetensors`、`adapter.json`。不复制 backbone，不改变
下载目录，不覆盖已有导出目录。DCP metadata 含 Python pickle，仅用于可信来源的模型。
运行时比对导出的 source manifest 与选中模型的 manifest。

## HTTP 服务

```sh
# latent H,W = 输出像素 H,W / 16，均须为偶数。
# 当前实现优先用于正确性验证，建议从小分辨率开始。
build-duo/src/models/duo/duo_server /path/to/Duo-1.0-distill-quant \
  30 52 8099 - 0

# BF16＋distill 增量路径：
build-duo/src/models/duo/duo_server /path/to/Duo-1.0-distill \
  30 52 8099 /path/to/imported-distill 0
```

服务只监听 `127.0.0.1`，默认历史窗口为 2 块（当前块和前一块）。参考图与 shared
文本前缀始终保留。换参考图、shared 提示或取消后必须重新初始化会话。
每次 `/v1/chunk` 完成四步采样及 clean commit 后返回；不是逐 token SSE 接口。
HTTP 入口每个请求最多 32 MiB，忙时返回 409，不积压生成请求。

回归测试的线协议客户端位于 `test/duo/client.py`。

`reference_latents` 为 `[24,1,H,W]`，shared 和每块 Intent 为 `[N,text_dim]`。
不能传 token embedding、最终归一化 hidden state 或未经官方选择的全部 token 状态。
正式模型要求 Intent 的指定中间层特征，宽度为 5120。

线协议 `fastllm-duo-v1`：

| 路由 | 请求 | 响应 |
|---|---|---|
| `GET /health` | 无 | JSON，模型阶段、输入输出类型、忙/就绪状态 |
| `POST /v1/session` | 二进制包，JSON `shared_rows`；reference、shared 两个张量 | JSON 新 session ID |
| `POST /v1/chunk` | 二进制包，JSON `session,chunk,rows,seed`；Intent 张量 | 二进制包，metadata、video、audio |
| `POST /v1/cancel` | JSON `session` | JSON；失效会话须 reset |

二进制包由 uint32 little-endian JSON 字节数、UTF-8 JSON、连续 little-endian float32
张量组成。video 行序为 `[t,y/2,x/2,c,patch_y,patch_x]`；audio 为 `[channel,t,c]`。
`chunk` 从 0 连续递增。输出 `first_sample/end_sample` 使用
`ceil(frame * 32000 / 24)`，与官方半开音频区间一致，不能累计取整后的 708 ms。

运行时的 Intent、视听预处理、对齐和 SolarX 解码由 C++/CUDA 执行。

服务 RNG 是 C++ `mt19937_64`＋`normal_distribution<float>`，同一构建、输入和 seed
可重复；不承诺与 Torch CUDA 使用相同 seed 时噪声一致，也不承诺跨标准库位一致。
数值对照应使用 CLI 的显式噪声输入。

## CLI 与可复现验证

```sh
build-duo/src/models/duo/duo_generate MODEL REQUEST.json OUTPUT [ADAPTER|-] [DEVICE]
```

请求中的路径相对 `REQUEST.json`；文件都是无头 float32 little-endian：

```json
{
  "latent_height": 4, "latent_width": 4, "shared_rows": 3,
  "reference": "reference.f32", "shared": "shared.f32", "history_chunks": 2,
  "chunks": [{"rows": 4, "intent": "intent-0.f32", "seed": 42}]
}
```

可选 `low_memory:true` 逐层释放权重（很慢）。每个 chunk 可选 `noise:[文件名...]`：
causal 两个张量；四步 DMD 八个张量，顺序为初始 video/audio、每次重新加噪的
video/audio。错误、非有限值或取消均不会发布部分结果。

```sh
python3 test/duo/reference_fixture.py /tmp/duo-tiny
build-duo/src/models/duo/duo_generate /tmp/duo-tiny /tmp/duo-tiny/request.json \
  /tmp/duo-native /tmp/duo-tiny/adapter
python3 test/duo/compare.py /tmp/duo-tiny /tmp/duo-native
python3 test/duo/test_server.py --binary build-duo/src/models/duo/duo_server --fixture /tmp/duo-tiny
build-duo/src/models/duo/duo_layout_test
```

`reference_fixture.py --lora` 验证非零 FP32 LoRA，`--fp8` 验证逐通道权重及动态激活
量化，`--causal` 验证 50 步 Euler。每个 fixture 连续生成 3 块，第三块验证淘汰。
FP8 oracle 使用官方 Transformer 和显式量化投影；不是官方离线反量化 BF16 路径。
比较时分别计算每块视频和音频的相对 L2，即误差向量的 L2 范数除以参考向量的 L2 范数。
ModelOpt FP8 输出的上限为 0.001，与 MiniMax-H3 音视频流水线的输出对照一致。报告同时记录最大绝对误差，
可用 `--max-absolute` 指定额外门槛。BF16、LoRA 和 causal 基准继续使用相对 L2 0.0005 和最大绝对误差 0.001。
所有对照都校验形状和有限值，全零参考要求输出也全零。
使用 `reference_fixture.py --fp8 --production-head` 生成正式模型头宽为 128 的对照样本。
在 SM120 上，额外的 `--production-head` 四步 BF16 和 LoRA 样本最大绝对误差分别为 0.00121 和 0.00233，超过 0.001 门槛。两组相对 L2 均低于 0.0003，原生化前后的 Realization 输出逐字节一致。启动器的 Distill FP8 样本通过既有门槛。
独立投影测试 `minimax_h3_image_fp8` 用 CPU E4M3 参考计算校验 ModelOpt 的逐通道权重缩放和动态激活量化。

## 回归测试

回归样本来自 inclusionAI/Duo 的 `897a7fefdee3b93e9c911cd9eedfa4c7a6420b5a`。生成脚本可用 `git show d6ca83b5:test/duo/reference_fixture.py` 查看。仓库按 SHA256 去重保存权重、输入与官方输出；展开 12 种配置中的任意一组只需 Python 标准库。

展开四组基础样本后，可统一执行：

```sh
python3 test/duo/run_suite.py --build build-duo --fixtures /path/to/fixtures
# 加跑 128 维 FP8 对照，使用同一误差门槛
python3 test/duo/run_suite.py --build build-duo --fixtures /path/to/fixtures --production-head /path/to/head128
```

fixture 子目录分别命名为 `tiny`、`tiny-lora`、`tiny-fp8`、`tiny-causal`；测试输出
`verification.json`，包括各块误差、HTTP 检查结果和可执行文件 SHA256。

浏览器端可用含语音的视频验证上传、ASR 文本、助手回复和分段播放。`python3 test/duo/test_live.py INPUT.mp4 --url http://127.0.0.1:18190 --playback avc` 会检查 H.264/PCM 包、归档音视频时长、损坏输入重试、会话取消与重开。使用 `--playback mp4` 验证分段 MP4。
