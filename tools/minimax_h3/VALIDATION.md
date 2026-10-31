# MiniMax-H3 数值验证

## 参考实现与范围

模型为 [MiniMaxAI/MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)，验证时的权重
revision 为 `42ed227ee7df40d41602854ae760620d6eb651fe`。参考实现使用
[Diffusers `aae97e79e8e6a55e0b43114f9c8916c06d4bca8c`](https://github.com/huggingface/diffusers/tree/aae97e79e8e6a55e0b43114f9c8916c06d4bca8c)
中的 `transformer_minimax_h3.py`、两个 `autoencoder_kl_minimax_h3*.py`、
`scheduling_minimax_h3.py` 和 `modular_pipelines/minimax_h3/`。这些参考源文件采用
Apache-2.0 许可。原生实现按其张量布局、运算顺序、调度与分块拼接语义重写；仓库不随附
模型权重或参考 Python 环境。

参考实现版权声明：Copyright 2025 The MiniMax Team and The HuggingFace Team. All rights reserved.
许可全文见仓库 [LICENSE](../../LICENSE)。

当前覆盖文字生成音视频（T2VA）、首尾帧及参考图像/视频/音频条件；各模式的验证范围见文末。
默认 BF16 混合精度，另保留 `--precision fp32` 严格数值基线，关闭 TF32。
BF16 下保留官方的 FP32 输入/输出投影、时间嵌入与采样状态；视频 VAE 默认 FP16
混合精度，音频 VAE 保持 FP32。视频 VAE 的具体精度边界和验证结果见文末。
这不等同于声称与官方默认输出逐位相同。

文本编码器为 Qwen3-VL 前 50 层的 pre-final-norm 残差。视频和音频分开归一化、使用不同
sigma shift，并按官方规则打包进同一个 DiT。视频 VAE 同时覆盖空间切片与时间交叠，
音频 VAE 覆盖双声道分别解码。测试直接使用官方权重，双方输入完全相同。

原生 C++ 与 PyTorch 的正态随机数序列不同。端到端测试显式共享初始噪声，避免把 RNG
差异误判为模型误差。RGB 比较在有损 H.264 编码之前进行，音频比较在 AAC 编码之前进行。

## 验证入口

仅开发验证需要已有的 PyTorch、Transformers、Diffusers、NumPy、safetensors；
推理库、CLI、HTTP 服务均不需要它们。构建时增加 `-DUNIT_TEST=ON`：

```bash
cmake --build build-h3 --target minimax_h3_reference minimax_h3_generate minimax_h3_server -j4
export CUDA_VISIBLE_DEVICES=7
H3_MODEL=/workspace2/hfmodels/MiniMax-H3
H3_LIB="$PWD/build-h3/libminimax_h3_reference.so"
H3_OUT=/workspace2/experiments/minimax-h3-native

python3 test/minimax_h3/test_geometry.py --library "$H3_LIB"
python3 test/minimax_h3/test_tokenizer.py --model "$H3_MODEL/tokenizer" --library "$H3_LIB"
python3 test/minimax_h3/compare_transformer.py --model "$H3_MODEL/transformer" \
  --library "$H3_LIB" --precision fp32 --output "$H3_OUT/block-fp32"
python3 test/minimax_h3/compare_dit.py --model "$H3_MODEL/transformer" \
  --library "$H3_LIB" --precision fp32 --output "$H3_OUT/dit-fp32"
python3 test/minimax_h3/compare_text.py --model "$H3_MODEL" \
  --library "$H3_LIB" --precision fp32 --output "$H3_OUT/text-fp32"
python3 test/minimax_h3/compare_video.py --model "$H3_MODEL/vae" \
  --library "$H3_LIB" --tiled --frames 12 --height 18 --width 4 --output "$H3_OUT/video-tiled-fp32"
python3 test/minimax_h3/compare_audio.py --model "$H3_MODEL/audio_vae" \
  --library "$H3_LIB" --output "$H3_OUT/audio-fp32"
python3 test/minimax_h3/compare_pipeline.py --model "$H3_MODEL" \
  --binary build-h3/src/models/minimax_h3/minimax_h3_generate \
  --precision fp32 --text-fixture "$H3_OUT/text-fp32" --width 128 --height 128 --frames 124 --steps 50 \
  --output "$H3_OUT/pipeline-128-50-fp32"
```

容器须仅暴露允许使用的物理 GPU；测试脚本要求只有一张可见卡。`compare_pipeline.py`
支持 `--skip-reference` 重用同一目录中的参考张量，须保证提示词、尺寸、帧数与步数不变。
其他提示词须先用相同提示词重新生成文本参考。测试在失败时仍保存比较报告。

`test_geometry.py` 检查 2..100 点、shift=3/12 共 198 组调度，以及 4 组不同宽高/时长的
打包布局，要求与官方实现逐位一致。组件与完整流程分别比较 relative L2 和最大绝对误差。
单次、单提示词数值通过不代表对所有尺寸和提示词穷举验证。

## 已完成的 FP32 验证（2026-10-30）

硬件为物理 GPU 7：RTX PRO 5000 72GB Blackwell，CUDA 13.0；参考环境为
PyTorch 2.14.0+cu130、Transformers 5.20.2。FP32 基线已提交至 `9235b6a7`；
下表记录切换 BF16 默认路径之前的结果。

128×128、124 帧、50 点采样（49 次 DiT），提示词为上述脚本默认的弹钢琴小猫。
同一份最终原生输出分别与官方默认 memory-efficient attention 和显式 math SDPA
比较；其余参数、权重、文本特征参考及初始噪声完全相同，均通过已有严格阈值。

| 最终输出 | 官方默认 attention | 官方 math attention |
|---|---:|---:|
| 视频 latent，相对 L2 | 9.87894e-6 | 1.61251e-5 |
| 音频 latent，相对 L2 | 6.09715e-5 | 1.85539e-5 |
| 解码音频波形，相对 L2 | 1.16337e-4 | 2.69258e-4 |
| RGB 最大通道差，0..255 | 1 | 1 |
| RGB 平均绝对通道差，0..255 | 0.0008753 | 0.0014220 |
| RGB 完全相同的通道比例 | 99.9125% | 99.8578% |

完整流程阈值为 video/audio latent、音频波形相对 L2 <1e-3，以及 RGB 最大通道差 ≤1。
阈值保持不变；通过修改计算顺序缩小误差。时间嵌入及 AdaLN 预计算保持单次前向的
至多两行 GEMM 形状，防止整段调度的大批量投影改变归约顺序。FP32 attention 的 Q/K
缩放顺序与官方 math SDPA 相同，并继续复用公共 cuBLAS attention。

`compare_pipeline.py --attention-backend math` 可显式选择 math 参考，默认 `auto`
保留 PyTorch 自行选择。`--skip-reference` 会检查已保存的配置，避免混用两种参考。
文本前 50 层的相对 L2 为 `3.27502e-8`；完整流程另外验证了本地 tokenizer 产出的
实际文本条件。7 组多语言分词输入、198 组调度与 4 组打包布局逐位匹配。

公共运行时回归通过：Qwen-Image 文本、DiT、VAE 相对 L2 分别为
`2.02954e-6`、`2.74033e-6`、`2.91727e-6`；HY-Nomas 运行时隔离检查通过，包括
张量跨 Runtime/工作线程存活、默认 CUDA pool 不被修改、外部 allocation 不被破坏。

视频 VAE 的独立空间与时间拼接用例（latent `12×18×4`，跨空间切片边界）相对 L2
`1.06633e-6`、最大绝对误差 `5.30481e-6`，通过 `<1e-4` 阈值。

## HTTP 回归

启动服务时使用 `--api-key test-h3`，然后运行：

```bash
python3 test/minimax_h3/test_http.py --base http://127.0.0.1:8009 \
  --key test-h3 --output "$H3_OUT/http-test"
```

该回归检查鉴权、参数校验、队列上限、运行中取消、删除、取消后恢复、multipart 提交、
实际 MP4 生成与下载。输出保存在仓库外，不把生成视频或中间张量加入 Git。

2026-10-30 的实际服务回归全部通过：先取消 4 个待处理任务，再由 multipart 请求成功生成
64×64、124 帧、2 点采样的 MP4，验证工作线程在取消后仍可继续使用。
FFprobe 确认 MP4 内为 H.264、24 fps、124 帧和 32 kHz 双声道 AAC。

加入原始权重缓存后，HTTP 回归会等到第一个完整 DiT 前向结束再取消，覆盖已经填充
缓存的状态。另测取消后的显存回收：整卡从 42228 MiB 回到测试前的 4820 MiB
（同卡两个空闲 H3 服务）；异常清理增加同步，确保异步内存池释放缓存后才报告取消。

FP32 原始权重缓存优化的短任务和完整 49 步回归分别保持 11/11、103/103 个输出文件
逐位一致；详细配置和速度记录见 [Benchmark](../../docs/benchmarks/minimax_h3.md)。

开启和关闭 `UNIT_TEST` 两种 CMake 构建均通过。测试关闭时可独立生成
`minimax_h3_generate` / `minimax_h3_server`，不构建 ctypes 比较桥接库。
`ldd` 确认服务只链接原生项目库、CUDA/cuBLAS 与系统 C/C++ 库。

共享运行时的变更还需要运行 `test/qwen_image/compare_reference.py`，覆盖已有文本、DiT
和 VAE 数值回归。速度及显存测量集中在 [Benchmark](../../docs/benchmarks/minimax_h3.md)。

## BF16 默认路径（2026-10-30，9235b6a7 之后）

依据 [Diffusers 示例](https://github.com/huggingface/diffusers/blob/aae97e79e8e6a55e0b43114f9c8916c06d4bca8c/docs/source/en/api/pipelines/minimax_h3.md)
的 `load_components(dtype=torch.bfloat16)` 与
[SGLang 精度配置](https://github.com/sgl-project/sglang/blob/3b6c28b0b9aff784115bafeecc67f37e1fe43454/python/sglang/multimodal_gen/configs/pipeline_configs/minimax_h3.py)：
DiT、文本编码器默认 BF16；SGLang 的视频 VAE 权重为 FP32、解码默认 FP16 autocast，音频 VAE 为 FP32。
该次提交首先切换文本和 DiT；以下历史测试的视频 VAE 仍为 FP32。
后续的视频 VAE 混合精度测试单独记录于文末。

复用连续张量运行时的 BF16 GEMM、FlashInfer attention、激活、切片和张量操作。
H3 RMSNorm 在 FP32 归约和乘权重之后舍入，RoPE、调制与残差保留 eager BF16 的舍入边界。
AdaLN 缓存按每一步的时间输入形状计算，第一步为一行，其余通常两行，避免合并时间点
改变 GEMM 归约。所有层和输出头复用每个计划的 SiLU 与类型转换结果。
未引入新的推理依赖或 H3 环境变量。

CLI、HTTP 服务和差分测试统一用 `--precision bf16|fp32`，默认 BF16。
参考模型在两种精度下均按层加载，避免重复维护驻留和流式两套分支。`compare_dit.py --isolated-blocks` 会把官方
各层输入分别交给原生同一层，检查局部误差，同时报告完整前向的累积误差；其通过状态
仅代表逐层检查，不代表完整前向精度通过。

- 文本第 50 层相对 L2：`8.44525e-4`。
- 50 个 DiT 层分别输入官方参考张量，最大相对 L2：`3.62394e-3`，全部低于已有的 BF16 单层阈值 `1e-2`。
- 合成随机输入的完整 50 层 DiT：视频/音频相对 L2 为 `0.0397141` / `0.136651`，
  仍超过 `1e-2` 阈值；没有放宽该阈值。
- 128×128、124 帧、4 点采样，端到端原生 BF16 对官方 BF16 auto：最终视频 latent
  相对 L2 `0.0870529`，音频 latent `0.0719304`，音频波形 `0.167790`，RGB 平均通道差
  `6.35234/255`、最大通道差 `158/255`。**未通过原 FP32 端到端严格阈值**。
- 控制实验中，官方 BF16 自身从 auto attention 切换到 math，最终视频/音频 latent
  相对 L2 分别为 `0.0723410` / `0.0414258`。这证明低精度后端差异会在迭代中放大，
  但不能据此证明原生误差全部来自后端或画质已经等价。

`compare_pipeline.py --precision bf16` 保留原始逐步报告和严格断言，当前上述用例会返回
非零状态；没有把失败改为通过。该少步用例用于数值诊断，不是生成质量评测。
输出质量和长轨迹 BF16 一致性仍需后续验证；FP32 对比模式用于保留严格参考。

保留的 FP32 模式重新运行 128×128、124 帧、50 点调度，并与同一份官方 math 参考比较：
视频 latent 相对 L2 `1.65346e-5`，音频 latent `1.80171e-5`，音频波形 `1.47940e-4`，
RGB 最大通道差 `1`、平均差 `0.00145992`。原严格阈值全部通过。第一步 AdaLN 现在按
真实的一行 GEMM 计算，所以不再承诺与旧原生输出逐位一致。

BF16 的 `UNIT_TEST=ON/OFF` 构建及 HTTP 回归通过，包含已驻留完整 DiT 后取消、队列
校验、multipart 恢复生成与 MP4 下载。独立取消检查中，整卡显存从 `39532 MiB`
回到测试前的 `2828 MiB`（包含同卡两个空闲 H3 服务）。没有恢复 Qwen 服务。
198 组调度与 4 组打包布局再次逐位匹配；共享运行时源码未改动。

整理前后，在 128×128、124 帧、4 点采样及同一份初始噪声下，BF16 和 FP32 各
11 个原始文件全部逐位一致（文本、初始噪声、3 步音视频 latent、RGB 与音频波形）。
参考模型改成统一流式加载后，文本的 50 个、DiT 的 52 个参考张量也与原记录逐位一致。
因此上述清理不改变已测精度，完整 BF16 未通过项仍然保留。

## 视频 VAE 混合精度（2026-10-30，5b75955b 之后）

默认主干仍为 BF16，视频 VAE 改为 FP16。`--precision fp32` 默认同时保留 FP32 VAE；
`--vae-precision fp16|bf16|fp32` 可以独立覆盖视频解码精度。HTTP `/health` 额外返回
`vae_precision`。音频 VAE 未改动。

视频解码使用如下精度边界：投影、FFN、RoPE 和 attention 为选定的低精度，GEMM
使用 FP32 累加；norm1/norm2、QK normalization、norm_out、scale1/scale2 与残差流
保持 FP32。复用公共连续张量运行时，扩展已有 FlashInfer adapter 支持 head_dim=64，
没有添加推理依赖或环境变量。空间与时间 overlap 的混合权重、乘积和和式也分别舍入，
与官方 eager 计算一致；不是仅在输出端强制转换类型。

参考模型以 FP32 权重加载，使用 `torch.autocast("cuda", dtype=...)` 解码；关闭 TF32
以及 FP16/BF16 GEMM 的低精度归约。这对应已发布的 FP16 autocast 配方，并可独立
检查 BF16。比较双方使用相同 latent，排除 DiT 的累积误差。

| 用例，相对 L2 | FP32 | FP16 | BF16 |
|---|---:|---:|---:|
| 单 tile，latent 7×16×16，1797 tokens，36 层 | 1.85728e-6 | 6.09056e-4 | 4.84972e-3 |
| 真实 latent，37×8×8 → 124 帧 128×128，时间拼接 | 1.13960e-6 | 9.78525e-4 | 7.58088e-3 |
| 空间与时间拼接，latent 12×18×4 | 1.06633e-6（此前基线） | 5.96701e-4 | 4.78319e-3 |

真实 latent 来自已验证的 FP32 128×128、124 帧、50 点官方参考的最后一步，先按模型
mean/std 反归一化；没有重新运行整段采样。低精度行对比相同 autocast 精度的参考。
`compare_video.py` 保留 FP32 `<1e-4`，新增 FP16 `<5e-3`、BF16 `<1e-2` 的输出阈值，
检查有限值及拼接输出仍可由目标 dtype 精确表示；单 tile 还保存每层残差的比较结果。

在同一真实 latent 上，原生 FP16 输出相对官方 FP32 解码的 RGB 平均通道差
`0.056406/255`、最大差 `3/255`、PSNR `60.61 dB`；原生 BF16 对应为
`0.465564/255`、`36/255`、`49.60 dB`。FP16 对相同精度参考的 RGB 平均差为
`0.049958/255`，最大差 `2/255`。这些是单用例数值指标，不代表通用视觉质量评测，
也不是逐位一致。结合近似的解码速度，默认采用误差较小的 FP16。

复现视频 VAE 组件对齐：

```bash
for H3_VAE_PRECISION in fp32 fp16 bf16; do
  python3 test/minimax_h3/compare_video.py --model "$H3_MODEL/vae" \
    --library "$H3_LIB" --precision "$H3_VAE_PRECISION" \
    --frames 7 --height 16 --width 16 --output "$H3_OUT/video-$H3_VAE_PRECISION-tile"
done
```

`--tiled` 验证完整空间/时间拼接；`--input FILE` 可以读取 `[T,H,W,24]` 顺序的 FP32
反归一化 latent。`compare_pipeline.py` 同样支持 `--vae-precision`，其完整流程的严格
阈值保持不变；重用旧 BF16 参考目录时需显式指定 `--vae-precision fp32`。

回归使用 128×128、124 帧、4 点和已有初始噪声：FP32 的 11 个输出文件与修改前逐位
相同；BF16 主干的文本、噪声、逐步音视频 latent 与音频波形共 10 个文件也逐位相同，
仅视频解码输出按新精度变化。已有 BF16 主干完整轨迹的精度未通过项不因此消失。
Qwen-Image 的文本、DiT、VAE 回归仍分别为 `2.02954e-6`、`2.74033e-6`、`2.91727e-6`。
连续张量运行时的生命周期、跨线程转移、默认 CUDA pool 隔离检查通过。
`UNIT_TEST=ON/OFF` 两种构建通过，服务依赖仍仅为原生项目库、CUDA/cuBLAS 与系统库。

FP16 overlap 的 F16C host 转换与 CUDA 软件 host 转换在真实 latent、跨空间边界两组
测试中逐位一致；不支持 F16C 的编译目标保留软件实现。默认 BF16 主干 + FP16 VAE
通过 HTTP 鉴权、参数校验、队列上限、驻留 DiT 后取消、恢复生成、multipart 和视频下载
回归；`/health` 确认为 `precision=bf16, vae_precision=fp16`。

提交前整理统一了公共 dtype 转换与 BF16 权重加载，合并 16 位 gather/scatter 复制，
删除视频 VAE 不使用的 FP16 DiT 分支和重复的 bool cast 接口；精度默认值统一解析，
H3 未新增运行时环境变量。整理前后 116 个 VAE 逐层/输出张量、FP32/BF16 两种主干
短流程共 22 个输出文件，以及 Qwen-Image 三个组件输出均逐位一致。测试仍只使用
物理 GPU 7；两种构建、运行时隔离、精度默认值/覆盖/非法参数与 HTTP 回归再次通过。

## DiT 算子融合与 FFN 分块（2026-10-31）

基于 `905521b6`，仅在物理 GPU 7 验证。三项优化依次完成回归后合并：

- 128 维 Q/K RMSNorm 与 RoPE 融合为一个 warp/head 的 CUDA kernel，保留原归约树、
  norm 输出及 RoPE 两次乘法的 BF16 舍入；其他 head 维度和 FP32 使用原路径。
- FFN 的 BF16 主块从 512 行扩大到 4096 行，融合拆分、SiLU 与门控乘法。
  **最后不足 512 行的 GEMM 必须独立保留**，避免 cuBLAS 在新形状下改变归约结果。
  FP32 仍用 512 行分块；临时张量有上限，未引入依赖或环境变量。
- Hidden RMSNorm 与 indexed AdaLN 共用归约 kernel，显式保留 norm、`1 + scale`、
  乘法三处 BF16 中间舍入。不会直接采用只在最后舍入的融合表达式。

`minimax_h3_fused_ops` 无需权重，以原分离算子为参考，覆盖 BF16/FP32、
1/17/47/513/12847 行、64/128 head 维度、多种 RoPE 长度、非整齐 hidden 维度和
非连续的调制索引。融合算子逐元素相等，相对 L2 为 0。仅暴露一张卡时运行：

```bash
cmake --build build-h3 --target minimax_h3_fused_ops -j4
CUDA_VISIBLE_DEVICES=7 ctest --test-dir build-h3 -R '^minimax_h3_fused_ops$' --output-on-failure
```

FFN 真实权重检查覆盖第 0、25、49 层，513/12847 行；保留尾块后与旧 512 行实现
逐元素相等。相同输入下，第 0 层完整 block 对官方 BF16 实现的相对 L2 在
513/12847 行分别约为 `1.512e-3` / `1.442e-3`，通过原有 `1e-2` 阈值。

每项优化分别进行 128×128 和 768×448、124 帧、4 点采样回归，prompt=`cat`、seed=42。
128×128 的全部 11 个原始张量/媒体文件，及 768×448 的原始 RGB、音频波形，
均与上一步逐字节一致。三项合并后，再验证 256×256、124 帧、50 点采样：
全部 **103 个文件**（含 49 步音视频 latent）与 `905521b6` 逐字节一致。
FP32 路径复用弹钢琴小猫、128×128、4 点的既有相同噪声用例，11 个输出也逐字节一致。
这些是相对于优化前 FastLLM 的回归结论；上文 BF16 与官方实现的既有差异仍然适用。

没有保留简单合并全部尾行的 FFN 方案：它在单层只有约 `1e-5` 相对 L2，
却在 768×448 的三步轨迹中放大为约 `6.6%` RGB 相对 L2。修复尾块后重新通过
严格相等检查，没有放宽端到端回归阈值。

实验脚本、逐项日志、前后共享库与 nsys 波形放在仓库外的 `dit-optimization` 目录；
推理库继续仅使用 C++、CUDA 和已有 cuBLAS/FlashInfer。

最终 CTest `minimax_h3_fused_ops` 通过；已重启的 HTTP 服务完成 64×64、124 帧、
2 点采样请求、MP4 下载及任务删除，Demo/OpenAPI 可访问。FFprobe 确认输出为
124 帧、24 fps、32 kHz 双声道音频，服务健康状态为 `ready`。

## FFN 输出直写与门控残差向量化（2026-10-31）

基线为 `631635aa`，仍仅使用物理 GPU 7，分两项实施并分别检查生成结果：

- 共享运行时的 `linearInto` 复用原 GEMM 实现，检查输出形状、类型、存储别名及
  16-bit biased GEMM 的 256 字节对齐要求。保留原 `linear` 接口与权重诊断输出。
  H3 BF16 FFN 直接写入最终张量的行切片，省掉 concat；GEMM 形状、leading dimension、
  FP32 累加与 `4096+4096+4096+512+47` 分块均不变。FP32 保留原分块输出路径。
- BF16 门控残差按行处理、使用成对读写和独立的 BF16 乘法/加法，保留中间舍入。
  窄行、奇数宽度、奇数调制表步长及未对齐输入使用原 kernel；FP32 路径不变。

扩展 `minimax_h3_fused_ops`，测试无需模型权重：

- 输出切片覆盖 FP32/FP16/BF16、有无 bias、1/47/512/513 行，与可精确累加的 CPU
  参考逐元素一致；同时检查切片前后哨兵，以及非法形状、类型、别名和对齐的拒绝。
- 残差覆盖 1/17/513/12847 行、127/512/5376 列、非连续调制索引与 gate 2/5；
  还检查正负零、极小数和未对齐输入。与分离的 gather/multiply/add 输出逐元素一致，
  相对 L2 为 0。最终 CTest 通过。

每项优化的 128×128、768×448、124 帧、4 点用例分别与前一步比较：11 个诊断/媒体
文件和 2 个媒体文件的 SHA256 全部一致。两项合并后，256×256、124 帧、50 点的
**103 个文件**与基线逐字节一致，FP32 相同噪声短流程的 11 个文件也逐字节一致。
Qwen 文本编码器、DiT、VAE 的前后输出完全一致；HY-Nomas 共享运行时的数学输出、
张量生命周期、线程移交及默认内存池隔离检查通过。

上述严格等价结论针对优化前 FastLLM；此前与官方 BF16 实现的差异仍按上文记录。
实验、共享库快照和波形保存在仓库外 `dit-output-residual`，未新增推理依赖或运行时
环境变量。

最终插桩与非插桩的原始 RGB/音频逐字节一致。HTTP 的 64×64、124 帧、2 点生成、
MP4 下载、任务删除以及 Demo/OpenAPI 检查通过；FFprobe 确认 24 fps、32 kHz
双声道音频。服务已恢复，状态为 `ready`。

## 文本缓存、权重驻留及 VAE 优化（2026-10-31）

以 `46954da5` 为严格等价基线，仅使用物理 GPU 7，每项优化分别通过回归后保留。
测试脚本、旧/新共享库、中间张量及 Nsight Systems 波形在仓库外
`remaining-optimization`；生产推理未引入 Python、额外库或环境变量。

视频 VAE 复用 RMSNorm、SiLU×gate、`linearInto`，融合 64 维 QK norm/RoPE、FP32
scaled residual，并将 unpatch 移到 GPU。RMSNorm 保持原 FP32 归约树；QK norm、
RoPE、门控激活以及 overlap 的中间低精度舍入不变；残差乘加仍禁止 FMA 收缩。
FFN 扩大主块但保留不足 512 行的尾块 GEMM；FP32 保留 512 行分块。

逐项数值检查包括：

- FP16/BF16/FP32 三种视频精度，各 37 个 tile 层级/输出张量，覆盖 128×128 和
  256×256 tile；完整 124 帧的三精度 128×128、默认 FP16 768×448 输出。
- 文本 BF16 1/257 token、FP32 17 token，每组 51 个张量；CPU 缓存及异步上传前后
  共 153 个张量逐字节一致。音频解码的 16 个层级/输出张量逐字节一致。
- FP16 overlap SIMD 对照原标量实现，477 例覆盖三个轴、不同通道、非整齐宽度、
  所有 overlap 长度、正负零和极小数；RGB SIMD 的 32 例包含尾部、舍入边界及四种
  FP32 舍入模式。输出均逐位一致，未修改 BF16 overlap 或 FP32 对照的舍入方式。
- 扩展 `minimax_h3_fused_ops`，覆盖上述 GPU 融合算子及三精度的 host cache：分片、
  预算不足、clear/disable、dtype 转换、跨线程上传、销毁缓存后使用输出。最终 CTest
  通过；测试无需模型权重。

GPU 驻留仅在满足显存条件的非 FP32 配置启用；相同 AdaLN 时间表才复用调制输出。
测试连续相同请求、修改提示词/seed/时间表后恢复，以及文本、prepare、DiT、视频、
音频阶段取消后重试，输出与初次请求完全一致。取消释放 GPU 权重；CPU 文本缓存
仍保留。异步上传使用 CUDA event 保护缓存生命周期，线程身份用唯一 token 表示，
避免操作系统复用线程 ID 后误判 CUDA 流。

最终验证 768×448、124 帧、4 点，基线/新生产/新插桩版本各两次请求，12 份原始
RGB/音频 SHA256 全部相同。加入缓存分配失败回退后，又重新验证两次完整生产请求。
256×256、124 帧、50 点的 **103 个文件**（含 49 步音视频 latent）与基线相同；
FP32 相同噪声短流程的 **11 个文件**也逐字节一致。Qwen 文本、DiT、VAE 三个组件
输出相同；HY-Nomas 数学输出、生命周期、线程移交和默认 pool 隔离回归通过。

最终 FP32 测试复现了文本开始阶段的 `invalid argument`。仓库外诊断构建定位到
`cudaHostAlloc` 为 embedding 分配 `1555824640` 字节时返回 `cudaErrorInvalidValue`；
改用 default flags 也会失败。该缓存是可选优化：现在对合法大块分配的 InvalidValue
保留普通 CPU 缓存，沿用同步上传路径，仍允许较小权重使用 pinned 缓存；
MemoryAllocation 则停止继续扩充缓存。
其他 CUDA 错误照常抛出。初版仅回退文件的方案会反复重试 pinning，波形观察到单次
重试耗时 8.70 秒；保留普通 CPU 缓存后无需重试。真实 FP32 回归通过；独立故障注入
覆盖这两种分配失败，并检查 InvalidValue 不会在 clear 后重新尝试缓存中的同名权重。
驱动拒绝该分配的内部原因未确定。早期两次 stage6 启动异常未记录调用点，不能断言
均为同一原因；原始异常和后续诊断保留在实验记录中。

所有“相同”均指与修改前 FastLLM 对比；不代表与官方实现或 SGLang 逐位相同。
此前 BF16 完整轨迹与官方参考的既有差异仍按上文记录，没有放宽阈值。
8-tile 批处理原型因变慢且改变舍入结果被排除，没有进入生产代码。

## GPU 拼接、RGB、音频分块及 MP4（2026-10-31，第二轮）

每项优化分别验证后保留，使用现有 C++/CUDA 运行时与 libc；没有增加推理依赖或
运行时环境变量。实验记录位于仓库外 `remaining-optimization-2`。

- GPU 空间/时间拼接保留原始邻接 tile 和中间舍入。FP16/BF16/FP32 的完整 128×128
  VAE，以及 FP16 768×448 的约 512 MB 浮点输出均与上轮基线逐字节一致。
  三种精度各测试 latent 帧数 3/4/5/6/7/8/12，共 21 个补帧/裁剪输出也完全一致。
- 无权重算子测试覆盖不均匀重叠、相接 tile、截短时间重叠、输出切片哨兵、非法输入；
  RGB 覆盖舍入边界、尾部和四种 FP32 舍入模式。音频卷积另对照可精确累加的 CPU
  参考，覆盖 17003 行、8/64/128 输出通道、普通/转置卷积、dilation 及尾块。
  最终 `minimax_h3_fused_ops` 通过（28.62 秒）。
- 音频维持 FP32。扩大 GEMM 分块会产生微小累加差异：独立组件 16 个张量的最大
  相对 L2 为 `4.59e-7`，最终单声道波形为 `2.23e-7`。没有改变原有官方对比阈值。
- 最终 FP32 同噪声短流程、256×256/50 点轨迹、768×448 生产与插桩结果，加上
  21 个边界用例，共 143 份文件中 137 份逐字节一致。所有文本、图像及采样 latent
  均一致；只有 6 份最终音频不同，最大相对 L2 `3.83e-6`、最大绝对误差 `4.99e-7`。
  接近静音的短采样音频相对误差稍高，对应最大绝对误差仅 `2.33e-8`。
- MP4 直传的 20 份文件、spawn 对比的 10 份文件均逐字节一致。最终两次完整生产
  请求的 RGB、音频、MP4 与前一步相同；短音频的 `-shortest` 输出也与旧实现相同。
  12 次缺失编码器/提前退出检查无 FD、子进程和临时目录泄漏，不改写已有目标文件，
  不改变进程 SIGPIPE 策略；成功重试通过。
- 在 VAE tile 内部取消后重试，RGB/音频与取消前完全一致；取消后显存约 2.46 GiB。
  HTTP 64×64、124 帧、2 点的生成、下载、删除、Demo/OpenAPI 检查通过，FFprobe
  确认为 24 fps、32 kHz 双声道，服务恢复为 `ready`。

以上严格比较对象是本轮修改前的 FastLLM；官方和 SGLang 的既有数值差异仍按前文
记录。默认分辨率的 50 点性能未测；小分辨率 50 点测试包含验证快照，仅用于正确性。

## VAE 合批与流式编码（2026-10-31，第三轮）

基线为第二轮结束的工作区。按流式编码、融合 QKV、空间 tile 合批、FFN 分块和
RoPE 缓存逐项验证，实验文件位于仓库外 `video-optimization`。只使用物理 GPU 7，
生产代码仍为 C++/CUDA，沿用现有 FFmpeg，没有新增运行时环境变量或依赖。

- 融合 QKV 与原先独立 Q/K/V 的 norm/RoPE 比较覆盖 FP16/BF16/FP32，行数
  1/7/513/1797 和多个独立 tile 的位置广播，结果逐位一致。
- 每步都比较真实权重的 36 层 tile 快照、最终投影，以及 768×448、124 帧的完整
  FP32 像素和 RGB 输出。最终保留的实现均逐字节一致。最初将输入和最终投影一起
  合批的原型产生相对 L2 `8.9236e-4`、最大 RGB 差 2，因此没有保留；输入和最终
  投影继续使用原 GEMM 形状，中间层合批，每个 tile 仍独立执行 attention。
- 比较 1/2/4/8 tile 批量，输出全部逐字节一致。最终低精度批量为 4；FP32 维持
  单 tile 和原 512 行 FFN。缓存按帧、高、宽失效，卸载清空 packed QKV 和 RoPE。
- 最终严格回归 **188/188 份文件逐字节一致**：4 次完整生成的 RGB/音频/MP4
  共 12 份（含 VAE 内部取消后的重试）、FP32 同噪声短流程 11 份、256×256/50 点
  轨迹 103 份、尺寸边界输出 62 份。边界覆盖三精度、latent 帧数 3/4/5/6/7/8/12，
  低精度额外覆盖 2/4/6/8 个 tile、跨请求尺寸变化、卸载后重载与流式/整段 RGB。
- 新增不需要 GPU 的 `minimax_h3_media` 测试：不规则分帧输入与整段输入的 MP4
  字节相同；缺失/提前退出的编码器、不可读 stdin、队列背压和结束等待中的取消、
  不完整输入及短音频均覆盖。检查子进程、FD、临时目录和 SIGPIPE 策略；失败或取消
  不替换已有目标文件。CPU 媒体与 CUDA 融合算子两项 CTest 均通过。
- 本轮新测旧版、最终生产版和插桩版各两次完整请求，最终版与插桩版的另 12 份
  RGB/音频/MP4 文件也与新测基线逐字节一致。
- 最终 HTTP 生成、内容下载、Demo/OpenAPI、删除及取消接口检查通过；FFprobe 确认
  64×64、124 帧、24 fps、32 kHz 双声道。服务已恢复 `ready`，两个容器均只映射物理
  GPU 7。最终整理仅调整注释和排版，重建 CLI/HTTP 后媒体 CTest 再次通过。

串行和流式编码使用同一组 libx264 `fast`、CRF 18、4 线程及 AAC 参数。历史 SGLang
文件使用 CRF 25、14 线程，编码质量和 CPU 资源不同，不属于相同编码条件的性能对照。
以上逐字节一致指相对本轮修改前 FastLLM，不代表与官方或 SGLang 逐位相同。

## 提交前整理（2026-10-31）

删除已被 `stitchVideo` 替代、生产路径不再调用的独立 `unpatchVideo` 接口和 kernel。
测试改用 CPU 展开参考直接验证单 tile/多 tile 拼接，保留原精度与尺寸覆盖。
移除 VAE 输入的中间 concat 拷贝、重复 pthread 链接，简化帧回调和 FFmpeg 参数列表；
没有新增运行时环境变量，未删除其他模型仍使用的 `FASTLLM_NOMAS_*` 配置。

CUDA 测试的设备数量检查移到所有 GPU 工作之前。隐藏 GPU 时正确返回 skip 77；
仅映射物理 GPU 7 时，媒体与融合算子 CTest 通过（0.29/29.34 秒）。独立 Release
构建的 CLI、HTTP、参考桥和测试目标全部通过。三精度边界输出 62 份及 768×448、
124 帧的浮点/RGB 输出 2 份，共 **64/64 份文件与整理前逐字节一致**。

此次验证使用独立构建目录，现有 HTTP 服务保持运行，已有任务与视频结果保留。
本轮未重跑完整采样或重新计时；此前 200 份生成/轨迹回归仍作为整理前验证记录。
清理差异、构建、设备隔离、CTest 和输出校验日志位于仓库外 `cleanup-20261031`。

## 条件生成：首尾帧与 Ref2VA（2026-10-31）

新增视频 VAE / 音频 VAE 编码端，复用共享 Qwen3-VL 视觉塔、MRoPE、GEMM、attention、
PNG 与图像缩放。推理仍为 C++/CUDA；FFmpeg / ffprobe 仅处理媒体文件。只在物理 GPU 7 验证。

FP32 基准使用原生 Diffusers/Transformers、相同输入与初始噪声：64×64、124 帧、2 点采样（1 次 DiT）。
参考图测试显式将 `reference_image_short_edge` 设为 64，生产默认仍为 2048；这些少步小尺寸输出不用于评价画质。

| 模式 | 视频 latent 相对 L2 | 音频 latent 相对 L2 | 解码 RGB 最大通道差 |
|---|---:|---:|---:|
| 首帧 | 1.56890e-4 | 2.35222e-5 | 1 |
| 尾帧 | 9.78009e-5 | 2.01559e-5 | 1 |
| 首尾帧 | 5.01776e-4 | 9.24027e-5 | 1 |
| Ref2VA 参考图 | 1.41116e-4 | 1.35539e-5 | 1 |

上述四例通过既有严格端到端阈值，没有放宽断言。首帧额外测试 4 点采样（3 次 DiT），
视频 / 音频 latent 相对 L2 为 `3.40e-4` / `6.49e-5`，RGB 最大通道差仍为 1，严格回归通过。
参考视频和音频目前验证了各组成部分及 HTTP 完整生成，尚未做混合媒体的原生端到端数值比较。
组成部分的检查如下：

- 图像 / 视频 VAE 编码最终相对 L2：`9.86e-7` / `2.51e-6`；空间分块 `8.86e-7`，时间分块 `1.93e-6`。
- 音频 VAE 编码最终相对 L2：`2.80e-6`。
- 视频双帧视觉 patch 与视觉塔：所有检查点低于 `1e-4`；视频条件经过完整 50 层文本编码后，L2 为 `7.39e-7`。
- 12 组关键帧布局、5 组混合 Ref2VA 布局与原生逐元素一致。12 组 Pillow / torchvision v2 缩放逐像素一致。
- 条件随机数的相对 L2 约 `3e-8`；posterior 采样测试最大 `1.16e-8`。
- 16 / 32 / 44.1 / 48 kHz 音频重采样对 torchaudio，最大相对 L2 `6.92e-6`。
- 共享改动的 Qwen-Image 回归：文本 L2 `2.03e-6`；视觉全部检查点低于 `1e-4`；PNG 编解码及 RGBA 缩放回归通过。

BF16 的局部误差与累计误差分开报告：

- 给各算子输入原生 BF16 张量，纯色图用例的 qkv/projection/fc1/fc2/GELU 逐位相同；
  LayerNorm 最大 L2 `2.46e-6`，带视觉 RoPE 的 attention L2 `9.86e-4`。
- 27 个视觉块分别输入原生输入，逐块及 merger 检查全部低于既有阈值；merger 最大 L2 `4.70e-3`。
- 纯色图完整视觉塔主输出 L2 `7.25%`，完整文本特征 `14.74%`；首帧端到端视频 latent
  `16.15%`、音频 latent `12.75%`，RGB 平均差 `10.56/255`、最大差 `151/255`。
  **完整 BF16 严格回归仍失败**，没有将局部检查通过解释为端到端精度或画质等价。
  这与已有 BF16 主干的累计差异一起保留；FP32 路径是严格数值基准。

新增差分脚本：`compare_video_encode.py`、`compare_audio_encode.py`、`compare_vision.py`、
`compare_image_text.py`（也覆盖视频条件）、`compare_vision_ops.py`、`test_conditioning.py`、
`test_condition_sampling.py`。`compare_pipeline.py` 支持 `--first-frame`、`--last-frame`、
`--reference-image`；`test_http_conditioning.py` 用专用服务检查所有上传模式及 MP4 音视频流。

HTTP 实测首帧、JPEG 尾帧、首尾帧、参考图、参考图＋44.1 kHz 音频、带音轨参考视频均成功，
FFprobe 确认输出 64×64、124 帧、双声道。非法上传、模式冲突、数量限制、鉴权、排队、
取消及取消后恢复检查通过。专用服务重启后，6 个完成任务的元数据正确恢复，下载视频逐字节一致。
这些测试采用 2 点采样（参考音频为 4 点）、参考素材短边 64，仅验证功能，不是生产分辨率性能测试。

随后用默认短边 2048 测速，发现参考图的 4096 个视觉 token 加提示词会触发共享 embedding
原有 4096 token 限制。将其与 H3 文本入口统一为 65536，未修改 embedding 计算。
新增 FP32 / FP16 / BF16 的 4096、4097、65536 token 精确 CPU 对照，及空输入、
65537 token、非法词表索引拒绝检查；完整融合算子回归通过。

## 条件模式逐算子优化验证（2026-10-31）

只使用物理 GPU 7，按每项改动先比较输出再进入下一项的顺序执行。生产推理仍为
C++/CUDA；独立原生验证和 Nsight 插桩留在仓库外 `conditioning-optimization`。

- 视频编码 GroupNorm 连续访问与融合 SiLU、causal im2col 索引优化：单个
  17×256×256 tile 最终 FP32 输出分别与原实现逐位一致。每步还运行原生图像、
  17 帧视频、22 帧时间切块、288×64 空间切块四组逐层检查，均通过。
- 卷积 GEMM 扩大到 8192 行：大 tile 相对旧实现 L2 `2.47856e-7`，最大绝对差
  `1.26839e-4`，并非逐位一致。四组原生测试通过，各组最大逐层 L2 分别为
  `2.28015e-6`、`4.02274e-6`、`1.91948e-6`、`8.90266e-7`。
  初始实验日志有一次误读旧文件的 equality 检查，已用实际保存的输出独立重算；
  `validation-audit.json` 与 `chunks/large-difference.json` 保留纠正记录。
- VAE 驻留：两次完整参考视频生成的 RGB、音频和 MP4，共 6 个文件逐字节一致。
  RoPE 位置复用通过原生 FP32 视觉、视频文本与 BF16 独立视觉模块检查。
- 文本 MLP 合批与下一层预取分别检查 14／1180／4110 token，最终特征均与对应
  修改前逐位一致。预取在层 0、1、24、48、49 取消后卸载，重新编码仍精确恢复；
  future 在异常退出时等待上传结束。共享 CPU 缓存测试覆盖源对象销毁后的读取，
  不同 checkpoint 的错误共享会被拒绝。
- GPU 视觉预处理的 CPU oracle 覆盖 FP32／FP16／BF16、32×64 与 96×64、
  单帧和双帧、RGBA alpha 边界 0/127/128/255、RoPE 坐标与非法输入。
  原生 FP32 图像、FP32 视频、BF16 独立视觉三组共 207 份中间张量与修改前逐位一致。
  最后四模式的 RGB、音频及 16 个 MP4，共 24 个输出文件与该步骤前逐字节一致。
- 完整融合算子测试（含上述 alpha 边界）及媒体测试通过。CLI、HTTP、参考桥
  和测试目标的 Release 构建通过，没有新增推理依赖或运行时环境变量。

沿用已有原生固定输入、初始噪声和严格阈值，最终 FP32 首帧 64×64／124 帧／
4 点采样的视频 latent L2 `3.40438e-4`、音频 latent L2 `6.49398e-5`；参考图
2 点采样、显式短边 64 为 `1.41116e-4`、`1.35539e-5`。两例 RGB 最大通道差
均为 1。没有放宽断言，也没有把局部通过解释成完整 BF16 等价。

**此前完整 BF16 的累计误差及严格回归失败仍存在**，具体数值保留在上一节；
本轮优化未解决该问题。相对旧版逐位一致与相对原生的严格精度是两种不同检查。

## 条件生成提交前整理（2026-10-31）

移除 Qwen-Image 的 RoPE/GELU 与 H3 indices 重复转发接口，直接继承共享 Runtime；
删除未使用的文本编码图像重载，保留实际使用的 `encodeVisual`。清理测试中的中途
重复报告写入和无关排版差异。远端 6 份 AppleDouble `._*` 元数据文件已删除。

HTTP 在同一把任务锁内完成元数据保存和完成状态发布，避免用户立即删除完成任务时
与 sidecar 写入发生竞态。上传集成测试增加完成响应与落盘元数据的一致性检查。

- H3、Qwen-Image、HY-Nomas 相关库、CLI/HTTP、参考桥及测试目标 Release 重建通过。
- 媒体与 CUDA 融合算子 CTest 通过（0.75／30.70 秒）；12 组关键帧布局、5 组
  Ref2VA 布局、12 组图像缩放、条件随机数／posterior 与 PNG/RGBA 编解码检查通过。
- 768×448／124 帧／4 点采样下，文字、首尾帧、参考图＋音频、带音轨参考视频
  各生成一次；RGB、音频、MP4 共 **12/12 份文件与整理前逐字节一致**。
- HTTP 首帧、尾帧、首尾帧、参考图、参考图＋音频、带音轨参考视频六种上传用法通过。
  重启后六个任务的参数与视频内容保留；删除后任务记录及 MP4/JSON/临时文件均清空。
  使用独立测试服务，只映射物理 GPU 7，未操作其他卡。
- Python 语法、Demo JavaScript 语法、OpenAPI JSON 与 `git diff --check` 通过。

没有新增 H3 运行时环境变量。当前共享后端保留 `FASTLLM_NOMAS_PRECISION`，
HY-Nomas pipeline 保留 `FASTLLM_NOMAS_TRACE`；上游已删除的分配器/attention 开关不再适用。
最初 HTTP 验证包装误用了 `--key`，
改为实际的 `--api-key` 后只重跑 HTTP 部分；该包装不属于生产代码。

以上检查验证整理没有改变输出，不改变上一节对完整 BF16 原生精度的限制结论。
记录、输出哈希、环境变量审计和建议提交分组保存在仓库外 `conditioning-cleanup`。

## 上游合并后的数值隔离（2026-10-31）

同步 `a9075766` 后，HY-Nomas 的共用 BF16 Attention 新增 FP32 softmax 分母规则，
LayerNorm affine 也新增 FMA 舍入边界。直接作用于 H3 时，4 点文生视频相对更新前
FastLLM 的 RGB/audio L2 分别为 0.0828176 / 0.151783；这不是相对原生模型的误差。

共享 `Runtime::Numerics` 显式选择上述两项规则，默认保留 HY-Nomas 上游行为，
H3 保留已验证的舍入顺序；两者仍复用同一底层 Attention 与 norm 实现。
同时把新 LingBot/HY-Nomas 的旧 FP32 字段访问适配到已有 DType 接口。

- H3、HY-Nomas、Qwen Image、LingBot 的库及相关可执行目标编译通过；未重新执行其他
  三个模型的完整生成流程。
- H3 fused-ops/media CTest、LingBot shared cast/upsample 检查通过。
- 6 个相同输入 Attention 案例覆盖 BF16 GQA/causal、自注意力、FP16 解码与 FP32；
  H3 策略与更新前逐字节一致，默认策略与上游新行为逐字节一致。独立 PyTorch 对照
  最大相对 L2 为 0.002952，FP32 为 6.62e-7；原生 BF16 单 block 检查通过。
- 文生视频 768×448/124 帧/4 点，以及默认参考图＋音频条件的 RGB/audio/MP4 共
  6 个文件，与此前对应 FastLLM 输出逐字节一致。参考条件覆盖视觉 LayerNorm 和
  72 维 head 的共享 Attention 路径。

没有新增运行依赖或环境变量。以上验证用于保证上游整合不改变现有 H3 输出；
此前记录的完整 BF16 流程相对原生的累积偏差仍需单独处理。
记录位于仓库外 `upstream-reuse-20261031`。

## 共享算子复用验证（2026-10-31）

本轮复用 SiLU×up、打包 SiLU、混合精度残差和索引调制，精度门槛采用相对修改前
逐字节一致，不引入新的误差容限。保留模型数值策略，不改变精度、累加类型或权重。

- 73 组新旧算子输出 SHA-256 一致：覆盖 FP32/BF16/FP16、单行/逐行 scale、
  打包/独立输入 SiLU、单行/索引调制及 BF16 双元素快路径。
- H3 fused-ops/media CTest 通过。补充了独立输入融合 SiLU 与分步实现对照，以及
  混合残差逐行 scale 和非法形状检查；现有非对齐、奇数 stride 和 BF16 边界测试保留。
- LingBot shared tensor 检查通过；新增 FP32/BF16 residual、branch、gate 的不同组合，
  以独立 CPU oracle 检查单行和逐行广播，显式禁止 host 乘加融合。
- 原生 BF16 单 Transformer block 检查通过，末端相对 L2 为 0.001017588，和本轮前相同。
- 768×448/124 帧/4 点的文生视频、参考图＋音频两种场景，RGB/audio/MP4 共 6 个
  输出文件均与修改前对应场景逐字节一致，新增相对 L2 为 0。
- H3、HY-Nomas、Qwen Image、LingBot 的库及相关生成、服务、reference/test 目标编译
  通过；`git diff --check` 通过。所有 CUDA 验证仅使用物理 GPU 7。

上述零新增误差是相对本轮前 FastLLM 的结论，不表示完整 BF16 生成与原生完全一致；
此前记录的原生累积偏差结论保持不变。其他三个模型未重跑完整生成。
纯 C++/CUDA 实现，没有新增运行依赖和环境变量。生产代码净减少 33 行，测试和
记录单独增加；执行路径退化的尝试未保留。数据、源码快照、逐算子耗时、输出哈希和
成对热测记录保存在仓库外 `operator-reuse-20261031`。

## 提交前整理与 DType 边界检查（2026-10-31）

清理融合测试中重复的列切片，更新已过时的环境变量说明。保留仍被 HY-Nomas 使用的
`FASTLLM_NOMAS_PRECISION`、`FASTLLM_NOMAS_TRACE`，没有新增 H3 开关或运行依赖。
补齐 LingBot 的 BF16 算子类型检查，避免 DType 迁移后把 FP16 指针按 BF16 解读；
纯布局操作保留输入 dtype，FP16 affine 参数通过已有 FP32 cast 路径处理。

- H3、HY-Nomas、Qwen Image、LingBot 相关库与生成、服务、测试目标编译通过。
- 仅 GPU 7：H3 fused-ops/media CTest 2/2 通过，LingBot shared tensor 检查通过。
  新增 FP32/BF16/FP16 布局、混合 dtype 拒绝、BF16 专用算子拒绝 FP16、
  FP16 affine 参数检查；输入 patch 与输出 head 按各自布局独立验证。
- 73 组算子输出与本次整理前逐字节一致。H3/LingBot 库中提取的全部 13 个
  sm_120 CUDA cubin 也逐字节一致，未改变 GPU 内核指令。
- `git diff --check` 通过。本次未重跑完整视频生成；性能及端到端精度结论沿用
  上节有明确配置与范围的实测，不将本次算子验证等同于完整原生精度对齐。

编译、测试、输出哈希及 CUDA 二进制对照记录保存在仓库外 `commit-cleanup-20261031`。
