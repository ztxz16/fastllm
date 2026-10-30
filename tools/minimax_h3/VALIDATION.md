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

当前覆盖文字生成音视频（T2VA）。首尾帧、参考图像/视频/音频条件尚未实现。
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
