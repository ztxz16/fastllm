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
仅支持 FP32 权重展开和计算，关闭 TF32；这是官方实现的 **FP32 数值基线**，
不等同于声称与官方默认 BF16/混合精度输出逐位相同。

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
  --library "$H3_LIB" --output "$H3_OUT/block-fp32"
python3 test/minimax_h3/compare_dit.py --model "$H3_MODEL/transformer" \
  --library "$H3_LIB" --output "$H3_OUT/dit-fp32"
python3 test/minimax_h3/compare_text.py --model "$H3_MODEL" \
  --library "$H3_LIB" --output "$H3_OUT/text-fp32"
python3 test/minimax_h3/compare_video.py --model "$H3_MODEL/vae" \
  --library "$H3_LIB" --tiled --frames 12 --height 18 --width 4 --output "$H3_OUT/video-tiled-fp32"
python3 test/minimax_h3/compare_audio.py --model "$H3_MODEL/audio_vae" \
  --library "$H3_LIB" --output "$H3_OUT/audio-fp32"
python3 test/minimax_h3/compare_pipeline.py --model "$H3_MODEL" \
  --binary build-h3/src/models/minimax_h3/minimax_h3_generate \
  --text-fixture "$H3_OUT/text-fp32" --width 128 --height 128 --frames 124 --steps 50 \
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
PyTorch 2.14.0+cu130、Transformers 5.20.2。源码为 FastLLM
`ed6a83aebb73508dd05d7e72ecab113a7fc51add` 上的未提交 H3 工作区，不能把基线 SHA
单独当作已包含此功能的版本。

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

快速优化用例为 256×256、124 帧、4 个采样点，prompt=`cat`、seed=42。优化前后
逐步视频/音频 latent、最终 RGB、音频波形等 11 个原始快照文件逐位一致，所有数学
算子、舍入与阈值均保持不变。该用例用于数值与速度对照，不用于评判少步生成的画质。

短测试确认收益后，补跑了 128×128、124 帧、50 点的完整回归。优化后 103 个原始
输出文件与原精度基线逐位相同，并再次通过官方 math attention 的差分；因此前述
最终误差表保持不变。

提交前清理移除了 BF16 计算试验分支、精度开关及重复的 Runtime/权重对象，保留
FP32 路径与原始权重缓存。再次运行上述快速用例和完整用例，分别得到 11/11、
103/103 个原始文件逐位一致，官方 math attention 差分仍通过相同阈值。

开启和关闭 `UNIT_TEST` 两种 CMake 构建均通过。测试关闭时可独立生成
`minimax_h3_generate` / `minimax_h3_server`，不构建 ctypes 比较桥接库。
`ldd` 确认服务只链接原生项目库、CUDA/cuBLAS 与系统 C/C++ 库。

共享运行时的变更还需要运行 `test/qwen_image/compare_reference.py`，覆盖已有文本、DiT
和 VAE 数值回归。速度及显存测量集中在 [Benchmark](../../docs/benchmarks/minimax_h3.md)。
