# LingBot 原生实现对照

验证脚本调用官方 Python 模型作为参考，原生库本身不依赖 PyTorch。部署步骤见 [LingBot 文档](../../docs/lingbot_world.md)。

参考代码：`Robbyant/lingbot-world-v2`，当前对照版本为 `1895d300d8ac936401689b26389f51cbd36530eb`。小模型权重版本为 `7e36a5f919f86cb4255cc9bfc30adb44963fbde1`；共享 T5/VAE 资源版本为 `5c33dd40b213598c418fd25bff30fdbd23fd38a7`。

测试环境使用官方依赖及 `numpy`、`safetensors`、`Pillow`。`prepare_inputs.py` 的官方相机处理还需要 `scipy`。这些包不进入部署服务。FlashAttention-2 可使用官方 `flash_attn`，或通过 `--attention vllm-fa2` 使用验证环境已有的 `vllm.vllm_flash_attn`。

## 对照范围

| 脚本 | 范围 | 判定 |
| --- | --- | --- |
| `compare_tokenizer.py` | 英文、中文、Unicode、HTML 实体、特殊 token、截断 | token ID 完全一致 |
| `compare_ops.py` | 相同输入的归一化、激活、Linear、注意力，以及官方图像预处理 | FP32 L2 < 2e-6、BF16 L2 < 5e-4、图像 L2 < 1e-5 |
| `compare_vae_ops.py` | VAE 的 FP32/BF16 归一化、时空卷积、空间卷积、1×1 投影及注意力 | FP32 L2 < 2e-6，BF16 L2 < 0.005；FP32 卷积使用 FP64 数值参考 |
| `compare_camera.py` | 连续平移、转向的相对位姿与 BF16 射线特征 | 相对 L2 < 2e-6 |
| `compare_text.py` | 官方 UMT5 编码器及完整 padding/mask | 相对 L2 < 0.01 |
| `compare_vae.py` | Wan 编码、解码、往返及分段缓存 | 编码与分块一致性 L2 < 0.001；BF16 画面归一化 RMSE < 0.01；FP32 解码 L2 < 0.001 |
| `compare_dit.py` | 去噪与提交历史的前向，跨缓存淘汰 | 相对 L2 < 0.01 |
| `compare_pipeline.py` | 相同噪声下四步采样、连续历史和最终画面 | 中间结果 L2 < 0.01，画面 L2 < 0.03 |

所有数值测试同时检查有限值。相对 L2 为 `norm(native - reference) / norm(reference)`。不要将通过单个算子的测试解释为整条链路通过。

cuDNN 会按形状选择不同卷积算法，其中 FP32 Winograd 的误差可能超过算子门槛。FP32 卷积用相同输入的 FP64 结果判定数值精度，并同时保留原生对官方 FP32 的误差、官方 FP32 对 FP64 的误差。BF16 算子以官方 CUDA autocast 为参考。完整 VAE 和连续推演使用与原生路径相同的解码精度，不用高精度卷积替换官方生成结果。

DiT 参考直接构造官方 `WanModelFast`，严格加载原始 FP32 checkpoint，主投影使用 BF16 autocast，时间与输出 head 保留代码明确要求的 FP32。当前官方辅助加载器会整体转换 BF16，这与 forward 中关闭 autocast 的 FP32 时间路径不一致；该辅助加载器不是数值参考入口。对照关闭 TF32 及 BF16 GEMM 的低精度归约，使用 FP32 累加。测试输出记录所用 attention 后端。

官方 head 后的 `unpatchify` 使用 `einsum`，在外层 autocast 下先将结果舍入到 BF16，再转回 FP32 参与采样。DiT 测试同时检查这个输出精度边界，避免把保留更多小数位误认为行为等价。

## 执行

从仓库根目录运行，下列变量指向各自的本地目录：

```bash
MODEL_DIR=/path/to/LingBot-World-2.0-1.3B
OFFICIAL_DIR=/path/to/lingbot-world-v2
LIBRARY=/path/to/build-lingbot/libfastllm_lingbot.so
RESULT_DIR=/path/to/validation
mkdir -p "$RESULT_DIR"

python3 test/lingbot_world/compare_tokenizer.py \
  --official "$OFFICIAL_DIR" --tokenizer "$MODEL_DIR/google/umt5-xxl" \
  --library "$LIBRARY" --output "$RESULT_DIR/tokenizer.json"
python3 test/lingbot_world/compare_text.py \
  --official "$OFFICIAL_DIR" --model "$MODEL_DIR" \
  --library "$LIBRARY" --output "$RESULT_DIR/text.json"
python3 test/lingbot_world/compare_vae.py \
  --official "$OFFICIAL_DIR" --model "$MODEL_DIR" \
  --library "$LIBRARY" --height 320 --width 576 --frames 5 \
  --output "$RESULT_DIR/vae.json"
```

算子桥接库仅供验证，不进入服务依赖。使用同一个构建目录编译，并运行独立图像预处理和算子对照：

```bash
cmake --build build-lingbot --target lingbot_world_test_ops -j
python3 test/lingbot_world/compare_ops.py \
  --bridge build-lingbot/liblingbot_world_test_ops.so \
  --attention vllm-fa2 \
  --output "$RESULT_DIR/operators.json"
python3 test/lingbot_world/compare_vae_ops.py \
  --bridge build-lingbot/liblingbot_world_test_ops.so \
  --output "$RESULT_DIR/vae-operators.json"
python3 test/lingbot_world/compare_vae_ops.py \
  --bridge build-lingbot/liblingbot_world_test_ops.so --precision bf16 \
  --output "$RESULT_DIR/vae-operators-bf16.json"
node test/lingbot_world/test_player.cjs
python3 test/lingbot_world/test_server.py
```

`test_player.cjs` 使用 Node 内置模块检查页面实际脚本的播放、移动、编辑和错误恢复，不替代浏览器的视觉验收。VAE 对照默认使用 BF16 autocast 解码、FP32 编码；BF16 单算子用 `compare_vae_ops.py --precision bf16` 检查，相对 L2 门槛为 0.005。解码与整段重建同时报告相对 L2 和 PSNR，门槛采用像素范围归一化 RMSE < 0.01（PSNR > 40 dB），避免接近零的低对比度输出使相对 L2 失真；编码与分块一致性仍使用相对 L2 < 0.001。传入 `--decode-precision fp32` 可回归原有 FP32 路径（全部门槛为 0.001）。连续推演对照读取 trace 中的 `vae_decode_precision`，使用相同精度解码；旧 trace 缺省为 FP32。

VAE 对照可额外传入 `--condition-inputs "$RESULT_DIR/inputs.npz"`，验证真实起始图片后接零帧的图像条件，并记录转成 BF16 后不同的值数。

`test_server.py` 使用 Python 标准库和模拟原生入口检查任务失败后的会话状态：推理失败保留已有会话，推理成功但画面写入失败则删除已前进的会话及不完整文件，避免客户端继续使用错误的片段编号。

`compare_text.py --prompt-file PATH` 可加入实际场景的长提示词；`--token-boundaries` 覆盖 31 到 512 个有效 token 的归约和 padding 边界。原生文本编码统一使用服务所用的逐层加载路径，测试直接调用同一实现。应覆盖实际提示词长度，不能只凭短文本对照判断编码器通过。

先生成真实文本、图片及相机条件，再执行 DiT 测试。该测试与默认的随机条件压力测试都应保留。

```bash
python3 test/lingbot_world/prepare_inputs.py \
  --official "$OFFICIAL_DIR" --model "$MODEL_DIR" \
  --example 01 --segments 4 --output "$RESULT_DIR/inputs.npz"
python3 test/lingbot_world/compare_camera.py \
  --official "$OFFICIAL_DIR" --inputs "$RESULT_DIR/inputs.npz" \
  --bridge build-lingbot/liblingbot_world_test_ops.so \
  --output "$RESULT_DIR/camera.json"
python3 test/lingbot_world/compare_dit.py \
  --official "$OFFICIAL_DIR" --weights "$MODEL_DIR/transformers" \
  --library "$LIBRARY" --attention vllm-fa2 \
  --inputs "$RESULT_DIR/inputs.npz" --segments 4 \
  --output "$RESULT_DIR/dit.json"
```

连续推演对照需要先从 `inputs.npz` 导出其 `first_image` 为 float32 THWC 原始文件，并将 `prompt` 写入 UTF-8 文本；使用相同宽高与 `intrinsics` 调用原生生成器。`--trace` 输出每次原生采样的噪声，参考代码直接使用这些噪声，不假设相同 seed 在不同随机数实现中生成相同张量。

```bash
python3 - "$MODEL_DIR" "$RESULT_DIR" <<'PY'
import subprocess, sys
from pathlib import Path
import numpy as np

model, result = sys.argv[1], Path(sys.argv[2])
inputs = np.load(result/'inputs.npz')
image = inputs['first_image']
image.astype(np.float32).tofile(result/'image.f32')
(result/'prompt.txt').write_text(str(inputs['prompt']), encoding='utf-8')
subprocess.run(['build-lingbot/src/models/lingbot_world/lingbot_world_generate',
    '--model', model, '--input-f32', str(result/'image.f32'),
    '--prompt-file', str(result/'prompt.txt'), '--width', str(image.shape[1]),
    '--height', str(image.shape[0]),
    '--intrinsics', ','.join(map(str, inputs['intrinsics'])),
    '--actions', 'stay,forward,look_left', '--output', str(result/'frames'),
    '--trace', str(result/'trace')], check=True)
PY
python3 test/lingbot_world/compare_pipeline.py \
  --official "$OFFICIAL_DIR" --model "$MODEL_DIR" \
  --inputs "$RESULT_DIR/inputs.npz" --trace "$RESULT_DIR/trace" \
  --output "$RESULT_DIR/pipeline.json"
```

连续推演的 `--trace` 仅用于上述同输入、同噪声的精度回归，不启用逐层张量导出。原始张量、日志和生成帧应保存在仓库外，不提交到源码。

当验证设备的空闲显存不足以常驻官方 DiT 权重时，连续推演对照可加 `--offload-reference`。它在每个官方 Transformer 块调用前后搬运权重，并关闭 autocast 的权重副本缓存；权重类型、算子及误差门槛不变。

## 条件缓存与状态回归

图像条件使用首帧加补零帧。`lingbot_world_test_condition_cache` 对比通用因果编码器与逐层条件缓存，覆盖两张独立图片、不同分块方式、超过完整时间感受野的序列，以及快照重放；全部结果要求逐位一致。

```bash
cmake --build build-lingbot --target lingbot_world_test_condition_cache lingbot_world_test_pipeline_state -j
build-lingbot/src/models/lingbot_world/lingbot_world_test_condition_cache \
  "$MODEL_DIR/native/vae.safetensors" 64 96
build-lingbot/src/models/lingbot_world/lingbot_world_test_pipeline_state \
  "$MODEL_DIR" "$RESULT_DIR/trace" 320 576
```

第二项使用生成器 trace 中的 `first_image.f32`、`0.text.f32`，宽高应与该 trace 一致。它检查不同场景分支交替执行、生成完成前取消、重试和旧快照恢复，要求画面与成功路径逐位一致。图像条件缓存不能用于任意视频输入；通用 `VAE::encode` 仍保留独立路径。
