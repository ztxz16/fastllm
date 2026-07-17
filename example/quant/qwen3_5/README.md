# Qwen3.5 / Qwen3.6 低比特量化配置

本目录提供面向 Qwen3.5 架构文本主干的实验性 GGUF K-quant 配置。

- `Q2_K_ALL_TEXT.json`：所有文本 Transformer 线性权重均使用 Q2_K。压缩最激进，适合测试极限，不建议直接用于要求准确性的任务。
- `Q2_K_MIXED_TEXT.json`：两个最大的 MLP 输入投影 `gate_proj`、`up_proj` 使用 Q2_K；更敏感的注意力投影、线性注意力主投影和 `down_proj` 使用 Q4_K；小型递归门控、embedding、lm_head、视觉权重等保留 FP16。推荐从此配置开始。
- `Q2_K_MIXED_TEXT_MTP_FP8.json`：主模型保持上述 Q2_K/Q4_K 配方，但 MTP 草稿层保留源 FP8。推荐用于单卡低延迟解码；约 456MB 的高精度草稿层可以显著提高主模型精确验收的接受率。

Q2_K 属于 2-bit 量化档位，但还包含分组 scale/min 等元数据，实际权重存储约为 2.625 bit/parameter。

## 推荐的高速配方

Qwen3.5 的 MTP 会先用轻量草稿层预测多个 token，再由量化后的主模型批量验收。贪心解码时只提交与主模型 argmax 一致的候选，因此草稿层精度只影响接受率和速度，不会直接替代主模型答案。

导出时分别量化主模型和草稿层：

```bash
ftllm export /path/to/model \
  --dtype auto \
  --dtype_config example/quant/qwen3_5/Q2_K_MIXED_TEXT_MTP_FP8.json \
  -t 32 \
  -o /path/to/output
```

RTX 50 系建议编译对应的原生架构，而不是依赖旧架构 PTX：

```bash
cmake -S . -B build-sm120 \
  -DUSE_CUDA=ON \
  -DCUDA_ARCH=120 \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-sm120 --target fastllm_tools -j 16
```

单卡贪心解码推荐 `--mtp 5`。它对应 1 个当前 token 加 5 个草稿 token，恰好位于当前 6-token 小序列快路径内；`--mtp 6` 会越过该边界，实测反而更慢。

```bash
ftllm server /path/to/output \
  --device cuda \
  --tp 0 \
  --mtp 5 \
  --temperature 0 \
  --top_k 1
```

`temperature=0`、`top_k=1`、`repeat_penalty=1` 时使用主模型精确验收。采样模式使用 typical acceptance，不属于逐 token 等价模式。

## 实测结果

Qwen3.6-27B、单张 RTX 5090、原生 `sm_120`、batch=1、输入/输出均为 128 token：

| 配方 | Decode | Prefill | 12 项客观题 |
| --- | ---: | ---: | ---: |
| 主模型 Q2_K/Q4_K，不启用 MTP | 84.59 tok/s | 1128.36 tok/s | 10/12 |
| 主模型 Q2_K/Q4_K + FP8 草稿，`--mtp 5` | 158.26 tok/s | 979.60 tok/s | 10/12 |

该配方使 decode 提升约 87.1%，代价是本组短输入 prefill 下降约 13.2%。512-token 长输出测试中 decode 约为 156 tok/s。导出目录约 19GB；单卡服务配置 8192-token KV 配额时，空闲显存实测 15,766MiB。客观题失败项与不启用 MTP 的混合量化基线相同，说明 MTP 没有进一步降低这组任务的准确率；这不代表主模型量化精度与源 FP8 完全相同。

量化规则按顺序匹配，后面的规则可以覆盖前面的规则。
