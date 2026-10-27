# Qwen3.5 视觉张量并行

Qwen3.5 架构族（包括对应的 Qwen3.6/3.8 模型）的 CUDA 视觉编码器跟随普通 `--tp` 的设备列表和分片比例。注意力按完整 head 分配，MLP 和 merger 支持不均匀分片，可使用 3、5、7 等奇数卡配置。

```bash
CUDA_VISIBLE_DEVICES=0,1,2 ftllm server /path/to/model \
    --device cuda --tp 3 --atype float16 --multimodal
```

## 放置与精度

- 多卡 CUDA 视觉使用文本 TP 的所有设备；`--vision_device cuda:N` 在此配置下也跟随文本 TP。CPU 视觉和普通单卡执行继续沿用原有路径。
- 多卡 CUDA 模式下，分片前检查视觉权重的存储格式，接受未打包的 FLOAT32、FLOAT16、BFLOAT16 权重；量化或 GGUF 视觉权重会明确报错并指出权重名。
- 默认 FP16 路径的输出投影保留 FP32 部分和，归约后按原算子的舍入规则加入偏置与残差。跨 TP 的浮点结果可能不同，应结合 FP32 参考和原精度误差判断。
- 精度验证区分激活类型和权重存储类型；FP32/FP32、FP16/FP16、FP16/BFLOAT16 分别对照，其他混合类型组合需单独验证。这里的 BFLOAT16 指权重存储；当前 CUDA LayerNorm 不支持 BFLOAT16 激活。

## 多模态显存与缓存

`--multimodal` 在 KV cache 定容前加载视觉权重、预热并预留各卡固定工作区。工作区根据 processor 的单图/单视频上限和该卡实际分片确定，多张图片逐张复用。调整 processor 上限后需要重启；超出预留 patch 预算的 native 输入会报错。

每卡仍保留完整隐藏状态、残差和归约缓冲，应按各卡实际容量评估视觉显存。自动 KV 定容会使用释放出的空间，因此整卡显存还取决于 KV 预算、文本权重及运行时分配。

已有 verbose 日志在编码完成后输出每卡 `capacity`、`peak_cumulative`、`live`，并给出媒体数量和特征 token 数。`peak_cumulative` 是包含启动预热的累计高水位；整卡峰值需另外采样。

视觉最终特征保持 FP32 CPU 格式，图片 embedding 缓存继续驻留 CPU，文本 prefill 按分块上传。图片特征命中与 KV 命中分别统计；现有多模态 KV 安全 miss 行为保留。MTP、跨卡 KV/线性状态恢复和工具调用使用原有接口。

当 TP 卡数多于 KV heads 时，没有 KV 分片的卡不阻止前缀复用；保存 MTP 快照和恢复前缀前，会检查所有有效 K/V 与 MTP 分片是否齐全。

## 回归入口

- [原生分片、舍入与缓存测试](../test/basic/test_qwen35_vision_tp.cpp)
- [真实模型 API 回归](../test/api/qwen35_vision_tp_probe.py)：图片冷/热及换序、媒体页边界、跨卡文本前缀、MTP 原始 token 对照、流式与非流式工具调用。
