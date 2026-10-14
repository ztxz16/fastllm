# Qwen3.5 系列 CUDA 按层加载

Qwen3.5/3.8 dense 模型加载 safetensors 或 GGUF 时，每个 decoder 层完成读取、
转换和合并后，立即上传 CUDA 并释放 CPU 源权重；独立 lm_head 同样处理。
单 GPU 使用原有 `ftllm server <模型路径>`，多 GPU 使用 `--tp`，无需额外开关。
DFlash/MTP 的主模型也适用，draft、embedding 和视觉权重保持原有准备流程。

适用条件是完整目标权重布局和纯 CUDA 设备映射；单 GPU 的 low-memory、CPU KV
cache 模式不启用此路径。此改动不扩展其他架构、MoE 或 CPU offload。
GGUF 的分组机制不依赖具体量化类型，但仍要求当前后端支持该格式及切分布局。

## 实现要点

- GGUF 加载器遵循已有的模型加载分组回调：组内并行读取，组间串行上传。
- GGUF GDN 的 tiled-head 排列和衰减转换在上传前完成，按层记录，避免重复处理。
  安全 FP32 反量化标记在合并、切分前设置；混合量化继续使用分开投影的布局。
- 单 GPU、TP 和预热复用 GDN 权重准备函数。合并 qkvz/ba 后，流式路径删除源
  权重；普通前向及隐藏状态前向支持仅保留合并权重的预填充。
- 单 GPU 提前上传矩阵及卷积权重。一维 norm 等权重仍按原流程准备，保留 CPU
  RMSNorm 偏移处理；共享 embedding 的输出头仍在预热时生成。
- GGUF 外部 MTP 的 KV 头数校验读取已加载配置，避免 `--ori` 时误用基类默认值。

## 内存与验证范围

2026-09-08，RTX 4090 D + RTX 4090、每卡 24564 MiB、约 1.5 TiB 主机 RAM；单卡使用 RTX 4090。以下为进程 `VmHWM`，不含未映射文件页缓存。被测构建未记录完整源码版本，结果不代表当前 HEAD 的测量：

| 配置 | RSS 峰值 |
| --- | ---: |
| 单卡 Qwen3.5-9B BF16 checkpoint | 4.32 GiB |
| 单卡 Qwen3.5-4B FP8 测试副本 | 4.40 GiB |
| 单卡 Qwen3.8-27B GGUF Q4_K_M | 8.85 GiB |
| 单卡同 GGUF + DFlash2 | 12.55 GiB |
| TP2 Qwen3.8-27B-FP8 + DFlash2 | 8.92 GiB |

普通模型使用默认 server 参数；DFlash2 另指定 draft 和 `--speculative_algorithm dflash --max_batch 1 --tokens 2048`。4B FP8 副本将 decoder 线性矩阵保存为 F8_E4M3、128×128 block scale，其余保留 BF16；单纯给 BF16 checkpoint 指定 dtype 不等于验证原生 FP8。
上述数据是特定权重和构建的加载结果，不代表任意上下文或并发下的进程内存上限。

禁用 swap 的 systemd scope 冷启动检查中，单卡 9B 使用 8 GiB 限额，单卡 Q4_K_M 和 TP2 FP8+DFlash2 使用 12 GiB，单卡/TP2 GGUF+DFlash2 使用 16 GiB，均完成三次 256-token 请求。scope 限额包含文件缓存，与进程 RSS 口径不同；未在相同物理内存容量的整机上验证。

```bash
ctest --test-dir build-fastllm \
  -R '^qwen35_streaming_(tp|single_gpu)_weights$' --output-on-failure
```

回归覆盖 CPU 源释放、CUDA 字节一致、共享输出头、draft 隔离、CPU norm、GGUF 合并和混合量化布局，以及 GDN 转换幂等性与安全反量化标记。设备不足时跳过。
