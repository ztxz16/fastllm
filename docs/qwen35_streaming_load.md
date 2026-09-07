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

## 实测结果

测试日期：2026-09-08。硬件为 RTX 4090 D + RTX 4090，每卡 24564 MiB，主机
约 1.5 TiB RAM。单卡测试实际使用 RTX 4090。通过 `bash install.sh` 编译安装。
TP 基线为 `b3a02503`；单卡基线包含此前 TP 修复。前后程序使用独立动态库。

| 场景 | 修改前 RSS 峰值 | 修改后 RSS 峰值 |
| --- | ---: | ---: |
| 单卡 Qwen3.5-9B，BF16 checkpoint | 21.43 GiB | 4.32 GiB |
| 单卡 Qwen3.5-4B，共享输出头 | 12.52 GiB | 4.38 GiB |
| 单卡 4B FP8 测试副本 | 8.14 GiB | 4.40 GiB |
| 单卡 Qwen3.8-27B GGUF Q4_K_M | 22.87 GiB | 8.78 GiB |
| 单卡同 GGUF + DFlash2 | 26.76 GiB | 12.56 GiB |
| TP2 Qwen3.8-27B-FP8 + DFlash2 | 34.07 GiB | 8.96 GiB |
| TP2 GGUF Q4_K_M + DFlash2，带 `--ori` | 24.74 GiB | 12.97 GiB |
| TP2 GGUF Q4_K_M + 外部 MTP，直接读取 GGUF 配置 | 21.05 GiB | 9.21 GiB |
| TP2 GGUF IQ4_XS | 21.46 GiB | 9.11 GiB |

单卡普通模型和 GGUF 使用默认 server 参数；DFlash2 额外指定 draft 路径和
`--speculative_algorithm dflash --max_batch 1 --tokens 2048`。
FP8 副本将 4B 的 200 个 decoder 线性矩阵保存为实际 F8_E4M3，使用 128×128
block scale，其余权重保留 BF16；它是本地量化测试副本。BF16 checkpoint 仅加
`--dtype fp8_e4m3` 会回退 FP16，不计入 FP8 验证。

上述测试的修改前后 greedy 输出均一致，覆盖中文代码解释、数学题和英文故事。
单卡每项预热后运行六次 256-token 请求，解码速度变化均小于 0.15%；其中
DFlash2 为 98.019 → 97.920 token/s。TP2 FP8 + DFlash2 按前、后、后、前顺序
运行，每轮三次 512-token 请求，合计速度为 119.95 → 120.11 token/s。
速度按总 token 数 / 首个非空 SSE 内容至响应结束的总时间统计。
单卡启动本轮增加约 0.6～3.9 秒；启动时间是单轮观察，不作为稳定统计结论。

## 内存限额及回归

通过 systemd 用户 scope 设置 `MemoryMax`，禁用 swap；启动前仅对被测权重
执行 `POSIX_FADV_DONTNEED`。以下限额下均完成冷启动及三次 256-token 请求，
输出与基线一致，`oom=0`、`oom_kill=0`：

| 场景 | 内存限额 |
| --- | ---: |
| 单卡普通 9B | 8 GiB |
| 单卡 GGUF Q4_K_M | 12 GiB |
| 单卡 / TP2 GGUF + DFlash2 | 16 GiB |
| TP2 FP8 + DFlash2 | 12 GiB |

RSS 峰值取主进程 `VmHWM`，用 `/usr/bin/time -v` 交叉检查，不含未映射文件缓存。
scope 限额包含文件缓存，触及限额时发生正常回收。这是进程组限额验证，未在
2080 Ti 或对应物理内存的整机上验证。

```bash
ctest --test-dir build-fastllm \
  -R '^qwen35_streaming_(tp|single_gpu)_weights$' --output-on-failure
```

回归覆盖 CPU 源释放、CUDA 字节一致、共享输出头、draft 隔离、CPU norm 保留、
GGUF 合并与混合量化布局、GDN 排列与转换幂等性，以及安全反量化标记传递。
测试通过设备映射选择单卡或 TP，不设置环境变量；设备不足时跳过。

代码整理后重新编译安装，两项 CTest 均通过；复测单卡 9B、4B FP8、GGUF、
GGUF + DFlash2，以及 TP2 FP8 + DFlash2，每项三次请求的输出哈希全部与整理前
一致。RSS 峰值分别为 4.32、4.40、8.85、12.55、8.92 GiB；DFlash2 单卡 / TP2
速度分别为 98.27 / 120.29 token/s。记录为 `/tmp/fastllm-streaming-cleanup/results.json`。

本机原始记录保存在 `/tmp/fastllm-single-gpu-streaming/`、
`/tmp/fastllm-dflash-perf/`、`/tmp/fastllm-gguf-*.{summary.json,log}`；
整理前完整记录归档在 `/tmp/fastllm-streaming-cleanup/`。
