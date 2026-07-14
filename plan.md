# Qwen3.5-35B-A3B-FP8 Batch-1 优化计划

## 背景与基线

- 目标模型：`Qwen3.5-35B-A3B-FP8`，TP2，RTX 5090 x2，batch 1。
- 原始 eager 解码：约 125–131 tokens/s。
- 修正 CUDA Graph 预捕获范围后的诊断基线：约 186–188 tokens/s，约 5.35 ms/token。
- 每个 token 有 40 个 MoE 层；每层 router/MoE 关键链约 30.27 us：
  - FP16 router GEMV：1.85 us
  - FP16→FP32：0.68 us
  - softmax(256)：1.44 us
  - SelectExpert top8/256：10.01 us
  - FP8 gate/up + SwiGLU：7.41 us
  - FP8 down + top8 reduce：8.88 us

## 阶段 1：融合 Softmax + SelectExpert（已完成）

### 设计

1. 新增 CUDA 融合快路径，直接消费 router GEMV 的 FP16/BF16 logits，不再物化 FP32 softmax 张量。
2. 单 CTA 处理一个 token，针对 `numExperts=256`、`topk=8` 提供编译期特化：
   - warp/block reduction 计算 softmax 的 max 和 sum；
   - 严格保持当前语义：使用 `softmax_probability + correction_bias` 排序，但输出未加 bias 的 probability；
   - warp 内选 top8，再合并各 warp 候选；
   - 无并列候选时走 9-candidate warp 快路径；检测到 top8 内或 cutoff 处并列时，切换到与旧 `SelectExpert` 完全相同的归并次序，保证 expert index 和 score 逐 bit 一致；
   - 根据 `needNorm` 对选中概率归一化，并应用 `routeScale`。
3. 支持无 bias、有 bias、`needNorm=true/false`；不支持的 dtype/shape/topk 自动回退到现有 `ToFloat32 + Softmax + SelectExpert`。
4. 快路径必须支持 CUDA Graph 捕获与 replay：运行阶段不得动态编译、分配设备内存或执行主机同步。
5. 支持条件满足时默认直接走融合路径；不支持的 dtype、shape 或 topk 自动回退原实现。

### 接入点

- CUDA kernel/launcher：`src/devices/cuda/fastllm-cuda.cu`
- 声明：`include/devices/cuda/fastllm-cuda.cuh`
- DirectRunner/operator 接入：`src/devices/cuda/cudadevice.cpp`、相关 operator 头文件
- Qwen 公共 CUDA wrapper：`include/models/qwen3_cuda_common.h`
- Qwen3.5 eager 与 CUDA Graph 路径：`src/models/qwen3_5.cpp`

### 验证

1. 编译：`bash install.sh -DUSE_CUDA=ON`。
2. 数值验证：随机 logits、真实 `256 experts/top8`、有/无 correction bias、norm 开关；专家 index 必须与旧路径一致，score 按容差比较。
3. 端到端正确性：TP2 模型启动、非流式请求、CUDA Graph capture/replay。
4. 性能验证：同一 prompt、同一生成长度，预热后比较融合开/关；重新聚合 kernel 时间。
5. 目标：转换+softmax+select 从 12.13 us/层降至 3–5 us/层，节省约 0.29–0.36 ms/token；Graph batch-1 从约 187 提升至约 197–201 tokens/s。

### 完成结果（2026-07-14）

- 编译安装：`bash install.sh -DUSE_CUDA=ON` 通过。
- 单算子数值：FP16/BF16/FP32、batch 1/4/4096、有/无 bias、norm 开/关、普通及人为构造的并列 logits 均通过；index 与 score 对旧路径逐 bit 一致。
- 单算子耗时（RTX 5090）：
  - 旧 `FP16->FP32 + Softmax + SelectExpert`：约 12.4 us；
  - 融合普通快路径：约 4.9 us，约 2.53x；
  - 触发并列语义回退的融合路径：约 9.7 us，仍少于旧链路。
- 真实模型输出：128-token 与 512-token 的 greedy 输出 hash 均与旧路径完全一致。
- eager 同轮 A/B：512-token API 吞吐均值约 `123.2 -> 126.3 tokens/s`，提升约 2.5%。
- CUDA Graph：修正预捕获 batch 上限，使 `--max_batch 4` 只捕获 batch 1–4；capture/replay 成功。
- CUDA Graph 512-token 三次实测：`197.85 / 197.67 / 197.46 tokens/s`，均值 `197.66 tokens/s`；相对原诊断基线约 187 tokens/s 提升约 5.7%。
- 默认路径：无需设置环境变量即可启用融合；不满足支持条件时自动回退旧链路。

## 阶段 2：Batch-1 FP8 MoE warp kernel（已完成）

1. `gate/up + SwiGLU` 改为 warp-per-output、8 warps/CTA、合并 FP8 加载和 warp shuffle 归约。
2. `down + top8 reduce` 改为 warp-per-hidden-output，消除当前 top8 循环中的 block barrier。
3. 为 TP2 实际形状 `hidden=2048, localInter=256, topk=8, block=128x128` 提供专用路径，其他形状回退。
4. 预计再节省约 0.15–0.25 ms/token，累计达到约 205–212 tokens/s。

### 完成结果（2026-07-14）

- 路径核对：Qwen3.5 的三维融合权重实际走 `CudaFusedMOE`，专用 kernel 已接入该真实热路径；同时也优化了相同形状的 indexed-weight `MergeMOE` 路径。
- 实现：一个 warp 负责一个输出、每 CTA 8 个 warp；每个 lane 复现旧 64-thread kernel 中两个 lane 的工作，并按旧归约树先合并 stride-32，再用 warp shuffle 完成归约。
- 加载策略：每 lane 使用 4-byte FP8 读取，整个 warp 合并成 128B 事务。没有强行使用每线程 `uint4`，因为当前映射已经覆盖完整内存事务，而 `uint4` 不会减少事务数且会破坏逐 bit 对齐旧归约树的简单映射。
- 默认启用：支持条件满足时直接选择专用 kernel，不需要环境变量；不支持的 batch、shape、topk、block 或 dtype 自动回退旧实现。
- 数值验证：FP16、BF16 的 legacy/warp 输出逐元素完全一致；真实模型 512-token greedy 输出 hash 保持为 `36f0db830cf5094626be1a992ff6af3e7af42a6930b340a6804442d253d68ba7`。
- 单算子耗时（RTX 5090，真实 TP2 局部形状）：
  - FP16：`13.3 -> 7.9 us`，降低约 40.6%；
  - BF16：`13.3 -> 8.4 us`，降低约 36.8%。
- Nsight kernel 分项（FP16）：
  - `gate/up + SwiGLU`：`5.31 -> 3.82 us`；
  - `down + top8 reduce`：`7.59 -> 3.58 us`；
  - 合计：`12.90 -> 7.41 us`。
- TP2 CUDA Graph 端到端：512-token 三次为 `204.490 / 204.277 / 204.107 tokens/s`，均值 `204.291 tokens/s`；相对阶段 2 前同轮均值 `196.085 tokens/s` 提升约 4.19%，每 token 延迟 `5.100 -> 4.895 ms`，节省约 `0.205 ms/token`。
- 验证：完整 `bash install.sh -DUSE_CUDA=ON`、`regressionOps`、FP16/BF16 `fusedmoe_fp8` 与非目标形状 `mergemoe_fp8` fallback 均通过。

## 阶段 3：Shared Expert 与 Routed Expert 并行

1. 在 CUDA Graph 内将 shared expert 和 router/routed expert 建成两个独立分支。
2. 在本地相加及 AllReduce 前 join，保持现有数值语义。
3. 评估显存带宽竞争；预计可额外隐藏 0.1–0.25 ms/token，但只在阶段 1、2 稳定后实施。

## 风险与回退原则

- correction bias 作用于 softmax 概率而非 raw logits，融合实现不得改变排序语义。
- top-k tie-breaking 必须稳定，否则可能造成逐 token 输出分叉。
- 所有新快路径均须保留旧 CUDA 实现，形状或类型不支持时自动 fallback。
- 性能结论以 CUDA Graph 稳态 batch-1 端到端吞吐为准；Nsight 全量追踪只用于 kernel 分项，不作为吞吐基准。
