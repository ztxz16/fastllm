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

### 完成结果（2026-07-14）

- 实现：在 graph capture 期间为每个 MoE 层插入 fork/shared-done/routed-begin/join 标记；capture 完成后重写依赖，将 shared expert 与 router/routed expert 变成两个分支，并在本地相加前 join。标记节点在 instantiate 前全部删除，replay 不增加额外 kernel。
- 默认启用：Qwen3.5 CUDA Graph 捕获成功后直接执行依赖重写，不新增环境变量；非捕获路径没有 marker kernel，重写失败时禁用该 GPU 的 graph 并安全回退。
- 拓扑验证：真实 batch-1 图识别到全部 40 个 MoE 层；Nsight API trace 记录到 160 个 marker node 被删除，并完成 81 组依赖添加。独立 CUDA Graph 自测验证双分支 join 的结果为 `11 + 31 = 42`。
- 数值验证：五次 512-token greedy 输出均保持 hash `36f0db830cf5094626be1a992ff6af3e7af42a6930b340a6804442d253d68ba7`，与阶段 2 完全一致。
- TP2 CUDA Graph 端到端：512-token 五次为 `215.592 / 216.379 / 216.590 / 215.994 / 216.088 tokens/s`，均值 `216.129 tokens/s`，标准差 `0.382 tokens/s`。
- 收益：相对阶段 2 的 `204.291 tokens/s` 提升约 `5.79%`；每 token 延迟 `4.895 -> 4.627 ms`，节省约 `0.268 ms/token`，略高于原先预计上界。
- 验证：`bash install.sh`、`regressionOps`、真实 TP2 的 batch 1/2/3/4 graph capture/replay 均通过。
- Profile 说明：Nsight Systems 2025.6 的 `--cuda-graph-trace=node` 会干扰这类 capture 后依赖重写，实际触发进程崩溃；普通 API trace 可验证图已重写，但逐节点时间不用于吞吐结论。性能结论以无 profiler 的固定输出 A/B 为准。

## 阶段 4：Router GEMV 与融合 Softmax/Top-K 微调（已完成）

### 完成结果（2026-07-14）

- Router GEMV：为真实 batch-1 形状 `1x2048 @ 256x2048` 增加 64-thread FP16 专用 kernel。每个线程复现旧 kernel 的四个 lane，并保持补偿求和次序，因此 256 个 FP16 logits 与旧路径逐 bit 一致；不满足形状、dtype、bias 或 `addTo` 条件时自动回退。
- 融合 Softmax/Top-K：将 `256 experts/top8` kernel 从 64 线程调整到 256 线程，全部线程并行完成 logits 加载、`exp` 和归一化；前 64 线程仍保持原来的 reduction 和 top-k 归并次序，并保留并列键 legacy fallback。
- 默认路径：两项优化在支持条件满足时直接启用，不新增环境变量；测试入口只用于显式比较 legacy 与专用 Router GEMV。
- 单 kernel 热态耗时（RTX 5090，Nsight Systems）：
  - Router GEMV：`1737.9 -> 1480.5 ns`，降低 `14.81%`；
  - fused softmax/top-k：`4688.3 -> 4433.2 ns`，降低 `5.44%`；
  - 合计每层节省约 `0.512 us`，40 层理论累计约 `20.5 us/token`。
- 直接融合原型已否决：cooperative-grid 版本会阻断阶段 3 的 shared/routed 并行；无 cooperative 的 last-CTA atomic 版本虽支持 graph capture 且结果一致，但热态 `6184.7 ns`，慢于分离后的 `5913.8 ns`，因此原型代码全部删除。
- 独立 worktree、相反启动顺序的两轮 TP2 A/B：
  - 第一轮：阶段 3 `219.228`，当前 `220.344 tokens/s`，提升 `0.509%`；
  - 反向顺序：阶段 3 `219.706`，当前 `220.590 tokens/s`，提升 `0.402%`；
  - 两组匹配均值：`219.467 -> 220.467 tokens/s`，提升 `0.456%`，延迟节省约 `0.0207 ms/token`，与 kernel 累计值吻合。
- 正确性：FP16/BF16/FP32、batch 1/4/4096、有/无 bias、norm 开/关、普通/并列 logits 共 72 组 index 和 score 均与旧路径逐 bit 一致；真实 512-token greedy 输出保持 hash `23f0b80085ccb381355cc49382e97730bef91844323d813b8fa8ac45f90ac83d`。
- 验证：`bash install.sh`、`regressionOps`、算子 microbenchmark 及真实 TP2 batch-1 CUDA Graph 服务均通过。

## 后续优化与剩余空间

当前匹配 A/B 基线约为 `220.5 tokens/s`（约 `4.535 ms/token`）。阶段 4 说明 Router 这条链已经接近“小 kernel 优化”的收益上限，下一步不再优先强行融合 Router GEMV 与 top-k；后续收益会互相重叠，不能把各项上界直接相加：

1. **RMSNorm/quant 与线性层入口融合，合并 linear-attention 的短 kernel 链**
   - 重点是 40 层中重复的 RMSNorm、dtype/quant 转换，以及 30 个 linear-attention 层的逐元素中间读写。
   - 单项可见空间约 `0.06–0.14 ms/token`，约 `1.3–3.2%`；需要先用 graph 拓扑确认哪些节点没有被现有分支并行隐藏。
2. **Graph/collective 调度优化**
   - 两次/层的 TP collective 已确认不是 129 us 的异常 AllReduce；空间主要来自隐藏规约尾延迟和缩短 token 间 graph 调度空隙，而不是重写 NCCL。
   - 单项可见空间约 `0.03–0.08 ms/token`，约 `0.7–1.8%`，正确性和调度风险高于算子融合。
3. **更深层的持久化 graph 或 attention/MoE 重构**
   - 可能突破上述范围，但会改变跨 token 调度、缓存生命周期或分支资源占用，不能按普通 kernel patch 的风险等级交付。

综合判断：剩余**较现实、可兑现**的空间约 `0.10–0.20 ms/token`，即再提升约 `2–5%`，稳定目标约 `225–231 tokens/s`。更深重构的激进目标仍可看 `235–240 tokens/s`，但不应作为当前计划的稳定承诺。下一步优先 profile RMSNorm/quant 和 linear-attention 短链，先确认真实 critical path 再选择融合边界。

## 风险与回退原则

- correction bias 作用于 softmax 概率而非 raw logits，融合实现不得改变排序语义。
- top-k tie-breaking 必须稳定，否则可能造成逐 token 输出分叉。
- 所有新快路径均须保留旧 CUDA 实现，形状或类型不支持时自动 fallback。
- 性能结论以 CUDA Graph 稳态 batch-1 端到端吞吐为准；Nsight 全量追踪只用于 kernel 分项，不作为吞吐基准。
