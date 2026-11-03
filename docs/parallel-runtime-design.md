# 通用 Attention DP 与 TP 并行运行时设计

状态：已接通运行时、DeepSeek V4.1 适配器、bounded prefill 及普通 decode 整轮 CUDA Graph。下节描述当前实现；其余章节保留完整目标设计，不能视为所有阶段均已交付。

## 当前实现

新增 `--attn-dp-size`，默认 1 保留原 TP 路径。八卡实例支持 DP2/TP4、DP4/TP2、DP8/TP1，MoE 保持原 EP8 后端。例如：

```sh
FASTLLM_CUDA_GRAPH=1 ftllm serve MODEL --tp 8 --attn-dp-size 4 \
  --moe_device multicuda:0,1,2,3,4,5,6,7 --ngram_device cuda \
  --kv_cache_dtype fp4_e2m1 --fast_prefill --max_batch 64
```

启动时需要选择纯 GPU 专家路径；不支持的模型、设备数、专家布局或推测解码会明确报错。使用已有 `FASTLLM_CUDA_GRAPH=1` 开启 decode 图，关闭时保留 eager 路径。尚不支持 DSpark 或混合专家推理。

`--fast_prefill` 仍使用模型已有的近似 bounded replay：完整生成共享 KV 后，父级和各 attention 组同步裁剪 hidden、HC 系数及 query 索引；cache-only chunk 同步推进各 rank 的请求位置。图像输入和 DSpark 验证不裁剪候选行，未启用时保持原 prefill 语义。

普通 decode 在预热阶段复用局部投影图和已有动态 KV/稀疏 attention 算子，将选行、Attention TP 归约、合并收集、Engram、全局 FFN/router/MoE 及输出头组成每卡每轮一张图。BF16、FP8、FP4 KV 共用同一组成路径；输入 embedding 和最终采样仍在图外。请求的 KV 地址与位置逐轮上传，行置换更新到图持有的稳定缓冲区。

整图匹配各 DP 组的实际行数，允许请求身份、组内行序和上下文长度变化，不把一个组的 KV 指针当作另一个组的状态。每个组的局部行数仍受现有动态 attention 算子的上限约束；未预热行数分布、图失效或不支持的形状保留分段/eager 路径。局部图键同时区分行数与全局逻辑行数决定的量化模式；父图持有局部图及 workspace 的生命周期，每份整图独立持有通信 scratch，避免其他 batch 的扩容改变捕获地址。

通用部分位于 `include/parallel/` 与 `src/devices/multicuda/parallel_runtime.cu`：显式拓扑、组私有通信器、已有每卡固定 worker 的复用、局部 Executor、带逆映射的选行与收集，以及 load-time TP 权重重组。CUDA/NCCL 使用现有依赖，无新增环境变量或通信库。

运行时的集合通信支持抓图：所有参与 rank 必须一致选择捕获或 eager，该状态纳入已有集合通信签名检查；完整图仍由每卡固定 worker 同时发射。遵循 [NCCL CUDA Graph 的参与者与提交要求](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/cudagraph.html)。

运行时保留等待 GPU 完成的 `Run`，并提供仅等待主机提交结束的 `Enqueue`。后者用每卡事件衔接调用线程和固定 worker 的流；调用方需保持缓冲区存活，资源释放前仍经过同步边界。行映射和通信 scratch 跨 forward 复用，布局或容量变化时更新；attention 输出与三组 mHC 系数打包为一次逻辑收集，传输精度和总有效数据量不变。

模型适配位于 `src/models/deepseekv41_parallel.inc`。Attention 权重按组内 TP 切分；请求的 KV 只在所属组分配。组间只汇集 attention 输出及 mHC 系数，后续 residual、Engram、FFN 和 EP 使用原全局路径。前缀快照通过 CPU 逻辑状态恢复到新请求所属组；活动请求不迁移。量化策略按全局逻辑行数选择，避免 DP 拆分改变原模型的精度边界。

Attention TP 宽度变化仍会改变 `W_oB` 局部矩阵乘法的 BF16 舍入位置。其规则保持为局部 BF16 输出、FP32 归约、最终 BF16 输出；不同 TP 宽度的完整模型 logits 不保证逐位一致。数值验证需要同时使用相同 TP 宽度下的不同请求分布，以及按对应分片模拟舍入的独立参考，不能将分布改变与归约宽度改变混为同一个误差来源。

当前按存活请求长度平衡 placement，**尚未实现设计中的逐设备显存准入账本**；原调度器 token 预算仍有效，不能直接将其放大 DP 倍数。MLA KV 的副本数确实降为 Attention TP 宽度，但 Attention 权重复制、workspace 与负载偏斜也需要占用显存。原生 DP→EP 路由和其他模型适配属于后续阶段。

原 TP 路径和 Attention DP 路径均支持普通 FP4 KV decode 整图。当前动态 attention 每组最多 16 行，因此整图总 batch 上限随 DP 组数变化；这不是服务合批上限，超出的 batch 保留分段/eager 路径。

可复现检查：CPU 的 `parallel_topology_test`，八卡的 `parallelRuntimeRegression`，以及 `test/basic/test_deepseek_v41_attention_parallel.py --binary BUILD/deepseekV41AttentionParallelRegression --full-graphs`。模型检查覆盖 BF16/FP4 KV、四种布局、混合长度、解码、KV 物理归属、跨组前缀恢复、请求复用和跨布局 teacher-forced logits；需要 NumPy，生成 fixture 另需 PyTorch 与 safetensors。

追加 `--fast-prefill` 检查 bounded / cache-only prefill。

目标是在同一模型实例内组合 Attention DP、Attention TP 和 MoE EP/TP，并让每张 GPU 始终由一个固定工作线程负责。DeepSeek V4.1 是首个接入模型；分组、通信、张量布局、请求归属和生命周期管理不依赖其层数、head 数、KV 格式或推测解码算法。

核心决定是将通信组从全局变量中分离，将请求归属从设备编号提升为 Attention TP 组，并通过显式布局转换连接 attention 与 FFN。先保留统一调度和现有专家计算后端，建立可验证的完整路径，再优化跨组通信和计算重叠。

## 并行维度与配置语义

设一个模型实例、一个流水线阶段内共有 `W` 个 GPU worker。首期限定单进程单节点，模型副本数和流水线阶段数均为 1。

| 符号 | 含义 | 约束 |
| --- | --- | --- |
| `A` | Attention DP 组数，组间处理不同请求 | `A * T = W` |
| `T` | 每个 Attention DP 组内的 Attention TP 宽度 | 由 `W / A` 推导 |
| `E` | 一组 MoE 内的专家分区数 | 首期 `E * M = W` |
| `M` | 单个专家分区内部的权重 TP 宽度 | 首期由 `W / E` 推导 |
| `F` | 某个 dense/shared FFN 的 TP 宽度 | 由该层执行计划声明，必须整除其设备域 |

Attention 和 MoE 是同一批设备的两种分解，不能计算成 `A * T * E * M` 张卡。将来的 MoE DP 可表达为 `MoEDP * E * M = W`，但首期将 `MoEDP` 固定为 1；CP、PP、多节点和多模型副本不隐式开启。

八卡应覆盖以下 attention 布局；MoE 可以在同一设备域继续使用 EP8：

| 设备数 | Attention DP | Attention TP | MLA 长期 KV 的全机副本数 |
| --- | --- | --- | --- |
| 8 | 1 | 8 | 8 |
| 8 | 2 | 4 | 4 |
| 8 | 4 | 2 | 2 |
| 8 | 8 | 1 | 1 |

此处副本数仅描述采用完整 latent 复制的 MLA。GQA/MHA 的 KV-head 分片、GDN 的 recurrent state 分片由模型适配器声明，不能套用 MLA 的计数。

对外拟保留 `--tp` 现有的设备选择语义，新增 `--attn-dp-size A`，例如 `--tp 8 --attn-dp-size 4` 表示八卡实例、Attention DP4、Attention TP2。`--tp 2 --attn-dp-size 4` 是非法组合，不解释为八卡。完整模型副本 DP 与 Attention DP 使用不同概念，首期不复用 `--dp-size` 表达两者。

内部统一通过 `ParallelOptions` 解析拓扑。当前 `--ep` 有 MiMo 专用的枚举与校验，新运行时不能通过放宽整数检查就宣称其他模型支持 EP/TP 混合。旧命令维持旧路径；显式选择新并行模式时，先解析布局，再依据模型、权重格式和后端 capability 校验。启动摘要必须显示 `world / attn_dp / attn_tp / moe_ep / moe_tp / dense_tp`，未支持的组合在加载权重前报错。

SGLang 的 Attention DP 同样在 TP 设备域内划分 attention 组；其 EP 与专家 TP 的可用组合还取决于通信后端。这里只对齐并行概念，不假设不同后端具有相同能力。[Attention DP 说明](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/dp_dpa_smg_guide.mdx)，[EP 后端说明](https://github.com/sgl-project/sglang/blob/main/docs/docs/advanced_features/expert_parallelism.mdx)。

## 八卡混合布局实例

逻辑 rank 到实际 CUDA device 的映射保存在拓扑中；下表用连续编号便于阅读，运行时也必须支持非连续设备列表。

对于 `W=8, A=4, T=2, E=8, M=1`：

| Attention DP 组 | Attention TP 成员 | 组内 leader | 请求状态所在位置 |
| --- | --- | --- | --- |
| 0 | GPU 0、1 | GPU 0 | 只在该组；MLA KV 在组内两份 |
| 1 | GPU 2、3 | GPU 2 | 只在该组；MLA KV 在组内两份 |
| 2 | GPU 4、5 | GPU 4 | 只在该组；MLA KV 在组内两份 |
| 3 | GPU 6、7 | GPU 6 | 只在该组；MLA KV 在组内两份 |

同时存在四个 attention TP 通信组、leader 组 `{0,2,4,6}` 和覆盖八卡的 MoE EP 组。每张 GPU 仍只有一个执行线程；GPU 0 的线程处理 attention 组 0、leader 组和 EP 组的相应操作，GPU 1 的线程处理 attention 组 0 和 EP 组的操作。

专家 TP 的分组独立于 attention。例如 `E=4, M=2` 时，专家 TP 组为 `{0,1}`、`{2,3}`、`{4,5}`、`{6,7}`，EP 组分别连接相同专家 TP 坐标的 `{0,2,4,6}` 与 `{1,3,5,7}`。一次专家输出先在对应专家 TP 组内归约，再按 EP 语义合并不同专家的贡献，不能重复求和。

组成员列表是显式、有序的。默认连续分组便于组内 TP 使用较快链路，但算法只依赖组成员表，不依赖 `deviceId % groupSize`。

## 当前代码与改造边界

| 现有实现 | 可以保留的部分 | 必须改变的部分 |
| --- | --- | --- |
| `MultiCudaDedicatedWorker` | 按 GPU 复用线程、事件和 per-thread stream | 提升为可共享的执行资源，禁止嵌套创建 GPU worker |
| `PersistentWorkerGroup` | 持久任务和异常汇总机制 | 模型接入新运行时后借用同一设备 worker，不再另建线程组 |
| LLM NCCL 封装 | 现有 collective、提交协调、错误处理经验 | `g_ncclComms / g_ncclRanks / generation / rendezvous` 改为通信组私有状态 |
| DiT `nomas::ParallelGroup` | 对象持有 communicator、异步 buffer 保活、精度协议 | 提取不依赖 DiT Tensor 的机制，避免再复制一套 LLM 通信框架 |
| `Data::tpLayout / tpAxis / tpRanges` | 单卡 payload、已有分片范围和算子入口 | 在其外增加同时表达请求分区与特征分片的布局，不直接改写全部 Data |
| DSV4.1 的本地 CUDA kernels | attention、量化缓存、indexer、top-k、压缩和 rotary 算子 | 移除新路径对全 world KV 副本的依赖，保留本地数值实现 |
| Qwen3.5 与 Clover 的 rank 前向 | 每 rank executor、缓存、异常传播模式 | 通过适配器接入统一的组、线程和布局接口 |

当前 `FastllmInitNccl(devices)` 初始化不同设备域时会销毁旧全局通信器。它不能直接交替用于 attention 子组和 world EP 组。当前 MultiCUDA worker 内再次分发 MultiCUDA 还可能同步设备并创建临时线程。这两个问题必须在模型接入前解决。

当前代码可从以下入口核对：[多卡 worker](../src/devices/multicuda/multicudadevice.cpp)、[全局 NCCL 管理](../src/devices/multicuda/fastllm-multicuda.cu)、[持久 worker 抽象](../include/utils/persistent_worker_group.h)、[DiT 并行上下文](../include/models/hy_nomas/parallel.h)、[Clover rank 前向](../src/models/clover_parallel.cpp)。

## 通用模块与职责

以下名称是接口设计，不是本次新增的实现符号。

| 模块 | 负责 | 不负责 |
| --- | --- | --- |
| `ParallelTopology` | 有序设备域、rank 坐标、各角色组和合法性校验 | 分配 KV 或选择模型 kernel |
| `DeviceWorkerRegistry` | 每设备唯一 worker、执行上下文、设备租约和提交事件 | 按通信组创建线程 |
| `ProcessGroup` | 组内 communicator、rank 映射、generation、collective 顺序及失败状态 | 推断请求属于谁 |
| `DistributedTensorView` | 本地 Data 与请求行、特征分片、部分和状态的组合描述 | 隐式复制整段 KV |
| `LayoutTransfer` | gather、scatter、repartition、必要的 reduction 与逆行号映射 | 改变算子精度或采样分布 |
| `ExecutionPlan` | 每轮各 rank 的阶段、通信顺序和资源依赖 | 保存可变模型全局状态 |
| `RequestPlacement` 与 `CachePlacement` | 请求所属 attention 组、缓存版本、逐设备资源账本 | 硬编码 890 B/token 或某种 recurrent state |
| `ParallelModelAdapter` | 权重分片规则、本地计算、缓存语义与 kernel capability | 创建 worker 或直接修改全局通信器 |

拟放置位置为 `include/parallel/`、`src/parallel/`，CUDA/NCCL 实现在相应设备目录中。模型文件只保留 adapter 和算子调用。通信抽象不直接包含模型头文件；缓存的物理分配器继续复用已有后端。

最小接口轮廓如下，类型细节在实现阶段收敛：

```cpp
struct RankContext {
    DeviceId device;
    Executor& executor;
    StreamContext& streams;
    RankGroups groups;       // attention TP, leaders, MoE EP/TP, dense TP
    RankWorkspace& workspace;
};

struct RequestPlacement {
    RequestId request;
    uint64_t requestEpoch;   // 防止 handle/slot 复用后命中旧描述符
    GroupId attentionGroup;
    uint64_t placementEpoch;
};

struct CollectiveSpec {
    GroupId group;
    uint64_t groupGeneration;
    StepId step;
    StageId stage;
    uint64_t ordinal;
    CollectiveKind operation;
    TensorContract contract; // dtype/count/layout/root/reduction precision
};

class ProcessGroup {
public:
    CompletionFence submit(RankContext&, const CollectiveSpec&,
                           BufferView input, BufferView output);
};

class ParallelModelAdapter {
public:
    ParallelCapabilities capabilities() const;
    ModelPartitionPlan partition(const ParallelTopology&) const;
    CacheReservation estimate(const RequestPlan&, const RankContext&) const;
    void prepare(RankContext&, const StagePlan&); // 分配与冷启动不在 capture 内
    CompletionFence runLocal(RankContext&, const StagePlan&);
};
```

`runLocal` 接收完整的阶段上下文，本地优化不允许根据环境中的默认 world 猜测通信组。`CompletionFence` 明确区分 CPU 已提交与 GPU 已完成；调用方不能将 callback 返回当成设备数据可释放。

## 每卡线程与执行顺序

一个协调线程构建本轮不可变 `ExecutionPlan`，再向所有相关 GPU worker 发布任务。每张卡的初始化、executor 绑定、本地 kernel、collective、graph capture/replay 和销毁都在其固定 worker 上执行。后台 tokenizer、CPU 专家或 I/O 线程不属于 GPU 提交线程。

每个 GPU 线程可以管理计算流和通信流，但每条 GPU 操作的依赖均由事件显式表示。首期使用保守的固定提交顺序，之后才允许在计划已证明无依赖时重叠。首期不为每个 attention 组启动独立请求主循环。

必须遵守以下规则：

1. 协调线程先提交所有参与者，再等待任一参与者；不能提交 GPU 0 并等待其 collective 返回后才提交 GPU 1。
2. 一个 worker 不等待将来排入自己队列的任务；本地 kernel 直接调用本卡实现，不递归进入 MultiCUDA 分发器。
3. worker 的 Executor 和可变工作区独立；共享权重描述只读，预打包和本地权重视图在发布阶段前完成。
4. GPU buffer、请求缓存和通信 scratch 保活到对应 fence；请求取消先停止发布新计算，待在途操作完成后回收。
5. 所有 group 建立、冷启动和大块扩容在一致的计划边界完成，不能让某 rank 在同组其他 rank 尚未提交 collective 时进入隐式同步分配。
6. 新路径中的设备拥有者不随请求、batch 或阶段改变。兼容旧 `RunWithCaller` 时只能在 session 初始化时选择固定 owner，不能中途把 rank 0 从调用线程换到 worker。

第一版允许一次运行时独占选定设备的 GPU 提交权。旧模型路径仍可单独使用，但不允许旧、新线程池同时操作同一租约设备。迁移适配器时将旧线程组委托给 registry，不在原线程池外再包一层。

## 多通信组的生命周期与死锁约束

`ProcessGroup` 的键至少包含运行时域、角色、有序成员、组代次和通信后端。相同成员的 attention/EP/其他角色可以使用不同 communicator；只有执行计划证明顺序完全一致时才允许资源复用。

每组独立持有 NCCL handle、成员到 local rank 映射、提交协调器、错误状态、graph 引用和可选 custom all-reduce 缓冲。创建一个子组不影响其他组。所有组按确定顺序在接收请求前创建；singleton 组走本地 identity 操作。

原来的 `FastllmNcclAllReduce(data, ..., deviceId)` 兼容封装绑定显式的旧默认组。新接口必须传 `GroupRef`，不通过动态 TLS 默认组切换 attention 与 EP。custom all-reduce 尚未适配多组时，新组使用 NCCL 基线路径，不能错用全 world 的 peer 表或 barrier counter。

每个 collective 的约定包括 `(group, generation, step, stage, ordinal, operation, dtype, count/root/reduction)`。同组成员必须一致；对 AllToAllv 检查发送与接收矩阵互为转置，不要求各 rank 的总量相同。检查失败发生在提交之前；提交后故障终止当前执行域，不能仅一张卡临时换后端重试。

多组在同一 GPU 上的调用遵循全局阶段顺序。比如先完成各 attention TP 组的输出归约，再提交 leader gather，再提交组内 broadcast，最后进入 MoE。每个 rank 取这个全局序列中属于自己的子序列；空请求组也不能漏掉后续跨组通信。NCCL 文档要求多 communicator 的主机提交以及 CUDA Graph 捕获、回放保持一致顺序；固定每设备提交线程有利于满足这一要求。[NCCL 多通信组说明](https://docs.nvidia.com/deeplearning/nccl/archives/nccl_2265/user-guide/docs/usage/communicators.html#using-multiple-nccl-communicators-concurrently)。

主机调用顺序本身不替代跨 stream 的设备依赖。首期将重叠组的通信串在明确的 stream/event 链上，不依赖某个 NCCL 版本的隐式 launch ordering 环境变量。后续放开重叠时，需要同时验证主机提交依赖与 GPU 执行依赖。

故障由运行时域传播到本轮所有相关组，唤醒主机等待者并停止后续提交，记录首个错误。cache 和 graph 的释放等待安全清理；通信器重建产生新 generation，旧 graph 不可继续回放。独立模型域是否可继续运行取决于 CUDA 错误级别，不承诺设备级错误隔离。

## 张量布局与请求行号

普通 `Data` 继续表示本地 storage。外层 `DistributedTensorView` 增加三个互相独立的描述：

- **行布局**：逻辑 token 行属于哪个请求和 attention 组，本地行与全局行的双向映射，以及有效行 mask。
- **特征布局**：replicated 或按明确 axis/ranges 分片，包含所属 group 和逻辑全尺寸。
- **数值状态**：完整结果或某组的 partial sum，附带归约 dtype、舍入边界和 bias/residual 的唯一添加者。

一个 attention 输入可以同时是“请求行分到 DP 组”与“hidden 在组内复制”；Q 可以是“相同行归属＋heads 在组内 TP 分片”。不能仅靠当前 `tpAxis` 将这两层含义压成一个轴，也不能将 partial sum 当成完整副本直接 gather。

逻辑行键使用 `(requestId, requestEpoch, absolutePosition, phase)`。请求编号与本轮位置是不同概念。prefill 的一个请求可以贡献多行，decode 通常一行，verify 可以多行；基础设施不写死这些数量。全局行顺序由计划声明，通信采用按 owner 排列的临时顺序时必须有逆置换。

这个行键的唯一性限定在一个执行 step 内；在途描述符还必须携带 `StepId`，避免回滚后重新计算相同位置时引用上一轮结果。将来树状推测解码通过 candidate/branch 标识扩展行键，不能把同位置的不同候选当作同一行。持久 KV 的位置键与临时激活行键分开。

通用布局转换至少提供：

| 转换 | 语义 |
| --- | --- |
| `ReplicateFeatures` | 收集特征分片，保持请求行身份 |
| `ReducePartial` | 按约定精度求和，之后才允许作为完整输出使用 |
| `CollectRows` | 每个请求行只从一个逻辑来源收集，排除组内 TP 副本 |
| `SelectOwnedRows` | 从复制的完整 batch 中选取本 attention 组请求 |
| `RepartitionRows` | 改变临时激活行的归属，不等于迁移请求 KV |
| `DispatchExperts / CombineExperts` | 维护 token 行、专家路由和权重，精确重建 routed sum |

首期 `CollectRows` 可以使用等长 padding、AllGather 和 unpack。各源统一 bucket，空源填 padding；padding 不更新 KV、recurrent state、RNG 或输出统计。后续 AllGatherV/AllToAllv 只替换传输实现，不能改变逻辑行约定。

## Attention 与 MoE 之间的通用基线

第一版采用“局部 attention、全 batch 专家后端”的桥接方式，避免将新拓扑与新的专家调度同时引入。

1. 一个请求固定属于一个 attention TP 组；组内各 rank 处理相同请求行，各自计算其权重/heads 分片。
2. attention 输出投影在本 TP 组内完成必要归约，得到组内复制的完整 hidden 行。
3. 仅各组 leader 参与 `CollectRows`，收集去重后的所有有效行。
4. leader 在自己的 attention TP 组内广播收集结果，得到原专家后端所需的全 world 复制 batch。`T=1` 时无此广播，`A=1` 时桥接整体可省去。
5. MoE 使用其自己的 EP/专家 TP 组执行，输出的逻辑行顺序不变。现有 world EP8 后端可作为首个 adapter。
6. `SelectOwnedRows` 使后续局部阶段继续处理原 attention 组的请求。模型若暂时保留全 batch residual/mHC/Engram，可显式声明该阶段输入复制，之后再切回局部。

复用现有专家后端指复用权重布局、本卡计算与归约语义。需要将其编排改接显式组；不能从新的 worker 内再调用会建立全局 NCCL 或重新分发整组线程的旧 MultiCUDA 入口。

**组内有 T 份相同激活时，只收集一次。** 直接在 world 上收集每卡的完整本地 batch 会把逻辑 token 重复 T 次。这必须由布局接口防止，而不是要求每个模型自己记得去重。

对于 dense FFN，适配器可以选择继续在同一 attention TP 组计算，省去跨 DP 收集；也可以声明更宽的 dense TP 组并通过上述布局转换进入。shared expert 的计算组和 routed expert 的组分别声明，只有输出布局与归约协议一致时才允许融合。

第二阶段增加原生 `DispatchExperts / CombineExperts`：每个逻辑 token 的 router 结果只有一个来源，按专家 owner 发出；专家 TP 组内广播输入、计算分片并归约，再根据行键返回请求组。传输可以用已有 NCCL send/recv 构成的 AllToAllv，不强制依赖 DeepEP 等额外库。专家过载不能通过丢 token 解决；容量不足要提前扩容、分块或拒绝本轮，不能改变模型输出。

### 容量收益与通信成本

Attention DP 首先减少跨组 KV 复制，并让不同组处理不同请求；它不是免费加速。同一请求内部的 attention TP 宽度变小，attention 权重的全机副本数增加，且 EP 桥接引入激活收集。单请求无法填满多个 DP 组，不能据此承诺单请求延迟降低。

设各组有效行数为 `n_i`，总行数 `N=sum(n_i)`，统一 bucket 为 `B>=max(n_i)`，则 gather 的行槽数为 `Npad=A*B`。每行 hidden 宽 `H`、每元素 `q` 字节时，首版 leader gather 加组内广播的逻辑接收量为：

- leader gather 合计接收 `(A-1)*Npad*H*q` 字节。
- 若广播前不压紧，组内 broadcast 合计接收 `A*(T-1)*Npad*H*q` 字节。
- 两者合计为 `(W-1)*Npad*H*q`；若先 unpack 成 N 行再广播，第二项可改用 N。

以上只统计新增桥接的端点接收量，不是物理链路计数，也不含原有 TP/EP 归约；`A=1` 时整个桥接省去。实际 kernel 时间、通信时间及依赖等待必须分别测量。`SelectOwnedRows` 在完整结果已经复制时仅是本地选行，无需额外跨卡 gather。

负载偏斜时 `Npad/N` 可能接近 A，需要同时观测 padding 放大和最长组耗时。调度器优先在容量允许的组间平衡新请求，再考虑前缀亲和性；极端偏斜可分块执行或采用不等长传输，不通过临时迁移活动 KV 或丢弃 token 掩盖成本。

## 权重与精度契约

权重布局由 `ModelPartitionPlan` 按层声明，而不是在 forward 中临时修改共享 `tpLinearType`。组内复制的小权重、head/output 分片、完整专家归属和专家内部 TP 使用独立 spec。

本地权重缓存键包含权重版本、device、分片范围、group layout、存储 dtype、量化块布局和 packing ABI。允许复用完全一致的本地 shard，禁止仅按全局权重指针缓存不同 group 的打包结果。weight/adapter 版本更新必须失效相应 graph。

`ColumnParallelLinear` 输出特征分片；`RowParallelLinear` 输出 partial sum。后者明确何时归约、何时舍入、哪个 rank 添加 bias/residual。归约后再把 group 内副本跨 DP 相加会重复计数；通信优化不能自行改变累加精度。

DSV4.1 的完整 attention 权重复制只是 `T=1` 的一个实例；`T=2/4/8` 仍按组内 TP 分片。新增存储随 `A` 和具体量化布局变化，不统一承诺“复制完整权重只增加若干 GiB”。head 数、GQA 比例、量化 group 和 packed projection 的可分性由适配器检查。SM 特化只影响本地 kernel 选择，通用路径保持相同布局和数学语义。

## 请求缓存与逐设备容量管理

请求归属为 `attentionGroup`，不是单一 `ownerDevice`。`T=1` 才退化成单卡归属。入场时根据组内最紧 rank 的可用空间、预期增长和计算负载选择组；请求存活期间归属保持不变。

缓存适配器声明各状态的 storage 语义：

| 模型状态 | 典型分片规则 |
| --- | --- |
| MLA latent、共享压缩 KV、index K | 所属 attention TP 组内复制；组外不创建 |
| GQA/MHA KV | 按 KV head 范围分片；head 不足时显式复制并计费 |
| GDN/linear attention | recurrent state、conv state 按模型 heads/channels 分片 |
| 滑窗、压缩尾块、位置元数据 | 按模型状态语义跟随请求组，不能只迁移主 KV |

基础设施使用 `estimate/reserve/bind/snapshot/restore/release` 生命周期和不透明状态 handle，不读取模型内部 tensor 名称。推测解码以后通过 `beginTransaction/commitPrefix/rollback` 接入；基础设施传接受长度，状态如何提交由模型决定。

缓存账本保留每个实际 GPU 的 weights、永久 workspace、graph、KV 页/容量、固定请求状态、通信缓冲和在途峰值。准入检查每个受影响 rank，不能只看八卡显存总和或简单把原每 token 预算除以 DP 数。对 MLA，同一请求只计入其所属组的 T 张卡；存在 head-shard 或固定 recurrent state 时使用适配器返回的逐 rank 估算。

总 token 上限仍为所有活动请求的逻辑输入输出预算，不混同单请求 context window。运行时另报告每组有效 token、实际分配字节、可回收前缀、余量和 padding 比例。

前缀目录区分逻辑前缀与物理布局。恢复必须匹配模型/权重版本、精度、token 与多模态标识、cache layout 版本和 TP 分片规则。相同 TP 宽度的请求优先放到已有前缀组；跨组复用可以复制合法快照或重算，不假设另一组的 GPU 指针可直接使用。共享缓存采用只读引用与写时复制；前缀热度不允许覆盖组内硬容量限制。

首期不做活动请求在线迁移。将来迁移需暂停该请求，等待所有消费者，复制完整模型状态并校验，原子更新 placementEpoch/描述符后再释放旧状态；只改 owner 字段不是迁移。

## 调度与模型状态

首期一个协调器生成全实例的轮次，attention 组可以有不同请求数、不同有效 token 数或为空，但必须执行兼容的模型阶段和跨组通信序列。prefill、decode、verify 分为不同阶段计划；先保留 prefill/decode 不混跑的策略。

TP 组内成员使用相同请求清单、位置和阶段签名。压缩器的奇偶位置等本地条件可以改变局部计算，但不能让某 rank 自行多发或少发 collective。某些模型的 bounded prefill 会省略尾部层；不同组的执行签名不兼容时应分轮处理，而不是让 EP 组进入不同层。

prefill 预算需要覆盖多个 attention 组，避免调度器始终只投喂一个组。新增请求、完成、取消和 prefix restore 在轮次边界调整；异常/取消的行在后续阶段使用已约定的 inactive 标记，本轮已经提交的 collective 仍按计划完成。

采样由每个逻辑请求的指定 rank 负责，结果广播到其 TP 组。随机数流按请求 seed、requestEpoch、阶段和抽样计数管理，不使用物理 GPU 编号决定随机序列。改变分片可能改变浮点舍入，但不能重复抽样或因 padding 消耗随机数。

当前 DSV4.1 的多请求路径没有 batched DSpark。新 Attention DP 首期显式关闭该组合，旧 TP DSpark 服务路径继续保留。请求数量下降到 1 时也不能自动切到旧 TP DSpark：其 KV 布局不同。后续支持时，各请求可以有不同接受长度，但组内必须共识提交范围、草稿分布和返回 token。

## CUDA Graph 与异步资源

Graph key 至少包含拓扑/组 generation、模型与权重版本、阶段签名、attention TP 宽度、本地 batch/token bucket、dtype、cache layout、通信布局与算法版本。只用全局 batch 不能区分 `{64,0,0,0}` 与 `{16,16,16,16}` 等局部分布。

请求 ID 和真实 KV 指针通过稳定地址的描述符缓冲更新，不放进 graph key 从而为每个请求单独抓图；变更 descriptor 使用顺序事件。padding 由 kernel capability 决定，不能让普通 kernel 将无效行写入真实缓存。

包含 collective 的 graph 采用全实例共同的 capture/replay 计划。部分 rank 可以复用本地 compute graph，其他 rank eager，但跨组 collective 的位置、顺序和 count 必须一致。不能让某 rank 因缺少 graph 独自跳过一次通信，也不能在回放中初始化 communicator 或搬迁权重。

通用基线保留分段图：本地计算图、显式通信、本地后续图。DSV4.1 已在预热时将支持的行数布局合为整轮图，保持同一 execution signature；其他模型适配器仍按算子、通信和缓存描述符的实际能力选择图粒度。

buffer lease、event 和 graph 节点引用归属 RankContext；被 graph 引用的 scratch 不得因另一组或另一 bucket 的执行被覆盖。设备异步释放必须晚于最后一个 consumer fence。

## 各模型接入边界

| 模型 | 可以直接复用 | 首要适配 |
| --- | --- | --- |
| DeepSeek V4.1 | 原 attention/indexer/压缩/FP4 cache kernels、现有 EP 专家后端 | 组内 KV、共享 source 层状态、本地 attention 前后边界、mHC/Engram 布局 |
| Qwen3.5 | `ForwardSingleGPU`、既有 thread TP 与缓存快照机制 | 替换隐式通信组；同时描述 GQA KV 与 GDN/conv state；绑定请求组 |
| Clover | per-rank 模型与 Executor、现有 TP 分片规则 | 将模型私有 worker 改为借用 registry；统一请求行和 recurrent state 生命周期 |
| DiT 类模型 | 现有 column/row linear、异步归约与分组 Tensor 搬运 | 复用低层组与线程，不强行采用 LLM 请求/KV 接口 |

模型适配器可以逐层声明支持能力。基础设施提供的拓扑不等于每个模型已经支持所有权重分片。例如初始 DSV4.1 adapter 可以覆盖 `A=1/2/4/8` 且 `E=8,M=1`；`E=4,M=2` 在专家内部 TP 尚未实现时应明确拒绝。其他支持专家 TP 的模型可以使用同一组接口扩展，不再修改 worker 和 NCCL 生命周期。

## 回退与兼容

未选择新模式的模型保持旧行为。迁移期间新增显式组 API，旧接口通过固定默认组适配；一个新 session 不允许混入会重置全局 communicator 的旧初始化调用。

回退分三层：启动时不支持的拓扑直接报错；本地算子不支持某形状时在同一 placement 上使用通用 kernel；尚无图时执行同语义 eager 阶段。任何回退都不能悄悄扩大请求缓存到其他 DP 组，或重新启用旧设备线程池。

不新增一批环境变量来控制核心并行语义。设备域、Attention DP、EP 和阶段策略由解析后的配置统一传递；诊断开关只用于追踪与强制通用实现。首期依赖现有 CUDA/NCCL，不新增通信库依赖。

## 实施阶段与验收

| 阶段 | 交付 | 必须通过的检查 |
| --- | --- | --- |
| 1 | Topology、组对象、显式 collective API、唯一设备 worker | 多子组与 world 组同时存在；初始化顺序；空组/非法配置；异常唤醒；旧 TP 行为 |
| 2 | 行布局、CollectRows/SelectOwnedRows、精度契约 | TP 副本去重、padding、逆置换、非连续 GPU、部分和/bias 只处理一次 |
| 3 | DSV4.1 的 eager Attention DP/TP 与 world EP8 桥接 | TP8/DP1、TP4/DP2、TP2/DP4、TP1/DP8；KV 物理归属；FP4 与 BF16；本地算子和 logits |
| 4 | 前缀、逐卡准入、多组 prefill、请求生命周期 | 取消/重用 handle、恢复/截断、扩容、空 rank、长短请求混合、故障清理 |
| 5 | 本地 graph buckets 与通信计划 | capture/replay 顺序；bucket 变更；KV 扩容；旧 descriptor 拒绝；eager/graph 混合边界 |
| 6 | Qwen3.5 或 Clover 第二模型接入 | 基础模块不新增模型名判断；GDN 与 KV 的独立状态规则；原模型回归 |
| 7 | 可选原生 DP 到 EP 路由、通信重叠 | 行键与专家权重一致；无 token 丢失；性能收益与额外显存实测 |

拓扑和行映射先用 CPU 离线模拟验证；真实 CUDA 验收另覆盖 collective 的 payload、执行顺序、算子 reference、内存检查和服务生命周期。模拟通过不能替代 NCCL 无死锁、GPU kernel 正确性或吞吐测试。

GPU 功能测试至少包含并发 `1/2/3/7/8/9/31/32/33/127/128/129/511/512/513`，单组有请求、其他组为空，以及不同长度导致各组不均衡。比较顺序为：字节搬运和缓存状态、top-k/数学参考、logits 误差、输出有效性，再测速度。TP 归约顺序改变时不承诺跨布局逐 token 相同，误差标准在测量前确定，不因候选失败而事后放宽。

观测项包含每卡 worker ID、组成员/generation、每轮有效和 padded 行数、collective 序号/字节数、KV 有效/分配字节、prefix 命中 token、prefill/decode 延迟和总吞吐。性能比较使用相同模型精度、总上下文、并发、采样和缓存条件。

## 设计约束检查

- Attention DP 与 TP 可以同时大于 1，MoE 分组独立存在。
- 每设备一个 GPU 提交线程，不为每个 group 或请求增加线程。
- 同一请求的 TP 副本不重复进入全局 token batch。
- KV 不跨 DP 组自动复制，模型特殊状态不遗失。
- 全部 collective 使用显式组，捕获与回放遵守同一提交顺序。
- 同一阶段全体参与者预先知道有效行、padding、通信类型和尺寸。
- 模型可以复用低层基础设施，无需采用 DSV4.1 的 KV 或网络结构。
- 配置、拓扑、布局与真实后端能力分开；不支持时明确拒绝。
