# Fast KV Manager CPU L2 缓存

本文描述已实现的本地 L2、接入契约和扩展边界。部署参数见
[FastLLM PD 部署](../../docs/fastkv_pd.md)，公共接口见
[`host_cache.h`](../include/fastkv/host_cache.h)。

## 分层与职责

- L1 由模型已有 GPU 页池和配套状态管理；没有原生驻留缓存的 adapter 可使用 `PrefixCache` GPU 检查点。
- L2 是每个 P 或启用了本地 L2 的单实例私有、可丢弃的 CPU 缓存，使用 `HostCache` 管理目录、不可变块、预算、淘汰和异步复制。
- 模型 adapter 提供完整状态和区域描述，负责验证恢复边界、编码和设备布局；FastKV 不依赖 `basellm`、tensor dtype 或模型页表。
- `PrefillRouter` 比较 L1 命中、L2 恢复成本和重算成本；查询不等于实际命中，状态完整导入后才统计有效命中。

L2 关闭时不创建主存备份或复制线程。缓存与计算线程分离，模型页表、请求状态和最终提交仍只由模型执行线程修改。

## 状态与版本契约

`CacheSnapshot` 包含完整 `Message` 和按 buffer 顺序排列的 `CachePayload`。
每个 payload 的 `CacheRegion` 必须从偏移 0 起连续、无重叠地覆盖全部字节；
非连续的模型页由 adapter 先导出为已有 PD 表示。

命名空间由调用者提供，必须区分模型、adapter 布局及租户/媒体等隔离身份。
前缀匹配先用哈希筛选，再比较完整 token 序列；只恢复实际发布过的完整边界。
所有必需组件必须属于同一边界，不能混合不同位置的 KV、SWA 或循环状态。

- 非空 `identity` 必须唯一标识该命名空间内不可变的字节版本。同一物理地址重新分配后必须换版本；不能仅以逻辑页号判断复用。
- 空 `identity` 表示边界独立的数据，适用于 SWA、SSM 等不能从旧边界推导的状态。
- `prefixRange` 允许沿用调用方**实际恢复过**的父 `Hit` 中同位置、同长度的区域。独立重算或分叉不得假借父版本混用数值状态。
- FP4/FP8 数据和量化伴随状态按原字节保存，不隐式转换精度、TP 布局或编码。

Naive adapter 对 Full KV 和 indexer 使用存储对象、物理页、allocation generation
及区域长度标识版本，SWA 保存独立边界状态。其他 adapter 可以直接使用完整 opaque
检查点，不必实现块去重；新增模型不要求修改缓存核心。

## 发布、恢复与生命周期

1. `PublishAsync` 接收已就绪、不可变且持有生命周期的源 buffer；不能异步读取随后会被 forward 覆盖的活动页。
2. 队列和在途额度不足时返回 `false`，调用者跳过可选备份。worker 预留 Host 内存、复用已有不可变块并复制新增区域；全部成功后才发布完整检查点。
3. `ProbeTokens` 只查询元数据。`Find` 更新 LRU 并持有完整检查点 lease，之后目录淘汰不会让该 `Hit` 失效。
4. `RestoreAsync` 按调用者给出的设备放置分配私有目标，返回 future；暂时没有队列/目标额度时返回 `false`，由调用者稍后重试。
5. 所有复制完成后返回相同 metadata 和 buffer 边界。模型执行线程校验并导入状态，随后计算未命中的后缀。
6. 取消或丢弃 future 不会提前释放 DMA 中的源、目标或额度。后台复制排空后释放；缓存停止时排空已接收任务。

CUDA worker 使用每设备私有 non-blocking stream。Linux 上按驻留锁页额度注册整块主存，
锁页区域直接与 GPU 传输，源和目标在各自底层分配内都连续时合并 DMA。
普通主存路径按需分配 pinned staging，将预算分为
两个轮换窗口，以事件保护窗口复用。连续物理区域的 CPU 复制和 DMA 分别合并，
逻辑 buffer 边界保持不变。多设备任务复用 FastKV 的有界线程池，每 worker、每设备
至多一条持久复制线程，使各卡 CPU 搬运与 DMA 独立推进；单设备任务直接在 worker 执行。
一个设备出错后仍等待其他设备结束，之后才能释放源、目标和额度；不调用设备级同步。
无 CUDA 构建仍支持 Host 数据的发布、恢复和独立测试。

Linux CUDA 构建通过设备 PCI 地址查询 NUMA 节点，主存 arena 与 staging 优先分配在
源 GPU 本地节点，恢复线程按目标 GPU 选择本地 CPU，并遵守调用线程原有 CPU 许可范围。
只在复制期间临时设置线程亲和性，之后恢复；不改变整个进程或系统的策略。
使用内核接口，无 libnuma 依赖。拓扑未知、策略设置失败或非 Linux 平台保留普通分配路径，
本地内存不足时允许跨节点分配。staging 注册不受支持时回退 CUDA pinned allocation。

L2 驻留内存按 NUMA 节点使用大块分配、内部切分复用，常规 slab 目标为 64 MiB，
小缓存按总预算缩小，超大 payload 可单独占用更大的 slab。分配时按空洞大小查找，
释放时合并相邻空洞。空闲的普通块立即归还，空闲的锁页块保留用于下次分配，
预算压力或缓存销毁时归还。外部 lease 持有的块保持有效。
避免每个小 arena 单独 `mmap/mbind`，防止长期淘汰产生大量 VMA 并触及
`vm.max_map_count`。不修改系统映射上限，不新增环境变量。

PD 接入在请求准入时取得 L2 lease，并将待恢复字节与未命中计算量分开计费。
确定采用 L2 后立即提交后台恢复，与下一次 forward 重叠；额度不足时从 prepared 队列重试。
完整 prompt 仍按 GPU 页向上取整预留容量。恢复不能满足目标设备预算的检查点不备份；
恢复失败走请求错误与统一清理路径，不把未完整导入的状态当成缓存命中继续计算。

## 预算与淘汰

| 资源 | 约束与释放时机 |
|---|---|
| Host 驻留 | `bytes` 约束整块 Host slab（含内部空洞）与核算的元数据，空闲但保留的锁页块继续计费 |
| 在途复制 | `inflightBytes` 约束备份源与恢复目标；恢复结果被消费者持有期间继续计费 |
| 每设备恢复 | `restoreDeviceBytes` 可进一步限制目标设备，额度随结果 buffer 最后一个引用释放 |
| 驻留锁页 | `pinnedBytes` 约束已注册驻留块，按系统页计费，包含在 Host 驻留预算内 |
| pinned staging | 每 worker、每使用过的 GPU 最多 `stagingBytes`，默认 8 MiB，与 Host 容量独立 |
| 工作队列 | `queueDepth` 包含等待和执行中的任务，恢复优先于可选备份 |

L2 独立按完整检查点 LRU 淘汰，活动读取 lease 不可淘汰。共享块按引用释放，
不会单独删除必需的 SWA/scale 而仍报告完整命中。容量不足且条目被持有时跳过备份。
Host 预算不是进程 RSS 上限；容器、索引及临时描述仍需预留额外空间。
淘汰每个检查点后优先重试已有空洞，不能只为了申请新 slab 而忽略刚释放的空间。
分配器不调用模型接口，缓存销毁后外部仍持有的 `Hit` 或 buffer 保持数据和预算有效。

每个 P 可用 `cache_bytes` 配置共享主存预算（默认 0），也可用 `cache_kv_bytes` 和 `cache_swa_bytes` 分配独立的物理池。KV 池计入目录与 Indexer，窗口池计入 SWA 和草稿检查点。共享页只占用一份存储，读者租约持有的内存继续计费，两个池共同发布一个完整版本。单实例对应参数见 [部署说明](../../docs/fastkv_pd.md)。

`cache_workers` 默认 2、范围 1～64。
多个 P 的容量需要相加核算，不能各自占满整机内存。复制与目标设备预算由已有
transport staging 额度导出，不增加环境变量或模型专用调试开关。

## 观测与验证

`[FastKV PD] prefix_hit` 在成功恢复后记录 L1/L2 有效前缀，L2 附带 `cache_tier=l2`。
`[FastKV L2 cache]` 记录驻留量、队列、跳过/失败、D2H/H2D 及复用字节；
组件 `Stats()` 另提供发布、淘汰、恢复次数和恢复计时。旧 `Buffer::Traffic()` 的同步
计数不包含异步 L2 DMA。

独立 `fastkv_host_cache_test` 覆盖字节一致性、命名空间、父版本复用、边界状态、
淘汰、取消和消费者持有预算；可传 CUDA device ordinal 验证设备复制，另传第二个
device ordinal 验证跨设备放置、并行复制与中途取消后重试。覆盖非整齐窗口和尾部字节。
`fastkv_host_pool_test` 验证 NUMA 分组隔离、空洞合并、并发分片复用、跨池生命周期和整块预算；
Linux CUDA 构建在系统允许查询和设置策略时还校验实际 NUMA preferred 节点。
Host cache 压力用例覆盖持续写入、淘汰、恢复的逐字节一致性。
`fastkv_routing_test` 覆盖分层代价与路由额度，`pd_state_test` 覆盖模型适配、
Host 恢复、排队准入和取消清理。模型精度与压力验证应比较相同执行形状下的冷算、
L1 恢复和 L2 恢复，SWE 长时间吞吐不能替代算子或状态字节正确性测试。

## 当前边界与后续扩展

L2 每实例私有，不提供跨 P 共享目录、持久化或远程存储。Naive 单实例通过 `RestoreIntoAsync` 将检查点直接写入租约保护的目标页与窗口，DMA 完成后由模型线程接入请求。PD 恢复先生成私有 GPU
`Message`，再由 adapter 导入页池，尚非直接 DMA 到活动页表。

后续直接目标恢复、TCP/RDMA 远程缓存和不同 TP 布局重分片，应复用
现有不可变版本、buffer lease、预算和完整提交契约，显式声明兼容布局。
这些能力未实现前，不自动转换数据类型或宣称支持任意异构恢复；wide-EP 的计算
collective 仍由推理系统负责。
