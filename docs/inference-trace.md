# 逐次推理计时

设置 `FASTLLM_INFERENCE_TRACE=/path/to/replica.jsonl` 开启 JSONL 日志。
目录须提前创建，每个服务进程使用独立文件。未设置或空字符串时关闭：
不创建文件、日志记录或 CUDA Event，不读取计时时钟；热路径仅保留条件检查。
该开关与逐算子 `FASTLLM_PRINT_PROFILE` 独立，无需打开逐算子 profiler。

当前详细 forward/CUDA 计时覆盖 Naive N0.5 / MiMo 的普通 TP 推理路径，
调度计时位于其使用的旧版通用调度器。不是所有模型和调度器的统一 profiler。
启动预热不记录，正式请求中的图准备开销会记录。日志没有提示词或生成文本。

每个实际 `Forward/ForwardBatch` 一条 `event=forward`，每个调度轮次一条
`event=scheduler_step`，通过进程 PID 与 `step_id` 关联。
长 prompt 被拆成多次 forward 时，每一块有独立的 `forward_id`、长度和耗时。
`timestamp_ns` 为开始时的 Unix 时间，所有耗时使用单调时钟，单位毫秒。

## 长度和速度

- `phase` 为该调度轮次的 prefill/decode，单 token prompt 仍属于 prefill。
- `query_lengths` 是各请求本次真正计算的 token 数；`input_tokens` 是其总和。
- `context_before/after` 是各请求的逻辑上下文，来自绝对 position，包含命中的历史；
  不用分配容量或 SWA 热窗口长度冒充上下文。`context_min/max/sum` 汇总计算后长度。
- `handles` 为该模型进程内部请求 ID；`cached_prefix_tokens` 在调度记录中提供。
- `tokens_per_second = input_tokens / forward_ms * 1000`。prefill 只计新算的 token，
  decode 通常每请求一个 token；这是整批速度，不是单路速度，也不是整场端到端吞吐。
- `forward_ms` 是模型 API 返回 token 的 wall time，包含 TP 准备、GPU 执行、KV
  元信息/前缀处理及采样。当前普通 TP 路径返回前已完成结果所需的 GPU 工作。
- `gpu_model_max_ms` 是各 TP rank CUDA Event 时间的最大值。每 rank 另有
  `model_gpu_ms`、`logits_gather_gpu_ms`、`wall_ms`。CUDA Event 测的是当前流的
  时间跨度，包含该跨度中的通信等待和 CPU 提交间隙，不是所有 kernel 时间相加。

## forward 内外的阶段

forward 的 `timings_ms` 分解输入准备、TP/KV 准备、embedding、graph 设置、
graph 捕获/缓存扩容、TP 执行、KV 元信息/前缀提交、采样，`tail_ms` 是余下时间。
这些是连续 wall-time 阶段，不能再把各 rank 的并行 CUDA 时间相加到它们上面。

调度记录的 `timings_ms` 分解分配、字典锁等待、结束请求回收、调度扫描、
前缀恢复与准入规划、batch 输入构建、forward 锁等待、workspace/输入准备、
forward dispatch、原有日志、结果锁等待、结果回填和清理。
`outside_forward_ms` 为总调度时间减去逐次模型 API 时间；它包含日志观察开销。
无任务的条件变量等待不输出为一次推理；不含 HTTP/Python 协议处理和沙箱时间。

## 观察开销

开启日志仍有观测成本：少量时钟读取、各 rank 三个 Event、Event 完成检查、
序列化和文件写入。Event 按 worker 复用，图捕获期间不插入 Event；优先利用原有
logits 汇聚的同步点，避免逐算子同步。`observer_wait_ms` 单独记录 Event 完成等待，
不代表全部 Event API 开销。`forward_trace_emit_ms` 和
`previous_step_trace_emit_ms` 记录日志序列化/写入的开销。
文件使用缓冲，活跃时约每秒刷新一次，进程正常退出时刷新；强制终止可能丢失尾部缓冲。

`completed=false` 的失败记录不应计入正常吞吐。日志打开或写入失败会在 stderr
提示并停止日志，不改变推理结果；性能实验应在正式放量前检查日志确实产生。

无 GPU 的格式与计时边界测试：

```sh
python -m unittest discover -s test/api -p 'test_inference_trace.py' -v
```

## 抽样主机细分

在已有 trace 基础上设置 `FASTLLM_INFERENCE_TRACE_DETAIL_EVERY=64`，每 64 个调度轮次
附加一份 `host_detail`；默认 0 关闭。1 表示每步采集，非法值按关闭处理。仅开启此变量
不会启用 trace。采样以 step_id 为准，无 forward 的轮次仍不输出；统计时应筛选相同
阶段、batch 和上下文，并保留未采样步骤作为观察开销参照。

`metrics` 每项包含 `region`、`name`、`wall_ms`、`calls`、`count`。区域为
`prefill_preview`、`postprocess`、`retire_requests` 或未划分的 `other`：

- preview 分解 token 比较/重置、页 hash、trie 查询、模型历史 memo 重建、驻留检查、
  可回收页引用检查和容量去重，并记录请求 token、管理器、遍历/复用页及历史条目数量。
- postprocess 分解请求查找、停止/长度检查与缓存发布、token/队列更新、通知消费者。
  缓存发布进一步分解快照复制/hash、历史去重、页表复制和 trie 发布。
- retire 分解请求删除、完成请求的描述符拆卸、KV lease 释放，以及页引用释放的锁等待
  与实际工作。`finished` 是本步访问的结束请求数，可能包含已经回收过的请求。

细分 span 可以嵌套，`wall_ms` 为含子项的时间，**不能将所有细项直接求和**。区域
`total` 另在支持 `CLOCK_THREAD_CPUTIME_ID` 的平台输出 `thread_cpu_ms`；该 CPU 时间
仅含调度线程，wall 与 CPU 的差额可能包括等待、抢占以及其他线程/GPU 的工作，不能
直接当作 GPU 空闲。异步 TP worker 和 API worker 不继承此线程局部记录。

长前缀预览会按独立 KV manager 使用常驻 worker，最多使用 `min(threads, 8)` 路，
其中一路由调度线程执行；短查询和热锚点仍串行处理。采样时这些 worker 各有独立记录，
合并到 `prefill_preview_workers_sum` 区域：其中 wall/CPU 时间是并行 worker 的累计时间，
不能与主线程 elapsed 时间相加，也不能作为单步耗时。`preview.trie_queries` 保留调用者
看到的完整墙钟时间，`preview.parallel_managers` 记录本步交给并行路径的 manager 数。
原 `prefill_preview` 内的页计数仅属于串行部分；全部页计数需加上 worker 区域对应计数。

细分日志没有额外 CUDA 调用、同步或逐页时钟读取。关闭或未采样时探针只检查指针，
不分配内存、不读时钟；采样时使用固定 128 项存储，超限在 `dropped_metrics` 计数。
计时本身仍有成本，应同时检查抽样/未抽样步骤；该日志用于归因，不改变调度、采样
或 KV 生命周期。序列化成本仍计入下一步的 `previous_step_trace_emit_ms`。


### Paged-cache publication and retirement

Long prefix publication and page-reference release can use a model-owned persistent
CPU worker group, bounded by the existing thread setting and at most eight workers
including the caller. Small jobs stay serial. Jobs for the same cache manager keep
their input order; manager locks, page generations and LRU updates are unchanged.
Request/cache ownership remains with the caller until the group completes. CUDA
frees and descriptor destruction remain on the calling thread.

Sampled diagnostics use `paged_publish_workers_sum` and `paged_release_workers_sum`
for concurrent worker intervals. These are summed work, not elapsed scheduler time;
do not add them to the parent `postprocess`/`retire_requests` intervals. The existing
trace/detail switches remain default-off; no additional environment setting is
required. `history.snapshot_hashes` measures shared-prefix hashing for multiple SWA
checkpoints; per-snapshot token copies still appear in `history.snapshot_copy_hash`.

Pending-prefill history previews reject unrelated token fingerprints before
creating per-request equality memo entries. `history.fingerprint_rejects` counts
these rejections; fingerprint matches still require exact token equality, and
residency is checked on every pass. Admission sorting captures request pointers
under the request-table lock instead of looking them up for each comparison;
queue-time, phase, length and handle ordering are unchanged.


## Naive 无推测解码：启动准备与 GPU 交接

Naive TP 的目标模型图在 `WarmUp()` 中一次性准备。启用 CUDA Graph 后，按照
配置的最大 batch 和 KV 容量建立图表；batch 按既有 bucket 补齐，设备上的
`liveKeys=0` 表示无效行。所有变体共享权重和 scratch，不复制权重。
运行中只查询、重绑定、重放已准备的图；未覆盖的 batch/容量、混合 DSA 长度区间、
2048 边界、设备/算子能力不支持或启动捕获失败，都走原有 eager 路径。
启动后修改配置使图失效时，不再补捕图。Naive 的 draft/verification 路径也遵循下述启动准备策略；
其他模型的默认图生命周期不变。

启动日志分别打印模型图和 decode pipeline 在所有 TP rank 上准备成功的形状数。
算子是否适合捕获由实际 warmup 和已有架构/布局检查决定，不再以 MegaMoE 后端
名称作为前置条件。现有 SM 专用算子仍按各自支持范围调度；缺少专用实现时保留
普通 CUDA 路径，不能安全捕获的路径使用 eager。

普通 full-vocabulary top-p decode 使用两个结果槽：每 rank 的模型与 logits
汇聚完成后记录 Event，采样流等待全部 rank；设备随机状态随请求推进，采样结果
异步复制到 pinned 主机缓冲。下一步模型可以先提交，GPU 通过 Event 等待实际采样
结果，再读取 token、更新 position/live length。P2P 可用时融合设备读取与状态更新；
否则使用 CUDA 复制。两者都只在启动时准备图，不需要特定 SM 的 PDL 或设备端图启动。

框架最多提前提交一轮目标模型计算，并非草稿推测。输出结束、取消、batch 变化、
新请求加入或 KV 扩页边界会排空在途结果，再返回原调度路径。必须在排空后才能
发布终止状态、回收 KV；结束请求的额外结果不发布，其余请求的有效结果正常提交。
重复惩罚、grammar/mask、返回 logits、历史前缀存储等需要同步语义的请求，保持
原同步 Forward。通用调度器的可选 `DecodePipeline` 接口默认关闭，其他模型不受影响。

采样图和 GPU 交接统一遵循现有 CUDA Graph 配置，只有能力检查通过才使用，
不提供独立实验开关。额外存储包括概率、参数、两个
小 token 槽、Event 和图执行元数据；没有增加原始或 packed 权重副本。FlashInfer
仅在已有源码内增加可选的逐请求 RNG 参数，默认模板参数维持旧调用行为，没有新增依赖。

交接图的 trace `mode` 为 `cuda_graph_pipeline`。这一模式的 rank
`model_gpu_ms` 包含设备输入更新和已组合的 logits 汇聚；`forward_ms` 记录提交路径，
不包含后续消费者的 token 等待，不能用它倒数代替端到端 token 吞吐。采集 CUDA Event
会等待模型完成，扰动异步流水，应使用关闭 trace 的结果作为正式速度。

相关回归测试：

```sh
ctest --test-dir build-fastllm --output-on-failure -R \
  '^(decode_pipeline|cuda_decode_sampling|naive_n05_serving_metadata|naive_n05_serving_pipeline|naive_n05_serving_pipeline_fallback|naive_n05_serving_graph_fallback|naive_n05_tp_graph)$'
```

其中模型流水测试检查实际交接图执行次数，并把 batch 1/3/5/9/16、请求缩减后的
每一步完整 logits 与相同 token 历史的普通前向比较。采样测试独立检查请求重排、
padding、RNG 推进和取消后的槽复用；不能把两次独立随机采样 token 相同作为唯一的
正确性标准，BF16 舍入可能产生并列最高分。
独立 TP fixture 使用未被启动图覆盖的小 head/window，检查 legacy/paged KV 的
普通执行、请求回收和零运行期捕图；支持形状的重放及失败回退由 serving 测试覆盖。


## Naive 推测解码：启动图与并发状态

`WarmUp()` 在最终 KV 布局确定后准备 verification 图，以及草稿 backbone 和
full-vocabulary proposal 图。草稿 KV 通过设备指针表重绑定，图不持有某个请求的
KV 地址；候选 logits/hidden 使用模型持有的共享 activation 工作区，不复制权重。
草稿模型保留原有短/长 attention 算法。图只覆盖准备成功的实际 batch、验证长度和
容量；未覆盖形状及不支持的布局/算子走 eager，不在请求中捕获。
verification 在捕获前遍历所有形状，直到共享工作区地址稳定且支持的形状均完成
warmup，再统一 capture。最大形状未必覆盖小形状的工作区分支；后者新增分配会使
前者的 warmup 状态失效。若有界准备仍不能稳定，保留 eager 回退，不在运行期补图。

proposal 图执行多步 Markov 投影、温度采样和置信度计算，统一读回候选及置信度。
超过首个置信度截断位置的计算结果全部丢弃，拒绝采样和 KV 提交仍只使用有效前缀。
目标分布 p 继续应用用户 top-p 配置，q 保存的是实际用于抽取候选的温度 softmax。
目标验证的 logits 汇聚使用 rank Event；主机提交线程不必先等待所有 GPU 执行完毕。
输入使用固定容量的 pinned 缓冲异步上传，前一轮结果和提交完成后才能复用。

单请求优先使用实际候选长度的验证图，置信度截断后无需固定补到 8 行；只有精确
形状不可用而 8 行图可用时才补齐。置信度模式仅在相应 batch 的验证图已准备成功时
合批，并在预算允许时做因果 padding。
原始固定行数 TP Flex 后端保留 8 行限制；Shared Flex TP 使用既有的每 8 行分块适配，
可以准备更大的验证图。在最大并发数和后端支持范围内，验证、草稿及 proposal 均准备
1–8 请求的精确形状；更大未覆盖的请求数或混合尾部预算优先拆成已准备的批次，剩余请求
独立处理。每个子批完成拒绝采样和 KV 提交后才复用共享工作区，不通过伪造请求或共享 KV
来补齐。缩 batch、不同接受长度、请求加入、
取消和尾部预算均通过实际请求的长度、独立 pending 队列和 KV 提交处理。
整批 proposal 的 q 由该轮共同持有；逐请求回退使用各请求独立的概率缓冲，不能共享
单轮 proposal 缓冲而覆盖此前请求的 q。

`GraphCaptureScope` 将请求期的禁止捕图约束传递到 persistent workers，包括底层
cuBLASLt 调优和 MoE 子图入口。未见过的 GEMM 在请求期采用受支持的 heuristic，
不会临时捕获测量图。作用域只限制当前调用及其子任务，不改变全局开关或其他模型；
异常退出和工作线程复用会恢复原策略。无新增环境变量或依赖库。

新增 `naive_n05_speculative_serving_{graph,batch5,batch8,failbegin,failinstantiate}` 覆盖启动封存、
完整 verification logits、请求并发/加入/不同预算、置信度截断、参数变化和捕获失败。
其中 batch5 覆盖未对齐的最大并发数和单卡草稿的历史缓存；
支持形状必须全部准备成功，并逐一检查多请求验证确实重放所有 rank 的图。
batch8 对重排和逐步缩减到 1 请求的完整验证 logits、不同接受长度和 KV 回滚做独立 eager 对照。
`naive_n05_draft_unbound_fallback` 强制不支持设备 KV 重绑定的 attention 回退，
检查请求重排后的完整 hidden/KV 以及零 batch 图重放。
直接算子捕获测试显式进入准备状态；正式 serving 测试要求请求期捕获计数保持不变。
