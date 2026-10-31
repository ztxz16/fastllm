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
