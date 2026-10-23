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
