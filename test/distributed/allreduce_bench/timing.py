"""统一 CUDA event 计时；输入恢复在事件区间外，批次结束后统一同步。"""

import math
import statistics

from .data import compare, inputs


def percentile(values, quantile):
    values = sorted(values)
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def statistics_for(samples, round_medians=None):
    if not samples or any(not math.isfinite(v) or v <= 0 for v in samples):
        raise ValueError("性能样本必须是正的有限时间")
    medians = round_medians if round_medians is not None else samples
    middle = statistics.median(medians)
    return {"median_us": middle, "p95_us": percentile(samples, 0.95),
            "cv_percent": (statistics.stdev(medians) / statistics.mean(medians) * 100
                           if len(medians) > 1 else 0.0),
            "calls_per_second": 1e6 / middle}


def rank_max_samples(rank_rows):
    samples = [row["samples_us"] for row in rank_rows]
    if not samples or not samples[0] or any(len(row) != len(samples[0]) for row in samples):
        raise ValueError("各 rank 的性能样本数量不一致")
    return [max(values) for values in zip(*samples)]


def measure(case, job, backend, operation, stream, group):
    import torch
    import torch.distributed as dist

    before = backend.info()
    for _ in range(job["warmup"]):
        operation.reset()
        operation.run()
    stream.synchronize()
    dist.barrier(group=group)
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(job["iterations"])]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(job["iterations"])]
    for start, end in zip(starts, ends):
        operation.reset()
        start.record(stream)
        operation.run()
        end.record(stream)
    stream.synchronize()
    samples = [start.elapsed_time(end) * 1000 for start, end in zip(starts, ends)]
    statistics_for(samples)  # 拒绝空样本、零时间或非有限时间。

    # 校验计时区实际产生的最后一份输出，不能额外执行一次正确调用掩盖计时错误。
    source = inputs(case["n"], case["pattern"], job["seed"])
    metrics = compare(operation.output.cpu().tolist(), source)
    if metrics["failed_elements"]:
        raise RuntimeError(f"timed output mismatch: {metrics}")
    after = backend.info()
    if before is not None:
        calls = job["warmup"] + job["iterations"]
        custom = calls if operation.path == "ft_custom" else 0
        if after["custom"] - before["custom"] != custom or after["nccl"] - before["nccl"] != calls - custom:
            raise RuntimeError("计时区 FT GPU 调用计数与 warmup/iterations 不一致")
    return {"samples_us": samples, "timed_output": metrics, "path": operation.path,
            "timed_counters_before": before, "timed_counters_after": after,
            "warmup": job["warmup"], "iterations": job["iterations"], "repeat": job["repeat"]}
