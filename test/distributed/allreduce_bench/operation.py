"""固定输入地址与正式 Graph 生命周期；准备、恢复数据均位于计时区外。"""

from contextlib import nullcontext

from .data import compare, digest, inputs


class Prepared:
    def __init__(self, backend, n, mode, device, rank, stream):
        import torch

        self.backend, self.n, self.mode = backend, n, mode
        self.tensor = torch.full((n,), rank + 1, dtype=torch.float32, device=device)
        self.source = self.tensor.clone()
        self.graph, self.output = None, None
        self.path = None
        if mode != "eager":
            # 普通调用 warmup；不用 vLLM capture 上下文中的占位输出来冒充执行。
            for _ in range(3):
                self.reset()
                backend(self.tensor)
            stream.synchronize()
            self.reset()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            context = backend.custom.capture() if hasattr(backend, "custom") else nullcontext()
            with context:
                with torch.cuda.graph(graph, stream=stream):
                    self.output = backend(self.tensor)
                    self.path = backend.path(n)
            self.graph = graph

    def set_source(self, values):
        import torch

        host = torch.frombuffer(values, dtype=torch.float32)
        self.source.copy_(host)

    def reset(self):
        self.tensor.copy_(self.source)

    def run(self):
        if self.graph is None:
            self.output = self.backend(self.tensor)
            self.path = self.backend.path(self.n)
        else:
            self.graph.replay()
        return self.output

    def close(self):
        if self.graph is not None:
            self.graph.reset()


def validate(case, job, backend, cache, device, rank, stream):
    maximum, total, failures, elements = 0.0, 0.0, 0, 0
    records, mismatches = [], []
    expected_custom, expected_nccl = 0, 0
    lengths = case.get("lengths", [case["n"]])
    for n in lengths:
        if n not in cache:
            cache[n] = Prepared(backend, n, job["mode"], device, rank, stream)
    before = backend.info()
    paths = set()
    for change in range(case.get("changes", 1)):
        n = lengths[change % len(lengths)]
        operation = cache[n]
        source = inputs(n, case["pattern"], job["seed"], change)
        operation.set_source(source[rank])
        operation.reset()
        actual = operation.run().cpu().tolist()
        metrics = compare(actual, source)
        path = operation.path
        paths.add(path)
        if job["backend"] == "ft":
            expected_custom += int(n <= 65536)
            expected_nccl += int(n > 65536)
        if n in job.get("strict_sizes", []) and path != (
                "nccl" if job["backend"] == "nccl" else job["backend"] + "_custom"):
            raise RuntimeError("共同支持区间没有执行目标算子")
        maximum = max(maximum, metrics["max_abs_error"])
        total += metrics["mean_abs_error"] * metrics["finite_elements"]
        elements += metrics["finite_elements"]
        failures += metrics["failed_elements"]
        mismatches.extend(metrics["mismatches"][:max(0, 8 - len(mismatches))])
        if change < 4 or change == case.get("changes", 1) - 1:
            records.append({"change": change, "n": n, "input_hashes": [digest(x) for x in source], "path": path})
        if failures:
            break
    after = backend.info()
    if before is not None and (after["custom"] - before["custom"] != expected_custom or
                               after["nccl"] - before["nccl"] != expected_nccl):
        raise RuntimeError("FT GPU 实际执行计数与验证调用次数不匹配")
    return {"status": "PASS" if failures == 0 else "FAIL", "path": ",".join(sorted(paths)),
            "max_abs_error": maximum, "mean_abs_error": total / elements if elements else 0.0,
            "failed_elements": failures, "mismatches": mismatches, "input_evidence": records,
            "counters_before": before, "counters_after": after}
