"""可复现 FP32 输入和独立 CPU 参考；无 Torch/CUDA 依赖。"""

from array import array
import hashlib
import math
import random


def inputs(n, pattern, seed, change=0):
    ranks = []
    for rank in range(2):
        rng = random.Random(seed + rank * 1000003 + change)
        shared = random.Random(seed + change)
        values = array("f")
        for index in range(n):
            if pattern == "zeros":
                value = 0.0
            elif pattern in ("known", "changed"):
                value = rank + 1 + (change % 17) / 4
            elif pattern == "alternating":
                value = (rank + 1) * (-1 if index % 2 else 1)
            elif pattern == "random":
                value = rng.uniform(-1, 1)
            elif pattern in ("cancel", "near_cancel"):
                base = shared.randint(-4096, 4096) / 16
                value = base if rank == 0 else -base + (1 / 256 if pattern == "near_cancel" else 0)
            elif pattern == "mixed":
                scale = (2 ** -10, 1, 2 ** 10)[index % 3]
                value = rng.randint(-4096, 4096) / 16 * scale
            elif pattern == "special":
                patterns = ((float("nan"), 1.0), (float("inf"), 1.0),
                            (-float("inf"), 1.0), (float("inf"), -float("inf")),
                            (0.0, -0.0), (-0.0, -0.0), (1.0, -1.0))
                value = patterns[index % len(patterns)][rank]
            else:
                raise ValueError("unknown input pattern: " + pattern)
            values.append(value)
        ranks.append(values)
    return ranks


def digest(values):
    return hashlib.sha256(values.tobytes()).hexdigest()


def compare(actual, ranks):
    if len(actual) != len(ranks[0]) or len(ranks) != 2:
        raise ValueError("参考比较要求两卡等长的逐元素 FP32 SUM")
    maximum, total, failures, finite = 0.0, 0.0, 0, 0
    mismatches = []
    for index, (value, first, second) in enumerate(zip(actual, *ranks)):
        reference64 = float(first) + float(second)
        reference32 = array("f", [reference64])[0]
        if math.isnan(reference32):
            matched = math.isnan(value)
        elif math.isinf(reference32):
            matched = value == reference32
        else:
            matched = math.isfinite(value) and value == reference32
            if math.isfinite(value):
                error = abs(value - reference64)
                maximum = max(maximum, error)
                total += error
                finite += 1
        if not matched:
            failures += 1
            if len(mismatches) < 8:
                mismatches.append({"index": index, "actual": repr(value), "expected": repr(reference32)})
    return {"max_abs_error": maximum, "mean_abs_error": total / finite if finite else 0.0,
            "failed_elements": failures, "finite_elements": finite, "mismatches": mismatches,
            "reference": "CPU FP64 sum of actual FP32 inputs; exact comparison after FP32 rounding"}
