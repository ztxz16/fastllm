"""固定比较范围、输入形状与计时约定。"""

CORE_SIZES = [4, 16, 64, 256, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
BOUNDARY_SIZES = [252, 256, 260, 2044, 2048, 2052, 4092, 4096, 4100,
                  65532, 65536, 65540]
UNALIGNED_SIZES = [1, 3, 31, 255, 257, 4095, 65535, 65537]
ROUTE_SIZES = [65532, 65536, 65540, 131072, 262144, 1048576]
BACKENDS = ["ft", "vllm", "nccl"]
MODES = ["eager", "graph_copy", "graph_registered"]
SEED = 20261114


def devices(value):
    result = [int(item) for item in value.split(",")]
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise ValueError("设备编号必须非负且不重复")
    return result


def manifest():
    return {
        "schema": 1, "scope": "operator_only", "model": None,
        "dtype": "float32", "world_size": 2, "seed": SEED,
        "backends": BACKENDS, "modes": MODES,
        "core_sizes": CORE_SIZES, "boundary_sizes": BOUNDARY_SIZES,
        "unaligned_sizes": UNALIGNED_SIZES, "route_sizes": ROUTE_SIZES,
        "warmup": 100, "iterations": 1000, "repeats": 5,
        "timing": "CUDA event interval around the complete production operator",
        "excluded": ["startup", "communicator initialization", "input reset",
                     "validation", "logging", "report generation"],
        "included": ["production staging copies", "collective kernels",
                     "device synchronization inside the operator"],
        "comparison": "same data, pair, shape, mode, warmup, iterations and repeats",
        "rank_aggregation": "maximum rank duration for each sample",
        "speedup": "comparison_backend_us / ft_us",
        "graph_copy": "both retain staging copy",
        "graph_registered": "each production implementation uses its supported graph path",
        "fallback": "record actual backend; never label NCCL as custom",
        "precondition": "SM70 executable artifacts and compatible runtime, then P2P checks",
        "steps": ["plan", "preflight", "smoke", "correctness", "performance",
                  "concurrent", "report"],
    }


def plan_rows():
    specs = [
        ("sum_known", "correctness", "0/1/2、正负交替；所有 rank 核对答案"),
        ("sum_random", "correctness", "相同种子、各 rank 不同输入，独立 CPU 参考"),
        ("sum_cancellation", "correctness", "正负抵消及大小量级混合"),
        ("size_boundaries", "correctness", "线程划分与 65536 路由边界"),
        ("unaligned_lengths", "correctness", "非对齐长度与明确的 NCCL 回退"),
        ("repeat_changed_input", "correctness", "同组连续改变数据和长度，验证同步轮次"),
        ("special_values", "correctness", "NaN/Inf/正负零按类别检查"),
        ("graph_replay", "correctness", "改变输入后重放，检查实际新输出"),
        ("invalid_config", "correctness", "拒绝非法配置后新建合法通信组"),
        ("perf_core", "performance", "双方自定义共同支持区间；三种执行模式"),
        ("perf_route", "performance", "超过 FT 自定义阈值时记录真实路径"),
        ("perf_two_groups", "concurrent", "两组独立 TP2：单独与同时运行"),
    ]
    return [dict(case=name, stage=stage, status="PLANNED", reason=reason)
            for name, stage, reason in specs]


def correctness_cases(mode, stress_iterations=10000):
    result = []
    for pattern in ("zeros", "known", "alternating", "cancel", "near_cancel", "mixed"):
        for n in (4, 256, 4096, 65536):
            result.append(dict(case=f"{pattern}_n{n}", n=n, pattern=pattern))
    for n in CORE_SIZES:
        result.append(dict(case=f"random_n{n}", n=n, pattern="random"))
    for prefix, sizes in (("boundary", BOUNDARY_SIZES), ("unaligned", UNALIGNED_SIZES)):
        for n in sizes:
            result.append(dict(case=f"{prefix}_n{n}", n=n, pattern="random"))
    result.append(dict(case="special_n256", n=256, pattern="special"))
    result.append(dict(case="repeat_changed_input", n=256, pattern="changed", changes=stress_iterations))
    result.append(dict(case="repeat_vary_shapes", n=65540, pattern="changed", changes=18,
                       lengths=[4, 4096, 256, 65536, 65540, 16]))
    if mode != "eager":
        for n in (4, 4096, 65536):
            result.append(dict(case=f"graph_new_input_n{n}", n=n, pattern="random", changes=5))
    return result
