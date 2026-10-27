"""Bounded startup calibration of the native quantized decode linear kernel."""

import functools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

_BENCHMARK_BUDGET_MS = 450
_WORKER_TIMEOUT_S = 1.5


def cpu_quota(proc=Path('/proc')):
    """Tightest visible CPU bandwidth quota, in cores, across cgroup ancestors."""
    quotas = []
    try:
        groups = [s.split(':', 2) for s in (proc / 'self/cgroup').read_text().splitlines()]
        mounts = (proc / 'self/mountinfo').read_text().splitlines()
        for line in mounts:
            before, after = line.split(' - ', 1)
            fields, fs = before.split(), after.split()
            v2 = fs[0] == 'cgroup2'
            if not v2 and (fs[0] != 'cgroup' or 'cpu' not in fs[2].split(',')):
                continue
            group = next((g[2] for g in groups if (v2 and g[0] == '0') or
                          (not v2 and 'cpu' in g[1].split(','))), None)
            if group is None:
                continue
            unescape = lambda s: re.sub(r'\\([0-7]{3})', lambda m: chr(int(m[1], 8)), s)
            root, mount = Path(unescape(fields[3])), Path(unescape(fields[4]))
            try:
                relative = Path(group).relative_to(root)
            except ValueError:
                if group != '/':
                    continue
                relative = Path('.')
            if '..' in relative.parts:
                continue
            current = mount / relative
            while True:
                try:
                    if v2:
                        quota, period = (current / 'cpu.max').read_text().split()
                    else:
                        quota = (current / 'cpu.cfs_quota_us').read_text().strip()
                        period = (current / 'cpu.cfs_period_us').read_text().strip()
                    if quota != 'max' and float(quota) > 0 and float(period) > 0:
                        value = float(quota) / float(period)
                        if math.isfinite(value):
                            quotas.append(value)
                except (OSError, ValueError):
                    pass
                if current == mount:
                    break
                current = current.parent
    except (OSError, ValueError, IndexError):
        pass
    return min(quotas) if quotas else None


def thread_limit(physical_cores, quota=None):
    physical_cores = max(1, physical_cores)
    limit = physical_cores - min(4, physical_cores // 4)
    quota = cpu_quota() if quota is None else quota
    if quota is not None:
        slots = max(1, math.floor(quota))
        # Apply the same controller/I/O headroom to quota cores as physical
        # cores. A GEMV-only sample omits live CUDA launch and expert I/O;
        # leaving just one quota core can throttle the full hybrid pipeline.
        limit = min(limit, slots - min(4, slots // 4))
    return max(1, min(128, limit))


def candidates(limit):
    counts, n = {1, limit}, 2
    while n < limit:
        counts.add(n)
        n *= 2
    if 6 <= limit <= 12:
        counts.add(6)
    return tuple(sorted(counts))


@functools.lru_cache(maxsize=8)
def _measure(hidden, intermediate, gguf_type, counts):
    # Keep calibration weights, cache state and worker lifetime isolated from
    # model loading. The parent also enforces a bounded subprocess timeout.
    request = json.dumps([hidden, intermediate, gguf_type, counts, _BENCHMARK_BUDGET_MS])
    result = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--worker', request],
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                            timeout=_WORKER_TIMEOUT_S, check=True)
    for line in reversed(result.stdout.splitlines()):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and 'samples_ms' in value:
                return value
        except ValueError:
            pass
    raise ValueError('native linear calibration returned no samples')


def select_threads(hidden, intermediate, gguf_type, limit):
    counts = candidates(limit)
    fallback = min(limit, 8)
    started = time.monotonic()
    detail = {'method': 'quantized-gemv', 'candidates': list(counts), 'thread_limit': limit,
              'quota_cores': cpu_quota(), 'budget_ms': _BENCHMARK_BUDGET_MS,
              'timeout_s': _WORKER_TIMEOUT_S}
    try:
        result = _measure(hidden, intermediate, gguf_type, counts)
        samples = {int(k): float(v) for k, v in result['samples_ms'].items()}
        samples = {k: v for k, v in samples.items() if k in counts and math.isfinite(v) and v > 0}
        if len(samples) < min(3, len(counts)):
            raise ValueError('too few completed linear samples')
        best = min(samples.values())
        # Prefer fewer workers when throughput is within 5% of the best.
        chosen = min(k for k, value in samples.items() if value <= best * 1.05)
        detail.update(result)
        detail['selected'] = chosen
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        chosen = fallback
        detail.update(selected=chosen, fallback=type(error).__name__)
    detail['elapsed_ms'] = round(1000 * (time.monotonic() - started), 3)
    print('[auto] CPU linear calibration: ' + json.dumps(detail, sort_keys=True), flush=True)
    return chosen, detail


def _worker(request):
    import ctypes
    hidden, intermediate, kind, counts, budget_ms = json.loads(request)
    # Avoid importing the model/tokenizer stack just to time a CPU kernel.
    library = next((p for p in Path(__file__).resolve().parent.glob('*fastllm_tools*')
                    if p.suffix in ('.so', '.dll', '.dylib')), None)
    if library is None:
        raise RuntimeError('native linear calibration library is unavailable')
    native = ctypes.CDLL(str(library))
    run = native.benchmark_cpu_linear_threads
    run.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_char_p
    raw = run(hidden, intermediate, kind, (ctypes.c_int * len(counts))(*counts), len(counts), budget_ms)
    if not raw:
        raise RuntimeError('native linear calibration failed')
    print(raw.decode(), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        _worker(sys.argv[2])
    else:
        raise SystemExit('This module is an internal startup calibration worker.')
