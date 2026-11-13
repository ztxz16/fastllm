"""Bounded startup calibration of native quantized CPU inference kernels."""

import functools
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time

_BENCHMARK_BUDGET_MS = 1000
# The native sampling budget does not include starting Python or loading the
# shared library. A cold load can take seconds even for a ~50 ms calibration.
_WORKER_TIMEOUT_S = 10.0


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
def _measure(hidden, intermediate, gguf_type, counts, numa_rows=0, topk=1, down_type=None):
    # Keep calibration weights, cache state and worker lifetime isolated from
    # model loading. The parent also enforces a bounded subprocess timeout.
    request = json.dumps([hidden, intermediate, gguf_type, counts, _BENCHMARK_BUDGET_MS, numa_rows, topk, down_type])
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
    raise ValueError('native CPU calibration returned no samples')


def numa_node_count(root=Path('/sys/devices/system/node')):
    """Count usable logical shards, including nonzero/sparse physical nodes."""
    try:
        paths = sorted(root.glob('node[0-9]*'), key=lambda p: int(p.name[4:]))
        requested = int(os.environ.get('FT_NUMAS', '0'))
        affinity = os.sched_getaffinity(0)
        count = 0
        for path in paths:
            cpus = set()
            for part in (path / 'cpulist').read_text().strip().split(','):
                if not part:
                    continue
                ends = [int(x) for x in part.split('-')]
                cpus.update(range(ends[0], ends[-1] + 1))
            count += bool(cpus & affinity)
        return min(count, requested) if requested > 0 else count
    except (OSError, ValueError, AttributeError):
        return 0


def select_threads(hidden, intermediate, gguf_type, limit, *, numa=False, rows=1, topk=1, down_type=None):
    nodes = numa_node_count() if numa else 0
    counts = candidates(limit)
    if nodes:
        counts = tuple(sorted({n // nodes * nodes for n in counts if n >= nodes}))
    numa_rows = max(1, rows) if nodes and counts else 0
    if not counts:
        counts = candidates(limit)
    started = time.monotonic()
    detail = {'method': 'numa-moe' if numa_rows else 'quantized-gemv', 'candidates': list(counts), 'thread_limit': limit,
              'quota_cores': cpu_quota(), 'budget_ms': _BENCHMARK_BUDGET_MS,
              'timeout_s': _WORKER_TIMEOUT_S, 'hidden': hidden, 'intermediate': intermediate,
              'kind': gguf_type, 'requested_rows': rows, 'rows': numa_rows or 1, 'topk': topk}
    if down_type is not None:
        detail['down_kind'] = down_type
    try:
        if not 1 <= rows <= 2147483647:
            raise ValueError('calibration rows must fit a positive native integer')
        result = (_measure(hidden, intermediate, gguf_type, counts, numa_rows, topk, down_type)
                  if numa_rows and down_type is not None else
                  _measure(hidden, intermediate, gguf_type, counts, numa_rows, topk) if numa_rows
                  else _measure(hidden, intermediate, gguf_type, counts))
        detail.update(result)
        samples = {int(k): float(v) for k, v in result['samples_ms'].items()}
        expected = set(samples)
        samples = {k: v for k, v in samples.items() if k in counts and math.isfinite(v) and v > 0}
        if set(samples) != expected or not samples:
            raise ValueError('invalid CPU calibration samples')
        # Bounded NUMA sweeps start at the largest count. A budget-limited
        # descending prefix can select among measured candidates, while an
        # incomplete legacy ascending sweep must retain the original limit.
        if set(samples) != set(counts):
            measured = set(counts[-len(samples):])
            if not (numa_rows and result.get('measurement_order') == 'descending' and
                    result.get('budget_exhausted') is True and set(samples) == measured and
                    set(result.get('unmeasured_threads', [])) == set(counts) - measured):
                raise ValueError('incomplete CPU calibration samples')
        best = min(samples.values())
        # Prefer fewer workers when throughput is within 5% of the best.
        chosen = min(k for k, value in samples.items() if value <= best * 1.05)
        detail['selected'] = chosen
    except (OSError, ValueError, subprocess.SubprocessError) as error:
        # Failure is not evidence that fewer threads are faster. Keep the
        # original limit, already bounded by affinity and cgroup CPU quota.
        chosen = limit
        detail.update(selected=chosen, fallback=type(error).__name__, error=str(error))
        if isinstance(error, subprocess.CalledProcessError):
            detail['returncode'] = error.returncode
        if isinstance(error, (subprocess.CalledProcessError, subprocess.TimeoutExpired)):
            for key in ('stdout', 'stderr'):
                value = getattr(error, key, None)
                if value:
                    if isinstance(value, bytes):
                        value = value.decode('utf-8', errors='replace')
                    detail[key + '_tail'] = value[-2048:]
    detail['elapsed_ms'] = round(1000 * (time.monotonic() - started), 3)
    print('[auto] CPU thread calibration: ' + json.dumps(detail, sort_keys=True), flush=True)
    return chosen, detail


def _worker(request):
    import ctypes
    values = json.loads(request)
    hidden, intermediate, kind, counts, budget_ms, numa_rows, topk = values[:7]
    down_type = values[7] if len(values) > 7 else None
    if numa_rows:
        # CPU-only startup calibration neither uploads nor registers CUDA memory.
        os.environ['FT_PINNED_WEIGHT'] = '0'
    # Avoid importing the model/tokenizer stack just to time a CPU kernel.
    library = next((p for p in Path(__file__).resolve().parent.glob('*fastllm_tools*')
                    if p.suffix in ('.so', '.dll', '.dylib')), None)
    if library is None:
        raise RuntimeError('native CPU calibration library is unavailable')
    load_started = time.monotonic()
    native = ctypes.CDLL(str(library))
    load_ms = (time.monotonic() - load_started) * 1000
    standard = numa_rows and down_type is not None
    run = (native.benchmark_cpu_standard_moe_threads if standard else
           native.benchmark_cpu_moe_threads if numa_rows else native.benchmark_cpu_linear_threads)
    run.argtypes = [ctypes.c_int] * (6 if standard else 5 if numa_rows else 3) + [ctypes.POINTER(ctypes.c_int),
                    ctypes.c_int, ctypes.c_int]
    run.restype = ctypes.c_char_p
    shape = [hidden, intermediate, kind] + ([down_type] if standard else []) + ([numa_rows, topk] if numa_rows else [])
    raw = run(*shape, (ctypes.c_int * len(counts))(*counts), len(counts), budget_ms)
    if not raw:
        raise RuntimeError('native CPU calibration failed')
    result = json.loads(raw.decode())
    if 'error' in result:
        raise ValueError(result['error'])
    result['library_load_ms'] = round(load_ms, 3)
    if numa_rows:
        ctypes.CDLL(None).fflush(None)  # finish NUMA binding logs before the JSON record
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--worker':
        _worker(sys.argv[2])
    else:
        raise SystemExit('This module is an internal startup calibration worker.')
