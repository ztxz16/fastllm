"""普通DP真实KV压力协议；统计只来自worker页管理器。"""

import copy
import math
import time


def validate_load(value, world, *, idle=False, running=False, previous=None):
    """/** 校验各worker页面计数及组内最大压力；空页池仅允许在尚未执行时出现。 */"""
    if not isinstance(value, dict):
        raise RuntimeError('missing native KV load statistics')
    workers = value.get('workers')
    if not isinstance(workers, list) or len(workers) != world:
        raise RuntimeError('invalid KV load worker count')
    usages = []
    for rank, worker in enumerate(workers):
        if not isinstance(worker, dict) or type(worker.get('rank')) is not int or worker['rank'] != rank:
            raise RuntimeError('invalid KV load worker rank')
        total, free = worker.get('total_pages'), worker.get('free_pages')
        if (type(total) is not int or type(free) is not int or not 0 <= free <= total <= 2147483647 or
                (running and total == 0) or (idle and free != total)):
            raise RuntimeError('invalid KV load page counts')
        if previous is not None and previous['workers'][rank]['total_pages'] > 0 and total == 0:
            raise RuntimeError('KV load pool disappeared before model unload')
        usages.append(1.0 - free / total if total else 0.0)
    usage = value.get('kv_cache_usage')
    if (type(usage) not in (int, float) or not 0 <= usage <= 1 or not math.isfinite(usage) or
            not math.isclose(usage, max(usages), rel_tol=1e-9, abs_tol=1e-9)):
        raise RuntimeError('invalid KV load usage')
    return dict(workers=copy.deepcopy(workers), kv_cache_usage=float(usage), sampled_at=time.monotonic())
