#!/usr/bin/env python3
"""Measure one round of 4096-token prefill and 512-token decode per MTP mode.

Run in an installed/built ftllm environment. Use --chunked_prefill_size 4096
--mtp 3 --prefix_cache false --cache_history false, plus the deployment flags.
Each measured request has one warmup. Prefix reuse must remain disabled.
Native [Prompt] lines report prefill Forward throughput; results.json also
records TTFT, token arrival times and fixed-length output for audit.
"""

import ctypes
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from ftllm import llm
from ftllm.benchmark import _encode_prompt
from ftllm.util import make_normal_parser, make_normal_llm_model

parser = make_normal_parser('Qwen3.8-Flash-Next hybrid prefill/decode benchmark')
parser.add_argument('--output-dir', required=True, help='New directory for raw results; must not exist')
args = parser.parse_args()
if args.chunked_prefill_size != 4096 or args.mtp != 3:
    parser.error('Use --chunked_prefill_size 4096 --mtp 3; the benchmark measures MTP 0 and 3 after loading')
if any(str(value).lower() not in ('false', '0', 'off')
       for value in (args.prefix_cache, args.cache_history)):
    parser.error('Use --prefix_cache false --cache_history false to measure uncached prefill')
W = Path(args.output_dir).resolve()
W.mkdir(parents=True, exist_ok=False)
cache_gib = args.moe_cuda_cache // 2**30
def save(name, value):
    tmp = W / (name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(W / name)
def state(phase, **kw):
    save('state.json', dict(phase=phase, time=time.time(), **kw))
state('loading')
start_load = time.perf_counter()
placements = []
original_set_device = llm.set_device_map
def record_device(value, is_moe=False):
    placements.append(dict(kind='moe' if is_moe else 'main', value=value))
    return original_set_device(value, is_moe)
llm.set_device_map = record_device
original_layered = llm.set_layered_moe_device_map
def record_layered(value):
    placements.append(dict(kind='layered_moe', value=value))
    return original_layered(value)
llm.set_layered_moe_device_map = record_layered
model = make_normal_llm_model(args)
save('placement.json', dict(maps=placements, tp=os.environ.get('FASTLLM_TP'), serial=os.environ.get('FASTLLM_CUDAPP_SERIAL')))
model.enable_thinking = False
lib = llm.fastllm_lib
load_seconds = time.perf_counter() - start_load
save('loaded.json', dict(load_seconds=load_seconds, options=vars(args),
    python_package=llm.__file__, ready_at=time.time(),
    native_maps=[s for s in Path('/proc/self/maps').read_text().splitlines() if 'libfastllm_tools.so' in s]))
print('MODEL_READY', load_seconds, flush=True)
prompt = ('Implement a robust Python LRU cache with capacity validation, get, put, resize, '
          'and optional TTL using an injectable monotonic clock. Include detailed unit tests '
          'for eviction, expiration, invalid arguments and resizing. Output Python code only.')
base = _encode_prompt(model, model.get_prompt(prompt))
padding = _encode_prompt(model, '# Python standard-library coding task.\n' * 2048)
assert len(base) < 512
decode_ids = padding[:512-len(base)] + base
save('prompt.json', dict(prompt=prompt, input_tokens=decode_ids, template_tokens=base, thinking=False))
prefill_ids = padding[:4096-len(base)] + base
assert len(prefill_ids) == 4096
save('prefill-prompt.json', dict(input_tokens=prefill_ids, template_tokens=base))
rows = []
libc = ctypes.CDLL(None)
libc.fflush.argtypes = [ctypes.c_void_p]

def request(ids, mtp, phase, output_count, repeat):
    os.environ['FASTLLM_QWEN4_ENABLE_MTP'] = str(mtp)
    name = f'cache{cache_gib}g-mtp{mtp}-{phase}-{repeat}'
    state(phase, name=name, mtp=mtp, repeat=repeat)
    started_at = time.time()
    libc.fflush(None)
    print('REQUEST_START', name, started_at, flush=True)
    start = time.perf_counter()
    handle = lib.launch_response_llm_model(model.model, len(ids),
        (ctypes.c_int * len(ids))(*ids), output_count, output_count,
        False, 1.0, 1, 1.0, 1.0, False, 0, None)
    output, arrivals = [], []
    last_progress = time.monotonic()
    while True:
        if time.perf_counter()-start > 900:
            lib.abort_response_llm_model(model.model, handle)
            raise TimeoutError(name)
        if not lib.can_fetch_response_llm_model(model.model, handle):
            time.sleep(.0002)
            continue
        token = int(lib.fetch_response_llm_model(model.model, handle))
        if token < 0:
            assert token == -1, token
            break
        output.append(token)
        arrivals.append(time.perf_counter()-start)
        if time.monotonic()-last_progress > 20:
            print('PROGRESS', name, len(output), round(arrivals[-1], 2), flush=True)
            last_progress = time.monotonic()
    elapsed = time.perf_counter()-start
    assert len(output) == output_count, (name, len(output), output_count)
    (W / (name + '.txt')).write_text(model.hf_tokenizer.decode(output, skip_special_tokens=False))
    row = dict(name=name, cache_gib_per_gpu=cache_gib, mtp=mtp, phase=phase,
        repeat=repeat, input_tokens=len(ids), output_tokens=len(output),
        started_at=started_at, ended_at=time.time(), ttft_s=arrivals[0], elapsed_s=elapsed,
        decode_s=arrivals[-1]-arrivals[0], decode_tps=((len(output)-1)/(arrivals[-1]-arrivals[0]) if len(output)>1 else None),
        end_to_end_tps=len(output)/elapsed, generated=output, arrivals=arrivals,
        generated_sha256=hashlib.sha256(json.dumps(output).encode()).hexdigest())
    libc.fflush(None)
    rows.append(row)
    save('results.json', rows)
    print('RESULT', json.dumps({k:v for k,v in row.items() if k not in ['generated','arrivals']}), flush=True)

try:
    # Long prefill uses one 4096-token chunk and native Forward timing.
    model.set_verbose(True)
    request(prefill_ids, 0, 'prefill-warmup', 1, 0)
    request(prefill_ids, 0, 'prefill', 1, 1)
    model.set_verbose(False)
    # Warm both decode modes once; measure each once.
    request(decode_ids, 0, 'warmup', 128, 0)
    request(decode_ids, 3, 'warmup', 128, 0)
    for mtp in (0, 3):
        request(decode_ids, mtp, 'benchmark', 512, 1)
    state('complete')
    print('BENCHMARK_COMPLETE', flush=True)
finally:
    model.release_memory()
