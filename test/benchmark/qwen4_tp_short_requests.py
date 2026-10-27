#!/usr/bin/env python3
"""Stress TP/NUMA prefill across short requests and cache/MTP transitions.

Example: python qwen4_tp_short_requests.py --report short.json MODEL \
    --tp 2 --moe_device numa --mtp 3 \
    --cache_history false --prefix_cache false

With TP expert-cache support, add --moe_cuda_cache 8g to alternate between
uncached and cached execution as well as between MTP modes.
Requires a real Qwen4 checkpoint. Empty output used to occur intermittently
when NUMA prefill workers reused a TP rank's unfinished CUDA temporaries.
Do not request logits or insert device synchronizations between operators:
those can hide the stream lifetime race this regression exercises.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import time


def run(model, prompts, cache_bytes, mtp, repeats, report):
    from ftllm import llm

    lib = llm.fastllm_lib
    records = []
    budgets = [0, cache_bytes] if cache_bytes else [0]
    modes = [0, mtp] if mtp else [0]

    def request(ids, count, budget, mode):
        llm.set_moe_cuda_cache(budget)
        os.environ['FASTLLM_QWEN4_ENABLE_MTP'] = str(mode)
        start = time.monotonic()
        handle = lib.launch_response_llm_model(
            model.model, len(ids), (ctypes.c_int * len(ids))(*ids),
            count, count, False, 1., 1, 1., 1., False, 0, None)
        output = []
        while True:
            if time.monotonic() - start > 300:
                lib.abort_response_llm_model(model.model, handle)
                raise TimeoutError(f'request {len(records)}, cache={budget}, mtp={mode}')
            if not lib.can_fetch_response_llm_model(model.model, handle):
                time.sleep(.0001)
                continue
            token = int(lib.fetch_response_llm_model(model.model, handle))
            if token < 0:
                break
            output.append(token)
        row = dict(input_tokens=len(ids), requested=count, cache_bytes=budget,
                   mtp=mode, generated=output, end=token,
                   elapsed=time.monotonic() - start)
        records.append(row)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(json.dumps(records, indent=2) + '\n')
        if token != -1 or len(output) != count:
            raise AssertionError(f'short request {len(records)} failed: {row}')

    for ids in prompts:
        for count in (1, 2, 8):
            for budget in budgets:
                for mode in modes:
                    request(ids, count, budget, mode)
    for _ in range(repeats):
        for budget in budgets:
            for mode in modes:
                request(prompts[0], 1, budget, mode)
    print(f'PASS: {len(records)} short requests completed at the requested length', flush=True)
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--repeats', type=int, default=50)
    test, model_argv = parser.parse_known_args()
    if test.repeats < 1:
        parser.error('--repeats must be positive')
    from ftllm.benchmark import _encode_prompt
    from ftllm.util import make_normal_parser, make_normal_llm_model

    options = make_normal_parser(__doc__).parse_args(model_argv)
    if any(str(value).lower() not in ('0', 'false', 'off')
           for value in (options.cache_history, options.prefix_cache)):
        parser.error('disable history and prefix caches to exercise every prefill')
    model = make_normal_llm_model(options)
    try:
        ids = _encode_prompt(model, model.get_prompt(
            'Write a Python LRUCache with get, put, capacity validation, and tests.'))
        padding = _encode_prompt(model, '# Code completion context.\n' * 2048)
        prompts = [padding[:n-len(ids)] + ids for n in (512, 2040)]
        if len(ids) > 512 or any(len(p) != n for p, n in zip(prompts, (512, 2040))):
            raise ValueError('the tokenizer cannot construct the requested prompt lengths')
        run(model, prompts, options.moe_cuda_cache, options.mtp, test.repeats, test.report)
    finally:
        model.release_memory()


if __name__ == '__main__':
    main()
