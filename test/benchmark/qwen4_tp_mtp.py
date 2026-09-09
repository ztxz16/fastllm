#!/usr/bin/env python3
"""Compare TP MTP against greedy decode using one real checkpoint.

Example: python qwen4_tp_mtp.py --report result.json MODEL --tp 4 --mtp 3
         --atype float16 --prefix_cache 0 --chunked_prefill_size 1024

Load draft weights once, then change the existing MTP switch only between
completed requests. Warmups are excluded from the paired timing summary.

This strict cross-mode reproducibility check fails on any token mismatch.
The tested checkpoint's 256-token LRU case differs at token 169 between TP
Graph and MTP. A mismatch alone does not establish an algorithmic error:
batched verification can change floating-point results. Keep this diagnostic
separate from same-mode optimization regressions and acceptance/cache checks.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--prompts', type=Path, help='Optional JSON with name/tokens entries')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--tokens', type=int, default=256)
    parser.add_argument('--benchmark-only', action='store_true')
    test, model_argv = parser.parse_known_args()
    from ftllm import llm
    from ftllm.benchmark import _encode_prompt
    from ftllm.util import make_normal_parser, make_normal_llm_model
    options = make_normal_parser(__doc__).parse_args(model_argv)
    assert options.mtp > 0, 'Load MTP weights with --mtp 3'
    assert test.repeats > 0 and test.tokens > 1
    model = make_normal_llm_model(options)
    model.enable_thinking = False
    lib = llm.fastllm_lib
    mode = int(options.mtp)
    records = []

    def prompt(name, text, length):
        ids = _encode_prompt(model, model.get_prompt(text))
        padding = _encode_prompt(model, '# Code completion context.\n' * length)
        assert len(ids) <= length
        # Keep the complete chat template and append deterministic padding to
        # its beginning. Input lengths are exact, independent of text decoding.
        ids = padding[:length-len(ids)] + ids
        return dict(name=name, tokens=ids)

    prompts = json.loads(test.prompts.read_text()) if test.prompts else [
        prompt('getters', 'Continue this Python code, output code only. Generate '
            'get_field_3 through get_field_50 with the same format.\n'
            'class Record:\n    def get_field_1(self):\n        return self.data["field_1"]\n\n'
            '    def get_field_2(self):\n        return self.data["field_2"]\n', 512),
        prompt('pytest', 'Continue the pytest tests, code only, through test_increment_50.\n'
            'def increment(x):\n    return x + 1\n\n'
            'def test_increment_1():\n    assert increment(1) == 2\n\n'
            'def test_increment_2():\n    assert increment(2) == 3\n', 512),
        prompt('lru', 'Implement an LRUCache with collections.OrderedDict, get, put, '
            'capacity validation, and detailed pytest tests. Output Python code only.', 512),
    ]

    def save(row):
        records.append(row)
        test.report.parent.mkdir(parents=True, exist_ok=True)
        test.report.write_text(json.dumps(records, ensure_ascii=False, indent=2)+'\n')
        print(json.dumps({k: v for k, v in row.items() if k != 'generated'},
                         ensure_ascii=False), flush=True)

    def request(case, mtp, phase, count=test.tokens, cancel=False, sample=False):
        os.environ['FASTLLM_QWEN4_ENABLE_MTP'] = str(mtp)
        ids = case['tokens']
        stops, stop_ids = model.stop_token_ctypes(None)
        start = time.perf_counter()
        handle = lib.launch_response_llm_model(model.model, len(ids),
            (ctypes.c_int*len(ids))(*ids), ctypes.c_int(count), ctypes.c_int(count),
            ctypes.c_bool(sample), ctypes.c_float(.9 if sample else 1),
            ctypes.c_int(20 if sample else 1), ctypes.c_float(.7 if sample else 1),
            ctypes.c_float(1), ctypes.c_bool(False), stops, stop_ids)
        output, arrivals = [], []
        while True:
            if time.perf_counter()-start > 900:
                lib.abort_response_llm_model(model.model, handle)
                raise TimeoutError(case['name'])
            if not lib.can_fetch_response_llm_model(model.model, handle):
                time.sleep(.0001)
                continue
            token = lib.fetch_response_llm_model(model.model, handle)
            if token < 0:
                assert token == -1, token
                break
            output.append(int(token))
            arrivals.append(time.perf_counter()-start)
            if cancel and len(output) == 7:
                lib.abort_response_llm_model(model.model, handle)
                break
        elapsed = time.perf_counter()-start
        assert len(output) == (7 if cancel else count), len(output)
        row = dict(case=case['name'], mode=mtp, phase=phase, sample=sample,
            input_tokens=len(ids), output_tokens=len(output), generated=output,
            prompt_sha256=hashlib.sha256(json.dumps(ids).encode()).hexdigest(),
            ttft_s=arrivals[0], elapsed_s=elapsed,
            decode_tokens_per_s=(len(output)-1)/(arrivals[-1]-arrivals[0]),
            end_to_end_tokens_per_s=len(output)/elapsed)
        save(row)
        return row

    def compare(case, reference, actual):
        mismatch = next((i for i, (a, b) in enumerate(zip(reference['generated'],
            actual['generated'])) if a != b), None)
        assert reference['generated'] == actual['generated'], (case['name'], mismatch)

    for index, case in enumerate(prompts):
        reference = None
        # Keep each mode's warmup and measurements together: switching to MTP
        # evicts ordinary TP's idle graph, which would penalize its baseline.
        for mtp in ([0, mode] if index % 2 == 0 else [mode, 0]):
            warmup = request(case, mtp, 'warmup')
            if reference is None:
                reference = warmup
            compare(case, reference, warmup)
            for repeat in range(test.repeats):
                compare(case, reference, request(case, mtp, 'benchmark'))

    if not test.benchmark_only:
        for length in (1057, 2016, 2113):
            case = prompt('context_'+str(length), 'Write Python merge sort and explain '
                'its complexity with examples and tests.', length)
            reference = request(case, 0, 'regression', count=64)
            for _ in range(2):
                compare(case, reference, request(case, mode, 'regression', count=64))
        case = prompts[0]
        reference = request(case, 0, 'regression', count=64)
        request(prompts[-1], mode, 'cancel', count=128, cancel=True)
        compare(case, reference, request(case, mode, 'after_cancel', count=64))
        request(case, mode, 'sampling_fallback', count=32, sample=True)
        compare(case, reference, request(case, mode, 'after_sampling', count=64))

    for case in prompts:
        means = {}
        for mtp in (0, mode):
            selected = [r['decode_tokens_per_s'] for r in records
                if r['case'] == case['name'] and r['mode'] == mtp and r['phase'] == 'benchmark']
            means[mtp] = statistics.mean(selected)
        print(json.dumps(dict(case=case['name'], decode_tokens_per_s=means,
            speedup=means[mode]/means[0]), ensure_ascii=False), flush=True)
    model.release_memory()
    print('PASS: complete token equality, native paired timings and requested regressions', flush=True)


if __name__ == '__main__':
    main()
