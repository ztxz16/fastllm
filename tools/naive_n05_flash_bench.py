#!/usr/bin/env python3
"""Measure Naive-N0.5 token latency or capture a warmed CUDA-profiler range.

Uses normal ftllm model arguments, plus --bench-input-ids / --bench-report.
Run under nsys with --capture-range=cudaProfilerApi and --bench-profile to
exclude loading and warmup. Ordinary speed measurements should run without
nsys, --bench-profile, FASTLLM_PRINT_PROFILE or FASTLLM_PROFILE_NUMAS_MOE.
"""
import ctypes
import datetime
import json
import os
from pathlib import Path
import statistics
import time


def main():
    # make_normal_llm_model sets FT_THREADS before importing the runtime.
    from ftllm.util import make_normal_parser, make_normal_llm_model

    parser = make_normal_parser(__doc__)
    parser.add_argument('--bench-input-ids', required=True)
    parser.add_argument('--bench-report', required=True)
    parser.add_argument('--bench-output-tokens', type=int, default=32)
    parser.add_argument('--bench-repeat-to', type=int, default=0)
    parser.add_argument('--bench-runs', type=int, default=2)
    parser.add_argument('--bench-profile', action='store_true')
    args = parser.parse_args()
    payload = json.loads(Path(args.bench_input_ids).read_text())
    ids = payload['input_ids'] if isinstance(payload, dict) else payload
    if not ids or args.bench_output_tokens < 2 or args.bench_runs < 1:
        parser.error('Nonempty input, at least two output tokens and one run are required')
    if args.bench_repeat_to:
        if args.bench_repeat_to < 1:
            parser.error('--bench-repeat-to must be positive')
        ids = [ids[i % len(ids)] for i in range(args.bench_repeat_to)]
    args.max_batch = 1
    args.tokens = max(args.tokens, len(ids) + args.bench_output_tokens + 128)
    started = time.perf_counter()
    model = make_normal_llm_model(args)
    from ftllm import llm

    report = {
        'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'args': vars(args), 'threads': args.threads,
        'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
        'ft_numas': os.environ.get('FT_NUMAS'),
        'load_and_model_warmup_seconds': time.perf_counter() - started,
        'cuda_profiler_range': args.bench_profile,
        'input_ids': ids, 'runs': [],
        'method': 'concurrency 1; direct runtime token fetch; greedy; no exported logits; model load excluded',
    }
    destination = Path(args.bench_report)
    destination.parent.mkdir(parents=True, exist_ok=True)

    def run(name, tokens):
        started = time.perf_counter()
        handle = llm.fastllm_lib.launch_response_llm_model(
            model.model, len(ids), (ctypes.c_int * len(ids))(*ids),
            tokens, 0, False, 1.0, 1, 1.0, 1.0, False, 0, None)
        generated, times = [], []
        while True:
            token = llm.fastllm_lib.fetch_response_llm_model(model.model, handle)
            if token < 0:
                break
            generated.append(token)
            times.append(time.perf_counter() - started)
        intervals = [b - a for a, b in zip(times, times[1:])]
        result = {
            'name': name, 'input_tokens': len(ids), 'output_ids': generated,
            'output_tokens': len(generated), 'token_seconds': times,
            'ttft_seconds': times[0] if times else None,
            'input_tokens_per_ttft_second': len(ids) / times[0] if times else None,
            'decode_tokens_per_second': len(intervals) / sum(intervals) if intervals else None,
            'median_inter_token_seconds': statistics.median(intervals) if intervals else None,
        }
        report['runs'].append(result)
        destination.write_text(json.dumps(report, indent=2) + '\n')
        print('BENCH_RESULT', json.dumps(result), flush=True)

    cudart = None
    capturing = False
    try:
        run('warmup_excluded', 3)
        if args.bench_profile:
            cudart = ctypes.CDLL('libcudart.so.12')
            status = cudart.cudaProfilerStart()
            if status:
                raise RuntimeError(f'cudaProfilerStart failed: {status}')
            capturing = True
        for i in range(args.bench_runs):
            run(f'run_{i + 1}', args.bench_output_tokens)
    finally:
        if capturing:
            status = cudart.cudaProfilerStop()
            if status:
                print(f'cudaProfilerStop failed: {status}', flush=True)
        model.release_memory()


if __name__ == '__main__':
    main()
