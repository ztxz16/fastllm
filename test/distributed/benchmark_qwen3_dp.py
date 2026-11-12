#!/usr/bin/env python3
"""正式进程模型工厂的固定工作量 DP 吞吐/延迟测试；仅使用已有原生产物。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import threading
import time
import uuid

from run_process_tp import ROOT, capture, write_xlsx
from fastllm_pytools.util import make_normal_parser, make_normal_llm_model
from fastllm_pytools.parallel_config import build_parallel_plan
from fastllm_pytools.qwen3_process_chat import prepare_request
from fastllm_pytools.qwen3_process_session import BACKEND, EMPTY
from fastllm_pytools.tp_graph import validate_graph, validate_allreduce, validate_prefix


def parse_args(argv=None):
    """/** 校验工作量及对照矩阵；每个 TP/配置必须有 DP=1 基线，禁止少于五轮。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'generate', 'worker', 'build-cache'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--configurations', nargs='+', default=['1x1', '2x1', '4x1', '1x2', '2x2'],
                        help='DPxTP，按TP分组，先运行各组DP=1')
    parser.add_argument('--profiles', nargs='+', choices=['common', 'optimized'], default=['common', 'optimized'])
    parser.add_argument('--concurrency', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--requests', type=int, default=8, help='每轮固定请求总量，各配置相同')
    parser.add_argument('--warmup', type=int, default=1, help='每case预热轮数，每轮与正式轮同样多请求')
    parser.add_argument('--iterations', type=int, default=1, help='每次重复的工作轮数')
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--max-new-tokens', type=int, default=32)
    parser.add_argument('--prompt', default='请用简短的文字介绍人工智能的应用。')
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args(argv)
    try:
        configs = [tuple(map(int, value.split('x'))) for value in args.configurations]
        if any(len(pair) != 2 or pair[0] < 1 or pair[1] not in (1, 2) for pair in configs):
            raise ValueError()
    except ValueError:
        parser.error('--configurations requires positive DPxTP with TP=1 or 2')
    if len(set(configs)) != len(configs) or any((1, tp) not in configs for _, tp in configs):
        parser.error('configurations must be unique and include DP=1 for each TP')
    if (args.repeats < 5 or args.warmup < 1 or args.iterations < 1 or args.requests < max(dp for dp, _ in configs) or
            not args.concurrency or min(args.concurrency) < 1 or max(args.concurrency) > args.requests or
            any(args.requests % c for c in args.concurrency)):
        parser.error('require repeats>=5, warmup/iterations>=1, requests>=max(DP,concurrency), requests divisible by concurrency')
    if len(set(args.profiles)) != len(args.profiles) or len(set(args.concurrency)) != len(args.concurrency):
        parser.error('profiles and concurrency values must be unique')
    if not 2 <= args.max_new_tokens <= 1920 or not args.prompt.strip() or not math.isfinite(args.timeout) or args.timeout < 120:
        parser.error('require output tokens in 2..1920, nonempty prompt, finite timeout>=120')
    args.configurations = sorted(configs, key=lambda pair: (pair[1], pair[0]))
    return args


def model_flags(args, output, dp, tp, profile):
    """/** 所有计时与校验均走正式工厂；逐请求落盘由公开参数关闭，不改变执行语义。 */"""
    flags = [str(args.model.resolve()), '--tp-backend', 'process', '--dp', str(dp), '--tp', str(tp),
             '--tp-binary', str(args.generate.resolve()), '--tp-worker', str(args.worker.resolve()),
             '--tp-output', str(output / 'models'), '--tp-timeout', str(args.timeout), '--tp-request-log', 'off',
             '--prefix-cache', 'true' if profile == 'optimized' else 'false']
    if profile == 'optimized':
        flags += ['--cuda-graph']
        if tp == 2:
            flags += ['--custom-allreduce']
    return flags


def run_wave(model, request, count, concurrency):
    """/** 将同一请求重复为固定工作量，复用混合请求计时入口。 */"""
    return run_requests(model, [request] * count, concurrency)


def run_requests(model, requests, concurrency):
    """/**
     * 运行固定客户端数的闭环请求轮，计入正式排队、IPC、生成和KV生命周期。
     * 线程/输入/计时槽预先创建；Token回调只记录时间，输出校验和落盘由调用者在返回后执行。
     * @return 总墙钟秒数、每请求耗时/Token时间戳/原生结果；单请求异常保留在记录中。
     */"""
    gate = threading.Event()
    count = len(requests)
    ready = threading.Barrier(concurrency + 1, timeout=30)
    records = [dict(index=i, token_times=[None] * request['max_new_tokens'], report=None, error=None) for i, request in enumerate(requests)]

    def client(lane):
        ready.wait()
        gate.wait()
        for index in range(lane, count, concurrency):
            record = records[index]
            emitted = 0

            def on_token(_):
                nonlocal emitted
                record['token_times'][emitted] = time.perf_counter()
                emitted += 1

            record['start'] = time.perf_counter()
            try:
                record['report'] = model.generate(requests[index], on_token=on_token)
            except Exception as error:
                record['error'] = f'{type(error).__name__}: {error}'
                record['report'] = getattr(error, 'report', None)
            record['end'] = time.perf_counter()

    with ThreadPoolExecutor(concurrency) as pool:
        futures = [pool.submit(client, lane) for lane in range(concurrency)]
        try:
            ready.wait()
            start = time.perf_counter()
        finally:
            gate.set()
        try:
            for future in futures:
                future.result()
        except BaseException:
            # 中断时先走正式关闭/取消，避免线程池等待整轮长请求后才释放GPU。
            model.close()
            raise
        end = max(record['end'] for record in records)
    return dict(elapsed_s=end - start, records=records)


def check_wave(wave, request, expected, dp, tp, profile, *, cache_state):
    """/** 计时外验证输出及实际Graph/规约/缓存路径；异常携带Token误差，禁止回退冒充目标后端。 */"""
    if cache_state not in (('cold', 'warm') if profile == 'optimized' else ('disabled',)):
        raise ValueError('reference cache state does not match execution profile')
    if len(expected) != request['max_new_tokens']:
        raise RuntimeError('baseline output length differs from fixed workload')
    errors, ranks = [], set()
    first_mismatch = None
    for record in wave['records']:
        if record['error']:
            raise RuntimeError(f"request {record['index']}: {record['error']}")
        report = record['report']
        if (report.get('status') != 'PASS' or report.get('backend') != BACKEND or report.get('request') != request or
                report.get('finish_reason') != 'length' or len(report.get('token_ids', [])) != request['max_new_tokens'] or
                report.get('cache_after_generate') != [EMPTY] * tp):
            raise RuntimeError(f"request {record['index']}: output/backend/request/KV mismatch")
        rank = report.get('dp_rank', 0)
        if type(rank) is not int or not 0 <= rank < dp:
            raise RuntimeError('invalid DP rank')
        ranks.add(rank)
        times = record['token_times']
        if any(t is None for t in times) or times != sorted(times) or not record['start'] <= times[0] <= times[-1] <= record['end']:
            raise RuntimeError('missing or invalid streaming timestamps')
        optimized = profile == 'optimized'
        if report.get('execution') != ('decode_cuda_graph' if optimized else 'eager'):
            raise RuntimeError('execution mode mismatch')
        states = validate_graph(report.get('graph'), tp, optimized)
        if not states[0]['replays' if optimized else 'eager_decodes']:
            raise RuntimeError('no decode execution evidence')
        custom = optimized and tp == 2
        ar = validate_allreduce(states, tp, custom, require_progress=custom)[0]
        if tp > 1 and not custom and not ar['nccl_calls']:
            raise RuntimeError('NCCL path was not executed')
        prefix = validate_prefix(states, tp, optimized)[0]
        if optimized:
            reused = len(request['token_ids']) - 1 if cache_state == 'warm' else 0
            if (prefix['last_reused_tokens'] != reused or
                    prefix['last_computed_tokens'] != len(request['token_ids']) - reused):
                raise RuntimeError(f'{cache_state} prefix cache state mismatch on request {record["index"]}, DP rank {rank}')
        if first_mismatch is None:
            first_mismatch = next((dict(request_index=record['index'], dp_rank=rank, token_1based=i + 1,
                                       actual=a, expected=b)
                                   for i, (a, b) in enumerate(zip(report['token_ids'], expected)) if a != b), None)
        errors.extend(abs(a - b) for a, b in zip(report['token_ids'], expected))
    metrics = dict(max_abs_error=float(max(errors, default=0)),
                   mean_abs_error=sum(errors) / max(1, len(errors)), failed_elements=sum(e != 0 for e in errors))
    if metrics['failed_elements']:
        error = AssertionError(f'generated token IDs differ from same-profile/same-TP/{cache_state} DP=1 baseline; '
                               f'first mismatch: {first_mismatch}')
        error.metrics = metrics
        raise error
    return metrics, ranks


def check_reference_wave(wave, request, references, dp, tp, profile, cache_state):
    """/** 仅由DP=1建立对应缓存状态的输出基线；整轮通过后保存，多副本必须使用已有基线。 */"""
    expected = references.get(cache_state)
    if expected is None:
        if dp != 1:
            raise RuntimeError(f'missing DP=1 {cache_state} baseline; cannot learn reference from DP={dp}')
        first = wave['records'][0]
        if first['error']:
            raise RuntimeError(first['error'])
        expected = first['report']['token_ids'][:]
    metrics, ranks = check_wave(wave, request, expected, dp, tp, profile, cache_state=cache_state)
    references[cache_state] = expected
    return metrics, ranks


def percentile(values, fraction):
    """/** 使用线性插值分位数；原始样本同时保存以便复算。 */"""
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def statistics_row(samples, requests, tokens, gpus):
    """/** 以各次重复的完整工作量墙钟计算吞吐；P95请求延迟覆盖全部请求，CV使用总体标准差。 */"""
    batches = [sample['elapsed_s'] * 1000 for sample in samples]
    latencies = [v for sample in samples for v in sample['latency_ms']]
    ttft = [v for sample in samples for v in sample['ttft_ms']]
    itl = [v for sample in samples for v in sample['itl_ms']]
    rates = [requests * tokens / sample['elapsed_s'] for sample in samples]
    return dict(median_batch_ms=statistics.median(batches), p95_batch_ms=percentile(batches, .95),
                batch_cv_pct=100 * statistics.pstdev(batches) / statistics.mean(batches),
                median_request_ms=statistics.median(latencies), p95_request_ms=percentile(latencies, .95),
                median_ttft_ms=statistics.median(ttft), p95_ttft_ms=percentile(ttft, .95),
                median_itl_ms=statistics.median(itl), p95_itl_ms=percentile(itl, .95),
                output_tokens_per_s=statistics.median(rates), output_tokens_per_s_per_gpu=statistics.median(rates) / gpus,
                requests_per_s=statistics.median([requests / sample['elapsed_s'] for sample in samples]))


def save_benchmark(output, rows, metadata):
    """/** 保存原始精度JSON和三位展示精度MD/CSV/XLSX；缺项或失败不能报告总体PASS。 */"""
    complete = len(rows) == metadata['expected_case_count']
    if 'expected_waves' in metadata:
        complete = complete and metadata.get('completed_waves') == metadata['expected_waves']
    status = 'PASS' if complete and rows and all(row['status'] == 'PASS' for row in rows) else 'FAIL'
    (output / 'summary.json').write_text(json.dumps(dict(status=status, complete=complete, metadata=metadata, cases=rows),
                                                   ensure_ascii=False, indent=2))
    fields = ['case', 'status', 'dp', 'tp', 'gpu_count', 'concurrency', 'prompt_tokens', 'output_tokens',
              'median_batch_ms', 'p95_batch_ms', 'batch_cv_pct', 'median_request_ms', 'p95_request_ms',
              'median_ttft_ms', 'p95_ttft_ms', 'median_itl_ms', 'p95_itl_ms', 'output_tokens_per_s',
              'output_tokens_per_s_per_gpu', 'requests_per_s', 'speedup_vs_dp1', 'baseline_case',
              'backend', 'execution', 'allreduce', 'prefix_cache', 'reference_cache_state',
              'cold_warm_token_mismatches', 'cold_warm_first_divergence_token_1based',
              'max_abs_error', 'mean_abs_error', 'failed_elements', 'reason']
    if any('throughput_ratio_vs_c1' in row for row in rows):
        fields.append('throughput_ratio_vs_c1')
    if any('routing_policy' in row for row in rows):
        fields += ['routing_policy', 'output_tokens_per_wave', 'reused_prompt_tokens', 'computed_prompt_tokens',
                   'prefix_hit_rate_pct', 'speedup_vs_round_robin']
    sample_fields = ['case', 'repeat', 'elapsed_s', 'output_tokens_per_s', 'request_index', 'latency_ms', 'ttft_ms', 'dp_rank']
    sample_rows = []
    for row in rows:
        for sample in row.get('samples', []):
            for index, latency in enumerate(sample['latency_ms']):
                sample_rows.append([row['case'], sample['repeat'], sample['elapsed_s'], sample['output_tokens_per_s'],
                                    index, latency, sample['ttft_ms'][index], sample['dp_ranks'][index]])

    def display(value):
        return f'{value:.3f}' if isinstance(value, float) else ('n/a' if value is None else str(value))

    values = [[row.get(key) for key in fields] for row in rows]
    for name, headers, data in [('summary', fields, values), ('samples', sample_fields, sample_rows)]:
        with (output / f'{name}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(headers)
            writer.writerows([[display(v) for v in row] for row in data])
    lines = [f'# {metadata.get("report_title", "Qwen3 DP performance")}: {status}', '',
             metadata.get('report_description',
                          '正式模型工厂的固定工作量闭环测试，包含排队、IPC、原生KV创建/释放和生成；不含模型加载、预热、分词、测试校验或报告落盘。'),
             'TTFT包含正式入口排队；ITL为客户端Token事件间隔。请求P95不是batch P95。输出Token ID误差仅表示同配置DP一致性。',
             metadata.get('cache_description',
                          'optimized先通过公开清理和串行请求为全部副本建立前缀，再进行热缓存预热和计时；冷/热分别对照DP1。'),
             'cold_warm字段仅记录两种基线的Token差异，不作为数值精度结论；数值诊断需单独执行。',
             metadata.get('comparison_description',
                          'speedup_vs_dp1 = 同profile、TP和并发的DP=1 Median批耗时 / 当前Median批耗时；不同profile之间不计算加速比。'), '',
             '| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join(['---'] * len(fields)) + ' |']
    lines.extend('| ' + ' | '.join(display(v).replace('|', '\\|').replace('\n', ' ') for v in row) + ' |' for row in values)
    lines.extend(['', '```json', json.dumps(metadata, ensure_ascii=False, indent=2), '```'])
    (output / 'summary.md').write_text('\n'.join(lines))
    write_xlsx(output / 'summary.xlsx', [('Cases', fields, values), ('Samples', sample_fields, sample_rows),
        ('Metadata', ['key', 'value'], [[key, json.dumps(value, ensure_ascii=False)] for key, value in metadata.items()])])
    return status


def benchmark_case(model, request, args, row, reference, output):
    """/**
     * 每case按同一缓存状态对照DP1；optimized先清理并串行填充全部副本，再预热和计时。
     * 建立前缀、预热及校验均在计时区域之外；每个冷/热请求仍须通过正式路径和Token精确检查。
     * @return 按disabled/cold/warm区分的DP1基线；失败保留原始响应及已完成样本。
     */"""
    dp, tp, profile, concurrency = (row[key] for key in ('dp', 'tp', 'profile', 'concurrency'))
    references = dict(reference or {})
    cache_state = 'warm' if profile == 'optimized' else 'disabled'
    row.update(reference_cache_state=cache_state, reference_token_ids_by_cache_state=references)

    if profile == 'optimized':
        model.clear_prefix_cache()
        # 使用正式调度逐个请求；不直接选择副本，也不假定调度已覆盖全部卡。
        wave = run_wave(model, request, dp, 1)
        (output / f"{row['case']}_prime.json").write_text(json.dumps(wave, ensure_ascii=False))
        metrics, ranks = check_reference_wave(wave, request, references, dp, tp, profile, 'cold')
        if ranks != set(range(dp)):
            raise RuntimeError('cold priming did not execute every DP replica')
        row['priming'] = dict(requests=dp, concurrency=1, dp_ranks=sorted(ranks), **metrics)

    seen = set()
    for warmup in range(args.warmup):
        wave = run_wave(model, request, args.requests, concurrency)
        (output / f"{row['case']}_warmup{warmup}.json").write_text(json.dumps(wave, ensure_ascii=False))
        metrics, ranks = check_reference_wave(wave, request, references, dp, tp, profile, cache_state)
        seen.update(ranks)
    if seen != set(range(dp)):
        raise RuntimeError('warmup did not execute every DP replica')
    expected = references[cache_state]
    row.update(metrics, reference_token_ids=expected)
    if profile == 'optimized':
        differences = [i for i, (cold, warm) in enumerate(zip(references['cold'], references['warm'])) if cold != warm]
        row.update(cold_warm_token_mismatches=len(differences),
                   cold_warm_first_divergence_token_1based=differences[0] + 1 if differences else None)

    for repeat in range(args.repeats):
        sample = dict(repeat=repeat, elapsed_s=0.0, latency_ms=[], ttft_ms=[], itl_ms=[], dp_ranks=[])
        for iteration in range(args.iterations):
            wave = run_wave(model, request, args.requests, concurrency)
            (output / f"{row['case']}_r{repeat}_i{iteration}.json").write_text(json.dumps(wave, ensure_ascii=False))
            metrics, _ = check_wave(wave, request, expected, dp, tp, profile, cache_state=cache_state)
            sample['elapsed_s'] += wave['elapsed_s']
            for record in wave['records']:
                sample['latency_ms'].append((record['end'] - record['start']) * 1000)
                sample['ttft_ms'].append((record['token_times'][0] - record['start']) * 1000)
                sample['itl_ms'].extend((b - a) * 1000 for a, b in zip(record['token_times'], record['token_times'][1:]))
                sample['dp_ranks'].append(record['report'].get('dp_rank', 0))
        sample['output_tokens_per_s'] = args.requests * args.iterations * request['max_new_tokens'] / sample['elapsed_s']
        row['samples'].append(sample)
    row.update(statistics_row(row['samples'], args.requests * args.iterations, request['max_new_tokens'], dp * tp))
    row.update(metrics)
    return references


def main(argv=None):
    """/** 顺序运行配置，避免不同模型争用GPU；失败中止并保存部分报告，绝不触发编译。 */"""
    args = parse_args(argv)
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_dp_perf_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    raw = output / 'raw'
    raw.mkdir()
    metadata = dict(command=shlex.join([sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]),
                    git=capture(['git', 'rev-parse', 'HEAD']), git_status=capture(['git', 'status', '--short']),
                    gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']),
                    topology=capture(['nvidia-smi', 'topo', '-m']), cuda=capture(['nvcc', '--version']),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), seed=0,
                    warmup=args.warmup, iterations=args.iterations, repeats=args.repeats, requests_per_iteration=args.requests,
                    configurations=args.configurations, profiles=args.profiles, concurrency=args.concurrency,
                    dtype='FP16', kv_cache='FP16 continuous request-local', request_log='off',
                    timing='closed-loop fixed-work wall clock; production queue/IPC/KV lifecycle/protocol checks included',
                    synchronization='same synchronous public generate return after native completion; no added per-kernel sync',
                    cache_profiles={'common': 'prefix disabled, eager',
                                    'optimized': 'all replicas warm identical prefix, decode graph; TP2 custom IPC AllReduce'},
                    graph_lifecycle='production per-request capture/warmup included; model warmup outside timing',
                    reference_scope='exact token IDs against same-profile/same-TP/same-cache-state DP1; '
                                    'cold/warm differences are observations, not independent mathematical accuracy',
                    prefix_priming='optimized: clear through public API, then DP sequential cold requests per case; '
                                   'excluded from warmup count and timing; common: none',
                    expected_case_count=len(args.configurations) * len(args.profiles) * len(args.concurrency))
    rows, references, baselines = [], {}, {}
    request = None
    row = None
    try:
        cache = args.build_cache.read_text()
        (output / 'CMakeCache.txt').write_text(cache)
        metadata['compile_architecture'] = [line for line in cache.splitlines()
                                           if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        for profile in args.profiles:
            for dp, tp in args.configurations:
                flags = model_flags(args, output, dp, tp, profile)
                row = dict(case=f'{profile}_dp{dp}_tp{tp}_startup', status='FAIL', samples=[], reason='',
                           profile=profile, dp=dp, tp=tp, gpu_count=dp * tp, model_flags=flags)
                options = make_normal_parser('DP performance').parse_args(flags)
                allocation = build_parallel_plan(options)
                with make_normal_llm_model(options) as model:
                    if request is None:
                        request = prepare_request(model.model_path, model.hf_tokenizer, args.prompt,
                                                  max_new_tokens=args.max_new_tokens, ignore_eos=True)
                        metadata['request'] = request
                        (output / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
                    for concurrency in args.concurrency:
                        name = f'{profile}_dp{dp}_tp{tp}_c{concurrency}'
                        row = dict(case=name, status='FAIL', reason='', profile=profile, dp=dp, tp=tp,
                                   gpu_count=dp * tp, concurrency=concurrency, prompt_tokens=len(request['token_ids']),
                                   output_tokens=request['max_new_tokens'], backend=BACKEND,
                                   execution='decode_cuda_graph' if profile == 'optimized' else 'eager',
                                   allreduce='custom_ipc' if profile == 'optimized' and tp == 2 else ('nccl' if tp > 1 else 'none'),
                                   prefix_cache='warm' if profile == 'optimized' else 'disabled',
                                   allocation=allocation, model_flags=flags, samples=[])
                        rows.append(row)
                        print(f'RUN: {name}', flush=True)
                        reference = benchmark_case(model, request, args, row, references.get((profile, tp)), raw)
                        references[profile, tp] = reference
                        key = (profile, tp, concurrency)
                        if dp == 1:
                            baselines[key] = row
                        baseline = baselines[key]
                        row.update(status='PASS', baseline_case=baseline['case'],
                                   speedup_vs_dp1=baseline['median_batch_ms'] / row['median_batch_ms'])
                        save_benchmark(output, rows, metadata)
                        print(f"PASS: {name}, {row['output_tokens_per_s']:.3f} output token/s, DP speedup {row['speedup_vs_dp1']:.3f}", flush=True)
    except (Exception, KeyboardInterrupt) as error:
        if row is None:
            row = dict(case='runner_failure', samples=[])
        if not any(item is row for item in rows):
            rows.append(row)
        row.update(status='FAIL', reason=f'{type(error).__name__}: {error}', **getattr(error, 'metrics', {}))
    status = save_benchmark(output, rows, metadata)
    print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
