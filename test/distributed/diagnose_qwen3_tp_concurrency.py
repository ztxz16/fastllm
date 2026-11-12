#!/usr/bin/env python3
"""同一正式模型进程中交替测试并发1/4，采集独立进程的GPU/CPU观测；不编译。"""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import statistics
import sys
import uuid

import benchmark_qwen3_dp as bench
from benchmark_observer import observer
from fastllm_pytools.qwen3_process_chat import validate_request


def schedule(blocks):
    """/** 相邻组交替ABBA/BAAB；每组两次C1和两次C4，抵消简单先后顺序偏差。 */"""
    return [(block, position, concurrency) for block in range(blocks)
            for position, concurrency in enumerate((1, 4, 4, 1) if block % 2 == 0 else (4, 1, 1, 4))]


def measure(model, request, args, output, name, concurrency, references, state):
    """/** 复用正式测速计时和输出检查；原始结果、缓存校验及统计均放在计时之后。 */"""
    wave = bench.run_wave(model, request, args.requests, concurrency)
    (output / 'raw' / f'{name}.json').write_text(json.dumps(wave, ensure_ascii=False))
    metrics, _ = bench.check_reference_wave(wave, request, references, 1, args.tp, args.profile, state)
    sample = dict(elapsed_s=wave['elapsed_s'], latency_ms=[], ttft_ms=[], itl_ms=[], dp_ranks=[])
    for record in wave['records']:
        sample['latency_ms'].append((record['end'] - record['start']) * 1000)
        sample['ttft_ms'].append((record['token_times'][0] - record['start']) * 1000)
        sample['itl_ms'].extend((b - a) * 1000 for a, b in zip(record['token_times'], record['token_times'][1:]))
        sample['dp_ranks'].append(record['report'].get('dp_rank', 0))
    sample['output_tokens_per_s'] = args.requests * request['max_new_tokens'] / sample['elapsed_s']
    sample['end'] = max(record['end'] for record in wave['records'])
    sample['start'] = sample['end'] - wave['elapsed_s']
    return sample, metrics


def save_observations(output, timeline):
    """/** 保存原始采样及与计时窗口的重叠关系；边界样本不强行归属某一并发，表格同时导出XLSX。 */"""
    path = output / 'telemetry.jsonl'
    frames = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    gpu, cpu, cgroup = [], [], []
    for index, frame in enumerate(frames):
        overlaps = ','.join(str(item['sequence']) for item in timeline
                            if item['start'] <= frame['end'] and item['end'] >= frame['start'])
        base = dict(frame=index, start=frame['start'], end=frame['end'], overlapping_sequences=overlaps)
        gpu.extend(dict(base, **row) for row in frame.get('gpu', []))
        for row in frame.get('processes', []):
            interval_start = row['cpu_interval_start']
            cpu_overlaps = ','.join(str(item['sequence']) for item in timeline
                                   if interval_start is not None and item['start'] <= row['cpu_interval_end']
                                   and item['end'] >= interval_start)
            cpu.append(dict(base, **row, cpu_overlapping_sequences=cpu_overlaps))
        group = frame.get('cgroup', {})
        cgroup.append(dict(base, path=group.get('path'), cpu_max=group.get('cpu_max'),
                           unavailable=group.get('unavailable'), **group.get('counters', {})))
    sheets = []
    for name, rows in [('Timeline', timeline), ('GPU', gpu), ('CPU', cpu), ('Cgroup', cgroup)]:
        keys = list(dict.fromkeys(key for row in rows for key in row))
        values = [[row.get(key) for key in keys] for row in rows]
        with (output / f'{name.lower()}.csv').open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(keys)
            writer.writerows([[f'{v:.3f}' if isinstance(v, float) else v for v in row] for row in values])
        sheets.append((name, keys or ['unavailable'], values))
    bench.write_xlsx(output / 'observations.xlsx', sheets)
    (output / 'timeline.json').write_text(json.dumps(timeline, indent=2))
    return frames


def summarize_gpu(frames, timeline, concurrency, uuids):
    """/** 仅汇总整个采集窗口落在该并发计时内的目标GPU样本；只报告关联，不能据此证明因果。 */"""
    selected = []
    for frame in frames:
        if any(item['concurrency'] == concurrency and item['start'] <= frame['start'] and frame['end'] <= item['end']
               for item in timeline):
            selected.extend(row for row in frame.get('gpu', []) if row['uuid'].removeprefix('GPU-') in uuids)
    rows = []
    for gpu_uuid in sorted({row['uuid'] for row in selected}):
        values = [row for row in selected if row['uuid'] == gpu_uuid]
        row = dict(uuid=gpu_uuid, samples=len(values))
        for key in ('clocks.current.sm', 'clocks.current.memory', 'utilization.gpu', 'temperature.gpu', 'power.draw'):
            numbers = [value[key] for value in values if isinstance(value.get(key), (int, float)) and math.isfinite(value[key])]
            row[key] = dict(min=min(numbers), median=statistics.median(numbers), max=max(numbers)) if numbers else None
        rows.append(row)
    return rows


def parse_args(argv):
    """/** 固定对照为DP1、并发1/4；保留输入文件和所有采样参数，至少执行五组交替测量。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'generate', 'worker', 'build-cache', 'request-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--tp', type=int, choices=(1, 2), default=2)
    parser.add_argument('--profile', choices=('common', 'optimized'), default='optimized')
    parser.add_argument('--blocks', type=int, default=5)
    parser.add_argument('--requests', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--timeout', type=float, default=300)
    parser.add_argument('--telemetry-interval', type=float, default=1.0)
    parser.add_argument('--no-telemetry', action='store_true', help='用于检查观察进程自身影响；报告明确标记未采样')
    parser.add_argument('--output', type=Path, default=bench.ROOT / 'test/distributed/results')
    args = parser.parse_args(argv)
    if args.blocks < 5 or args.requests < 4 or args.requests % 4 or args.warmup < 1:
        parser.error('require blocks>=5, requests>=4 divisible by4, warmup>=1')
    if not math.isfinite(args.timeout) or args.timeout < 120 or not math.isfinite(args.telemetry_interval) or args.telemetry_interval < .5:
        parser.error('require finite timeout>=120 and telemetry-interval>=0.5')
    return args


def main(argv=None):
    """/** 同一模型完成全部交替轮次；保持正式串行、Graph、缓存和流式协议，失败保留部分报告。 */"""
    args = parse_args(argv)
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_tp_concurrency_' + uuid.uuid4().hex[:8])
    (output / 'raw').mkdir(parents=True, exist_ok=False)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]),
                    git=bench.capture(['git', 'rev-parse', 'HEAD']), git_status=bench.capture(['git', 'status', '--short']),
                    gpu=bench.capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']),
                    topology=bench.capture(['nvidia-smi', 'topo', '-m']), cuda=bench.capture(['nvcc', '--version']),
                    cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'), warmup=args.warmup, iterations=1,
                    repeats=2 * args.blocks, blocks=args.blocks, requests_per_iteration=args.requests,
                    expected_case_count=2, expected_waves=4 * args.blocks, completed_waves=0, tp=args.tp, profile=args.profile,
                    dtype='FP16', kv_cache='FP16 continuous request-local', request_log='off',
                    telemetry=not args.no_telemetry, telemetry_interval_s=args.telemetry_interval,
                    observation_limits='external sampler can perturb load; GPU utilization has its own sampling period; '
                                       'CPU 100%=one logical core; cgroup path and unavailable fields recorded; no causal claim',
                    timing='same run_wave as DP benchmark: queue/IPC/generation/KV lifecycle/protocol checks; '
                           'no model loading/warmup/validation/report IO in measured thread',
                    synchronization='synchronous production generate return; no extra CUDA synchronization',
                    comparison_description='throughput_ratio_vs_c1 = C1 Median批耗时 / 当前并发Median批耗时；'
                                           'within_block_c4_over_c1_ratios按每组相同工作量的累计耗时计算。',
                    report_title='Qwen3 interleaved concurrency diagnosis',
                    report_description='同一正式DP1模型及固定工作量交替ABBA/BAAB，每种并发至少10个样本。'
                                       'throughput_ratio_vs_c1为同一TP/profile下C1 Median耗时/C4 Median耗时，不是DP扩展加速比。'
                                       '外部GPU/CPU采样与时间轴见observations.xlsx、telemetry.jsonl。')
    rows, references, timeline = [], {}, []
    current = None
    try:
        cache = args.build_cache.read_text()
        (output / 'CMakeCache.txt').write_text(cache)
        metadata['compile_architecture'] = [line for line in cache.splitlines() if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        request = json.loads(args.request_file.read_text())
        config = json.loads((args.model / 'config.json').read_text())
        validate_request(request, config['vocab_size'])
        if request['temperature'] != 0 or request['eos_token_ids'] or request['max_new_tokens'] < 2:
            raise ValueError('requires unchanged greedy/no-EOS request with output>=2')
        metadata.update(request=request, seed=request['seed'], source_request_file=str(args.request_file.resolve()))
        (output / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))
        options = bench.make_normal_parser('TP concurrency diagnosis').parse_args(bench.model_flags(args, output, 1, args.tp, args.profile))
        allocation = bench.build_parallel_plan(options)
        metadata['allocation'] = allocation
        state = 'warm' if args.profile == 'optimized' else 'disabled'
        with bench.make_normal_llm_model(options) as model:
            if state == 'warm':
                model.clear_prefix_cache()
                wave = bench.run_wave(model, request, 1, 1)
                (output / 'raw/prime.json').write_text(json.dumps(wave))
                bench.check_reference_wave(wave, request, references, 1, args.tp, args.profile, 'cold')
            for concurrency in (1, 4):
                current = dict(case=f'{args.profile}_dp1_tp{args.tp}_c{concurrency}_interleaved', status='FAIL', reason='',
                               profile=args.profile, dp=1, tp=args.tp, gpu_count=args.tp, concurrency=concurrency,
                               prompt_tokens=len(request['token_ids']), output_tokens=request['max_new_tokens'],
                               backend=bench.BACKEND, execution='decode_cuda_graph' if state == 'warm' else 'eager',
                               allreduce='custom_ipc' if state == 'warm' and args.tp == 2 else ('nccl' if args.tp == 2 else 'none'),
                               prefix_cache=state, reference_cache_state=state, samples=[])
                rows.append(current)
                for warmup in range(args.warmup):
                    _, metrics = measure(model, request, args, output, f'c{concurrency}_warmup{warmup}', concurrency, references, state)
                current.update(metrics, reference_token_ids_by_cache_state=references)
            by_concurrency = {row['concurrency']: row for row in rows}
            with observer(output, args.telemetry_interval, enabled=not args.no_telemetry):
                for sequence, (block, position, concurrency) in enumerate(schedule(args.blocks)):
                    current = by_concurrency[concurrency]
                    sample, metrics = measure(model, request, args, output, f'b{block}_p{position}_c{concurrency}', concurrency, references, state)
                    sample.update(repeat=len(current['samples']), block=block, position=position, sequence=sequence)
                    current['samples'].append(sample)
                    current.update(metrics)
                    timeline.append(dict(sequence=sequence, block=block, position=position, concurrency=concurrency,
                                         start=sample['start'], end=sample['end'], elapsed_s=sample['elapsed_s'],
                                         output_tokens_per_s=sample['output_tokens_per_s']))
                    metadata['completed_waves'] += 1
                    bench.save_benchmark(output, rows, metadata)
                    print(f'BLOCK {block + 1}/{args.blocks} C{concurrency}: {sample["output_tokens_per_s"]:.3f} output token/s', flush=True)
        if not args.no_telemetry:
            metadata['observer'] = json.loads((output / 'observer.status.json').read_text())
            if (metadata['observer']['errors'] or not metadata['observer']['samples'] or
                    metadata['observer']['end_monotonic'] < timeline[-1]['end']):
                raise RuntimeError(f'observer failed: {metadata["observer"]}')
        for row in rows:
            row.update(bench.statistics_row(row['samples'], args.requests, request['max_new_tokens'], args.tp))
            row.update(status='PASS', throughput_ratio_vs_c1=rows[0]['median_batch_ms'] / row['median_batch_ms'])
        metadata['within_block_c4_over_c1_ratios'] = [
            sum(item['elapsed_s'] for item in timeline if item['block'] == block and item['concurrency'] == 1) /
            sum(item['elapsed_s'] for item in timeline if item['block'] == block and item['concurrency'] == 4)
            for block in range(args.blocks)]
    except (Exception, KeyboardInterrupt) as error:
        if current is None:
            current = dict(case='runner_failure', samples=[])
            rows.append(current)
        current.update(status='FAIL', reason=f'{type(error).__name__}: {error}', **getattr(error, 'metrics', {}))
    try:
        frames = save_observations(output, timeline)
        if 'allocation' in metadata:
            uuids = {device['uuid'] for replica in metadata['allocation']['replicas'] for device in replica['devices']}
            metadata['gpu_observations'] = {str(c): summarize_gpu(frames, timeline, c, uuids) for c in (1, 4)}
    except Exception as error:
        rows[-1].update(status='FAIL', reason=f'observation export failed: {error}; previous: {rows[-1].get("reason", "")}')
    status = bench.save_benchmark(output, rows, metadata)
    for row in rows:
        print(json.dumps({key: row.get(key) for key in ('case', 'status', 'reason', 'output_tokens_per_s', 'median_ttft_ms',
                         'median_itl_ms', 'batch_cv_pct', 'throughput_ratio_vs_c1')}, ensure_ascii=False), flush=True)
    print(json.dumps({key: metadata.get(key) for key in ('within_block_c4_over_c1_ratios', 'gpu_observations', 'observer')}, ensure_ascii=False), flush=True)
    print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
