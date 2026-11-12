#!/usr/bin/env python3
"""正式DP入口的混合前缀路由对照；每轮从空缓存开始，按实际冷热状态验证Token。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import sys
import threading
import uuid

import benchmark_qwen3_dp as bench
from run_process_tp import save_report
from run_qwen3_entrypoint import verify_native
from run_qwen3_prefill import require
from fastllm_pytools.qwen3_process_session import GenerationCancelled


def workload(seed, vocab):
    """/** 构造六种互不共享首Token的64-token前缀，各重复四次并交替输出8/32个Token；固定seed及EOS策略。 */"""
    require(vocab >= 16 and seed['token_ids'], 'requires vocabulary>=16 and nonempty seed prompt')
    tail = (seed['token_ids'] * 64)[:63]
    return [dict(token_ids=[family, *tail], max_new_tokens=count, eos_token_ids=[],
                 temperature=0.0, top_k=0, top_p=1.0, seed=0)
            for family in range(1, 7) for count in (8, 32, 8, 32)]


def key(request):
    """/** 用完整Token及生成参数标识基准；不同输出长度独立建立冷/热基准。 */"""
    return json.dumps(request, sort_keys=True)


def verify(wave, requests, references, dp, tp, policy):
    """/** 计时外检查输出、实际缓存长度及路由证据；混合请求只允许零命中或完整前缀命中。 */"""
    reused, computed, hits = 0, 0, 0
    for record, request in zip(wave['records'], requests):
        require(not record['error'], f'request {record["index"]}: {record["error"]}')
        report = record['report']
        prefix = report['graph'][0]['prefix_cache']
        length = prefix['last_reused_tokens']
        require(length in (0, len(request['token_ids']) - 1), 'unexpected partial prefix in disjoint workload')
        state = 'warm' if length else 'cold'
        bench.check_wave(dict(records=[record]), request, references[key(request)][state]['token_ids'],
                         dp, tp, 'optimized', cache_state=state)
        route = report['dp_routing']
        require(route['policy'] == policy and report['dp_rank'] in route['idle_ranks'], 'wrong policy/busy dispatch')
        if policy == 'prefix':
            require(route['estimated_reused_tokens'] == length, 'routing directory disagrees with native reuse')
        reused += length
        computed += prefix['last_computed_tokens']
        hits += int(length > 0)
    return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0,
                reused=reused, computed=computed, hits=hits)


def sample(wave, total):
    """/** 从公开请求边界计算延迟，计时保持run_requests原有队列、IPC、KV和同步语义。 */"""
    rows = wave['records']
    return dict(elapsed_s=wave['elapsed_s'], output_tokens_per_s=total / wave['elapsed_s'],
                latency_ms=[(r['end'] - r['start']) * 1000 for r in rows],
                ttft_ms=[(r['token_times'][0] - r['start']) * 1000 for r in rows],
                itl_ms=[(b-a)*1000 for r in rows for a, b in zip(r['token_times'], r['token_times'][1:])],
                dp_ranks=[r['report']['dp_rank'] for r in rows])


def parse_args(argv):
    """/** 固定比较两种正式路由和并发1/4；GPU分组由正式模型工厂校验。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'generate', 'worker', 'build-cache', 'request-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--dp', type=int, default=4)
    parser.add_argument('--tp', type=int, choices=(1, 2), default=1)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--output', type=Path, default=bench.ROOT / 'test/distributed/results')
    args = parser.parse_args(argv)
    if args.dp < 2 or args.repeats < 5 or args.warmup < 1:
        parser.error('requires dp>=2, repeats>=5, warmup>=1')
    args.timeout = 300
    return args


def main(argv=None):
    """/** 建立同TP独立基准、检查目录生命周期并执行两种路由；任何失败保存已完成证据并返回非零。 */"""
    args = parse_args(argv)
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_dp_routing_' + uuid.uuid4().hex[:8])
    (output / 'raw').mkdir(parents=True, exist_ok=False)
    functional = output / 'functional'
    functional.mkdir()
    metadata = dict(command=shlex.join([sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]),
                    git=bench.capture(['git', 'rev-parse', 'HEAD']), git_status=bench.capture(['git', 'status', '--short']),
                    gpu=bench.capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']),
                    cuda=bench.capture(['nvcc', '--version']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    seed=0, warmup=args.warmup, iterations=1, repeats=args.repeats, dp=args.dp, tp=args.tp,
                    expected_case_count=4, expected_waves=4 * args.repeats, completed_waves=0,
                    dtype='FP16', kv_cache='FP16 continuous', cuda_graph=True, custom_allreduce=args.tp == 2,
                    execution_order=['round_robin/c1', 'round_robin/c4', 'prefix/c1', 'prefix/c4'],
                    cache_initial_state='public clear before EVERY warmup and measured wave; empty snapshots verified',
                    cache_description='每轮先公开清缓存，再执行相同混合请求；按实际复用长度分别对照DP1冷/热基准。',
                    timing='formal run_requests: queue/IPC/generation/KV/production protocol validation; no cache clear, test validation, file IO',
                    synchronization='synchronous generate return, no additional CUDA synchronization',
                    report_title='Qwen3 DP mixed-prefix routing',
                    report_description='固定混合Token序列和输出长度、相同初始空缓存；缓存演化是被比较策略的效果。'
                                       '冷/热分别对照同TP的DP1输出，不要求冷热输出相同。路由方案依次运行，须结合波动判断性能。',
                    comparison_description='speedup_vs_round_robin = 相同DP/TP/并发下轮转策略Median耗时 / 当前策略Median耗时。')
    rows, functions, references, current = [], [], {}, None

    def model(dp, policy):
        flags = bench.model_flags(args, output, dp, args.tp, 'optimized') + ['--dp-routing', policy]
        return bench.make_normal_llm_model(bench.make_normal_parser('DP routing').parse_args(flags))

    def save_function(name, action):
        row = dict(case=name, backend=bench.BACKEND, status='FAIL', reason='',
                   max_abs_error=None, mean_abs_error=None, failed_elements=None)
        functions.append(row)
        try:
            row.update(action() or {})
            row['status'] = 'PASS'
        except BaseException as error:
            row.update(reason=f'{type(error).__name__}: {error}', **getattr(error, 'metrics', {}))
            raise
        finally:
            save_report(functional, functions, dict(metadata, expected_case_count=3,
                        report_title='DP routing native lifecycle', warmup=0, repeats=1))

    try:
        cache = args.build_cache.read_text()
        (output / 'CMakeCache.txt').write_text(cache)
        metadata['compile_architecture'] = [line for line in cache.splitlines() if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        config = json.loads((args.model / 'config.json').read_text())
        requests = workload(json.loads(args.request_file.read_text()), config['vocab_size'])
        total = sum(request['max_new_tokens'] for request in requests)
        metadata.update(requests_per_iteration=len(requests), output_tokens_per_wave=total)
        (output / 'requests.json').write_text(json.dumps(requests, ensure_ascii=False, indent=2))
        sequence = [dict(requests[0], token_ids=[1, i, *requests[0]['token_ids'][2:]], max_new_tokens=2) for i in range(2, 12)]
        sequence += [sequence[0]]
        with model(1, 'round_robin') as oracle:
            oracle.session.prefix_directory()
            for request in requests:
                if key(request) in references:
                    continue
                oracle.clear_prefix_cache()
                references[key(request)] = {state: oracle.generate(request) for state in ('cold', 'warm')}
                (output / 'references.json').write_text(json.dumps(dict(generation=references)))
            oracle.clear_prefix_cache()
            lifecycle = [oracle.generate(request) for request in sequence]
        (output / 'references.json').write_text(json.dumps(dict(generation=references, lifecycle=lifecycle), ensure_ascii=False))

        for policy in ('round_robin', 'prefix'):
            current = None
            with model(args.dp, policy) as engine:
                metadata.setdefault('allocations', {})[policy] = engine.plan
                if policy == 'prefix':
                    def directory_lifecycle():
                        engine.clear_prefix_cache()
                        observed = []
                        for request, reference in zip(sequence, lifecycle):
                            result = engine.generate(request)
                            observed.append(result)
                            (functional / 'directory.raw.json').write_text(json.dumps(observed))
                            verify_native(result, reference)
                            p = result['graph'][0]['prefix_cache']
                            ref = reference['graph'][0]['prefix_cache']
                            require(p['snapshots'] == ref['snapshots'], 'directory differs from same-history DP1')
                            require(p['last_reused_tokens'] == result['dp_routing']['estimated_reused_tokens'], 'stale routing after eviction')
                        require(p['evictions'] > 0, 'eviction was not exercised')
                        engine.clear_prefix_cache()
                        require(all(not replica.session.prefix_directory() for replica in engine.replicas), 'clear retained directory')
                        return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)
                    save_function('directory_eviction_clear_same_history_output', directory_lifecycle)

                    def cancel_reuse():
                        engine.clear_prefix_cache()
                        event = threading.Event()
                        try:
                            engine.generate(dict(requests[0], max_new_tokens=128), cancel_event=event, on_token=lambda _: event.set())
                        except GenerationCancelled as error:
                            (functional / 'cancel.raw.json').write_text(json.dumps(error.report))
                        else:
                            raise AssertionError('cancellation was not acknowledged')
                        wave = bench.run_requests(engine, [requests[0]], 1)
                        (functional / 'after_cancel.raw.json').write_text(json.dumps(wave))
                        verify(wave, [requests[0]], references, args.dp, args.tp, policy)
                        require(wave['records'][0]['report']['dp_routing']['estimated_reused_tokens'] == 63, 'valid cancelled prefill lost affinity')
                        return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)
                    save_function('cancelled_prefill_reuse_output', cancel_reuse)

                    def busy_fallback():
                        emitted, cancel = threading.Event(), threading.Event()
                        busy_rank = engine.generate(requests[0])['dp_rank']
                        with ThreadPoolExecutor(1) as pool:
                            future = pool.submit(engine.generate, dict(requests[0], max_new_tokens=128),
                                                 cancel_event=cancel, on_token=lambda _: emitted.set())
                            try:
                                require(emitted.wait(30), 'busy request did not emit')
                                require(not future.done(), 'long request completed before overlap check')
                                wave = bench.run_requests(engine, [requests[0]], 1)
                                (functional / 'busy.raw.json').write_text(json.dumps(wave))
                                verify(wave, [requests[0]], references, args.dp, args.tp, policy)
                                require(wave['records'][0]['report']['dp_rank'] != busy_rank, 'waited for busy cached replica')
                            finally:
                                cancel.set()
                            try:
                                future.result(timeout=300)
                            except GenerationCancelled:
                                pass
                        return dict(max_abs_error=0.0, mean_abs_error=0.0, failed_elements=0)
                    save_function('busy_cached_replica_uses_idle_output', busy_fallback)

                for concurrency in (1, 4):
                    current = dict(case=f'{policy}_dp{args.dp}_tp{args.tp}_c{concurrency}', status='FAIL', reason='',
                                   routing_policy=policy, dp=args.dp, tp=args.tp, gpu_count=args.dp*args.tp, concurrency=concurrency,
                                   prompt_tokens=64, output_tokens_per_wave=total, backend=bench.BACKEND, execution='decode_cuda_graph',
                                   allreduce='custom_ipc' if args.tp == 2 else 'none', prefix_cache='empty at wave start; mixed cold/warm',
                                   samples=[], reused_prompt_tokens=0, computed_prompt_tokens=0)
                    rows.append(current)
                    hit_count = 0
                    for repeat in range(-args.warmup, args.repeats):
                        engine.clear_prefix_cache()
                        require(all(not replica.session.prefix_directory() for replica in engine.replicas), 'wave did not start empty')
                        wave = bench.run_requests(engine, requests, concurrency)
                        (output / 'raw' / f'{current["case"]}_{repeat}.json').write_text(json.dumps(wave))
                        metrics = verify(wave, requests, references, args.dp, args.tp, policy)
                        if repeat < 0:
                            continue
                        values = sample(wave, total)
                        current['samples'].append(dict(values, repeat=repeat))
                        current.update({k: metrics[k] for k in ('max_abs_error', 'mean_abs_error', 'failed_elements')})
                        current['reused_prompt_tokens'] += metrics['reused']
                        current['computed_prompt_tokens'] += metrics['computed']
                        hit_count += metrics['hits']
                        metadata['completed_waves'] += 1
                        bench.save_benchmark(output, rows, metadata)
                    current.update(bench.statistics_row(current['samples'], len(requests), total/len(requests), args.dp*args.tp))
                    baseline = next(row for row in rows if row['routing_policy'] == 'round_robin' and row['concurrency'] == concurrency)
                    current.update(status='PASS', prefix_hit_rate_pct=100*hit_count/(args.repeats*len(requests)),
                                   baseline_case=baseline['case'], speedup_vs_round_robin=baseline['median_batch_ms']/current['median_batch_ms'])
                    print(f'PASS: {current["case"]}, {current["output_tokens_per_s"]:.3f} output token/s', flush=True)
    except (Exception, KeyboardInterrupt) as error:
        if current is None:
            current = dict(case='runner_failure', samples=[])
            rows.append(current)
        current.update(status='FAIL', reason=f'{type(error).__name__}: {error}', **getattr(error, 'metrics', {}))
    status = bench.save_benchmark(output, rows, metadata)
    for row in rows:
        print(json.dumps({key: row.get(key) for key in ('case', 'status', 'reason', 'output_tokens_per_s', 'p95_ttft_ms',
                         'batch_cv_pct', 'prefix_hit_rate_pct', 'speedup_vs_round_robin')}, ensure_ascii=False), flush=True)
    print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
