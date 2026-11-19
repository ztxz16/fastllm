#!/usr/bin/env python3
"""通过正式 MoE 会话验证跨批次完整输出；不编译、不运行 HF、不把局部通过当作全链路通过。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import itertools
import json
import math
from pathlib import Path
import shlex
import sys
import threading
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_full import request
from fastllm_pytools.qwen_moe_process_session import QwenMoeProcessSession
from fastllm_pytools.moe_batch_invariant import BACKEND


def compare_tokens(actual, expected):
    """/** 比较完整 Token 序列，保留长度差与首差；不以 Top-K 或语义接近豁免。 */"""
    pairs = list(itertools.zip_longest(actual, expected, fillvalue=None))
    changed = [i for i, (a, b) in enumerate(pairs) if a != b]
    return dict(token_match=not changed, token_mismatches=len(changed),
                first_divergence_token_1based=changed[0] + 1 if changed else None,
                actual_token=pairs[changed[0]][0] if changed else None,
                expected_token=pairs[changed[0]][1] if changed else None,
                actual_length=len(actual), expected_length=len(expected))


def submit(session, jobs, order, timeout):
    """/** 通过公开 generate 提交；staggered 等首波产出后提交后波，不修改内部调度。 */"""
    indices = list(range(len(jobs)))
    if order == 'reverse':
        indices.reverse()
    with ThreadPoolExecutor(len(jobs)) as pool:
        if order != 'staggered' or len(jobs) == 1:
            barrier = threading.Barrier(len(jobs))

            def generate(index):
                """/** 仅同步客户端提交起点；服务端仍走正式队列和联合轮次选择。 */"""
                barrier.wait(timeout)
                return session.generate(jobs[index])

            futures = {i: pool.submit(generate, i) for i in indices}
        else:
            events = [threading.Event() for _ in range(2)]
            futures = {}
            for i in range(2):
                futures[i] = pool.submit(session.generate, jobs[i], on_token=lambda *_, event=events[i]: event.set())
            for event in events:
                if not event.wait(timeout):
                    raise TimeoutError('first admission wave produced no token')
            futures.update({i: pool.submit(session.generate, jobs[i]) for i in indices[2:]})
        return [futures[i].result() for i in range(len(jobs))]


def verify_paths(session, reports, graph, cache):
    """/** 在计时区域外核验实际后端、Graph、KV 释放及缓存命中证据。 */"""
    states = session._states
    if not all(s.get('expert_gemm_backend') == BACKEND and s.get('expert_gemm_submissions', 0) > 0 for s in states):
        raise AssertionError('requested expert GEMM backend was not observed on every rank')
    if any(s['request_active'] for s in states):
        raise AssertionError('finished requests retained active KV state')
    if graph and not all(s['captures'] == 1 and s['replays'] > 0 for s in states):
        raise AssertionError('Graph capture/replay was not observed on every rank')
    for result in reports:
        if result.get('status') != 'PASS' or len(result['token_ids']) != result['request']['max_new_tokens']:
            raise AssertionError('request did not complete the fixed output workload')
        reuse = result['states'][result['dp_rank']]['prefix_cache']['last_reused_tokens']
        if cache == 'off' and reuse:
            raise AssertionError('disabled prefix cache reused tokens')
        if cache == 'hot' and reuse != len(result['request']['token_ids']) - 1:
            raise AssertionError('hot prefix cache did not reuse the expected prefix for this request')


def main():
    """/** 先建立同模式单请求基准，再逐组合验收；失败和中断均保存四格式部分报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--prompt-lengths', type=int, nargs='+', default=[512, 1024, 1792])
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--concurrencies', type=int, nargs='+', choices=(1, 4, 8), default=[1, 4, 8])
    parser.add_argument('--cache-states', nargs='+', choices=('off', 'cold', 'hot'), default=['off', 'cold', 'hot'])
    parser.add_argument('--orders', nargs='+', choices=('simultaneous', 'reverse', 'staggered'),
                        default=['simultaneous', 'reverse', 'staggered'])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if args.repeats < 5 or args.output_tokens < 3 or not math.isfinite(args.timeout) or args.timeout <= 0 or any(
            length < 1 or length + args.output_tokens - 1 > 2048 for length in args.prompt_lengths):
        parser.error('require at least 5 repeats, positive timeout and context <= 2048')
    for values in (args.prompt_lengths, args.concurrencies, args.cache_states, args.orders):
        if len(values) != len(set(values)):
            parser.error('duplicate matrix values')
    args.model, args.build = args.model.resolve(), args.build.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_batch_invariant_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'批次不变性完整生成验收：{output}', flush=True)
    rows, counter = [], 0
    matrix = list(itertools.product(args.prompt_lengths, (False, True), args.cache_states, args.concurrencies, args.orders))
    metadata = dict(scope='full_generation_batch_invariance', execution_complete=False,
        expected_case_count=len(args.prompt_lengths) + len(matrix), command=shlex.join([sys.executable, *sys.argv]),
        git=capture(['git', 'rev-parse', 'HEAD']), git_status=capture(['git', 'status', '--short']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), build_cache=(args.build / 'CMakeCache.txt').read_text(),
        model=str(args.model), batch_invariant=True, backend_scope='moe_experts',
        warmup=0, iterations=1, repeats=args.repeats, seed=20261114, output_tokens=args.output_tokens,
        prompt_lengths=args.prompt_lengths, concurrencies=args.concurrencies, cache_states=args.cache_states,
        orders=args.orders, eos_policy='disabled', sampling='greedy', ep=4, dtype='float16',
        full_planned_matrix=(args.prompt_lengths == [512, 1024, 1792] and args.output_tokens == 128 and
            set(args.concurrencies) == {1, 4, 8} and set(args.cache_states) == {'off', 'cold', 'hot'} and
            set(args.orders) == {'simultaneous', 'reverse', 'staggered'}),
        report_title='Attention DP/EP 完整生成批次不变性',
        report_description='通过公开会话比较同模式单请求及跨并发完整 Token；不等价于 HF 对齐或性能验收。')

    def session(graph, prefix):
        """/** 每个配置创建独立进程组，复用正式捕获与缓存管理；无跨模式缓存传递。 */"""
        nonlocal counter
        counter += 1
        return QwenMoeProcessSession(args.build / 'fastllm_moe_generate', args.build / 'fastllm_tp_worker',
            args.model, '0,1,2,3', output / f'session{counter:04d}', cuda_graph=graph, prefix_cache=prefix,
            batch_invariant=True, timeout=args.timeout, queue_size=16, trace=True, request_log=True,
            trace_token_positions=[p for p in (2, 79) if p <= args.output_tokens])

    references = {}
    try:
        save_report(output, rows, metadata)
        for length, graph, cache, concurrency, order in [
                (length, False, 'off', 1, 'baseline') for length in args.prompt_lengths] + matrix:
            name = f'prompt{length}_output{args.output_tokens}_graph{int(graph)}_{cache}_c{concurrency}_{order}'
            evidence = dict(reports=[], comparisons=[], session=None)
            row = dict(case=name, backend=BACKEND, status='FAIL', reason='', prompt_length=length,
                       concurrency=concurrency, repeats=args.repeats, max_abs_error=None, mean_abs_error=None,
                       failed_elements=0, token_mismatches=0)
            try:
                jobs = [request(([1, 2, 3, 4] * ((length + 3) // 4))[:length], args.output_tokens)
                        for _ in range(concurrency)]
                if order != 'baseline' and length not in references:
                    raise AssertionError('single-request reference failed or is missing')
                with session(graph, cache != 'off') as active:
                    evidence['session'] = str(active.output)
                    evidence['ready'] = active.ready
                    if cache == 'hot':
                        # round_robin 串行遍历四个 rank，确保每个副本都有相同前缀。
                        for _ in range(4):
                            active.generate(jobs[0])
                    expected = references.get(length)
                    for repeat in range(args.repeats):
                        if cache == 'cold':
                            cleared = active.clear_prefix_cache()
                            if any(s['prefix_cache']['entries'] or s['prefix_cache']['device_bytes'] for s in cleared):
                                raise AssertionError('cold batch started with retained prefix cache')
                        reports = submit(active, jobs, order, args.timeout)
                        evidence['reports'].append(reports)
                        verify_paths(active, reports, graph, cache)
                        if expected is None:
                            expected = reports[0]['token_ids']
                        for index, result in enumerate(reports):
                            comparison = compare_tokens(result['token_ids'], expected)
                            evidence['comparisons'].append(dict(repeat_1based=repeat + 1, request_1based=index + 1,
                                rank=result['dp_rank'], **comparison))
                            row['token_mismatches'] += comparison['token_mismatches']
                            if comparison['first_divergence_token_1based'] is not None:
                                row.setdefault('first_divergence_token_1based', comparison['first_divergence_token_1based'])
                    evidence['states'] = active._states
                    if order == 'staggered' and concurrency > 1 and not any(
                            1 in step.get('joint_round', {}).get('actions', []) and
                            2 in step.get('joint_round', {}).get('actions', [])
                            for reports in evidence['reports'] for result in reports for step in result['steps']):
                        raise AssertionError('staggered submission did not exercise mixed Prefill/Decode; admission coverage incomplete')
                evidence['shutdown'] = active.shutdown
                if row['token_mismatches']:
                    raise AssertionError('full generated token sequences differ; inspect comparisons and saved request traces')
                if order == 'baseline':
                    references[length] = expected
                row['status'] = 'PASS'
            except BaseException as error:
                row.update(reason=f'{type(error).__name__}: {error}', failed_elements=max(1, row['token_mismatches']))
                if isinstance(error, (KeyboardInterrupt, SystemExit)):
                    raise
            finally:
                (output / (name + '.json')).write_text(json.dumps(evidence, ensure_ascii=False, indent=2, allow_nan=False))
                rows.append(row)
                save_report(output, rows, metadata)
                print(f'{row["status"]}: {name} {row["reason"]}', flush=True)
        metadata['execution_complete'] = True
    finally:
        status = save_report(output, rows, metadata)
        print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
