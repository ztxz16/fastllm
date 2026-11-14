#!/usr/bin/env python3
"""阶段3～6：正式会话的完整模型、联合调度、Graph/Prefix、真实FP16参考和性能验收。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import signal
import statistics
import sys
import threading
import time
import traceback
import uuid

from run_process_tp import ROOT, capture, save_report, case_process, wait_case, parse_report
from qwen_moe_reference import fixture, full_logits
from fastllm_pytools.qwen_moe_process_session import QwenMoeProcessSession, BACKEND
from fastllm_pytools.qwen3_process_session import GenerationCancelled, GenerationObserverError


def require(value, message):
    """/** 验收条件失败立即记录具体原因，不放宽误差或隐式跳过路径。 */"""
    if not value:
        raise AssertionError(message)


def request(ids, length=8):
    """/** 固定贪心、禁用EOS，便于不同执行模式使用相同工作量和完整Token历史。 */"""
    return dict(token_ids=ids, max_new_tokens=length, eos_token_ids=[], temperature=0.0,
                top_k=0, top_p=1.0, seed=20261114)


def batch(session, requests, forced=None):
    """/** 通过与正式模型相同的会话入口并发提交，不直接调用测试专用 kernel。 */"""
    with ThreadPoolExecutor(len(requests)) as pool:
        futures = [pool.submit(session.generate, r, teacher_forced_tokens=None if forced is None else forced[i][:-1])
                   for i, r in enumerate(requests)]
        return [f.result() for f in futures]


def compare_trace(report, reference, metrics, atol, rtol, label):
    """/** 以本次实际输出作下一步教师强制输入；比较每步完整logits及贪心选择。 */"""
    ids = list(report['request']['token_ids'])
    for index, step in enumerate(report['steps']):
        expected = reference(ids)
        actual = step['logits']
        require(len(actual) == len(expected), f'{label} step={index} logits shape mismatch')
        bad = 0
        for a, b in zip(actual, expected):
            error = abs(a-b) if math.isfinite(a) and math.isfinite(b) else float('inf')
            mismatch = not math.isfinite(error) or error > atol + rtol * abs(b)
            metrics['max'] = max(metrics['max'], error)
            metrics['sum'] += error
            metrics['count'] += 1
            metrics['failed'] += int(mismatch)
            bad += int(mismatch)
        if bad:
            errors = [abs(a-b) if math.isfinite(a) and math.isfinite(b) else float('inf')
                      for a, b in zip(actual, expected)]
            worst = max(range(len(errors)), key=errors.__getitem__)
            raise AssertionError(f'{label} step={index} input_length={len(ids)} backend={step["execution"]}: '
                                 f'{bad} failed logits; max_abs={errors[worst]:.9g} '
                                 f'mean_abs={sum(errors)/len(errors):.9g}; worst_vocab_id={worst} '
                                 f'actual={actual[worst]:.9g} expected={expected[worst]:.9g}')
        require(step['token_id'] == max(range(len(actual)), key=lambda i: actual[i]), f'{label}: invalid greedy token')
        forced = report.get('teacher_forced_tokens')
        if index < len(report['steps']) - 1:
            ids.append(forced[index] if forced is not None else step['token_id'])


def align_tied_routing(actual, expected, tokens, experts, topk):
    """/**
     * 仅为同分专家的槽位置换对齐比较权重，不修改任何后端的路由或原始证据。
     * 两侧必须选择相同且无重复的专家集合，置换槽位在两侧各自的router logits
     * 中均须严格同分。不同集合、非同分换序、非法ID或形状立即拒绝。
     * @param actual FastLLM层的原始诊断字段。
     * @param expected HF层的原始参考字段。
     * @param tokens 本rank真实Token数。
     * @param experts 模型专家数。
     * @param topk 每Token选择数。
     * @return 按参考专家顺序排列的实际权重副本，以及同分置换槽位数。
     */"""
    require(tokens >= 0 and 1 <= topk <= experts, 'invalid routing dimensions')
    for name, values in (('actual', actual), ('reference', expected)):
        for field, size in (('selected_experts', tokens*topk), ('routing_weights', tokens*topk),
                            ('router_logits', tokens*experts)):
            require(len(values[field]) == size, f'{name}/{field}: shape mismatch')
        for expert in values['selected_experts']:
            require(isinstance(expert, (int, float)) and not isinstance(expert, bool)
                    and math.isfinite(expert) and int(expert) == expert and 0 <= expert < experts,
                    f'{name}/selected_experts: invalid ID {expert}')
        require(all(math.isfinite(value) for value in values['router_logits']),
                f'{name}/router_logits: nonfinite value')

    aligned, permutations = [], 0
    for token in range(tokens):
        start = token*topk
        a = [int(x) for x in actual['selected_experts'][start:start+topk]]
        b = [int(x) for x in expected['selected_experts'][start:start+topk]]
        require(len(set(a)) == topk and len(set(b)) == topk,
                f'token={token}/selected_experts: duplicate expert')
        require(set(a) == set(b), f'token={token}/selected_experts: different sets actual={a} reference={b}')
        for left, right in zip(a, b):
            if left != right:
                require(all(values['router_logits'][token*experts+left] ==
                            values['router_logits'][token*experts+right]
                            for values in (actual, expected)),
                        f'token={token}/selected_experts: non-tied permutation actual={a} reference={b}')
                permutations += 1
        aligned.extend(actual['routing_weights'][start+a.index(expert)] for expert in b)
    return aligned, permutations


class Suite:
    """/** 每case保存原始证据，即使中途失败也生成四种汇总格式。 */"""

    def __init__(self, args):
        self.args = args
        self.output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_moe_full_' + uuid.uuid4().hex[:8])
        self.output.mkdir(parents=True)
        self.rows, self.counter = [], 0
        self.metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
            gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
            build_cache=args.build_cache.read_text(), cuda=capture(['nvcc', '--version']),
            seed=20261114, warmup=0, iterations=1, repeats=1,
            correctness='teacher-forced full logits; synthetic atol=rtol=.008, real atol=.05 rtol=.01',
            performance='end-to-end concurrent batch including admission and KV release; no token logs/trace; Eager/Graph use identical requests and EOS policy',
            report_title='Attention DP + EP full model')

    def case(self, name, action):
        """/** action共享误差累计器；异常和已完成输出保留在独立case文件内。 */"""
        row = dict(case=name, backend=BACKEND, status='FAIL', max_abs_error=None, mean_abs_error=None,
                   failed_elements=0, reason='')
        metrics = dict(max=0.0, sum=0.0, count=0, failed=0)
        try:
            evidence = action(metrics)
            (self.output / (name + '.json')).write_text(json.dumps(evidence, indent=2, allow_nan=False))
            row['status'] = 'PASS'
            if isinstance(evidence, dict):
                row.update(evidence.get('summary', {}))
        except Exception as error:
            row['reason'] = f'{type(error).__name__}: {error}'
        finally:
            if metrics['count']:
                row.update(max_abs_error=metrics['max'], mean_abs_error=metrics['sum']/metrics['count'])
            row['failed_elements'] = metrics['failed'] or int(row['status'] == 'FAIL')
            self.rows.append(row)
            save_report(self.output, self.rows, self.metadata)
            print(f'{row["status"]}: {name} {row["reason"]}', flush=True)
        return row['status'] == 'PASS'

    def session(self, model, world, graph=False, prefix=False, trace=True, routing='round_robin'):
        """/** 新建独立权重生命周期，不复用旧结果目录；不自动编译或下载。 */"""
        self.counter += 1
        return QwenMoeProcessSession(self.args.binary, self.args.worker, model,
            ','.join(map(str, range(world))), self.output / f'session-{self.counter:04d}',
            cuda_graph=graph, prefix_cache=prefix, routing=routing, timeout=self.args.timeout,
            trace=trace, request_log=trace, queue_size=16)

    def correctness(self, model, world, reference, real=False, baseline=None, requests=None):
        """/** 覆盖变长、空闲、跨请求图复用、冷热前缀及rank独立KV，逐步比对参考。 */"""
        vocab = json.loads((model / 'config.json').read_text())['vocab_size']
        requests = requests or [request([(i*7+j+1) % vocab for j in range(2+i%4)], 3+i%5) for i in range(8)]
        for graph in (False, True):
            for prefix in (False, True):
                name = f'{model.name}_ep{world}_graph{int(graph)}_prefix{int(prefix)}'
                def run(metrics):
                    nonlocal baseline
                    with self.session(model, world, graph, prefix) as session:
                        self.metadata['workers'] = session.ready['workers']
                        reports = batch(session, requests, baseline)
                        if baseline is None:
                            baseline = [r['token_ids'] for r in reports]
                        # 相同请求再跑，验证缓存与Graph跨请求生命周期。
                        warm = batch(session, requests, baseline)
                        for i, report in enumerate(reports + warm):
                            compare_trace(report, reference, metrics, .05 if real else .008, .01 if real else .008, f'{name}/request{i}')
                        token_mismatches = sum(a != b for report, expected in zip(reports + warm, baseline * 2)
                                               for a, b in zip(report['token_ids'], expected))
                        if graph:
                            require(all(s['captures'] == 1 and s['replays'] > 0 for s in session._states), 'Graph capture/replay evidence missing')
                        if prefix:
                            # 强制同rank最长前缀路由，避免轮转时不同完成顺序造成无命中。
                            session.routing = 'prefix'
                            hot = session.generate(requests[0], teacher_forced_tokens=baseline[0][:-1])
                            compare_trace(hot, reference, metrics, .05 if real else .008, .01 if real else .008, name+'/prefix-route')
                            require(hot['states'][hot['dp_rank']]['prefix_cache']['hits'] > 0, 'prefix cache never hit')
                            repeated = session.generate(requests[0], teacher_forced_tokens=baseline[0][:-1])
                            require(repeated['token_ids'] == hot['token_ids'], 'same-path warm repeat changed tokens')
                            session.clear_prefix_cache()
                        evidence = dict(cold=reports, warm=warm, ready=session.ready,
                                        summary=dict(token_mismatches=token_mismatches))
                    evidence['shutdown'] = session.shutdown
                    return evidence
                self.case(name, run)

    def lifecycle(self, model):
        """/** 覆盖EOS、非法输入、取消后复用、关闭与真实worker故障；失败不能假装PASS。 */"""
        def run(metrics):
            with self.session(model, 4, True, True) as session:
                initial = session.generate(request([1, 2, 3], 4))
                eos = request([1, 2, 3], 40)
                eos['eos_token_ids'] = [initial['token_ids'][0]]
                stopped = session.generate(eos)
                require(len(stopped['token_ids']) == 1 and stopped['finish_reason'] == 'eos', 'EOS not honored')
                for invalid in (request([999999999]), request([1], 2049), request([])):
                    try:
                        session.generate(invalid)
                    except ValueError:
                        pass
                    else:
                        raise AssertionError('invalid request admitted')
                cancel = threading.Event()
                with ThreadPoolExecutor(2) as pool:
                    cancelled = pool.submit(session.generate, request([2, 3], 100), cancel_event=cancel, on_token=lambda _: cancel.set())
                    peer = pool.submit(session.generate, request([3, 2], 6))
                    try:
                        cancelled.result()
                    except GenerationCancelled:
                        pass
                    else:
                        raise AssertionError('cancellation ignored')
                    require(peer.result()['status'] == 'PASS', 'peer failed after cancellation')
                def fail_observer(_):
                    raise ValueError('injected observer failure')
                try:
                    session.generate(request([2, 3], 100), on_token=fail_observer)
                except GenerationObserverError:
                    pass
                else:
                    raise AssertionError('observer failure not isolated')
                require(session.generate(request([1, 2, 3], 4))['token_ids'] == initial['token_ids'], 'reuse after cancellation changed output')
                # 2048边界必须真执行，不能只检查参数。
                boundary = session.generate(request([1]*2048, 1))
                require(len(boundary['token_ids']) == 1, '2048 boundary failed')
                session.routing = 'prefix'
                session.clear_prefix_cache()
                session.generate(request([4, 5, 6], 3))
                partial = session.generate(request([4, 5, 7, 8], 3))
                require(partial['states'][partial['dp_rank']]['prefix_cache']['last_reused_tokens'] == 2, 'partial prefix not reused')
                session.clear_prefix_cache()
                cold = session.generate(request([4, 5, 7, 8], 3), teacher_forced_tokens=partial['token_ids'][:-1])
                for index, (a, b) in enumerate(zip(partial['steps'], cold['steps'])):
                    for x, y in zip(a['logits'], b['logits']):
                        error = abs(x-y)
                        metrics['max'] = max(metrics['max'], error)
                        metrics['sum'] += error
                        metrics['count'] += 1
                        metrics['failed'] += int(error > .008 + .008*abs(y))
                require(not metrics['failed'], 'partial prefix changed actual logits')
                for i in range(12):
                    session.generate(request([4, 5, i], 2))
                require(all(s['prefix_cache']['entries'] <= 8 for s in session._states) and
                        any(s['prefix_cache']['evictions'] > 0 for s in session._states), 'prefix LRU not exercised')
                session.clear_prefix_cache()
            return dict(shutdown=session.shutdown)
        self.case('joint_lifecycle', run)

        def failure(_):
            session = self.session(model, 4, True).__enter__()
            workers = session.ready['workers']
            try:
                os.kill(workers[1]['pid'], signal.SIGKILL)
                with ThreadPoolExecutor(4) as pool:
                    futures = [pool.submit(session.generate, request([1, 2], 40)) for _ in range(4)]
                    for future in futures:
                        try:
                            future.result(self.args.timeout + 40)
                        except RuntimeError:
                            pass
                        else:
                            raise AssertionError('dead worker returned success')
                require(session.state == 'failed', 'failed group reusable')
            finally:
                try:
                    session.close()
                except RuntimeError:
                    require(session.state == 'failed', 'shutdown failed unexpectedly')
            require(all(not Path(f'/proc/{w["pid"]}').exists() for w in workers), 'worker not reaped')
            return dict(workers=workers, failure=session.failure)
        self.case('worker_failure_reaps_group', failure)

    def real_layers(self, model, layers):
        """/** 同一HF层输入送入正式单层路径，比较全部层的Router/专家/隐藏输出。 */"""
        directory = self.output / 'real-layers'
        directory.mkdir()
        config = json.loads((model / 'config.json').read_text())
        tasks = [{k: v for k, v in row.items() if k != 'reference'} for row in layers]
        (directory / 'tasks.json').write_text(json.dumps(tasks))
        def run(metrics):
            command = [str(self.args.binary.resolve().parent / 'fastllm_moe_layer_check'),
                       '--worker', str(self.args.worker.resolve()), '--model', str(model.resolve()),
                       '--tasks', str(directory / 'tasks.json'), '--devices', '0,1,2,3',
                       '--output', str(directory / 'native'), '--dump-weights', '0']
            (directory / 'command.txt').write_text(shlex.join(command))
            with (directory / 'stdout.log').open('wb') as out, (directory / 'stderr.log').open('wb') as err:
                with case_process(command, stdout=out, stderr=err) as process:
                    code = wait_case(process, self.args.timeout)
            report = parse_report((directory / 'stdout.log').read_text())
            require(code == 0 and report['status'] == 'PASS' and report.get('workers_reaped'), 'real layer process failed')
            require(len(report['results']) == len(layers), 'missing real layer results')
            failures, permutations = [], 0
            for layer, actual in zip(layers, report['results']):
                require(actual['task'] == layer['name'] and len(actual['ranks']) == 4, 'layer/rank mismatch')
                for rank, (values, ref) in enumerate(zip(actual['ranks'], layer['reference'])):
                    aligned_weights = None
                    try:
                        aligned_weights, moved = align_tied_routing(values, ref,
                            len(layer['inputs'][rank]['hidden']) // layer['hidden'],
                            config['num_experts'], config['num_experts_per_tok'])
                        permutations += moved
                    except AssertionError as error:
                        failures.append(f'{layer["name"]}/rank{rank}/{error}')
                        metrics['failed'] += 1
                    for field, expected in ref.items():
                        # 专家ID是类别，不能把ID相减计入浮点张量的max/mean误差。
                        if field == 'selected_experts':
                            continue
                        require(len(values[field]) == len(expected), f'{layer["name"]} rank{rank} {field} shape')
                        observed = aligned_weights if field == 'routing_weights' and aligned_weights is not None else values[field]
                        bad = 0
                        for a, b in zip(observed, expected):
                            error = abs(a-b) if math.isfinite(a) and math.isfinite(b) else float('inf')
                            wrong = not math.isfinite(error) or error > .05 + .01*abs(b)
                            bad += int(wrong)
                            metrics['max'] = max(metrics['max'], error)
                            metrics['sum'] += error
                            metrics['count'] += 1
                            metrics['failed'] += int(wrong)
                        if bad:
                            failures.append(f'{layer["name"]}/rank{rank}/{field}: {bad}')
            require(not failures, '; '.join(failures))
            return dict(tasks=len(layers), native_report=str(directory / 'stdout.log'), workers=report['workers'],
                        tied_routing_slots=permutations)
        self.case('real_all_layers_router_experts_outputs', run)

    def performance(self, model):
        """/** 两种正式执行模式统一工作量；每轮同步结果后停止计时，校验和落盘在计时区外。 */"""
        baseline, tokens, cache_histories = {}, {}, {}
        for graph, cache, concurrency in ((g, c, n) for g in (False, True)
                                         for c in ('off', 'cold', 'hot') for n in (1, 4, 8)):
            def run(_):
                key = (cache, concurrency)
                jobs = [request([1, 2, 3, 4] * 8, 16) for _ in range(concurrency)]
                raw, responses = [], []
                with self.session(model, 4, graph, cache != 'off', trace=False, routing='prefix' if cache != 'off' else 'round_robin') as session:
                    batch(session, jobs)  # 一次相同工作量warmup。
                    if cache == 'hot':
                        batch(session, [jobs[0]] * 4)
                    memory = capture(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv'])
                    with ThreadPoolExecutor(concurrency) as pool:
                        for repeat in range(5):
                            if cache == 'cold':
                                session.clear_prefix_cache()
                            start = time.perf_counter()
                            future = [pool.submit(session.generate, r) for r in jobs]
                            result = [f.result() for f in future]
                            elapsed = (time.perf_counter() - start) * 1000
                            raw.append(elapsed)
                            responses.append(result)
                    if graph:
                        require(all(s['replays'] > 0 and s['captures'] == 1 for s in session._states), 'benchmark missed Graph')
                    evidence = session._states
                all_tokens = [[r['token_ids'] for r in run] for run in responses]
                reuse = [sorted(r['states'][r['dp_rank']]['prefix_cache']['last_reused_tokens'] for r in run) for run in responses]
                if cache == 'hot':
                    require(all(value == len(jobs[0]['token_ids'])-1 for run in reuse for value in run), 'hot benchmark did not hit warm cache')
                if cache == 'off':
                    require(not any(value for run in reuse for value in run), 'disabled cache reused tokens')
                if graph:
                    require(all_tokens == tokens[key], 'benchmark modes used different output histories')
                    require(reuse == cache_histories[key], 'benchmark modes had different prefix reuse')
                else:
                    tokens[key] = all_tokens
                    cache_histories[key] = reuse
                ttft = [r['steps'][0]['elapsed_ms'] for run in responses for r in run]
                itl = [b['elapsed_ms']-a['elapsed_ms'] for run in responses for r in run for a, b in zip(r['steps'], r['steps'][1:])]
                queued = [r['queue_ms'] for run in responses for r in run]
                median = statistics.median(raw)
                if not graph:
                    baseline[key] = median
                return dict(samples_ms=raw, responses=responses, states=evidence, memory_used_after_warmup=memory,
                    warmup=1, iterations=1, repeats=5,
                    cache_state=cache, reused_tokens=reuse,
                    ttft_samples_ms=ttft, itl_samples_ms=itl, queue_samples_ms=queued,
                    summary=dict(median_ms=median, p95_ms=sorted(raw)[math.ceil(.95*len(raw))-1],
                                 cv=statistics.pstdev(raw)/statistics.mean(raw), tokens_per_second=concurrency*16*1000/median,
                                 speedup=baseline[key]/median, concurrency=concurrency, repeats=5,
                                 ttft_median_ms=statistics.median(ttft), itl_median_ms=statistics.median(itl),
                                 queue_median_ms=statistics.median(queued),
                                 reserved_kv_bytes=sum(s['reserved_kv_bytes'] for s in evidence),
                                 prefix_bytes=sum(s['prefix_cache']['device_bytes'] for s in evidence),
                                 graph_workspace_bytes=sum(s['workspace_bytes'] for s in evidence)))
            self.case(f'perf_graph{int(graph)}_cache_{cache}_concurrency{concurrency}', run)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--build-cache', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='真实Qwen1.5-MoE目录；省略则运行小模型及生命周期')
    parser.add_argument('--requests', type=Path, default=ROOT / 'test/distributed/data/qwen_moe_requests.jsonl')
    parser.add_argument('--reference-python', default=sys.executable, help='独立Transformers参考环境的Python路径')
    parser.add_argument('--performance', action='store_true')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    suite = Suite(args)
    try:
        if args.model:
            # 先在CPU保存请求，再分开加载两个后端，避免彼此争用显存。
            # 真实参考使用显式按层分布的FP16模型，仅在FastLLM会话关闭后运行。
            cached = {}
            def record(ids):
                key = tuple(ids)
                if key not in cached:
                    raise RuntimeError('reference cache must be prepared before comparison')
                return cached[key]
            # 先采集正式Eager的真实Token历史，再由Transformers独立计算完整logits。
            from transformers import AutoTokenizer
            from fastllm_pytools.qwen3_process_chat import prepare_request
            tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
            prompts = [prepare_request(args.model, tokenizer, None, messages=row['messages'],
                        max_new_tokens=row['max_new_tokens'], ignore_eos=True, seed=20261114)
                       for row in (json.loads(line) for line in args.requests.read_text().splitlines() if line.strip())]
            require(prompts, 'empty JSONL request set')
            (suite.output / 'requests.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
            with suite.session(args.model, 4, trace=True) as session:
                traces = batch(session, prompts)
            histories = []
            for report in traces:
                ids = list(report['request']['token_ids'])
                for step in report['steps']:
                    histories.append(tuple(ids))
                    ids.append(step['token_id'])
            inputs, reference_path = suite.output / 'histories.json', suite.output / 'transformers-reference.json'
            inputs.write_text(json.dumps(list(dict.fromkeys(histories))))
            command = [args.reference_python, str(ROOT / 'test/distributed/qwen_moe_transformers_reference.py'),
                       '--model', str(args.model.resolve()), '--inputs', str(inputs), '--output', str(reference_path)]
            (suite.output / 'reference.command.txt').write_text(shlex.join(command))
            with (suite.output / 'reference.log').open('wb') as log, case_process(command, stdout=log, stderr=log) as process:
                code = wait_case(process, args.timeout)
                require(code == 0, f'Transformers reference exited {code}; inspect {suite.output / "reference.log"}')
            reference = json.loads(reference_path.read_text())
            cached.update({tuple(row['ids']): row['logits'] for row in reference['rows']})
            suite.metadata['reference'] = reference['metadata']
            suite.real_layers(args.model, reference['layers'])
            suite.correctness(args.model, 4, record, real=True, baseline=[r['token_ids'] for r in traces], requests=prompts)
        else:
            for name, options in [('f16_bias', {}), ('bf16_no_bias', dict(dtype='BF16', bias=False)),
                                  ('f32_tied_normalized', dict(dtype='F32', tied=True, normalize=True)),
                                  ('zero_experts', dict(experts=3, topk=1)), ('hotspot', dict(hotspot=True, topk=1))]:
                model = suite.output / name
                config, tensors = fixture(model, **options)
                for world in (1, 2, 4):
                    suite.correctness(model, world, lambda ids: full_logits(config, tensors, ids))
            suite.lifecycle(suite.output / 'f16_bias')
        if args.performance:
            suite.performance(args.model or suite.output / 'f16_bias')
    except BaseException as error:
        (suite.output / 'setup.traceback.log').write_text(traceback.format_exc())
        suite.rows.append(dict(case='suite_setup', backend=BACKEND, status='FAIL', reason=str(error), failed_elements=1))
        print(f'FAIL: suite_setup: {error}', file=sys.stderr, flush=True)
        if isinstance(error, (KeyboardInterrupt, SystemExit)):
            raise
    finally:
        status = save_report(suite.output, suite.rows, suite.metadata)
        print(f'{status}: {suite.output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
