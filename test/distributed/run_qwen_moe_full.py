#!/usr/bin/env python3
"""阶段3～6：正式会话的完整模型、联合调度、Graph/Prefix、真实FP16参考和性能验收。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import zip_longest
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
from fastllm_pytools.qwen3_process_chat import MAX_CONTEXT_TOKENS
from generation_alignment import POLICY_SOURCE, compare_generation, compare_generation_all, validate_policy


SYNTHETIC_FIXTURES = {
    'f16_bias': {}, 'bf16_no_bias': dict(dtype='BF16', bias=False),
    'f32_tied_normalized': dict(dtype='F32', tied=True, normalize=True),
    'zero_experts': dict(experts=3, topk=1), 'hotspot': dict(hotspot=True, topk=1),
    'qkv_bias_rounding': dict(qkv_bias_rounding=True),
    'mixed_qkv_bias': dict(mixed_qkv_bias=True),
    'expert_accumulation': dict(experts=4, topk=4, expert_accumulation=True),
}
SYNTHETIC_CASES = {
    f'{name}_ep{world}_graph{int(graph)}_prefix{int(prefix)}': (name, world, graph, prefix)
    for name in SYNTHETIC_FIXTURES for world in (1, 2, 4)
    for graph in (False, True) for prefix in (False, True)
}
PERFORMANCE_CASES = tuple((graph, cache, concurrency) for graph in (False, True)
                          for cache in ('off', 'cold', 'hot') for concurrency in (1, 4, 8))


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


def performance_token_differences(actual, expected):
    """/**
     * 按提交顺序比较测速输出，记录每个请求的首次分歧，不重排请求或放宽Token判定。
     * 分歧后的不一致位置也计入数量，但不解释为同历史数值误差；本函数不运行推理。
     * @param actual 待检查模式的多轮请求报告，形状为[轮次][提交序号]。
     * @param expected 同工作量的参考模式报告，或用于观察模式内波动的首请求副本。
     * @return 严格匹配状态、不同请求/Token位置数量，以及Token、rank和执行路径证据。
     */"""
    differences = []
    same_shape = len(actual) == len(expected)
    for repeat, (got_batch, ref_batch) in enumerate(zip_longest(actual, expected, fillvalue=[]), 1):
        same_shape = same_shape and len(got_batch) == len(ref_batch)
        for index, (got, ref) in enumerate(zip_longest(got_batch, ref_batch, fillvalue={}), 1):
            got_ids, ref_ids = got.get('token_ids'), ref.get('token_ids')
            if got_ids == ref_ids:
                continue
            mismatch = [i for i, (a, b) in enumerate(zip_longest(got_ids or [], ref_ids or [])) if a != b]
            first = mismatch[0] if mismatch else None
            item = dict(repeat_1based=repeat, request_1based=index,
                        first_token_1based=None if first is None else first + 1,
                        token_mismatches=len(mismatch), actual_length=None if got_ids is None else len(got_ids),
                        expected_length=None if ref_ids is None else len(ref_ids))
            for side, report in (('actual', got), ('expected', ref)):
                ids, steps = report.get('token_ids', []), report.get('steps', [])
                item[side + '_token'] = ids[first] if first is not None and first < len(ids) else None
                item[side + '_rank'] = report.get('dp_rank')
                item[side + '_execution'] = steps[first].get('execution') if first is not None and first < len(steps) else None
            differences.append(item)
    return dict(match=same_shape and not differences, same_shape=same_shape,
                mismatched_requests=len(differences), token_mismatches=sum(d['token_mismatches'] for d in differences),
                first_divergence=differences[0] if differences else None, differences=differences)


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

    def __init__(self, args, *, output=None):
        self.args = args
        self.output = output if output is not None else args.output.resolve() / (
            datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_moe_full_' + uuid.uuid4().hex[:8])
        self.output.mkdir(parents=True)
        self.rows, self.counter = [], 0
        self.metadata = dict(command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
            gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
            build_cache=args.build_cache.read_text(), cuda=capture(['nvcc', '--version']),
            seed=20261114, warmup=0, iterations=1, repeats=1,
            correctness='teacher-forced full logits; synthetic atol=rtol=.008, real atol=.05 rtol=.01',
            performance='end-to-end concurrent batch including admission and KV release; no token logs/trace; Eager/Graph use identical requests and EOS policy',
            report_title='Attention DP + EP full model')

    def case(self, name, action, *, failure_evidence=None):
        """/** action共享误差累计器；可选failure_evidence在异常或中断后保存已采集证据。 */"""
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
            if row['status'] == 'FAIL' and failure_evidence is not None:
                failure_evidence['error'] = row['reason'] or 'case interrupted before completion'
                (self.output / (name + '.json')).write_text(json.dumps(failure_evidence, indent=2, allow_nan=False))
            if metrics['count']:
                row.update(max_abs_error=metrics['max'], mean_abs_error=metrics['sum']/metrics['count'])
            row['failed_elements'] = metrics['failed'] or int(row['status'] == 'FAIL')
            self.rows.append(row)
            save_report(self.output, self.rows, self.metadata)
            print(f'{row["status"]}: {name} {row["reason"]}', flush=True)
            if row.get('alignment_policy') == 'all':
                print(f'ALIGNMENT: {name} exact={row["exact_status"]} '
                      f'top5={row["top5_status"]} top5+0.1={row["top5_near_tie_status"]}', flush=True)
            if row.get('logit_diagnostic_status') == 'WARNING':
                print(f'WARNING: {name} logits数值超差（不影响生成验收判定）; '
                      f'max_abs={row["max_abs_error"]:.3f} mean_abs={row["mean_abs_error"]:.3f} '
                      f'exceeded_elements={row["logit_failed_elements"]}', flush=True)
        return row['status'] == 'PASS'

    def compare_generation_report(self, report, expected, tokenizer, reference_path, index, scope):
        """/**
         * 从同一份自由生成输出计算选定规则或全部三种生成规则，并保存一次原始证据。
         * @param report FT正式会话输出，不触发额外生成或教师强制。
         * @param expected 对应请求的独立HF生成输出。
         * @param tokenizer 与参考相同的文本解码器。
         * @param reference_path HF原始证据路径。
         * @param index HF请求索引。
         * @param scope 共同Eager配置或FT生产扩展配置。
         * @return 独立策略结论、共同数值诊断及原始输出；规则失败由case保留为FAIL。
         */"""
        report['text'] = tokenizer.decode(report['token_ids'], skip_special_tokens=False,
                                          clean_up_tokenization_spaces=False)
        if self.args.alignment_policy == 'all':
            result = compare_generation_all(report, expected)
        else:
            result = compare_generation(report, expected, policy=self.args.alignment_policy,
                topk=self.args.alignment_topk, near_tie_tol=self.args.near_tie_tol)
        summary = {key: value for key, value in result.items()
                   if key not in ('candidate_checks', 'logit_diagnostics', 'comparisons')}
        summary['configuration_scope'] = scope
        return dict(summary=summary, comparison=result, actual=report,
                    reference_file=str(reference_path), reference_request=index)

    def session(self, model, world, graph=False, prefix=False, trace=True, routing='round_robin'):
        """/** 新建独立权重生命周期，不复用旧结果目录；不自动编译或下载。 */"""
        self.counter += 1
        return QwenMoeProcessSession(self.args.binary, self.args.worker, model,
            ','.join(map(str, range(world))), self.output / f'session-{self.counter:04d}',
            cuda_graph=graph, prefix_cache=prefix, routing=routing, timeout=self.args.timeout,
            trace=trace, request_log=trace, queue_size=16)

    def generation_alignment(self, model, requests, tokenizer):
        """/**
         * 使用双方公开生成入口进行独立贪心验收，原逐层数值检查由numeric模式保留。
         * HF先单独运行并释放显存；FT共同配置为Eager/无Prefix/串行请求，再独立报告
         * Graph与Prefix生产配置。每请求保存选定策略结论、严格一致性及原始误差诊断；
         * 预登记全部生成用例，未完成的汇总不得显示通过；性能由独立入口执行。
         * @param model 同一份本地FP16权重目录。
         * @param requests 已统一聊天模板、Prompt Token ID、生成长度和EOS策略的请求。
         * @param tokenizer 用于FT输出文本解码的同一模型tokenizer。
         * @return 无；任一请求或执行路径失败均保留FAIL行，并由main返回非零。
         */"""
        require(requests, 'empty generation request set')
        # 四种模式各两轮请求、四次执行检查、两次Prefix路由。
        expected_cases = len(self.rows) + 8 * len(requests) + 6
        self.metadata.update(
            expected_case_count=expected_cases,
            correctness='independent greedy generation; selected policy decides PASS/FAIL',
            alignment_policy=self.args.alignment_policy, alignment_topk=self.args.alignment_topk,
            near_tie_tol=self.args.near_tie_tol, policy_source=POLICY_SOURCE,
            model_path=str(model.resolve()), dtype='float16', kv_dtype='float16',
            eos_policy='disabled; fixed max_new_tokens; no penalties or filtering',
            shared_configuration='FT and HF eager, prefix off, one request at a time, FP16 KV enabled',
            additional_configurations='FT Graph/Prefix and concurrent requests vs HF eager serial reference; correctness only',
            report_description='生成验收：failed_elements只统计选定策略失败；logit_failed_elements及logit_diagnostic_status保留同历史全词表误差。'
                               '数值超差只记录并提示WARNING，不影响生成验收判定。'
                               '首次分歧后不比较logits，uncompared_steps明确标记未比较尾部。严格数值验收请另跑numeric。',
            report_title='FT vs HF generation alignment')
        save_report(self.output, self.rows, self.metadata)
        reference_path = self.output / 'transformers-generation.json'
        command = [self.args.reference_python, str(ROOT / 'test/distributed/qwen_moe_transformers_reference.py'),
                   '--model', str(model.resolve()), '--generation-requests', str(self.output / 'requests.json'),
                   '--output', str(reference_path)]
        (self.output / 'reference.command.txt').write_text(shlex.join(command))
        with (self.output / 'reference.log').open('wb') as log, case_process(command, stdout=log, stderr=log) as process:
            code = wait_case(process, self.args.timeout)
            require(code == 0, f'Transformers reference exited {code}; inspect {self.output / "reference.log"}')
        reference = json.loads(reference_path.read_text())
        expected = reference['reports']
        require(len(expected) == len(requests), 'missing HF generation reports')
        self.metadata['reference'] = reference['metadata']

        for graph in (False, True):
            for prefix in (False, True):
                mode = f'real_ep4_graph{int(graph)}_prefix{int(prefix)}'

                def check_reports(reports, phase):
                    """/** 每请求独立判定，失败不阻止其余请求保存报告或覆盖原结论。 */"""
                    require(len(reports) == len(requests), 'missing FT generation reports')
                    for index, report in enumerate(reports):
                        def compare(_):
                            """/** 输出生成门禁与独立数值诊断，数值超差只警告，不覆盖候选级结论。 */"""
                            require(report['request'] == requests[index], 'FT prompt or sampling configuration changed')
                            scope = 'shared_eager' if not graph and not prefix else 'ft_production_vs_hf_eager'
                            return self.compare_generation_report(report, expected[index], tokenizer,
                                                                  reference_path, index, scope)
                        self.case(f'{mode}_{phase}_request{index:03d}', compare)

                def run(_):
                    """/** 运行同一公开会话的首次、重复及前缀命中请求，并核对执行路径。 */"""
                    with self.session(model, 4, graph, prefix) as session:
                        self.metadata['workers'] = session.ready['workers']
                        for phase in ('cold', 'warm'):
                            reports = ([session.generate(r) for r in requests] if not graph and not prefix
                                       else batch(session, requests))
                            check_reports(reports, phase)
                        if graph:
                            require(all(state['captures'] == 1 and state['replays'] > 0 for state in session._states),
                                    'Graph capture/replay evidence missing')
                        if prefix:
                            session.routing = 'prefix'
                            hot = session.generate(requests[0])
                            require(hot['states'][hot['dp_rank']]['prefix_cache']['last_reused_tokens'] > 0,
                                    'prefix route request did not reuse cached tokens')
                            # 再次生成也不使用教师强制；候选验收仍使用相同HF独立参考。
                            def hot_check(_):
                                """/** 对缓存命中的自由生成使用同一HF参考，保留首个分歧证据。 */"""
                                require(hot['request'] == requests[0], 'FT prefix prompt or sampling configuration changed')
                                return self.compare_generation_report(hot, expected[0], tokenizer,
                                    reference_path, 0, 'ft_production_vs_hf_eager')
                            self.case(mode + '_prefix_route', hot_check)
                            session.clear_prefix_cache()
                    return dict(shutdown=session.shutdown)
                self.case(mode + '_execution', run)

    def correctness(self, model, world, reference, real=False, baseline=None, requests=None,
                    modes=None, case_suffix=''):
        """/** 覆盖变长、空闲、跨请求图复用、冷热前缀及rank独立KV，逐步比对参考。 */"""
        vocab = json.loads((model / 'config.json').read_text())['vocab_size']
        requests = requests or [request([(i*7+j+1) % vocab for j in range(2+i%4)], 3+i%5) for i in range(8)]
        for graph in (False, True):
            for prefix in (False, True):
                if modes is not None and (graph, prefix) not in modes:
                    continue
                name = f'{model.name}_ep{world}_graph{int(graph)}_prefix{int(prefix)}{case_suffix}'
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

    def synthetic(self):
        """/**
         * 运行完整小模型矩阵，或重复一个明确选择的用例以诊断间歇故障。
         * 定向模式仍走相同会话、完整logits比较及Graph/Prefix检查；每轮独立
         * 进程和输出，不重试掩盖失败，也不宣称覆盖跨模式及完整生命周期验收。
         */"""
        if self.args.synthetic_case:
            name, world, graph, prefix = SYNTHETIC_CASES[self.args.synthetic_case]
            self.metadata.update(scope='focused_synthetic', selected_case=self.args.synthetic_case,
                                 repeats=self.args.repeat, expected_case_count=self.args.repeat,
                                 report_title='Attention DP + EP focused synthetic reproduction')
            model = self.output / name
            config, tensors = fixture(model, **SYNTHETIC_FIXTURES[name])
            for index in range(self.args.repeat):
                self.correctness(model, world, lambda ids: full_logits(config, tensors, ids),
                                 modes=[(graph, prefix)], case_suffix=f'_repeat{index+1:03d}')
        else:
            for name, options in SYNTHETIC_FIXTURES.items():
                model = self.output / name
                config, tensors = fixture(model, **options)
                for world in (1, 2, 4):
                    self.correctness(model, world, lambda ids: full_logits(config, tensors, ids))
            self.lifecycle(self.output / 'f16_bias')

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
        """/**
         * 独立比较正式Eager/Graph性能，不运行HF生成或数值对齐。
         * 每种输入长度内的两种模式统一工作量；每轮同步结果后停止计时，输出历史、缓存复用和
         * Graph路径检查及落盘在计时区外。预登记全部用例，保留失败及中断报告。
         * @param model 同一份本地FP16真实模型或合成小模型目录。
         * @return 无；任一性能用例失败保留FAIL行，并由main返回非零。
         */"""
        lengths, output_tokens = self.args.performance_prompt_lengths, self.args.performance_output_tokens
        modes = [mode for mode in PERFORMANCE_CASES if mode[1] in self.args.performance_cache_states
                 and mode[2] in self.args.performance_concurrencies]
        prompts = {length: ([1, 2, 3, 4] * ((length + 3) // 4))[:length] for length in lengths}
        self.metadata.update(
            scope='performance', expected_case_count=len(self.rows) + len(lengths) * len(modes),
            selected_cache_states=self.args.performance_cache_states,
            selected_concurrencies=self.args.performance_concurrencies,
            performance_matrix_complete=len(modes) == len(PERFORMANCE_CASES),
            model_path=str(model.resolve()), dtype='float16', kv_dtype='float16',
            warmup=1, iterations=1, repeats=5, hot_cache_preparation_requests=4,
            prompt_lengths=lengths, prompt_token_ids_by_length=prompts, max_new_tokens=output_tokens,
            eos_policy='disabled; fixed max_new_tokens; no penalties or filtering',
            correctness='FT Eager/Graph output histories, prefix reuse and Graph execution evidence',
            report_description='独立性能测试：不运行HF对齐；保留Eager/Graph输出历史、缓存及执行路径检查。'
                               '只覆盖本次所选长度、缓存和并发组合。'
                               '加速比为同配置Eager耗时/当前模式耗时，不代表相对HF或vLLM的速度。',
            report_title='Attention DP + EP Eager/Graph performance')
        save_report(self.output, self.rows, self.metadata)

        baseline, reference_responses, cache_histories = {}, {}, {}
        cases = ((length, *mode) for length in lengths for mode in modes)
        for prompt_length, graph, cache, concurrency in cases:
            prefix = 'perf' if lengths == [32] and output_tokens == 16 else f'perf_prompt{prompt_length}_output{output_tokens}'
            name = f'{prefix}_graph{int(graph)}_cache_{cache}_concurrency{concurrency}'
            record = dict(samples_ms=[], responses=[], warmup=1, iterations=1, repeats=5,
                          prompt_length=prompt_length, output_tokens=output_tokens, concurrency=concurrency,
                          cuda_graph=graph, cache_state=cache)
            if graph:
                record['eager_reference_file'] = str(self.output / f'{prefix}_graph0_cache_{cache}_concurrency{concurrency}.json')

            def run(_):
                key = (prompt_length, cache, concurrency)
                jobs = [request(prompts[prompt_length], output_tokens) for _ in range(concurrency)]
                raw, responses = record['samples_ms'], record['responses']
                record['requests'] = jobs
                with self.session(model, 4, graph, cache != 'off', trace=False, routing='prefix' if cache != 'off' else 'round_robin') as session:
                    record['session_output'] = str(session.output)
                    batch(session, jobs)  # 一次相同工作量warmup。
                    if cache == 'hot':
                        batch(session, [jobs[0]] * 4)
                    memory = capture(['nvidia-smi', '--query-gpu=uuid,memory.used', '--format=csv'])
                    record['memory_used_after_warmup'] = memory
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
                    record['states'] = session._states
                    if graph:
                        require(all(s['replays'] > 0 and s['captures'] == 1 for s in session._states), 'benchmark missed Graph')
                    evidence = session._states
                record['shutdown'] = session.shutdown
                all_tokens = [[r['token_ids'] for r in run] for run in responses]
                record['output_variant_count'] = len({tuple(ids) for run in all_tokens for ids in run})
                record['within_mode_vs_first_request'] = performance_token_differences(
                    responses, [[responses[0][0]] * len(run) for run in responses])
                if graph and key in reference_responses:
                    record['comparison'] = performance_token_differences(responses, reference_responses[key])
                require(all(len(ids) == output_tokens for run in all_tokens for ids in run),
                        'benchmark output length differs from requested workload')
                reuse = [sorted(r['states'][r['dp_rank']]['prefix_cache']['last_reused_tokens'] for r in run) for run in responses]
                record['reused_tokens'] = reuse
                if cache == 'hot':
                    require(all(value == len(jobs[0]['token_ids'])-1 for run in reuse for value in run), 'hot benchmark did not hit warm cache')
                if cache == 'off':
                    require(not any(value for run in reuse for value in run), 'disabled cache reused tokens')
                if graph:
                    require(key in baseline and key in reference_responses, 'benchmark Eager baseline failed or is missing')
                    comparison = record['comparison']
                    require(comparison['match'], 'benchmark modes used different output histories; '
                            f'first_divergence={json.dumps(comparison["first_divergence"])}; evidence={self.output / (name + ".json")}')
                    require(reuse == cache_histories[key], 'benchmark modes had different prefix reuse')
                else:
                    reference_responses[key] = responses
                    cache_histories[key] = reuse
                ttft = [r['steps'][0]['elapsed_ms'] for run in responses for r in run]
                itl = [b['elapsed_ms']-a['elapsed_ms'] for run in responses for r in run for a, b in zip(r['steps'], r['steps'][1:])]
                queued = [r['queue_ms'] for run in responses for r in run]
                median = statistics.median(raw)
                if not graph:
                    baseline[key] = median
                record.update(
                    ttft_samples_ms=ttft, itl_samples_ms=itl, queue_samples_ms=queued,
                    summary=dict(prompt_length=prompt_length, output_tokens=output_tokens,
                                 median_ms=median, p95_ms=sorted(raw)[math.ceil(.95*len(raw))-1],
                                 cv=statistics.pstdev(raw)/statistics.mean(raw), tokens_per_second=concurrency*output_tokens*1000/median,
                                 speedup=baseline[key]/median, concurrency=concurrency, repeats=5,
                                 ttft_median_ms=statistics.median(ttft), itl_median_ms=statistics.median(itl),
                                 queue_median_ms=statistics.median(queued),
                                 reserved_kv_bytes=sum(s['reserved_kv_bytes'] for s in evidence),
                                 prefix_bytes=sum(s['prefix_cache']['device_bytes'] for s in evidence),
                                 graph_workspace_bytes=sum(s['workspace_bytes'] for s in evidence)))
                return record
            self.case(name, run, failure_evidence=record)


def summarize_alignment_stage(stage, phase):
    """/**
     * 将阶段明细归并成独立策略结论，运行失败及缺失用例对所有受影响策略生效。
     * 宽松候选通过不能覆盖exact或严格Top-5失败；均值保留在逐case明细，避免
     * 混合不同张量大小的不加权均值。根报告failed_elements表示失败case数。
     * @param stage 已保存四格式报告的阶段Suite。
     * @param phase generation或numeric。
     * @return 三条生成策略汇总或一条numeric汇总；未完成的阶段均标记FAIL。
     */"""
    report = json.loads((stage.output / 'summary.json').read_text())
    policies = ('exact', 'top5', 'top5_near_tie') if phase == 'generation' else ('numeric',)
    rows = []
    for policy in policies:
        failures = []
        for case in report['cases']:
            if case.get(policy + '_status', case['status']) != 'PASS':
                reason = case.get(policy + '_reason', case.get('reason', ''))
                failures.append(f'{case["case"]}: {reason}')
        if not report['complete']:
            failures.append(f'incomplete stage: {report["completed_case_count"]}/{report["expected_case_count"]} cases')
        errors = [case['max_abs_error'] for case in report['cases'] if case.get('max_abs_error') is not None]
        row = dict(case=policy, alignment_policy=policy, backend=BACKEND,
                   status='FAIL' if failures else 'PASS', complete=report['complete'],
                   completed_case_count=report['completed_case_count'], expected_case_count=report['expected_case_count'],
                   max_abs_error=max(errors) if errors else None, mean_abs_error=None,
                   failed_elements=len(failures), reason='; '.join(failures),
                   detail_report=str(stage.output / 'summary.json'))
        if phase == 'generation':
            compared = [case for case in report['cases'] if 'exact_status' in case]
            row.update(topk=5, near_tie_tol=0.0 if policy == 'top5' else 0.1,
                       token_match=all(case['token_match'] for case in compared) if compared else None,
                       text_match=all(case['text_match'] for case in compared) if compared else None)
        rows.append(row)
    return rows


def run_all_alignment(suite, args, prompts, tokenizer):
    """/**
     * 顺序执行一套自由生成和独立numeric验收，并逐阶段保存四项汇总。
     * 普通失败保留后继续下一阶段；用户中断会保存部分报告并立即停止。
     * 两阶段使用独立输出目录和显存生命周期，不复用教师强制数据作生成证据。
     * @param suite 根报告Suite，统一保存命令、硬件及Git元信息。
     * @param args 已校验的all模式参数；固定K=5、宽松容差0.1。
     * @param prompts 双方共用的Prompt ID和生成配置。
     * @param tokenizer 生成文本解码器。
     * @return 无；四项状态写入suite.rows，最终任一失败返回非零。
     */"""
    suite.metadata.update(alignment_policy='all', expected_case_count=4, execution_complete=False,
        correctness='exact + strict Top-5 + Top-5/0.1 on shared independent generation; separate strict numeric',
        model_path=str(args.model.resolve()), policy_source=POLICY_SOURCE, stages={},
        report_title='FT vs HF: four alignment checks',
        report_description='前三项复用同一批独立生成；numeric单独运行。每项结论均保留，任一失败则总体FAIL。'
                           '根报告failed_elements统计失败case及未完成标记；最大误差取阶段case最大值，'
                           '均值和元素级失败数见各阶段明细。生成数值诊断只警告，numeric仍严格失败。')
    save_report(suite.output, suite.rows, suite.metadata)

    for phase in ('generation', 'numeric'):
        stage_args = argparse.Namespace(**vars(args))
        stage_args.alignment_policy = 'all' if phase == 'generation' else 'numeric'
        stage = Suite(stage_args, output=suite.output / phase)
        stage.metadata.update(expected_case_count=8 * len(prompts) + 6 if phase == 'generation' else 5,
                              execution_complete=False, alignment_stage=phase,
                              parent_report=str(suite.output / 'summary.json'))
        (stage.output / 'requests.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
        suite.metadata['stages'][phase] = str(stage.output / 'summary.json')
        save_report(stage.output, stage.rows, stage.metadata)
        try:
            if phase == 'generation':
                stage.generation_alignment(args.model, prompts, tokenizer)
            else:
                run_numeric_reference(stage, stage_args, prompts)
            stage.metadata['execution_complete'] = True
        except BaseException as error:
            (stage.output / 'setup.traceback.log').write_text(traceback.format_exc())
            stage.rows.append(dict(case=phase + '_setup', backend=BACKEND, status='FAIL',
                                   failed_elements=1, reason=f'{type(error).__name__}: {error}'))
            print(f'FAIL: {phase}: {error}', file=sys.stderr, flush=True)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            save_report(stage.output, stage.rows, stage.metadata)
            rows = summarize_alignment_stage(stage, phase)
            suite.rows.extend(rows)
            suite.metadata['execution_complete'] = len(suite.rows) == 4 and all(row['complete'] for row in suite.rows)
            save_report(suite.output, suite.rows, suite.metadata)
            for row in rows:
                print(f'{row["status"]}: alignment/{row["case"]}; details: {row["detail_report"]}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', type=Path, required=True)
    parser.add_argument('--worker', type=Path, required=True)
    parser.add_argument('--build-cache', type=Path, required=True)
    parser.add_argument('--model', type=Path, help='真实Qwen1.5-MoE目录；省略则运行小模型及生命周期')
    parser.add_argument('--requests', type=Path, default=ROOT / 'test/distributed/data/qwen_moe_requests.jsonl')
    parser.add_argument('--reference-python', default=sys.executable, help='独立Transformers参考环境的Python路径')
    parser.add_argument('--alignment-policy', choices=('topk', 'exact', 'numeric', 'all'), default='topk',
                        help='默认Top-K；exact严格输出；numeric严格数值；all共用生成检查三种规则后独立运行numeric')
    parser.add_argument('--alignment-topk', type=int, default=5, help='生成验收候选数，不改变实际采样')
    parser.add_argument('--near-tie-tol', type=float, default=0.1, help='候选边界logprob容差；0为严格双向Top-K包含')
    parser.add_argument('--performance', action='store_true',
                        help='仅运行Eager/Graph性能矩阵，不附带对齐或合成验收；省略--model只准备性能小模型')
    parser.add_argument('--performance-prompt-lengths', type=int, nargs='+',
                        help='性能输入Token长度列表，每个长度默认运行18项；默认32')
    parser.add_argument('--performance-output-tokens', type=int,
                        help='性能每请求固定输出Token数，至少2；默认16')
    parser.add_argument('--performance-cache-states', nargs='+', choices=('off', 'cold', 'hot'),
                        help='定向性能复测的缓存状态；默认全部，始终同时测试Eager/Graph')
    parser.add_argument('--performance-concurrencies', type=int, nargs='+', choices=(1, 4, 8),
                        help='定向性能复测的并发数；默认1/4/8，始终同时测试Eager/Graph')
    parser.add_argument('--synthetic-case', choices=sorted(SYNTHETIC_CASES),
                        help='只复现指定小模型用例；不代表完整验收通过')
    parser.add_argument('--repeat', type=int, default=1, help='定向用例独立执行次数；任何一轮失败则总体失败')
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    performance_options = (args.performance_prompt_lengths, args.performance_output_tokens,
                           args.performance_cache_states, args.performance_concurrencies)
    if not args.performance and any(option is not None for option in performance_options):
        parser.error('性能参数必须与--performance一起使用')
    args.performance_prompt_lengths = args.performance_prompt_lengths or [32]
    args.performance_output_tokens = 16 if args.performance_output_tokens is None else args.performance_output_tokens
    args.performance_cache_states = args.performance_cache_states or ['off', 'cold', 'hot']
    args.performance_concurrencies = args.performance_concurrencies or [1, 4, 8]
    if args.performance:
        if any(len(set(values)) != len(values) for values in (args.performance_cache_states, args.performance_concurrencies)):
            parser.error('性能缓存状态和并发数不能重复')
        if args.performance_output_tokens < 2:
            parser.error('性能输出至少2个Token，以检查Graph Decode路径并计算ITL')
        if len(set(args.performance_prompt_lengths)) != len(args.performance_prompt_lengths):
            parser.error('性能输入长度不能重复')
        if any(length < 1 or length + args.performance_output_tokens - 1 > MAX_CONTEXT_TOKENS
               for length in args.performance_prompt_lengths):
            parser.error(f'性能输入长度必须为正，输入长度+输出长度-1不得超过{MAX_CONTEXT_TOKENS}')

    if args.repeat < 1 or (args.repeat != 1 and not args.synthetic_case):
        parser.error('--repeat必须为正整数，重复执行必须指定--synthetic-case')
    if args.synthetic_case and (args.model or args.performance):
        parser.error('--synthetic-case不能与--model或--performance混用')
    if args.performance and (args.alignment_policy != 'topk' or args.alignment_topk != 5 or args.near_tie_tol != 0.1):
        parser.error('--performance独立测速，不能同时指定其他对齐策略或候选阈值')
    if args.alignment_policy == 'all':
        if not args.model or args.performance:
            parser.error('all仅用于真实模型功能验收，需要--model，不能与--performance混用')
        if args.alignment_topk != 5 or args.near_tie_tol != 0.1:
            parser.error('all固定比较exact、Top-5/0、Top-5/0.1及numeric，不能覆盖K或宽松容差')
    try:
        validate_policy('topk' if args.alignment_policy in ('numeric', 'all') else args.alignment_policy,
                        args.alignment_topk, args.near_tie_tol)
    except ValueError as error:
        parser.error(str(error))
    suite = Suite(args)
    try:
        if args.performance:
            model = args.model or suite.output / 'f16_bias'
            if not args.model:
                fixture(model, **SYNTHETIC_FIXTURES['f16_bias'])
            suite.performance(model)
        elif args.model:
            # 在CPU统一Prompt ID；两种验收模式均分开加载后端，避免显存争用。
            from transformers import AutoTokenizer
            from fastllm_pytools.qwen3_process_chat import prepare_request
            tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=False)
            prompts = [prepare_request(args.model, tokenizer, None, messages=row['messages'],
                        max_new_tokens=row['max_new_tokens'], ignore_eos=True, seed=20261114)
                       for row in (json.loads(line) for line in args.requests.read_text().splitlines() if line.strip())]
            require(prompts, 'empty JSONL request set')
            (suite.output / 'requests.json').write_text(json.dumps(prompts, ensure_ascii=False, indent=2))
            if args.alignment_policy == 'all':
                run_all_alignment(suite, args, prompts, tokenizer)
            elif args.alignment_policy != 'numeric':
                suite.generation_alignment(args.model, prompts, tokenizer)
            else:
                run_numeric_reference(suite, args, prompts)
        else:
            suite.synthetic()
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


def run_numeric_reference(suite, args, prompts):
    """/** 保留原教师强制全logits与逐层严格门禁，使用独立运行目录和非零失败退出码。 */"""
    cached = {}

    def record(ids):
        """/** 只返回同一真实历史的独立HF结果，缺失历史必须失败。 */"""
        key = tuple(ids)
        if key not in cached:
            raise RuntimeError('reference cache must be prepared before comparison')
        return cached[key]

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


if __name__ == '__main__':
    raise SystemExit(main())
