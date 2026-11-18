#!/usr/bin/env python3
"""通过正式会话诊断并发生成波动；独立于性能/HF验收，不编译、不改变采样或容差。"""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import heapq
import json
import math
from pathlib import Path
import shlex
import sys
import traceback
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_full import batch, request, performance_token_differences
from fastllm_pytools.qwen_moe_process_session import QwenMoeProcessSession
from fastllm_pytools.qwen3_process_chat import MAX_CONTEXT_TOKENS


def file_hash(path):
    """/** 分块记录已有产物的SHA256，不加载模型或启动原生程序。 */"""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def candidates(step, selected, other):
    """/** 返回真实原始logits的前五名与双方选词分数；分差只作证据，不自动豁免失败。 */"""
    logits = step['logits']
    top = heapq.nlargest(min(5, len(logits)), range(len(logits)), key=lambda i: (logits[i], -i))
    return dict(selected_token=selected, selected_logit=logits[selected], other_token=other,
                other_logit=logits[other], selected_minus_other=logits[selected] - logits[other],
                top5=[dict(token_id=i, logit=logits[i]) for i in top],
                top1_top2_gap=logits[top[0]] - logits[top[1]] if len(top) > 1 else None,
                execution=step['execution'], joint_round=step['joint_round'],
                trace_schedule=step['trace_schedule'])


def compare_requests(actual, expected, positions):
    """/**
     * 对自由生成输出作严格Token比较，并仅在双方输入历史相同时比较所选位置的完整logits。
     * 分叉后的logits明确跳过；失败元素指不同Token位置，不把后续分叉解释为独立数值错误。
     * @param actual 当前正式请求报告；包含完整Token序列和选择性trace。
     * @param expected 本模式首请求或同轮次同提交序号的Eager报告。
     * @param positions 待观察的1-based生成位置；缺少证据或非有限数值直接失败。
     * @return 严格Token结果、同历史max/mean绝对误差、候选分数及首次分歧证据。
     */"""
    if actual['request'] != expected['request']:
        raise ValueError('diagnostic compared different input/sampling requests')
    delta = performance_token_differences([[actual]], [[expected]])
    traces, largest, total, count = [], 0.0, 0.0, 0
    for position in positions:
        index = position - 1
        a, b = actual['steps'][index], expected['steps'][index]
        left, right = a['logits'], b['logits']
        if not left or len(left) != len(right) or any(not math.isfinite(v) for v in [*left, *right]):
            raise ValueError(f'token {position}: invalid full-vocabulary logits')
        same_history = actual['token_ids'][:index] == expected['token_ids'][:index]
        evidence = dict(token_1based=position, same_input_history=same_history,
            actual=candidates(a, actual['token_ids'][index], expected['token_ids'][index]),
            expected=candidates(b, expected['token_ids'][index], actual['token_ids'][index]))
        if same_history:
            errors = [abs(x-y) for x, y in zip(left, right)]
            evidence.update(max_abs_error=max(errors), mean_abs_error=sum(errors)/len(errors),
                            nonzero_logit_elements=sum(x != y for x, y in zip(left, right)))
            largest, total, count = max(largest, max(errors)), total + sum(errors), count + len(errors)
        else:
            evidence['numeric_comparison'] = 'SKIP: different generated input histories'
        traces.append(evidence)
    first = delta['first_divergence']
    return dict(token_comparison=delta, traces=traces,
                first_divergence_trace=next((t for t in traces if first and t['token_1based'] == first['first_token_1based']), None),
                max_abs_error=largest if count else None, mean_abs_error=total/count if count else None,
                compared_steps=sum(t['same_input_history'] for t in traces))


def validate_report(report, job, graph, positions):
    """/** 验证真实完成长度、关闭缓存和实际Graph路径；不能只凭模式名称认定路径通过。 */"""
    if (report['status'] != 'PASS' or report['request'] != job or report.get('teacher_forced_tokens') is not None or
            len(report['token_ids']) != job['max_new_tokens'] or len(report['steps']) != job['max_new_tokens']):
        raise ValueError('incomplete request or changed input history')
    rank = report['dp_rank']
    if report['states'][rank]['prefix_cache']['last_reused_tokens'] != 0:
        raise ValueError('disabled prefix cache reused tokens')
    if graph and not any(step['execution'] == 'decode_cuda_graph' for step in report['steps']):
        raise ValueError('requested Graph but no actual Graph decode')
    if not graph and any(step['execution'] != 'eager' for step in report['steps']):
        raise ValueError('Eager request used another execution path')
    for index, step in enumerate(report['steps'], 1):
        if step['token_id'] != report['token_ids'][index-1]:
            raise ValueError('step token does not match generated sequence')
        if index in positions and 'logits' not in step:
            raise ValueError(f'missing selected logits: token {index}')
        if 'joint_round' not in step or step['trace_schedule']['token_positions_1based'][rank] != index:
            raise ValueError(f'missing/mismatched actual schedule: token {index}')


def main():
    """/** 单独采集两种模式的重复并发请求，逐项落盘；差异/异常返回非零，中断仍保存部分报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--prompt-length', type=int, default=1024)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--concurrency', type=int, choices=(1, 4, 8), default=4)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--trace-tokens', type=int, nargs='+', default=[78, 79, 80])
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if (args.prompt_length < 1 or args.output_tokens < 2 or args.repeats < 5 or
            args.prompt_length + args.output_tokens - 1 > MAX_CONTEXT_TOKENS or
            len(set(args.trace_tokens)) != len(args.trace_tokens) or
            any(p < 1 or p > args.output_tokens for p in args.trace_tokens) or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('invalid context/trace positions/timeout; repeats must be at least 5')
    args.trace_tokens.sort()
    args.build, args.model = args.build.resolve(), args.model.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_concurrency_probe_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'独立并发诊断：{output}', flush=True)
    rows, eager_files, variants = [], {}, {}
    job = request(([1, 2, 3, 4] * ((args.prompt_length+3)//4))[:args.prompt_length], args.output_tokens)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), scope='concurrent_generation_diagnostic',
        git=capture(['git', 'rev-parse', 'HEAD']), cpu=capture(['lscpu']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), model_path=str(args.model), request=job,
        warmup=1, iterations=1, repeats=args.repeats, concurrency=args.concurrency,
        trace_tokens_1based=args.trace_tokens, prefix_cache=False, routing='round_robin',
        execution_complete=False, expected_case_count=3*args.repeats*args.concurrency,
        report_title='MoE并发输出波动诊断',
        report_description='独立诊断，不运行HF，不输出性能或加速比。FAIL表示Token差异或采集失败；'
            'failed_elements计不同Token位置，logits只记录同输入历史误差，不设置新数值阈值。'
            'trace取回和日志可能影响时序，未复现不证明原问题消失。')

    def record(label, actual, expected, actual_path, expected_path):
        """/** 每个比较保留独立证据与四格式摘要；首次分歧候选仅在实际采集到该位置时打印。 */"""
        detail = compare_requests(actual, expected, args.trace_tokens)
        delta = detail['token_comparison']
        first = delta['first_divergence']
        row = dict(case=label, backend='formal_attention_dp_ep', status='PASS' if delta['match'] else 'FAIL',
            token_mismatches=delta['token_mismatches'], failed_elements=delta['token_mismatches'],
            first_divergence_token_1based=first['first_token_1based'] if first else None,
            max_abs_error=detail['max_abs_error'], mean_abs_error=detail['mean_abs_error'],
            compared_steps=detail['compared_steps'],
            reason='Token一致；logits仅诊断' if delta['match'] else '自由生成Token不同；保留FAIL')
        detail.update(actual_file=str(actual_path), expected_file=str(expected_path))
        (output / (label + '.json')).write_text(json.dumps(detail, indent=2, allow_nan=False))
        rows.append(row)
        save_report(output, rows, metadata)
        print(f'{row["status"]}: {label}; first_token={row["first_divergence_token_1based"]}', flush=True)
        if first:
            trace = detail['first_divergence_trace']
            print('首次分歧证据：' + json.dumps(trace if trace else dict(
                not_captured=first['first_token_1based'], next_trace_position=first['first_token_1based']), ensure_ascii=False), flush=True)

    try:
        metadata['build_cache'] = (args.build / 'CMakeCache.txt').read_text()
        metadata['binary_sha256'] = {name: file_hash(args.build / name)
                                    for name in ('fastllm_moe_generate', 'fastllm_tp_worker')}
        metadata['model_config_sha256'] = file_hash(args.model / 'config.json')
        index = args.model / 'model.safetensors.index.json'
        if index.exists():
            metadata['model_index_sha256'] = file_hash(index)
        save_report(output, rows, metadata)

        for graph in (False, True):
            mode = 'graph' if graph else 'eager'
            first_report, first_path = None, None
            metadata[mode + '_variants'] = {}
            try:
                with QwenMoeProcessSession(args.build / 'fastllm_moe_generate', args.build / 'fastllm_tp_worker',
                        args.model, '0,1,2,3', output / mode, timeout=args.timeout, trace=True,
                        trace_token_positions=args.trace_tokens, cuda_graph=graph, prefix_cache=False,
                        routing='round_robin', queue_size=16, request_log=True) as session:
                    for report in batch(session, [job] * args.concurrency):
                        validate_report(report, job, graph, args.trace_tokens)
                    with ThreadPoolExecutor(args.concurrency) as pool:
                        for repeat in range(1, args.repeats+1):
                            futures = [pool.submit(session.generate, job) for _ in range(args.concurrency)]
                            reports = [future.result() for future in futures]
                            for number, report in enumerate(reports, 1):
                                validate_report(report, job, graph, args.trace_tokens)
                                path = session.output / f'request-{report["id"]:06d}.json'
                                variant = variants.setdefault(tuple(report['token_ids']), 'V' + str(len(variants)+1))
                                counts = metadata[mode + '_variants']
                                counts[variant] = counts.get(variant, 0) + 1
                                print(f'{mode} 第{repeat}轮 请求{number} rank={report["dp_rank"]} {variant}', flush=True)
                                if first_report is None:
                                    first_report, first_path = report, path
                                label = f'{mode}_repeat{repeat:02d}_request{number:02d}'
                                record(label + '_vs_first', report, first_report, path, first_path)
                                if graph:
                                    reference_path = eager_files[(repeat, number)]
                                    record(label + '_vs_eager', report, json.loads(reference_path.read_text()), path, reference_path)
                                else:
                                    eager_files[(repeat, number)] = path
                    if graph and not all(s['captures'] == 1 and s['replays'] > 0 for s in session._states):
                        raise ValueError('Graph capture/replay evidence missing')
                metadata[mode + '_shutdown'] = session.shutdown
            except Exception as error:
                (output / (mode + '.traceback.log')).write_text(traceback.format_exc())
                rows.append(dict(case=mode + '_collection', status='FAIL', reason=f'{type(error).__name__}: {error}',
                                 failed_elements=1))
                save_report(output, rows, metadata)
                print(f'FAIL: {mode}: {error}', flush=True)
        metadata['execution_complete'] = len(rows) == metadata['expected_case_count'] and not any(
            row['case'].endswith('_collection') for row in rows)
    except BaseException as error:
        (output / 'setup.traceback.log').write_text(traceback.format_exc())
        rows.append(dict(case='diagnostic_setup_or_interrupt', status='FAIL', failed_elements=1,
                         reason=f'{type(error).__name__}: {error}'))
        print(f'FAIL: diagnostic setup/interruption: {type(error).__name__}: {error}', flush=True)
    finally:
        metadata['reproduction_status'] = ('OBSERVED_TOKEN_DIVERGENCE' if any(r.get('token_mismatches', 0) for r in rows)
            else 'NOT_OBSERVED' if metadata['execution_complete'] else 'INCOMPLETE')
        status = save_report(output, rows, metadata)
    print(f'{status}: {metadata["reproduction_status"]}; {output}', flush=True)
    return int(status != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
