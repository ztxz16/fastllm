#!/usr/bin/env python3
"""使用已有常驻进程入口定位冷/热前缀输出分歧；仅做单卡数值诊断，不测速、不编译。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import uuid

from run_process_tp import ROOT, capture, save_report
from fastllm_pytools.qwen3_process_chat import validate_request
from fastllm_pytools.qwen3_process_session import BACKEND, EMPTY, Qwen3ProcessSession
from fastllm_pytools.tp_graph import validate_graph, validate_prefix


def validate_trace(report, request, vocab, graph, reused):
    """/** 验证正式生成结果、完整词表分数和实际缓存路径；缺失证据或错误贪心采样均拒绝。 */"""
    ids, steps = report.get('token_ids', []), report.get('steps', [])
    if (report.get('status') != 'PASS' or report.get('backend') != BACKEND or
            report.get('request') != request or report.get('finish_reason') != 'length' or
            report.get('cache_after_generate') != [EMPTY] or report.get('trace_logits') is not True or
            len(ids) != request['max_new_tokens'] or len(steps) != len(ids)):
        raise RuntimeError('output/request/KV/logits trace mismatch')
    if report.get('execution') != ('decode_cuda_graph' if graph else 'eager'):
        raise RuntimeError('execution mode mismatch')
    states = validate_graph(report.get('graph'), 1, graph)
    prefix = validate_prefix(states, 1, True)[0]
    if (prefix['last_reused_tokens'] != reused or
            prefix['last_computed_tokens'] != len(request['token_ids']) - reused):
        raise RuntimeError(f'expected reused_tokens={reused}, observed {prefix}')
    for index, (token, step) in enumerate(zip(ids, steps)):
        logits = step.get('logits', [])
        if (type(token) is not int or step.get('token_id') != token or len(logits) != vocab or
                any(not math.isfinite(value) for value in logits) or
                token != max(range(vocab), key=logits.__getitem__)):
            raise RuntimeError(f'invalid logits or greedy token at generation step {index + 1}')


def top_two(logits):
    """/** 提取前两名及分差；相同分数按较小Token ID优先，与正式贪心采样一致。 */"""
    first = max(range(len(logits)), key=logits.__getitem__)
    second = max((i for i in range(len(logits)) if i != first), key=logits.__getitem__)
    return dict(token_ids=[first, second], logits=[logits[first], logits[second]],
                margin=logits[first] - logits[second])


def compare_traces(name, actual, expected, *, exact_tokens=False):
    """/**
     * 比较输入历史一致的生成步骤，包含首次选词分歧的那一步，之后停止数值对照。
     * 沿用真实模型前缀验收的atol=0.05、rtol=0.01；相同缓存状态的重复还要求Token完全一致。
     * @return 数值误差、首次分歧与候选词分差；PASS不表示分歧后数值精度或独立数学正确性。
     */"""
    if actual['request'] != expected['request'] or len(actual['token_ids']) != len(expected['token_ids']):
        raise ValueError('trace comparison requires identical requests and output lengths')
    a_ids, b_ids = actual['token_ids'], expected['token_ids']
    different = [i for i, (a, b) in enumerate(zip(a_ids, b_ids)) if a != b]
    steps = different[0] + 1 if different else len(a_ids)
    maximum, total, count, failed = 0.0, 0.0, 0, 0
    for index in range(steps):
        a_values, b_values = actual['steps'][index]['logits'], expected['steps'][index]['logits']
        if not a_values or len(a_values) != len(b_values):
            raise ValueError('logits shape mismatch')
        for a, b in zip(a_values, b_values):
            error = abs(a - b) if math.isfinite(a) and math.isfinite(b) else float('inf')
            maximum = max(maximum, error)
            total += error
            count += 1
            failed += int(not math.isfinite(error) or error > .05 + .01 * abs(b))
    reasons = []
    if failed:
        reasons.append(f'{failed} logits exceed atol=0.05, rtol=0.01 on identical histories')
    if exact_tokens and different:
        reasons.append('repeated same-cache-state output token IDs differ')
    row = dict(case=name, backend=BACKEND, status='FAIL' if reasons else 'PASS', reason='; '.join(reasons),
               max_abs_error=maximum, mean_abs_error=total / count, failed_elements=failed,
               token_mismatches=len(different), compared_steps=steps, compared_elements=count,
               first_divergence_token_1based=different[0] + 1 if different else None,
               exact_tokens_required=exact_tokens, compared_after_divergence=False)
    if different:
        index = different[0]
        row['first_divergence'] = dict(actual=top_two(actual['steps'][index]['logits']),
                                       reference=top_two(expected['steps'][index]['logits']))
    return row


def main(argv=None):
    """/** 保持原请求不变，记录Eager/Graph冷热及清理后的响应；失败保留部分四格式报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'generate', 'worker', 'build-cache', 'request-file'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--device', type=int, default=0, help='CUDA可见逻辑编号，诊断只使用一张卡')
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args(argv)
    if args.device < 0:
        parser.error('--device must be nonnegative')

    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_prefix_diag_' + uuid.uuid4().hex[:8])
    raw = output / 'raw'
    raw.mkdir(parents=True, exist_ok=False)
    metadata = dict(command=shlex.join([sys.executable, *sys.argv] if argv is None else [sys.executable, __file__, *argv]),
                    git=capture(['git', 'rev-parse', 'HEAD']), git_status=capture(['git', 'status', '--short']),
                    gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv,noheader']),
                    cuda=capture(['nvcc', '--version']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                    device=args.device, dp=1, tp=1, dtype='FP16', warmup=0, iterations=1, repeats=1,
                    expected_case_count=8, atol=.05, rtol=.01, trace_logits=True,
                    request_file=str(args.request_file.resolve()),
                    report_title='Qwen3 cold/warm prefix diagnosis',
                    report_description='通过正式常驻会话记录完整logits。仅比较输入历史相同的步骤（含首次分歧），'
                                       '分歧后不比较logits。冷热允许记录Token分歧，相同状态重复要求Token一致。'
                                       'PASS仅表示所比较步骤在既有容差内，不是独立数学精度或性能结论。')
    rows, paths = [], {}
    current = 'setup'
    try:
        metadata['build_cache'] = args.build_cache.read_text()
        metadata['compile_architecture'] = [line for line in metadata['build_cache'].splitlines()
                                           if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        request = json.loads(args.request_file.read_text())
        config = json.loads((args.model / 'config.json').read_text())
        vocab = config['vocab_size']
        validate_request(request, vocab)
        if (vocab < 2 or request['temperature'] != 0 or request['eos_token_ids'] or
                len(request['token_ids']) < 2 or request['max_new_tokens'] < 2 or
                vocab * request['max_new_tokens'] > 8388608):
            raise ValueError('requires greedy/no-EOS request, prompt/output>=2, trace<=8M logits; request is never truncated')
        metadata.update(request=request, seed=request['seed'], vocabulary=vocab)
        (output / 'request.json').write_text(json.dumps(request, ensure_ascii=False, indent=2))

        for mode, graph in [('eager', False), ('graph', True)]:
            current = f'{mode}/startup'
            with Qwen3ProcessSession(args.generate, args.worker, args.model, str(args.device), output / mode,
                                     trace=True, cuda_graph=graph, prefix_cache=True, request_log=False) as session:
                for label in ('cold', 'warm', 'warm_repeat', 'after_clear'):
                    current = f'{mode}/{label}'
                    if label == 'after_clear':
                        session.clear_prefix_cache()
                    print(f'RUN: {current}', flush=True)
                    report = session.generate(request)
                    path = raw / f'{mode}_{label}.json'
                    path.write_text(json.dumps(report, ensure_ascii=False))
                    reused = len(request['token_ids']) - 1 if label.startswith('warm') else 0
                    validate_trace(report, request, vocab, graph, reused)
                    paths[mode, label] = path
                    del report

        pairs = [(f'{mode}/{name}', (mode, actual), (mode, expected), exact)
                 for mode in ('eager', 'graph')
                 for name, actual, expected, exact in [('cold_vs_warm', 'warm', 'cold', False),
                                                      ('warm_repeat', 'warm_repeat', 'warm', True),
                                                      ('after_clear', 'after_clear', 'cold', True)]]
        pairs += [(f'graph_vs_eager/{state}', ('graph', state), ('eager', state), False) for state in ('cold', 'warm')]
        for current, actual, expected, exact in pairs:
            row = compare_traces(current, json.loads(paths[actual].read_text()), json.loads(paths[expected].read_text()),
                                 exact_tokens=exact)
            row.update(actual_trace=str(paths[actual]), reference_trace=str(paths[expected]))
            rows.append(row)
            save_report(output, rows, metadata)
            print(json.dumps({key: value for key, value in row.items() if key not in ('actual_trace', 'reference_trace')},
                             ensure_ascii=False), flush=True)
    except (Exception, KeyboardInterrupt) as error:
        rows.append(dict(case=current, backend=BACKEND, status='FAIL', reason=f'{type(error).__name__}: {error}',
                         max_abs_error=None, mean_abs_error=None, failed_elements=None))
    status = save_report(output, rows, metadata)
    print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
