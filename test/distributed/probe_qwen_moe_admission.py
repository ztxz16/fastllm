#!/usr/bin/env python3
"""固定MoE联合准入轮次的归因诊断；复用正式原生协议，不计性能、不运行HF、不编译。"""

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shlex
import sys
import traceback
import uuid

from probe_qwen_moe_concurrency import file_hash, compare_requests, validate_report
from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_full import request
from fastllm_pytools.qwen3_process_chat import MAX_CONTEXT_TOKENS, validate_request
from fastllm_pytools.qwen_moe_process_session import RoundTransport


def collect_schedule(transport, job, first_wave, graph, positions, vocab_size, output, previous):
    """/**
     * 固定四个相同请求的Begin轮次，驱动正式协调器并保留实际采样和联合轮次。
     * rank小于first_wave者第1轮Begin，其余第2轮Begin；后续始终输入本请求的
     * 实际采样Token，达到长度后End。只规定准入，不替换原生前向、同步或Graph选择。
     * @param transport 正式RoundTransport；前一批次已End，缓存关闭。
     * @param job 已校验的公开请求，贪心采样、EOS关闭；四rank使用相同输入。
     * @param first_wave 第1轮准入数1至4；4表示全部同时Begin。
     * @param graph 是否允许Graph，逐轮核对实际路径及replay增量。
     * @param positions 采集完整FP32 logits的1-based生成位置，轮次不受trace耗时影响。
     * @param vocab_size 完整词表大小，用于校验采样和logits。
     * @param output 本批次独立目录；异常时也保存已采集请求和轮次。
     * @param previous 上一批次最终状态或ready状态，用于校验首次replay增量。
     * @return 完整四请求报告及最终状态；路径、KV或输出异常时抛出并由调用者回收进程组。
     */"""
    validate_request(job, vocab_size)
    if (first_wave not in (1, 2, 3, 4) or job['eos_token_ids'] or job['temperature'] != 0 or
            job['max_new_tokens'] < 3 or len(previous) != 4 or any(s['request_active'] for s in previous)):
        raise ValueError('controlled admission requires four idle ranks, greedy sampling and no EOS')
    output.mkdir(parents=True, exist_ok=False)
    reports = [dict(id=rank+1, dp_rank=rank, status='INCOMPLETE', request=job,
                    token_ids=[], steps=[], teacher_forced_tokens=None) for rank in range(4)]
    active, finished = [False]*4, [False]*4
    round_number = 0
    try:
        with (output / 'rounds.jsonl').open('w') as log:
            while not all(finished):
                round_number += 1
                steps = []
                for rank, report in enumerate(reports):
                    tokens = report['token_ids']
                    if round_number == (1 if rank < first_wave else 2):
                        step = dict(action=1, token_ids=job['token_ids'], position_begin=0,
                            capacity=len(job['token_ids']) + job['max_new_tokens'] - 1,
                            sampling={k: job[k] for k in ('seed', 'top_k', 'top_p', 'temperature')})
                    elif active[rank]:
                        step = dict(action=3, token_ids=[]) if len(tokens) == job['max_new_tokens'] else dict(
                            action=2, token_ids=[tokens[-1]])
                    else:
                        step = dict(action=0, token_ids=[])
                    steps.append(step)

                current_positions = [len(r['token_ids'])+1 if s['action'] in (1, 2) else None
                                     for r, s in zip(reports, steps)]
                schedule = dict(round_1based=round_number, token_positions_1based=current_positions,
                    request_ids=[r['id'] if s['action'] else None for r, s in zip(reports, steps)],
                    input_lengths=[len(job['token_ids'])+p-1 if p is not None else None for p in current_positions])
                joint = dict(actions=[s['action'] for s in steps],
                             submitted_token_counts=[len(s['token_ids']) for s in steps])
                trace = any(p in positions for p in current_positions)
                result = transport.exchange('step', steps=steps, trace=trace)
                log.write(json.dumps(dict(schedule=schedule, joint_round=joint,
                    result={k: v for k, v in result.items() if k != 'logits'}), allow_nan=False) + '\n')
                log.flush()

                expected_graph = graph and 1 not in joint['actions'] and 2 in joint['actions']
                if result['execution'] != ('decode_cuda_graph' if expected_graph else 'eager'):
                    raise ValueError(f'round {round_number}: unexpected execution path')
                if any(len(result[k]) != 4 for k in ('states', 'token_ids', 'logits')):
                    raise ValueError(f'round {round_number}: rank count mismatch')
                for rank, (step, report, state, token) in enumerate(zip(
                        steps, reports, result['states'], result['token_ids'])):
                    running = step['action'] in (1, 2)
                    if (state['request_active'] != running or
                            state['replays'] - previous[rank]['replays'] != int(expected_graph) or
                            state['prefix_cache']['last_reused_tokens'] != 0 or
                            (expected_graph and state['captures'] != 1)):
                        raise ValueError(f'round {round_number}, rank {rank}: invalid KV/cache/Graph state')
                    if running:
                        if type(token) is not int or not 0 <= token < vocab_size:
                            raise ValueError(f'round {round_number}, rank {rank}: invalid sampled token')
                        if state['request_tokens'] != schedule['input_lengths'][rank]:
                            raise ValueError(f'round {round_number}, rank {rank}: input history length changed')
                        item = dict(token_id=token, execution=result['execution'], joint_round=joint,
                                    trace_schedule=schedule)
                        if current_positions[rank] in positions:
                            values = result['logits'][rank]
                            if len(values) != vocab_size or not all(math.isfinite(v) for v in values):
                                raise ValueError(f'round {round_number}, rank {rank}: invalid selected logits')
                            item['logits'] = values
                        report['token_ids'].append(token)
                        report['steps'].append(item)
                    elif token is not None or any(state[k] for k in (
                            'request_tokens', 'request_capacity', 'request_kv_bytes')):
                        raise ValueError(f'round {round_number}, rank {rank}: ended/idle request retained output or KV')
                    if step['action'] == 3:
                        finished[rank] = True
                    active[rank] = running
                previous = result['states']

        for report in reports:
            report.update(status='PASS', states=previous)
            validate_report(report, job, graph, positions)
        return reports, previous
    finally:
        for rank, report in enumerate(reports):
            (output / f'rank{rank}.json').write_text(json.dumps(report, allow_nan=False))


def main():
    """/** 固定准入组合逐批采集；同历史logits只作诊断，Token不同或采集异常保留FAIL与部分报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, default=Path('build-attention-ep-cu126'))
    parser.add_argument('--prompt-length', type=int, default=1024)
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--first-wave', type=int, choices=(1, 2, 3, 4), nargs='+', default=[4, 2])
    parser.add_argument('--trace-tokens', type=int, nargs='+', default=[2, 79])
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if (args.prompt_length < 1 or args.output_tokens < 3 or args.repeats < 5 or
            args.prompt_length + args.output_tokens - 1 > MAX_CONTEXT_TOKENS or
            len(set(args.first_wave)) != len(args.first_wave) or 4 not in args.first_wave or
            len(set(args.trace_tokens)) != len(args.trace_tokens) or
            any(p < 1 or p > args.output_tokens for p in args.trace_tokens) or
            not math.isfinite(args.timeout) or args.timeout <= 0):
        parser.error('invalid context/positions/timeout; first-wave must include 4; repeats must be at least 5')
    waves = [4] + [n for n in args.first_wave if n != 4]
    args.build, args.model = args.build.resolve(), args.model.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_admission_probe_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    print(f'固定准入诊断：{output}', flush=True)
    job = request(([1, 2, 3, 4] * ((args.prompt_length+3)//4))[:args.prompt_length], args.output_tokens)
    rows, paths, variants = [], {}, {}
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), scope='controlled_admission_diagnostic',
        git=capture(['git', 'rev-parse', 'HEAD']), git_status=capture(['git', 'status', '--short']),
        cpu=capture(['lscpu']), cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), model_path=str(args.model), request=job,
        warmup=1, iterations=1, repeats=args.repeats, concurrency=4, first_waves=waves,
        trace_tokens_1based=args.trace_tokens, prefix_cache=False, execution_complete=False,
        expected_case_count=2*len(waves)*args.repeats*4, variants={},
        report_title='MoE固定准入轮次诊断', report_description=
            '直接驱动正式RoundTransport，仅固定Begin轮次，不经过Python异步队列；不作为调度或性能验收。'
            '同组合同rank比较跨轮输出；与全4准入同rank/同轮及Eager同组合逐项比较。'
            'FAIL表示Token差异或采集失败。failed_elements为各比较中最大Token差异数；'
            'logits只在输入历史相同时记录误差，数值差异不自动豁免Token失败。未复现不证明原问题消失。')
    try:
        vocab = json.loads((args.model / 'config.json').read_text())['vocab_size']
        validate_request(job, vocab)
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
            for wave in waves:
                label = f'{mode}_first{wave}_then{4-wave}'
                directory = output / label
                try:
                    with RoundTransport(args.build / 'fastllm_moe_generate', args.build / 'fastllm_tp_worker',
                            args.model, '0,1,2,3', directory / 'native', timeout=args.timeout,
                            cuda_graph=graph, prefix_cache=False) as session:
                        _, previous = collect_schedule(session, job, wave, graph, args.trace_tokens, vocab,
                                                       directory / 'warmup', session.ready['states'])
                        for repeat in range(1, args.repeats+1):
                            batch_dir = directory / f'repeat{repeat:02d}'
                            reports, previous = collect_schedule(session, job, wave, graph, args.trace_tokens,
                                                                 vocab, batch_dir, previous)
                            for rank, report in enumerate(reports):
                                path = batch_dir / f'rank{rank}.json'
                                paths[(mode, wave, repeat, rank)] = path
                                refs = dict(vs_first=paths[(mode, wave, 1, rank)])
                                if wave == 4:
                                    refs['vs_rank0'] = paths[(mode, 4, repeat, 0)]
                                else:
                                    refs['vs_all4'] = paths[(mode, 4, repeat, rank)]
                                if graph:
                                    refs['vs_eager'] = paths[('eager', wave, repeat, rank)]
                                details = {name: dict(expected_file=str(p), **compare_requests(
                                    report, json.loads(p.read_text()), args.trace_tokens)) for name, p in refs.items()}
                                failures = [name for name, d in details.items() if not d['token_comparison']['match']]
                                errors = [t for d in details.values() for t in d['traces'] if t['same_input_history']]
                                first = min((d['token_comparison']['first_divergence']['first_token_1based']
                                    for d in details.values() if not d['token_comparison']['match']), default=None)
                                mismatches = max(d['token_comparison']['token_mismatches'] for d in details.values())
                                case = f'{label}_repeat{repeat:02d}_rank{rank}'
                                detail_path = output / (case + '.json')
                                detail_path.write_text(json.dumps(dict(actual_file=str(path), comparisons=details),
                                                                  indent=2, allow_nan=False))
                                rows.append(dict(case=case, backend=mode, status='FAIL' if failures else 'PASS',
                                    prompt_length=args.prompt_length, output_tokens=args.output_tokens,
                                    failed_elements=mismatches, token_mismatches=mismatches,
                                    max_abs_error=max((t['max_abs_error'] for t in errors), default=None),
                                    mean_abs_error=sum(t['mean_abs_error'] for t in errors)/len(errors) if errors else None,
                                    first_divergence_token_1based=first, detail_report=str(detail_path),
                                    reason='Token差异：' + ', '.join(failures) if failures else 'Token一致；logits仅诊断'))
                                variant = variants.setdefault(tuple(report['token_ids']), 'V' + str(len(variants)+1))
                                counts = metadata['variants'].setdefault(label, {})
                                counts[variant] = counts.get(variant, 0) + 1
                                selected = {p: report['token_ids'][p-1] for p in args.trace_tokens}
                                print(f'{rows[-1]["status"]}: {case} {variant}; selected_tokens={selected}; first_token={first}; '
                                      f'comparisons={failures or "all_match"}', flush=True)
                                for name in failures:
                                    evidence = details[name]['first_divergence_trace']
                                    print(f'首次分歧 {name}: ' + json.dumps(evidence if evidence else dict(
                                        not_captured=details[name]['token_comparison']['first_divergence']),
                                        ensure_ascii=False), flush=True)
                            save_report(output, rows, metadata)
                    metadata[label + '_shutdown'] = session.shutdown
                except Exception as error:
                    directory.mkdir(parents=True, exist_ok=True)
                    (directory / 'collection.traceback.log').write_text(traceback.format_exc())
                    rows.append(dict(case=label + '_collection', status='FAIL', failed_elements=1,
                                     reason=f'{type(error).__name__}: {error}'))
                    save_report(output, rows, metadata)
                    print(f'FAIL: {label}: {error}', flush=True)
        metadata['execution_complete'] = (len(rows) == metadata['expected_case_count'] and
                                         not any(r['case'].endswith('_collection') for r in rows))
    except BaseException as error:
        (output / 'setup.traceback.log').write_text(traceback.format_exc())
        rows.append(dict(case='setup_or_interrupt', status='FAIL', failed_elements=1,
                         reason=f'{type(error).__name__}: {error}'))
    finally:
        metadata['reproduction_status'] = ('OBSERVED_TOKEN_DIVERGENCE' if any(r.get('token_mismatches', 0) for r in rows)
            else 'NOT_OBSERVED' if metadata['execution_complete'] else 'INCOMPLETE')
        status = save_report(output, rows, metadata)
    print(f'{status}: {metadata["reproduction_status"]}; {output}', flush=True)
    return int(status != 'PASS')


if __name__ == '__main__':
    raise SystemExit(main())
