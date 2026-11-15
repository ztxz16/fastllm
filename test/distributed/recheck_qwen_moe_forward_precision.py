#!/usr/bin/env python3
"""只读复核已保存的完整前向轨迹与输出头；不启动GPU进程、不加载模型、不编译。"""

import argparse
import copy
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import sys
import traceback
import uuid

from probe_qwen_moe_forward_precision import (
    BACKEND, compare, compare_native_head_replay, digest, hf_trajectory, load_trace,
)
from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_precision_diagnostic import ATOL, RTOL, failing_history


def recheck(directory, report):
    """/**
     * 从原始文件重新计算数值和门禁，额外定位每个Token首次超差的真实前向边界。
     * 校验轨迹哈希、参考历史及阶段连续性；读取失败即拒绝归因。原报告中的其他
     * case原样保留，所有原模型阈值不变，文本编码修正只作用于原生输出头精确门禁。
     * @param directory 已完成的forward_probe目录，及其引用的历史参考文件必须仍在。
     * @param report 该目录的原summary，含实际执行环境和原始构建信息。
     * @return 更新后的case、逐Token观察及独立复核元数据；任何证据缺失抛异常。
     */"""
    metadata = report['metadata']
    if metadata.get('atol') != ATOL or metadata.get('rtol') != RTOL:
        raise ValueError('original numerical tolerances differ from this recheck')
    trace_meta = metadata['trace']
    ids = metadata['target']['ids']
    hidden, layers = trace_meta['hidden'], trace_meta['layers']
    artifacts = {}

    def read(path):
        """/** 只读JSON并记录本次输入哈希，不覆盖历史文件。 */"""
        artifacts[str(path)] = digest(path)
        return json.loads(path.read_text())

    trace_checked, native = load_trace(directory / 'trace', ids, hidden, layers)
    if trace_checked != trace_meta:
        raise ValueError('saved trace metadata/hash differs from the original report')
    diagnostic = Path(metadata['diagnostic'])
    source = read(diagnostic / 'summary.json')['metadata']
    rank = metadata['source_rank']
    if source['target']['ids'] != ids or source['rank'] != rank or not 0 <= rank < 4:
        raise ValueError('saved layer diagnostic history/rank mismatch')
    if read(diagnostic / 'layer-inputs.json') != [ids if i == rank else [] for i in range(4)]:
        raise ValueError('saved HF input history mismatch')
    reference = read(diagnostic / 'reference.json')
    hf = hf_trajectory(reference, rank, len(ids), hidden, layers)
    source_dir = Path(metadata['source_dir'])
    if source_dir != Path(source['source_dir']):
        raise ValueError('original request directory mismatch')
    original = read(source_dir / source['request'])
    full_reference = read(source_dir / 'transformers-reference.json')
    target = failing_history(original, full_reference)
    if target != metadata['target'] or original['dp_rank'] != rank:
        raise ValueError('original failing request/history changed')
    saved_hf = next(row['logits'] for row in full_reference['rows'] if row['ids'] == ids)
    vocab = len(saved_hf)
    first, second, traced = [read(directory / f'{name}.json') for name in
                             ('native-plain-1', 'native-plain-2', 'native-traced')]
    head_data = read(directory / 'head-replay.json')
    if head_data['metadata'] != metadata['head_reference']:
        raise ValueError('saved head replay metadata mismatch')
    head = head_data['values']

    rows, tokens, first_failure = [], [], {}

    def record(label, actual, expected, width, exact=False):
        """/** 所有非传输门禁沿用原比较函数及阈值。 */"""
        row = compare(label, actual, expected, width, exact)
        rows.append(row)
        return row['status'] == 'PASS'

    record('saved_native_vs_saved_hf', original['steps'][target['step']]['logits'], saved_hf, vocab)
    stable = record('plain_repeat_exact', second, first, vocab, exact=True)
    unchanged = record('trace_vs_plain_exact', traced, second, vocab, exact=True)
    record('plain_fresh_vs_saved_hf', second, saved_hf, vocab)
    record('traced_fresh_vs_saved_hf', traced, saved_hf, vocab)
    for name, expected in hf.items():
        record(name, native[name], expected, hidden)
        for token in range(len(ids)):
            begin, end = token*hidden, (token+1)*hidden
            row = compare(f'{name}/token{token}', native[name][begin:end], expected[begin:end], hidden)
            row.update(token_index=token, token_id=ids[token], stage=name)
            tokens.append(row)
            if row['status'] != 'PASS' and token not in first_failure:
                first_failure[token] = dict(token_index=token, token_id=ids[token], stage=name,
                    max_abs_error=row['max_abs_error'], mean_abs_error=row['mean_abs_error'],
                    failed_elements=row['failed_elements'])

    hf_valid = record('hf_head_reproduces_saved_logits', head['hf_head_full'], saved_hf, vocab, exact=True)
    native_row = compare_native_head_replay(head['native_head_replay'], traced, vocab)
    rows.append(native_row)
    record('final_norm_accumulated', native['final_norm'], head['hf_norm'], hidden)
    record('final_norm_same_native_input', native['final_norm'], head['hf_norm_on_native'], hidden)
    record('head_hf_single_row_vs_full_rows', head['hf_head_last'], head['hf_head_full'], vocab)
    record('head_native_vs_hf_same_hf_norm', head['native_head_on_hf'], head['hf_head_last'], vocab)
    record('head_native_vs_hf_same_native_norm', traced, head['hf_head_on_native_norm'], vocab)

    previous = {row['case']: row for row in report['cases']}
    if len(previous) != len(report['cases']):
        raise ValueError('duplicate case in original report')
    for row in rows:
        old = previous.get(row['case'])
        if old is None:
            raise ValueError(f'original probe is incomplete: {row["case"]}')
        if row['case'] != native_row['case'] and any(old[key] != row[key] for key in
                ('status', 'max_abs_error', 'mean_abs_error', 'failed_elements', 'exact_different_elements')):
            raise ValueError(f'saved artifacts no longer reproduce original metrics: {row["case"]}')
    recomputed = {row['case']: row for row in rows}
    rows = [recomputed.get(row['case'], copy.deepcopy(row)) for row in report['cases']]
    detail = dict(artifact_sha256=artifacts, first_failure_by_token=list(first_failure.values()),
        trace_attribution_valid=stable and unchanged,
        head_attribution_valid=stable and unchanged and hf_valid and native_row['status'] == 'PASS',
        head_gate_raw_max_abs=native_row['raw_numeric_comparison']['max_abs_error'],
        head_gate_wire_max_abs=native_row['max_abs_error'],
        head_gate_ambiguous_elements=native_row['serialization_ambiguous_elements'])
    return rows, tokens, detail


def main():
    """/** 创建新的四格式CPU复核报告；保留历史FAIL和来源信息，不重写或重新运行原验收。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--probe', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    directory = args.probe.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_forward_recheck_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    print(f'只读复核目录：{output}', flush=True)
    rows, metadata = [], {}
    analysis = dict(recheck_command=shlex.join([sys.executable, *sys.argv]),
                    recheck_git=capture(['git', 'rev-parse', 'HEAD']),
                    recheck_source=str(directory), recheck_execution='CPU saved artifacts only; no GPU execution')
    try:
        report = json.loads((directory / 'summary.json').read_text())
        metadata = copy.deepcopy(report['metadata'])
        rows = copy.deepcopy(report['cases'])
        metadata.update(analysis, source_summary_sha256=digest(directory / 'summary.json'),
            report_title='MoE完整前向历史数据复核',
            report_description='只读复用历史数据；command/git/GPU/构建字段属于原采集。'
                               '修正原生输出头传输精确门禁，保留原模型阈值及失败，不是新GPU验收。')
        rows, tokens, detail = recheck(directory, report)
        metadata.update(detail)
        token_dir = output / 'tokens'
        token_dir.mkdir()
        save_report(token_dir, tokens, dict(metadata, report_title='逐Token累积误差观察'))
        for name in ('trace_attribution_valid', 'head_attribution_valid', 'head_gate_raw_max_abs',
                     'head_gate_wire_max_abs', 'head_gate_ambiguous_elements'):
            print(f'{name}: {detail[name]}', flush=True)
        print('各Token首次原阈值超差（层号和Token位置从0开始）：', flush=True)
        for item in detail['first_failure_by_token']:
            print(f'  token={item["token_index"]} id={item["token_id"]} {item["stage"]}: '
                  f'max_abs={item["max_abs_error"]:.9g} mean_abs={item["mean_abs_error"]:.9g} '
                  f'failed={item["failed_elements"]}', flush=True)
    except Exception as error:
        metadata.update(analysis, trace_attribution_valid=False, head_attribution_valid=False)
        (output / 'error.log').write_text(traceback.format_exc())
        rows.append(dict(case='recheck_error', backend=BACKEND, status='FAIL', max_abs_error=None,
                         mean_abs_error=None, failed_elements=1, reason=str(error)))
        print(f'FAIL: recheck_error: {error}', flush=True)
    status = save_report(output, rows, metadata)
    print(f'{status}: {output}（保留真实数值失败；仅历史数据复核，非完整模型验收）', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
