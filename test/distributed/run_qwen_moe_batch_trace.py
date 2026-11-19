#!/usr/bin/env python3
"""逐层/第0层内部定位批次不变候选的剩余差异；复用正式观察器，不运行 HF 或性能测试。"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import sys
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_full import request
from probe_qwen_moe_prefill_batches import prefill, load_traces, difference
from probe_qwen_moe_block_batches import load_blocks, compare_blocks


def main():
    """/** 固定两/四请求，独立采集逐层和专家内部快照；首差及部分报告始终落盘。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--prompt-length', type=int, default=1024)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if not 2 <= args.prompt_length <= 1792 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error('invalid prompt length/timeout')
    args.model, args.build, args.batch_invariant = args.model.resolve(), args.build.resolve(), True
    config = json.loads((args.model / 'config.json').read_text())
    job = request(([1, 2, 3, 4] * ((args.prompt_length + 3) // 4))[:args.prompt_length], 128)
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_batch_trace_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    metadata = dict(scope='batch_invariant_remaining_difference_diagnostic', execution_complete=False,
        batch_invariant=True, command=shlex.join([sys.executable, *sys.argv]),
        git=capture(['git', 'rev-parse', 'HEAD']), build_cache=(args.build / 'CMakeCache.txt').read_text(),
        cuda=capture(['nvcc', '--version']), gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        warmup=1, iterations=1, repeats=1, request=job, first_different_stage={},
        report_title='批次不变候选剩余差异定位',
        report_description='正式算子及通信不变，只观察逐层边界及第0层专家内部；此诊断不替代完整生成验收。')
    try:
        for block in (False, True):
            snapshots, results = {}, {}
            for wave in (2, 4):
                label = ('block' if block else 'layers') + f'_first{wave}'
                result, directory, shutdown = prefill(args, output / label, job, list(range(wave)),
                                                     config['vocab_size'], block_trace=block)
                metadata[label + '_shutdown'] = shutdown
                results[wave] = result
                snapshots[wave] = load_blocks(directory, wave, config, job) if block else load_traces(
                    directory, list(range(wave)), job['token_ids'], config['hidden_size'], config['num_hidden_layers'])
            if block:
                block_metadata = dict(expert_pairing_valid=True)
                compare_blocks(snapshots, rows, block_metadata)
                metadata['block_diagnostic'] = block_metadata
            else:
                for rank in (0, 1):
                    a, x = snapshots[2][rank]
                    _, y = snapshots[4][rank]
                    for entry in a['snapshots']:
                        begin, end = entry['offset'], entry['offset'] + entry['elements']
                        row = dict(case=f'rank{rank}_{entry["name"]}', backend='formal_sm70_expert_candidate',
                                   reason='', **difference(x[begin:end], y[begin:end]))
                        rows.append(row)
                        if row['status'] == 'FAIL':
                            metadata['first_different_stage'].setdefault(str(rank), entry['name'])
                    rows.append(dict(case=f'rank{rank}_logits', backend='formal_sm70_expert_candidate', reason='',
                                     **difference(results[2]['logits'][rank], results[4]['logits'][rank])))
            save_report(output, rows, metadata)
        metadata['execution_complete'] = True
    except Exception as error:
        rows.append(dict(case='diagnostic_failure', backend='formal_sm70_expert_candidate', status='FAIL',
                         failed_elements=1, reason=f'{type(error).__name__}: {error}'))
    finally:
        status = save_report(output, rows, metadata)
        print('各 rank 首差：' + json.dumps(metadata['first_different_stage'], ensure_ascii=False))
        print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
