#!/usr/bin/env python3
"""独立比较旧路径与 SM70 专家候选的正式端到端性能；不运行 HF 或功能矩阵。"""

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shlex
import statistics
import sys
from types import SimpleNamespace
import uuid

from run_process_tp import ROOT, capture, save_report
from run_qwen_moe_full import Suite
from fastllm_pytools.moe_batch_invariant import BACKEND


def compare_samples(old, new, old_status, new_status):
    """/** 校验相同输入、输出历史及缓存复用后计算旧耗时/新耗时；失败时保留耗时但不算加速比。 */"""
    def tokens(record):
        return [[r['token_ids'] for r in repeat] for repeat in record.get('responses', [])]

    fields = ('requests', 'prompt_length', 'output_tokens', 'concurrency', 'cuda_graph', 'cache_state',
              'warmup', 'iterations', 'repeats', 'reused_tokens')
    same_workload = all(field in old and field in new and old[field] == new[field] for field in fields if field != 'reused_tokens')
    same = same_workload and old.get('reused_tokens') == new.get('reused_tokens') and 'reused_tokens' in old
    same = same and bool(tokens(old)) and tokens(old) == tokens(new)
    a, b = old.get('samples_ms', []), new.get('samples_ms', [])
    valid = len(a) == len(b) >= 5 and all(math.isfinite(v) and v > 0 for v in a + b)
    complete = valid and all(
        len(tokens(record)) == len(record['samples_ms']) == record.get('repeats') and
        all(len(repeat) == record.get('concurrency') and
            all(len(ids) == record.get('output_tokens') for ids in repeat) for repeat in tokens(record))
        for record in (old, new))
    measurements_valid = same_workload and complete and old_status == new_status == 'PASS'
    comparable = same and measurements_valid
    if not valid:
        return dict(comparison_eligible=False, measurements_valid=False, speedup=None, reason='missing/invalid timing samples')
    median = statistics.median(b)
    return dict(comparison_eligible=comparable, measurements_valid=measurements_valid,
        speedup=statistics.median(a) / median if comparable else None,
        elapsed_change_percent=(median / statistics.median(a) - 1) * 100 if comparable else None,
        baseline_median_ms=statistics.median(a), median_ms=median,
        baseline_p95_ms=sorted(a)[math.ceil(.95 * len(a)) - 1],
        baseline_cv=statistics.pstdev(a) / statistics.mean(a),
        baseline_tokens_per_second=old['concurrency'] * old['output_tokens'] * 1000 / statistics.median(a),
        p95_ms=sorted(b)[math.ceil(.95 * len(b)) - 1], cv=statistics.pstdev(b) / statistics.mean(b),
        tokens_per_second=new['concurrency'] * new['output_tokens'] * 1000 / median,
        reason='' if comparable else 'outputs/cache/workload differ or a source case failed; speedup not comparable')


def main():
    """/** 双方顺序独立加载，沿用正式性能入口；保留两份原始报告和比较报告。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--build', type=Path, required=True)
    parser.add_argument('--prompt-lengths', type=int, nargs='+', default=[512, 1024, 1792])
    parser.add_argument('--output-tokens', type=int, default=128)
    parser.add_argument('--concurrencies', type=int, nargs='+', choices=(1, 4, 8), default=[1, 4, 8])
    parser.add_argument('--cache-states', nargs='+', choices=('off', 'cold', 'hot'), default=['off', 'cold', 'hot'])
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if args.output_tokens < 2 or not math.isfinite(args.timeout) or args.timeout <= 0 or any(n < 1 or n + args.output_tokens - 1 > 2048 for n in args.prompt_lengths):
        parser.error('invalid output/context/timeout')
    for values in (args.prompt_lengths, args.concurrencies, args.cache_states):
        if len(values) != len(set(values)):
            parser.error('duplicate matrix values')
    comparisons = 2 * len(args.prompt_lengths) * len(args.concurrencies) * len(args.cache_states)
    if comparisons * 3 > 50:
        parser.error('旧路径、候选与比较报告合计超过50项，请按输入长度和缓存状态分批')
    args.model, args.build = args.model.resolve(), args.build.resolve()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_moe_batch_performance_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    rows, reports = [], {}
    metadata = dict(scope='formal_old_vs_batch_invariant_performance', execution_complete=False,
        command=shlex.join([sys.executable, *sys.argv]), git=capture(['git', 'rev-parse', 'HEAD']),
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), build_cache=(args.build / 'CMakeCache.txt').read_text(),
        warmup=1, iterations=1, repeats=5, source_reports={}, model=str(args.model),
        expected_case_count=comparisons, independent_execution_cases=2 * comparisons,
        dtype='FP16', attention_dp=4, ep=4, tp=1, dp=1,
        report_title='Attention DP/EP 批次不变候选性能代价',
        report_description='正式公开会话端到端计时；未串入 HF。加速比=旧路径耗时/候选耗时，输出历史不同则不可直接比较。')
    try:
        for name, enabled in (('legacy', False), ('candidate', True)):
            options = SimpleNamespace(binary=args.build / 'fastllm_moe_generate', worker=args.build / 'fastllm_tp_worker',
                model=args.model, build_cache=args.build / 'CMakeCache.txt', timeout=args.timeout,
                batch_invariant=enabled, performance_prompt_lengths=args.prompt_lengths,
                performance_output_tokens=args.output_tokens, performance_cache_states=args.cache_states,
                performance_concurrencies=args.concurrencies)
            suite = Suite(options, output=output / name)
            try:
                suite.performance(args.model)
            finally:
                save_report(suite.output, suite.rows, suite.metadata)
                reports[name] = suite
                metadata['source_reports'][name] = str(suite.output / 'summary.json')
        legacy = {r['case']: r for r in reports['legacy'].rows}
        for row in reports['candidate'].rows:
            case = row['case']
            source = legacy.get(case, {'status': 'FAIL'})
            result = dict(case=case, backend=BACKEND, status='FAIL', max_abs_error=None, mean_abs_error=None,
                          failed_elements=1, legacy_status=source['status'], candidate_status=row['status'])
            old_path, new_path = output / 'legacy' / (case + '.json'), output / 'candidate' / (case + '.json')
            old = json.loads(old_path.read_text()) if old_path.exists() else {}
            new = json.loads(new_path.read_text()) if new_path.exists() else {}
            result.update(compare_samples(old, new, source['status'], row['status']))
            result.update(model=args.model.name, dtype='FP16', attention_dp=4, ep=4,
                          prompt_length=new.get('prompt_length'), output_tokens=new.get('output_tokens'),
                          concurrency=new.get('concurrency'), cuda_graph=new.get('cuda_graph'),
                          cache_state=new.get('cache_state'), repeats=5)
            result['legacy_backend'] = sorted({state.get('expert_gemm_backend', 'unreported') for state in old.get('states', [])})
            for key in ('ttft_median_ms', 'itl_median_ms', 'queue_median_ms', 'reserved_kv_bytes',
                        'prefix_bytes', 'graph_workspace_bytes'):
                result[key] = row.get(key)
            if result['measurements_valid']:
                result.update(status='PASS', failed_elements=0)
            result.update(legacy_evidence=str(old_path), candidate_evidence=str(new_path))
            rows.append(result)
        metadata['execution_complete'] = True
    except Exception as error:
        rows.append(dict(case='execution_failure', backend=BACKEND, status='FAIL', failed_elements=1,
                         reason=f'{type(error).__name__}: {error}'))
    finally:
        status = save_report(output, rows, metadata)
        print(f'{status}: {output}', flush=True)
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
