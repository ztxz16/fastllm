#!/usr/bin/env python3
"""DP 配置验收：默认仅 Python 回归；--real-cuda 通过正式 CLI 查询服务器 GPU。"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import traceback
import unittest
import uuid

from run_process_tp import ROOT, capture, save_report


class ReportResult(unittest.TextTestResult):
    def __init__(self, stream, descriptions, verbosity, *, rows, output, metadata):
        super().__init__(stream, descriptions, verbosity)
        self.rows, self.output, self.metadata = rows, output, metadata

    def startTest(self, test):
        super().startTest(test)
        self.row = {'case': test.id(), 'backend': 'python/formal-entry + mocked CUDA/native transport',
                    'status': 'PASS', 'reason': '', 'max_abs_error': None,
                    'mean_abs_error': None, 'failed_elements': None}
        self.rows.append(self.row)

    def _failure(self, test, error):
        self.row['status'] = 'FAIL'
        self.row['reason'] += test.id() + '\n' + self._exc_info_to_string(error, test)

    def addFailure(self, test, error):
        self._failure(test, error)
        super().addFailure(test, error)

    def addError(self, test, error):
        self._failure(test, error)
        super().addError(test, error)

    def addSubTest(self, test, subtest, error):
        if error is not None:
            self._failure(subtest, error)
        super().addSubTest(test, subtest, error)

    def addSkip(self, test, reason):
        self.row.update(status='SKIP', reason=reason)
        super().addSkip(test, reason)

    def stopTest(self, test):
        super().stopTest(test)
        save_report(self.output, self.rows, self.metadata)


def real_cuda_case(output, dp, tp, metadata):
    """/**
     * 在独立 CLI 进程中验证真实 CUDA 的逻辑设备分配，不启动模型推理。
     * @param output 保存原始命令、stdout 和 stderr 的独立报告目录。
     * @param dp 完整副本数。
     * @param tp 每副本 GPU 数。
     * @param metadata 回填实际查询的 GPU 身份及驱动支持的 CUDA 版本。
     * @return 包含实际输出、编号误差和 PASS/FAIL 的用例记录；失败也保留日志。
     */"""
    command = [sys.executable, '-m', 'fastllm_pytools.cli', 'serve', '--tp-backend', 'process',
               '--dp', str(dp), '--tp', str(tp), '--parallel-plan']
    env = dict(os.environ)
    env['PYTHONPATH'] = str(ROOT / 'tools') + (os.pathsep + env['PYTHONPATH'] if env.get('PYTHONPATH') else '')
    row = {'case': f'real_cuda_dp{dp}_tp{tp}', 'backend': 'formal CLI / CUDA Driver API',
           'command': shlex.join(command), 'status': 'FAIL', 'reason': '',
           'max_abs_error': None, 'mean_abs_error': None, 'failed_elements': None}
    (output / 'cuda.command.txt').write_text(row['command'] + '\n', encoding='utf-8')

    try:
        with (output / 'cuda.stdout.log').open('w', encoding='utf-8') as stdout, \
                (output / 'cuda.stderr.log').open('w', encoding='utf-8') as stderr:
            process = subprocess.run(command, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, timeout=45)
        if process.returncode != 0:
            raise RuntimeError(f'CLI exit {process.returncode}: ' + (output / 'cuda.stderr.log').read_text())
        plan = json.loads((output / 'cuda.stdout.log').read_text())
        row['actual_plan'] = plan
        if (plan['dp_size'], plan['tp_size'], plan['total_gpus']) != (dp, tp, dp * tp):
            raise AssertionError('wrong DP/TP sizes')
        replicas = plan['replicas']
        if len(replicas) != dp or [replica['dp_rank'] for replica in replicas] != list(range(dp)):
            raise AssertionError('wrong replica count or ranks')
        if any(len(replica['devices']) != tp for replica in replicas):
            raise AssertionError('wrong per-replica GPU count')
        devices = [device for replica in replicas for device in replica['devices']]
        actual = [device['logical_id'] for device in devices]
        expected = list(range(dp * tp))
        errors = [abs(a - b) for a, b in zip(actual, expected)]
        row.update(max_abs_error=float(max(errors)), mean_abs_error=sum(errors) / len(errors),
                   failed_elements=sum(error != 0 for error in errors))
        if any(type(value) is not int for value in actual) or row['failed_elements']:
            raise AssertionError('GPU logical order differs from expected contiguous assignment')
        if len({device['uuid'] for device in devices}) != dp * tp:
            raise AssertionError('replicas share CUDA device UUIDs')
        for device in devices:
            uuid.UUID(device['uuid'])
            if not device['name']:
                raise AssertionError('missing GPU name')
        if plan['preview_only'] is not True or plan['multi_replica_execution_available'] is not True:
            raise AssertionError('wrong preview/execution capability')
        metadata.update(gpu=devices, cuda=plan['driver_cuda_version'])
        row['status'] = 'PASS'
    except Exception as error:
        row['reason'] = f'{type(error).__name__}: {error}'
    return row


def main():
    """/** 运行配置验收并持续写入四种格式报告；任一失败或提前中断返回非零。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--real-cuda', action='store_true', help='增加实际 GPU 的正式 CLI 预览验收；仅在 GPU 服务器使用')
    parser.add_argument('--execution-tests', action='store_true', help='增加 DP 执行、会话及 HTTP 的纯 Python 协议回归')
    parser.add_argument('--benchmark-tests', action='store_true', help='增加测速统计、失败报告和正式工厂协议回归；不测真实GPU')
    parser.add_argument('--dp', type=int, default=2)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    args = parser.parse_args()
    if args.dp < 1 or args.tp < 1:
        parser.error('--dp and --tp must be positive')

    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_parallel_config_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    metadata = {'command': shlex.join([sys.executable, *sys.argv]), 'git': capture(['git', 'rev-parse', 'HEAD']),
                'git_status': capture(['git', 'status', '--short']), 'python': sys.version, 'seed': 0,
                'gpu': None, 'cuda': None, 'compile_architecture': 'not applicable; Python only, no native compilation',
                'cuda_version_kind': 'Driver API supported version, not installed toolkit version',
                'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES'),
                'warmup': 0, 'iterations': 1, 'repeats': 1, 'real_cuda_requested': args.real_cuda,
                'report_title': 'DP configuration and GPU allocation',
                'report_description': '纯 Python 检查使用 CUDA/原生协议替身；仅 real_cuda 用例查询真实 GPU。'
                                      '不执行真实模型推理或性能测试；非数值配置检查的误差字段为 n/a。'}
    rows = []
    try:
        modules = ['test_parallel_config', 'test_process_tp_options', 'test_process_tp_entrypoint']
        if args.execution_tests:
            modules += ['test_process_dp', 'test_dp_routing', 'test_dp_service_stability', 'test_qwen3_session', 'test_qwen3_http']
        if args.benchmark_tests:
            modules += ['test_benchmark_qwen3_dp', 'test_benchmark_dp_routing', 'test_diagnose_qwen3_prefix', 'test_tp_concurrency_diagnostic']
        suite = unittest.defaultTestLoader.loadTestsFromNames(modules)
        metadata['expected_case_count'] = suite.countTestCases() + int(args.real_cuda)
        save_report(output, rows, metadata)
        with (output / 'unittest.stdout.log').open('w', encoding='utf-8') as stdout, \
                (output / 'unittest.stderr.log').open('w', encoding='utf-8') as stderr:
            with redirect_stdout(stdout), redirect_stderr(stderr):
                runner = unittest.TextTestRunner(stream=stderr, verbosity=2,
                    resultclass=lambda *values: ReportResult(*values, rows=rows, output=output, metadata=metadata))
                runner.run(suite)
        if args.real_cuda:
            rows.append(real_cuda_case(output, args.dp, args.tp, metadata))
    except (Exception, KeyboardInterrupt) as error:
        rows.append({'case': 'runner_failure', 'backend': 'python', 'status': 'FAIL',
                     'reason': f'{type(error).__name__}: {error}\n{traceback.format_exc()}'})
    status = save_report(output, rows, metadata)
    print(f'{status}: {output}')
    return 0 if status == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
