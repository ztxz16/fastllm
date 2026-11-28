#!/usr/bin/env python3
"""准备独立验收脚本并顺序调度已有入口；不编译、不下载、不修改推理实现。"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys
import uuid

from run_process_tp import ROOT, capture, save_report


BLOCKERS = ['前缀缓存命中/未命中输出不一致尚未修复，完整批次不变性矩阵尚未通过。']


def write_json(path, value):
    """/** 原子保存调度状态，避免断连或读写并发产生半份JSON。 */"""
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def fingerprint(path):
    """/** 记录已有文件内容摘要，不执行产物或触发编译。 */"""
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def jobs_for(args, output):
    """/**
     * 将已有矩阵展开为独立脚本，每份脚本的报告明细及策略汇总保守合计不超过50项。
     * 调度器本身只顺序启动这些脚本；不把子用例合并为一个case来绕过上限。
     * @param args 包含已有产物、模型和参考Python路径的参数。
     * @param output 本次独立目录；请求分片和每批输出均归入此目录。
     * @return 全部批次定义；不会运行GPU测试。
     */"""
    from run_qwen_moe_full import SYNTHETIC_FIXTURES
    from run_qwen_moe_layer import cases as layer_cases
    from run_qwen3_decode import cases as decode_cases

    jobs = []
    common = ['--binary', str(args.build / 'fastllm_moe_generate'),
              '--worker', str(args.build / 'fastllm_tp_worker'),
              '--build-cache', str(args.build / 'CMakeCache.txt')]

    def add(name, group, script, flags, count, output_option='--output'):
        """/** 每批独立登记预算、命令、证据目录；超限直接拒绝。 */"""
        if not 0 < count <= 50:
            raise ValueError(f'{name}: {count} report rows exceeds 50-case budget')
        directory = output / name
        command = [args.python, str(ROOT / 'test/distributed' / script), *map(str, flags),
                   output_option, str(directory / 'reports')]
        jobs.append(dict(name=name, group=group, command=command, expected_report_rows=count,
                         directory=str(directory), state='PENDING'))

    for world in (1, 2, 4):
        add(f'layer_ep{world}', 'validation', 'run_qwen_moe_layer.py',
            ['--check', args.build / 'fastllm_moe_layer_check', '--worker', args.build / 'fastllm_tp_worker',
             '--build-cache', args.build / 'CMakeCache.txt', '--world-size', world], len(layer_cases()), '--output-root')
    fixtures = list(SYNTHETIC_FIXTURES)
    for offset in range(0, len(fixtures), 4):
        selected = fixtures[offset:offset + 4]
        add(f'synthetic_part{offset // 4 + 1}', 'validation', 'run_qwen_moe_full.py',
            [*common, '--synthetic-fixtures', *selected], 12 * len(selected))
    add('lifecycle_small', 'validation', 'run_qwen_moe_full.py', [*common, '--lifecycle-only'], 2)
    for mode in ('legacy', 'candidate'):
        flags = ['--batch-invariant'] if mode == 'candidate' else []
        add(f'lifecycle_{mode}', 'validation', 'run_qwen_moe_full.py',
            [*common, '--model', args.model, '--lifecycle-only', *flags], 2)
        add(f'http_{mode}', 'validation', 'run_qwen_moe_http.py',
            [*common, '--model', args.model, *flags], 6)

    for graph in (False, True):
        add(f'decode_graph{int(graph)}', 'validation', 'run_qwen3_decode.py',
            ['--check', args.build / 'fastllm_tp_decode_check', '--worker', args.build / 'fastllm_tp_worker',
             '--build-cache', args.build / 'CMakeCache.txt', *(['--cuda-graph'] if graph else [])], len(decode_cases()))
    for config in ('dp2_tp1_eager', 'dp4_tp1_graph', 'dp2_tp2_eager', 'dp2_tp2_graph'):
        add('dense_' + config, 'validation', 'run_attention_ep_dense_regression.py',
            ['--build', args.build, '--configuration', config], 12)

    requests = [json.loads(line) for line in args.requests.read_text().splitlines() if line.strip()]
    if not requests:
        raise ValueError('HF request set is empty')
    for offset in range(0, len(requests), 4):
        selected = requests[offset:offset + 4]
        path = output / f'hf_requests_part{offset // 4 + 1}.jsonl'
        path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in selected))
        for mode in ('legacy', 'candidate'):
            flags = ['--batch-invariant'] if mode == 'candidate' else []
            add(f'hf_{mode}_part{offset // 4 + 1}', 'validation', 'run_qwen_moe_full.py',
                [*common, '--model', args.model, '--requests', path, '--reference-python', args.reference_python,
                 '--alignment-policy', 'all', *flags], 8 * len(selected) + 7)

    for length in (512, 1024, 1792):
        for cache in ('off', 'cold', 'hot'):
            # 6项旧路径 + 6项候选 + 6项比较；重复5轮不另计case。
            add(f'performance_p{length}_{cache}', 'performance', 'run_qwen_moe_batch_performance.py',
                ['--build', args.build, '--model', args.model, '--prompt-lengths', length,
                 '--output-tokens', 128, '--concurrencies', 1, 4, 8, '--cache-states', cache], 18)
    return jobs


def prepare(args):
    """/** 生成可审阅的命令、独立脚本与覆盖清单；此阶段不启动任何验收进程。 */"""
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') +
                                     '_attention_ep_acceptance_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True, exist_ok=False)
    args.build, args.model = args.build.resolve(), args.model.resolve()
    all_jobs = jobs_for(args, output)
    unknown = set(args.only or []) - {job['name'] for job in all_jobs}
    if unknown:
        raise ValueError(f'unknown jobs: {sorted(unknown)}')
    jobs = [job for job in all_jobs if job['group'] in args.groups and
            (not args.only or job['name'] in args.only)]
    if not jobs:
        raise ValueError('no selected jobs')
    for job in jobs:
        directory = Path(job['directory'])
        directory.mkdir()
        # EXIT trap位于独立子脚本，SSH断连不影响；正常失败仍记录原始退出码。
        exit_trap = 'code=$?; printf "%s\\n" "$code" > ' + shlex.quote(str(directory / 'exit.code')) + '; exit "$code"'
        script = '\n'.join([
            '#!/usr/bin/env bash', 'set -uo pipefail', 'cd ' + shlex.quote(str(ROOT)),
            'printf "%s\\n" "$$" > ' + shlex.quote(str(directory / 'pid')),
            'trap ' + shlex.quote(exit_trap) + ' EXIT',
            shlex.join(job['command']), '',
        ])
        (directory / 'run.sh').write_text(script)
        (directory / 'command.txt').write_text(shlex.join(job['command']) + '\n')
        job['script_sha256'] = fingerprint(directory / 'run.sh')
    build_files = [args.build / name for name in ('CMakeCache.txt', 'fastllm_moe_generate',
                   'fastllm_tp_worker', 'fastllm_moe_layer_check', 'fastllm_tp_decode_check', 'fastllm_tp_generate')]
    # 固定源码与原生产物；长时间后台运行中若被更新，后续批次明确失败。
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    sources = [ROOT / name for name in tracked if name and Path(name).suffix in ('.py', '.cu', '.cuh', '.cpp', '.h', '.sh')]
    sources += [Path(__file__), ROOT / 'test/distributed/run_attention_ep_acceptance_v100.sh']
    sources += list(output.glob('hf_requests_part*.jsonl')) + [args.model / 'config.json']
    files = {str(path): fingerprint(path) for path in set(build_files + sources) if path.is_file()}
    manifest = dict(jobs=jobs, all_job_names=[job['name'] for job in all_jobs],
        requested_groups=args.groups, build=str(args.build), model=str(args.model), python=args.python,
        reference_python=args.reference_python, git=capture(['git', 'rev-parse', 'HEAD']),
        git_status=capture(['git', 'status', '--short']), frozen_files=files,
        required_files=[str(path) for path in build_files] + [str(args.model / 'config.json')],
        blockers=BLOCKERS, final_delivery_eligible=False,
        requested_command=shlex.join([sys.executable, *sys.argv]))
    write_json(output / 'manifest.json', manifest)
    lines = ['# Attention DP/EP 分批验收', '', '每份 run.sh 独立运行；调度器仅顺序启动，不并行争用GPU。',
             '性能是当前版本测量，完整交付仍受前缀缓存一致性问题阻塞。', '',
             *[f'- {job["name"]}: {job["expected_report_rows"]} 项（含策略/比较汇总），`{job["name"]}/run.sh`' for job in jobs]]
    (output / 'plan.md').write_text('\n'.join(lines) + '\n')
    print(output, flush=True)


def inspect_job(job, exit_code):
    """/** 核对退出码、完整报告与实际行数；缺失报告、少跑用例或任一FAIL均不能通过。 */"""
    paths = sorted((Path(job['directory']) / 'reports').rglob('summary.json'))
    reasons, count = [], 0
    for path in paths:
        try:
            report = json.loads(path.read_text())
            rows = report['cases']
            if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
                raise ValueError('invalid cases')
            if not isinstance(report.get('metadata', {}), dict):
                raise ValueError('invalid metadata')
        except (OSError, ValueError, KeyError, TypeError) as error:
            reasons.append(f'{path}: invalid report: {error}')
            continue
        count += len(rows)
        if report.get('status') != 'PASS' or report.get('complete') is False:
            reasons.append(str(path))
        expected = report.get('expected_case_count', report.get('metadata', {}).get('expected_case_count'))
        if expected is not None and len(rows) != expected:
            reasons.append(str(path) + ': incomplete report')
        if any(row.get('status') != 'PASS' for row in rows):
            reasons.append(str(path) + ': failed case')
    if exit_code != 0:
        reasons.append(f'exit={exit_code}')
    if count != job['expected_report_rows']:
        reasons.append(f'expected {job["expected_report_rows"]} report rows, found {count}')
    return dict(case=job['name'], backend='existing formal test entrypoints',
                status='FAIL' if reasons else 'PASS', failed_elements=int(bool(reasons)),
                completed_case_count=count, expected_case_count=job['expected_report_rows'],
                reason='; '.join(reasons), reports=[str(path) for path in paths], exit_code=exit_code)


def write_performance(output, manifest):
    """/** 汇总原始旧/新对照；保留不可比原因和全部证据链接，不重算无效加速比。 */"""
    rows, sources = [], []
    for job in manifest['jobs']:
        if job['group'] != 'performance':
            continue
        for path in sorted((Path(job['directory']) / 'reports').glob('*/summary.json')):
            try:
                report = json.loads(path.read_text())
                if not isinstance(report.get('cases'), list) or any(not isinstance(row, dict) for row in report['cases']):
                    raise ValueError('missing cases')
            except (OSError, ValueError, TypeError, AttributeError) as error:
                sources.append(dict(path=str(path), status='FAIL', reason=str(error)))
                continue
            rows.extend(dict(row, detail_report=str(path)) for row in report['cases'])
            sources.append(dict(path=str(path), metadata=report.get('metadata', {}), status=report.get('status', 'FAIL')))
    if not rows:
        return
    directory = output / 'performance'
    directory.mkdir(exist_ok=True)
    expected = sum(job['expected_report_rows'] // 3 for job in manifest['jobs'] if job['group'] == 'performance')
    save_report(directory, rows, dict(report_title='Attention DP/EP 当前版本性能对比',
        report_description='当前版本测量；不是修复后最终交付。无法比较的行保留原因且加速比为空。',
        expected_case_count=expected, execution_complete=all(job['state'] in ('PASS', 'FAIL')
            for job in manifest['jobs'] if job['group'] == 'performance'),
        git=manifest['git'], model=manifest['model'], sources=sources,
        final_delivery_eligible=False, blockers=manifest['blockers']))


def execute(output):
    """/**
     * 后台顺序执行清单中的独立脚本，普通失败后继续其余批次，保留日志与退出码。
     * 所有汇总仅表示所选范围；已知前缀一致性阻塞不会被这些PASS覆盖。
     * @param output prepare生成的独立运行目录，禁止重复执行或覆盖历史结果。
     * @return 所选批次全部通过返回0，失败或未完成返回非零。
     */"""
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    # 排他创建，避免误执行两次；不将重试覆盖到旧目录。
    with (output / 'started.json').open('x') as handle:
        json.dump(dict(pid=os.getpid(), started_utc=datetime.now(timezone.utc).isoformat()), handle)
    rows = []
    metadata = dict(report_title='Attention DP/EP 所选验证与性能批次',
        report_description='阶段汇总仅表示本次所选范围；完整交付仍受前缀一致性问题阻塞。',
        expected_case_count=len(manifest['jobs']), execution_complete=False,
        git=manifest['git'], git_status=manifest['git_status'], command=manifest['requested_command'],
        gpu=capture(['nvidia-smi', '--query-gpu=name,uuid,driver_version', '--format=csv']),
        cuda=capture(['nvcc', '--version']), final_delivery_eligible=False, blockers=manifest['blockers'])
    code = 1
    try:
        save_report(output, rows, metadata)
        missing = [path for path in manifest['required_files'] if not Path(path).is_file()]
        if missing:
            raise RuntimeError(f'缺少服务器模型或已有编译产物：{missing}；本入口不自动编译')
        for name in ('python', 'reference_python'):
            if not shutil.which(manifest[name]):
                raise RuntimeError(f'缺少可执行的 {name}: {manifest[name]}')
        for job in manifest['jobs']:
            changed = [path for path, digest in manifest['frozen_files'].items()
                       if not Path(path).is_file() or fingerprint(Path(path)) != digest]
            if changed:
                raise RuntimeError(f'后台运行期间源码或产物变化：{changed}')
            directory = Path(job['directory'])
            if fingerprint(directory / 'run.sh') != job['script_sha256']:
                raise RuntimeError(f'{job["name"]}: prepared script changed')
            job['state'] = 'RUNNING'
            write_json(manifest_path, manifest)
            print(f'RUN: {job["name"]}; report rows <= {job["expected_report_rows"]}', flush=True)
            with (directory / 'stdout.log').open('wb') as stdout, (directory / 'stderr.log').open('wb') as stderr:
                with subprocess.Popen(['bash', str(directory / 'run.sh')], stdout=stdout, stderr=stderr,
                                      stdin=subprocess.DEVNULL, start_new_session=True) as process:
                    (directory / 'pid').write_text(str(process.pid) + '\n')
                    try:
                        returncode = process.wait()
                    except BaseException:
                        os.killpg(process.pid, signal.SIGINT)
                        try:
                            process.wait(timeout=90)
                        except subprocess.TimeoutExpired:
                            os.killpg(process.pid, signal.SIGTERM)
                            process.wait(timeout=30)
                        raise
            (directory / 'exit.code').write_text(str(returncode) + '\n')
            row = inspect_job(job, returncode)
            rows.append(row)
            job['state'] = row['status']
            write_json(manifest_path, manifest)
            save_report(output, rows, metadata)
            write_performance(output, manifest)
            print(f'{row["status"]}: {job["name"]}; {row["reason"]}', flush=True)
        metadata['execution_complete'] = True
    except BaseException as error:
        metadata['error'] = f'{type(error).__name__}: {error}'
        print(f'FAIL: {metadata["error"]}', flush=True)
    finally:
        try:
            write_performance(output, manifest)
            # 持久性能索引不生成Excel；本次原始报告目录仍包含XLSX。
            if (output / 'performance/summary.json').exists():
                destination = ROOT / 'docs/performance/runs' / output.name
                destination.mkdir(parents=True, exist_ok=False)
                for suffix in ('json', 'md', 'csv'):
                    shutil.copy2(output / f'performance/summary.{suffix}', destination / f'summary.{suffix}')
        except Exception as error:
            metadata.update(execution_complete=False, report_error=f'{type(error).__name__}: {error}')
        status = save_report(output, rows, metadata)
        code = 0 if status == 'PASS' else 1
        (output / 'delivery.md').write_text(
            f'# 当前验收结果：{status}\n\n所选批次完成 {len(rows)}/{len(manifest["jobs"])}。\n\n'
            '最终交付状态：未完成。\n\n' + '\n'.join('- ' + issue for issue in manifest['blockers']) +
            '\n\n本轮性能作为当前版本测量保留，修复后仍须按影响范围复测。\n')
        (output / 'exit.code').write_text(str(code) + '\n')
        print(f'{status}: selected scope; final delivery BLOCKED; {output}', flush=True)
    return code


def main():
    """/** 只准备、执行或汇总明确指定的一次清单；从不连接SSH或自动构建。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    plan = sub.add_parser('prepare')
    plan.add_argument('--build', type=Path, required=True)
    plan.add_argument('--model', type=Path, required=True)
    plan.add_argument('--python', default=sys.executable)
    plan.add_argument('--reference-python', default=sys.executable)
    plan.add_argument('--requests', type=Path, default=ROOT / 'test/distributed/data/qwen_moe_requests.jsonl')
    plan.add_argument('--groups', nargs='+', choices=('validation', 'performance'), default=['validation', 'performance'])
    plan.add_argument('--only', nargs='+')
    plan.add_argument('--output', type=Path, default=ROOT / 'test/distributed/results')
    run = sub.add_parser('run')
    run.add_argument('directory', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args)
        return 0
    def stop(signum, frame):
        """/** 将终止信号转入受控退出，保存部分报告并回收当前批次。 */"""
        raise KeyboardInterrupt(f'signal {signum}')
    signal.signal(signal.SIGTERM, stop)
    return execute(args.directory.resolve())


if __name__ == '__main__':
    raise SystemExit(main())
