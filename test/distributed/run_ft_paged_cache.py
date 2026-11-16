#!/usr/bin/env python3
"""运行已有的FT页管理器检查程序并输出四种报告；不触发编译。"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shlex
import subprocess
import sys
import uuid

from run_process_tp import capture, save_report


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--cuda', action='store_true', help='同时运行原FT公开分页追加和Attention数值回归')
    parser.add_argument('--check', type=Path, required=True)
    parser.add_argument('--build-cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('process-tp-results'))
    args = parser.parse_args()
    output = args.output.resolve() / (datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_ft_pages_' + uuid.uuid4().hex[:8])
    output.mkdir(parents=True)
    command = [str(args.check.resolve())] + (['--cuda'] if args.cuda else [])
    metadata = dict(command=shlex.join([sys.executable, *sys.argv]), native_command=shlex.join(command),
                    git=capture(['git', 'rev-parse', 'HEAD']), seed=0, warmup=0, iterations=1, repeats=1,
                    gpu=capture(['nvidia-smi', '--query-gpu=name,driver_version', '--format=csv,noheader']),
                    cuda=capture(['nvcc', '--version']), build_cache=None, compile_architecture=None,
                    expected_case_count=7 if args.cuda else 5, report_title='FT shared paged cache manager correctness')
    rows = []
    try:
        cache = args.build_cache.read_text()
        metadata['build_cache'] = cache
        metadata['compile_architecture'] = [line for line in cache.splitlines() if line.startswith(('CUDA_ARCH:', 'CMAKE_CUDA_ARCHITECTURES:'))]
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120)
        (output/'native.stdout.log').write_text(result.stdout)
        (output/'native.stderr.log').write_text(result.stderr)
        native = json.loads(result.stdout.strip().splitlines()[-1])
        rows = native['results']
        if result.returncode or native['status'] != 'PASS':
            raise RuntimeError(f'native page checks failed: exit={result.returncode}')
    except Exception as error:
        rows.append(dict(case='native_execution', backend='FT/PagedCacheManager', status='FAIL', reason=str(error)))
    save_report(output, rows, metadata)
    passed = len(rows) == metadata['expected_case_count'] and all(row['status'] == 'PASS' for row in rows)
    print(('PASS' if passed else 'FAIL') + ': ' + str(output))
    return 0 if passed else 1


if __name__ == '__main__':
    raise SystemExit(main())
