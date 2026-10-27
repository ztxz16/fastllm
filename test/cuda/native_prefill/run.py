#!/usr/bin/env python3
"""Standalone native-prefill oracles; run on an idle GPU, CUDA >= 12.8.
CUDA_HOME=/path/to/cuda python3 test/cuda/native_prefill/run.py --arch 120f --tma
Use --compile-only for code-generation checks on a host without a matching GPU.
"""
import argparse
import os
from pathlib import Path
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--arch', default='120f')
parser.add_argument('--tma', action='store_true', help='SM120-family TMA versus cuBLASLt oracle')
parser.add_argument('--compile-only', action='store_true')
parser.add_argument('--output', type=Path, default=Path('/tmp/fastllm-native-prefill-tests'))
args = parser.parse_args()
if args.tma and args.arch not in ('120f', '120a'):
    parser.error('--tma requires --arch 120f or 120a')
here = Path(__file__).resolve().parent
repo = here.parents[2]
cuda = Path(os.environ.get('CUDA_HOME', '/usr/local/cuda'))
linear = repo / 'src/devices/cuda/linear'
args.output.mkdir(parents=True, exist_ok=True)
common = [str(cuda / 'bin/nvcc'), '-O3', '-std=c++17', '--default-stream=per-thread',
          '-gencode=arch=compute_' + args.arch + ',code=sm_' + args.arch]
common += ['-I' + str(repo / p) for p in (
    'include', 'include/utils', 'third_party/json11', 'include/devices/cuda', 'src/devices/cuda',
    'src/devices/cuda/linear')]
tests = ['prefill-policy-test', 'native-layout-test', 'native-test', 'native-fp4-test', 'gdn-key-sanitizer-test']
if args.tma:
    tests += ['native-tma-test', 'native-layout-prefill-test', 'native-fresh-test']
for name in tests:
    cmd = common + [str(here / (name + '.cu'))]
    if name.startswith('native-') and name != 'native-layout-test':
        cmd += [str(linear / 'fastllm-native-lowbit-prefill.cu'), '-lcublasLt']
    else:
        cmd += ['-lcublas']
    if name in ('native-tma-test', 'native-layout-prefill-test', 'native-fresh-test'):
        cmd += ['-DFASTLLM_NATIVE_PREFILL_SM120',
                str(linear / 'fastllm-native-nvfp4-tma-sm120.cu'), '-lcuda']
    binary = args.output / name
    subprocess.run(cmd + ['-o', str(binary)], check=True)
    print('BUILT', name, flush=True)
    if not args.compile_only:
        subprocess.run([str(binary)], check=True)
        if name == 'native-layout-test':
            for k in (5120, 17408):
                subprocess.run([str(binary), str(k)], check=True)
