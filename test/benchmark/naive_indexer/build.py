"""Standalone Indexer test: compare this worktree with its committed baseline."""
import argparse
from pathlib import Path
import subprocess

source = Path(__file__).resolve().parent
root = source.parents[2]
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--arch', default='sm_90')
p.add_argument('--nvcc', default='/usr/local/cuda/bin/nvcc')
p.add_argument('--baseline', default='HEAD')
a = p.parse_args()
build = source / 'build'
build.mkdir(exist_ok=True)
for name, namespace in [('topk', 'naive_topk'), ('dsa-mma', 'naive_dsa_mma')]:
    baseline = Path(a.baseline)
    code = ((baseline / f'naive-n05-{name}.cuh').read_text() if baseline.is_dir() else
            subprocess.check_output(['git', 'show', f'{a.baseline}:src/devices/cuda/models/naive-n05-{name}.cuh'], cwd=root, text=True))
    (build / f'baseline-{name}.cuh').write_text(code.replace(namespace, 'baseline_' + namespace))
subprocess.run([a.nvcc, '-O3', '-std=c++17', '-arch=' + a.arch, '-shared', '-Xcompiler=-fPIC',
                '-I' + str(root / 'src/devices/cuda/models'), '-I' + str(build), str(source / 'operators.cu'),
                '-o', str(build / 'indexer.so')], check=True)
print(build / 'indexer.so')
