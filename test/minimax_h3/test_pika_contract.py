"""Reject incompatible Pika manifests/weights before any model forward.

Requires the real base/adapter headers and one explicitly selected GPU.
Temporary fixtures link immutable weights; malformed headers use sparse files.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--model', required=True)
p.add_argument('--adapter', required=True)
a = p.parse_args()
assert os.environ.get('CUDA_VISIBLE_DEVICES'), 'Select the permitted GPU explicitly'
source = Path(a.adapter)
command = [a.binary, '--model', a.model, '--prompt', 'test', '--width', '64', '--height', '64']


def reject(extra, expected):
    result = subprocess.run(command + extra, text=True, capture_output=True, timeout=30)
    assert result.returncode != 0 and expected in result.stderr, result.stderr


reject(['--pika', str(source), '--steps', '5'], 'requires --steps 6')
reject(['--pika', str(source), '--precision', 'fp32'], 'requires BF16')
manifest = json.loads((source / 'fl2va/manifest.json').read_text())
cases = [
    ('rank', lambda j: j.update(rank=64), 'format/rank/alpha'),
    ('trunk', lambda j: j.update(trunk='transformer_ref'), 'task/trunk'),
    ('policy', lambda j: j.update(numerics_policy='fp32_conditioning_v1'), 'conditioning precision'),
    ('schedule', lambda j: j['sampling'].update(method='euler'), 'sampling schedule'),
    ('base', lambda j: j['base'].update(revision='wrong'), 'base revision'),
]
with tempfile.TemporaryDirectory(prefix='fastllm-pika-contract-') as tmp:
    root = Path(tmp)
    (root / 'fl2va').mkdir()
    (root / 'ref2va').symlink_to((source / 'ref2va').resolve(), target_is_directory=True)
    weights = root / 'fl2va/adapter.safetensors'
    weights.symlink_to((source / 'fl2va/adapter.safetensors').resolve())
    for name, mutate, expected in cases:
        changed = copy.deepcopy(manifest)
        mutate(changed)
        (root / 'fl2va/manifest.json').write_text(json.dumps(changed))
        reject(['--pika', str(root)], expected)
        print('PASS', name, flush=True)
    (root / 'fl2va/manifest.json').write_text(json.dumps(manifest))
    with weights.open('rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
        data_size = weights.stat().st_size - n - 8
    weights.unlink()
    key = next(k for k in header if k != '__metadata__')
    del header[key]
    body = json.dumps(header).encode()
    with weights.open('wb') as f:
        f.write(struct.pack('<Q', len(body)))
        f.write(body)
        f.truncate(8 + len(body) + data_size)
    reject(['--pika', str(root)], 'missing LoRA weights')
    print('PASS missing tensor', flush=True)
print('PASS Pika release contract')
