"""One-time lossless conversion of official T5/VAE .pth files to safetensors.

Run in a preparation environment with torch and safetensors. Neither package
is imported or linked by native inference. Original files remain untouched.
"""
import argparse
import hashlib
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(16 << 20), b''):
            result.update(chunk)
    return result.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('directory', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    output = args.output or args.directory / 'native'
    output.mkdir(parents=True, exist_ok=True)
    report = {}
    for source, target in [('models_t5_umt5-xxl-enc-bf16.pth', 't5.safetensors'),
                           ('Wan2.1_VAE.pth', 'vae.safetensors')]:
        path = args.directory / source
        weights = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
        if 'state_dict' in weights:
            weights = weights['state_dict']
        tensors = {}
        for name, tensor in weights.items():
            if not isinstance(tensor, torch.Tensor):
                raise ValueError(f'Unexpected non-tensor checkpoint field: {name}')
            if tensor.dtype == torch.float16:
                tensor = tensor.float()  # exact widening
            if tensor.dtype not in (torch.float32, torch.bfloat16):
                raise ValueError(f'Unsupported dtype: {name}: {tensor.dtype}')
            tensors[name] = tensor.contiguous()
        destination = output / target
        temporary = destination.with_suffix('.tmp')
        save_file(tensors, str(temporary), metadata={'source': source, 'conversion': 'lossless'})
        temporary.replace(destination)
        report[target] = dict(source=source, source_sha256=digest(path),
            sha256=digest(destination), tensors=len(tensors),
            parameters=sum(t.numel() for t in tensors.values()),
            dtypes=sorted({str(t.dtype) for t in tensors.values()}))
        print(target, report[target], flush=True)
        del weights, tensors
    (output / 'conversion.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
