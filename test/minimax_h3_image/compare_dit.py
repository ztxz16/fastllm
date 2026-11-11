"""Six-step PDD differential test against official Diffusers using native weights.

Loads the reference on one exposed GPU, releases it, then runs the C++ binary.
The 128x192 canvas is a numerical fixture, not a quality benchmark.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from safetensors import safe_open
from diffusers import MiniMaxH3ImageTransformer2DModel, MiniMaxH3Scheduler
from diffusers.models.transformers.transformer_minimax_h3_image import MiniMaxH3ImageRotaryPosEmbed
from diffusers.modular_pipelines.minimax_h3_image.before_denoise import build_image_packed_sequence

p = argparse.ArgumentParser()
for name in ['model', 'binary', 'diffusers', 'output']:
    p.add_argument('--' + name, required=True)
p.add_argument('--condition-binary', help='Optional condition_fixture executable to test mixed tags and two references')
a = p.parse_args()
assert torch.cuda.device_count() == 1, 'Expose only the permitted GPU'
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
root, out = Path(a.model), Path(a.output)
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('converter', Path(a.diffusers) / 'scripts/convert_minimax_h3_to_diffusers.py')
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)
cfg = conv.MINIMAX_H3_IMAGE_TRANSFORMER_CONFIG
with torch.device('meta'):
    model = MiniMaxH3ImageTransformer2DModel.from_config(cfg)
index = json.loads((root / 'transformer/model.safetensors.index.json').read_text())['weight_map']
state = {}
for shard in sorted(set(index.values())):
    print('load', shard, flush=True)
    with safe_open(str(root / 'transformer' / shard), framework='pt') as f:
        for key in f.keys():
            tensor = f.get_tensor(key)
            if key.endswith('.attn.qkv_proj.weight'):
                tensor = conv.reorder_interleaved_qkv(tensor, 56, 128)
            for name, value in conv.convert_transformer_key(key, tensor, cfg):
                state[name] = value.to('cuda')
for name, value in conv.convert_pdd_heads(str(root), cfg).items():
    state[name] = value.to('cuda')
model.load_state_dict(state, assign=True, strict=True)
del state, tensor, value
model.rope = MiniMaxH3ImageRotaryPosEmbed(16, 10000).cuda()
model.eval()
nt, h, w = 11, 128, 192
generator = torch.Generator().manual_seed(888)
text = (torch.randn(1, nt, 5120, generator=generator) * .1).bfloat16().cuda()
latent = torch.randn(1, (h // 32) * (w // 32), 96, generator=generator).cuda()
text.float().cpu().numpy().tofile(out / 'text.f32')
latent.cpu().numpy().tofile(out / 'noise.f32')
grids = [(4, 6), (6, 4)] if a.condition_binary else []
text_tags = torch.tensor([1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1]) if grids else torch.ones(nt)
positions, tags, image_idx, text_idx, nrefs = build_image_packed_sequence(text_tags, grids, h // 16, w // 16, (1, 2, 2), 0)
refs = torch.randn(1, nrefs, 96, generator=generator).cuda()
refs.cpu().numpy().tofile(out / 'references.f32')
positions, tags, image_idx, text_idx = [x.cuda() for x in [positions, tags, image_idx, text_idx]]
scheduler = MiniMaxH3Scheduler(shift_provided_sigmas=False, euler_update_form='velocity')
sigmas = json.loads((root / 'pdd_config.json').read_text())['sigmas']
scheduler.set_timesteps(sigmas=sigmas, device='cuda')
values = {}
step_index = 0
for i, block in enumerate(model.transformer_blocks):
    def hook(m, ins, result, i=i):
        if i in [0, 24, 49]:
            values[f'step.{step_index}.block.{i}'] = result.float().cpu().numpy().reshape(-1)
    block.register_forward_hook(hook)
with torch.inference_mode():
    for step_index, time in enumerate(scheduler.timesteps):
        row_times = torch.full((len(tags),), time.item())
        row_times[nt:nt+nrefs] = .999
        times, indices = torch.unique(row_times, sorted=True, return_inverse=True)
        velocity = model(torch.cat([refs, latent], 1), text, times.cuda(), indices.cuda(), tags, positions, image_idx, text_idx, step_index).sample[:, nrefs:]
        latent = scheduler.step(velocity, time, latent, return_dict=False)[0]
        values[f'latent.{step_index}'] = latent.cpu().numpy().reshape(-1)
        print('reference step', step_index + 1, flush=True)
for name, value in values.items():
    value.tofile(out / (name + '.reference.f32'))
del model, block, text, latent, velocity
gc.collect()
torch.cuda.empty_cache()
subprocess.run([a.condition_binary, a.model, str(out)] if a.condition_binary else [a.binary, '--model', a.model, '--prompt', 'fixture', '--width', str(w), '--height', str(h),
                '--text-in', str(out / 'text.f32'), '--noise-in', str(out / 'noise.f32'), '--skip-decode',
                '--trace', str(out / 'native'), '--trace-blocks'], check=True)
metrics = {}
for name, reference in values.items():
    actual = np.fromfile(out / 'native' / (name + '.f32'), np.float32)
    assert actual.shape == reference.shape and np.isfinite(actual).all()
    diff = actual.astype(np.float64) - reference
    metrics[name] = {
        'max_abs': float(np.abs(diff).max()),
        'relative_rms': float(np.sqrt(np.mean(diff ** 2)) / max(np.sqrt(np.mean(reference.astype(np.float64) ** 2)), 1e-8)),
        'cosine': float(np.dot(actual.astype(np.float64), reference) / np.linalg.norm(actual.astype(np.float64)) / np.linalg.norm(reference.astype(np.float64))),
    }
print(json.dumps(metrics, indent=2))
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
assert metrics['latent.5']['relative_rms'] < .08 and metrics['latent.5']['cosine'] > .995, metrics['latent.5']
