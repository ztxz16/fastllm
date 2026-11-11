"""Compare tiled reference-image encoding and posterior sampling with official weights.

Torch/Diffusers are reference-test dependencies only; native inference has neither.
"""
import argparse
import gc
import importlib.util
import json
from pathlib import Path
import subprocess
import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from diffusers import AutoencoderKLMiniMaxH3Image

p = argparse.ArgumentParser()
for name in ('model', 'binary', 'diffusers', 'output'):
    p.add_argument('--' + name, required=True)
a = p.parse_args()
assert torch.cuda.device_count() == 1, 'Expose only the permitted GPU'
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
root, out = Path(a.model), Path(a.output)
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('converter', Path(a.diffusers) / 'scripts/convert_minimax_h3_to_diffusers.py')
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)
with torch.device('meta'):
    model = AutoencoderKLMiniMaxH3Image.from_config(conv.MINIMAX_H3_IMAGE_VAE_CONFIG)
state = {}
with safe_open(str(root / 'vae/diffusion_pytorch_model.safetensors'), framework='pt') as f:
    for key in f.keys():
        if key.startswith(('encoder.', 'quant_conv.')):
            for name, value in conv.convert_video_vae_key(key, f.get_tensor(key), conv.MINIMAX_H3_IMAGE_VAE_CONFIG):
                state[name] = value.float().cuda()
missing = model.load_state_dict(state, assign=True, strict=False)
assert not missing.unexpected_keys and all(k.startswith(('decoder.', 'post_quant_conv.')) for k in missing.missing_keys)
del state
model.eval()
# Asymmetric multi-tile grid and non-patch-aligned final latent dimension exercise stitching/cropping.
y, x = np.mgrid[:272, :400]
pixels = np.stack([(x * 3 + y) % 256, (x + y * 2) % 256, (x // 2 + y // 3) % 256], -1).astype(np.uint8)
Image.fromarray(pixels).save(out / 'input.png')
image = torch.from_numpy(pixels.copy()).permute(2, 0, 1)[None].cuda().float() / 255
image = (image - torch.tensor([.485, .456, .406], device='cuda')[None, :, None, None]) / torch.tensor([.229, .224, .225], device='cuda')[None, :, None, None]
with torch.inference_mode():
    distribution = model.encode(image).latent_dist
    # Match official CPU generator sampling, then FP16 rounding before normalization.
    noise = torch.randn(distribution.mean.shape, generator=torch.Generator().manual_seed(42)).cuda()
    latent = (distribution.mean + distribution.std * noise).half().float()
    latent = latent[:, :, :latent.shape[2] // 2 * 2, :latent.shape[3] // 2 * 2]
    cfg = json.loads((root / 'vae/config.json').read_text())
    mean = torch.tensor(cfg['latents_mean'], device='cuda')[None, :, None, None]
    std = torch.tensor(cfg['latents_std'], device='cuda')[None, :, None, None]
    reference = ((latent - mean) / std).cpu().numpy().reshape(-1)
del model, distribution, image, latent, noise, value
gc.collect()
torch.cuda.empty_cache()
subprocess.run([a.binary, '--model', a.model, '--encode-image', str(out / 'input.png'),
                '--seed', '42', '--output', str(out / 'native.f32')], check=True)
actual = np.fromfile(out / 'native.f32', np.float32)
assert actual.shape == reference.shape and np.isfinite(actual).all()
diff = actual.astype(np.float64) - reference
metrics = {'relative_rms': float(np.linalg.norm(diff) / np.linalg.norm(reference.astype(np.float64))),
           'max_abs': float(np.abs(diff).max()), 'values': len(actual)}
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
print(json.dumps(metrics, indent=2))
assert metrics['relative_rms'] < .002, metrics

