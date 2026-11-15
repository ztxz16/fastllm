"""Compare native 2D ViT decoding against official Diffusers on real weights.

Requires Diffusers main including scripts/convert_minimax_h3_to_diffusers.py.
Expose one unused GPU; this script never chooses a device for other users.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
from PIL import Image
from safetensors import safe_open
from diffusers import AutoencoderKLMiniMaxH3Image
from diffusers.models.autoencoders.autoencoder_kl_minimax_h3_image import MiniMaxH3ImageViTDecoder2d

p = argparse.ArgumentParser()
p.add_argument('--model', required=True)
p.add_argument('--binary', required=True)
p.add_argument('--diffusers', required=True)
p.add_argument('--output', required=True)
p.add_argument('--precision', choices=['fp32', 'bf16'], default='fp32')
p.add_argument('--tiled', action='store_true')
a = p.parse_args()
assert torch.cuda.device_count() == 1, 'Expose only the permitted GPU'
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
dtype = torch.float32 if a.precision == 'fp32' else torch.bfloat16
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
spec = importlib.util.spec_from_file_location('converter', Path(a.diffusers) / 'scripts/convert_minimax_h3_to_diffusers.py')
conv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(conv)
model = MiniMaxH3ImageViTDecoder2d()
state, post = {}, {}
with safe_open(str(Path(a.model) / 'vae/diffusion_pytorch_model.safetensors'), framework='pt') as f:
    for key in f.keys():
        if key.startswith('decoder.'):
            for name, tensor in conv.convert_video_vae_key(key, f.get_tensor(key), conv.MINIMAX_H3_IMAGE_VAE_CONFIG):
                state[name.removeprefix('decoder.')] = tensor
        elif key.startswith('post_quant_conv.'):
            post[key.removeprefix('post_quant_conv.')] = f.get_tensor(key)
model.load_state_dict(state, strict=True)
del state
model.to(device='cuda', dtype=dtype).eval()
for name, module in model.named_modules():
    if any(part in name for part in ['norm1', 'norm2', 'norm_out']):
        module.float()
for name, parameter in model.named_parameters():
    if name.endswith(('scale1', 'scale2')):
        parameter.data = parameter.data.float()
post_conv = torch.nn.Conv2d(24, 24, 1)
post_conv.load_state_dict(post)
post_conv.to(device='cuda', dtype=dtype).eval()
values = {}
for i, block in enumerate(model.transformer_blocks):
    block.register_forward_hook(lambda m, ins, result, i=i: values.__setitem__(f'block.{i}', result.float().cpu().numpy().reshape(-1)))
h, w = 8, 12  # Non-square positions catch axis-order and rotary-width errors.
latent = torch.randn(1, 24, h, w, generator=torch.Generator().manual_seed(123))
latent.permute(0, 2, 3, 1).numpy().tofile(out / 'latent.f32')
with torch.inference_mode():
    decoded = model(post_conv(latent.to('cuda', dtype)))
    packed = decoded.reshape(1, 3, h, 16, w, 16).permute(0, 2, 4, 1, 3, 5)
    values['output'] = packed.float().cpu().numpy().reshape(-1)
    if a.tiled:
        with torch.device('meta'):
            tiled_model = AutoencoderKLMiniMaxH3Image()
        tiled_model.decoder, tiled_model.post_quant_conv = model, post_conv
        normalized = torch.randn(1, 24, 24, 32, generator=torch.Generator().manual_seed(999))
        normalized.reshape(1, 24, 12, 2, 16, 2).permute(0, 2, 4, 1, 3, 5).contiguous().numpy().tofile(out / 'packed.f32')
        cfg = json.loads((Path(a.model) / 'vae/config.json').read_text())
        mean = torch.tensor(cfg['latents_mean']).view(1, 24, 1, 1)
        std = torch.tensor(cfg['latents_std']).view(1, 24, 1, 1)
        # Remove hooks so the single-tile block references remain intact.
        for block in model.transformer_blocks:
            block._forward_hooks.clear()
        image = tiled_model._decode((normalized * std + mean).to('cuda', dtype)).float()
        rgb_mean = torch.tensor([.485, .456, .406], device='cuda').view(1, 3, 1, 1)
        rgb_std = torch.tensor([.229, .224, .225], device='cuda').view(1, 3, 1, 1)
        image = ((image * rgb_std + rgb_mean).clamp(0, 1) * 255).byte()
        tiled_reference = image[0].permute(1, 2, 0).cpu().numpy()
        Image.fromarray(tiled_reference).save(out / 'tiled-reference.png')
        del tiled_model, image
del model, post_conv, decoded, packed
torch.cuda.empty_cache()
subprocess.run([a.binary, '--model', a.model, '--vae-tile', str(out / 'latent.f32'),
                '--width', str(w * 16), '--height', str(h * 16), '--vae-precision', a.precision,
                '--output', str(out / 'output.f32'), '--trace', str(out / 'native'), '--trace-blocks'], check=True)
metrics = {}
for name, reference in values.items():
    path = out / 'output.f32' if name == 'output' else out / 'native' / (name + '.f32')
    actual = np.fromfile(path, np.float32)
    assert actual.shape == reference.shape and np.isfinite(actual).all()
    diff = actual.astype(np.float64) - reference
    relative = float(np.sqrt(np.mean(diff ** 2)) / max(np.sqrt(np.mean(reference.astype(np.float64) ** 2)), 1e-8))
    metrics[name] = {'max_abs': float(np.abs(diff).max()), 'relative_rms': relative}
print(json.dumps(metrics, indent=2))
(out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
assert metrics['output']['relative_rms'] < (0.002 if dtype == torch.float32 else 0.06), metrics['output']
if a.tiled:
    subprocess.run([a.binary, '--model', a.model, '--decode-latents', str(out / 'packed.f32'),
                    '--width', '512', '--height', '384', '--vae-precision', a.precision,
                    '--output', str(out / 'tiled-native.png')], check=True)
    actual = np.array(Image.open(out / 'tiled-native.png').convert('RGB'))
    difference = np.abs(actual.astype(np.float64) - tiled_reference)
    metrics['tiled_rgb'] = {'mean_abs_8bit': float(difference.mean()), 'max_abs_8bit': float(difference.max())}
    print(json.dumps(metrics['tiled_rgb'], indent=2))
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2))
    assert difference.mean() < (0.02 if dtype == torch.float32 else 2.0), metrics['tiled_rgb']
