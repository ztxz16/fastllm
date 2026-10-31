"""Replay native noise through official sampling, KV history and VAE decoding.

Input conditions come from prepare_inputs.py. This checks whole rollouts with
identical stochastic inputs, not just isolated operators or a matching seed.
"""
import argparse
import gc
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from compare_dit import official_module
from compare_vae import metrics, thwc


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--official', type=Path, required=True)
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--inputs', type=Path, required=True)
    p.add_argument('--trace', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--offload-reference', action='store_true',
                   help='keep official DiT block weights on CPU between calls; arithmetic is unchanged')
    a = p.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    import vllm.vllm_flash_attn as flash_attn
    sys.modules['flash_attn'] = flash_attn
    module = official_module(a.official)
    package = types.ModuleType('wan.utils')
    package.__path__ = [str(a.official/'wan/utils')]
    sys.modules['wan.utils'] = package
    scheduler_module = importlib.import_module('wan.utils.fm_solvers_unipc')
    camera_module = importlib.import_module('wan.utils.cam_utils')
    meta = json.loads((a.trace/'metadata.json').read_text())
    inputs = np.load(a.inputs)
    f, h, w = 3, meta['height']//8, meta['width']//8
    n, nt = f*h*w//4, len(inputs['text'])
    def read(name, shape):
        return np.fromfile(a.trace/(name+'.f32'), np.float32).reshape(shape)
    def tensor(x):
        return torch.from_numpy(np.ascontiguousarray(x)).permute(3, 0, 1, 2).cuda()
    def record(name, value, expected):
        result = metrics(value, expected)
        results[name] = result
        print(name, result['relative_l2'], result['max_abs'], flush=True)
    results = {}
    record('text', read('0.text', (nt, 4096)), inputs['text'])
    scheduler = scheduler_module.FlowUniPCMultistepScheduler(num_train_timesteps=1000, shift=1, use_dynamic_shifting=False)
    scheduler.set_timesteps(1000, shift=5)
    timesteps = scheduler.timesteps[[0, 250, 500, 750]]
    schedule_ok = True
    for index, t in enumerate(timesteps):
        flow_index = torch.argmin((scheduler.timesteps-t).abs())
        noise_index = scheduler.index_for_timestep(t, scheduler.timesteps)
        entry = meta['schedule'][index]
        schedule_ok &= entry['timestep'] == int(t)
        schedule_ok &= np.float32(entry['flow_sigma']) == np.float32(scheduler.sigmas[flow_index])
        schedule_ok &= np.float32(entry['noise_sigma']) == np.float32(scheduler.sigmas[noise_index])
    print('schedule_exact', bool(schedule_ok), flush=True)
    with torch.device('meta'):
        model = module.WanModelFast(model_type='i2v', patch_size=(1, 2, 2), text_len=512,
            in_dim=36, dim=1536, ffn_dim=8960, freq_dim=256, text_dim=4096, out_dim=16,
            num_heads=12, num_layers=30, local_attn_size=18, sink_size=6, qk_norm=True,
            cross_attn_norm=True, eps=1e-6)
    index = json.loads((a.model/'transformers/model.safetensors.index.json').read_text())
    state = {}
    for shard in sorted(set(index['weight_map'].values())):
        state.update(load_file(str(a.model/'transformers'/shard)))
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.eval()
    if a.offload_reference:
        for name, child in model.named_children():
            if name != 'blocks':
                child.cuda()
        def load_block(block, args):
            block.cuda()
        def unload_block(block, args, output):
            block.cpu()
        for block in model.blocks:
            block.register_forward_pre_hook(load_block)
            block.register_forward_hook(unload_block)
    else:
        model.cuda()
    model.freqs = torch.cat([module.rope_params(1024, 44), module.rope_params(1024, 42), module.rope_params(1024, 42)], dim=1).cuda()
    cache = [dict(k=torch.zeros(1, n*6, 12, 128, device='cuda', dtype=torch.bfloat16),
        v=torch.zeros(1, n*6, 12, 128, device='cuda', dtype=torch.bfloat16),
        global_end_index=torch.tensor(0, device='cuda'), local_end_index=torch.tensor(0, device='cuda')) for _ in range(30)]
    cross = [dict(k=torch.zeros(1, 512, 12, 128, device='cuda', dtype=torch.bfloat16),
        v=torch.zeros(1, 512, 12, 128, device='cuda', dtype=torch.bfloat16), is_init=torch.tensor(0, device='cuda')) for _ in range(30)]
    text = torch.from_numpy(inputs['text']).bfloat16().cuda()
    final_latents = []
    pose = torch.eye(4, dtype=torch.float32)
    for segment, action in enumerate(meta['actions']):
        condition = inputs['condition'][segment*3:segment*3+3]
        record(f'{segment}.condition', read(f'{segment}.condition', condition.shape), condition)
        old = pose.clone()
        poses = []
        # Compose the same explicit local action; use official camera conversion.
        delta = torch.eye(4, dtype=torch.float32)
        if action in ['forward', 'backward', 'left', 'right', 'up', 'down']:
            axis = dict(forward=2, backward=2, left=0, right=0, up=1, down=1)[action]
            delta[axis, 3] = -.1 if action in ['backward', 'left', 'up'] else .1
        elif action in ['look_left', 'look_right', 'look_up', 'look_down']:
            angle = np.deg2rad(-4 if action in ['look_left', 'look_down'] else 4)
            c, s = float(np.cos(angle)), float(np.sin(angle))
            if action in ['look_left', 'look_right']:
                delta[:3, :3] = torch.tensor([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            else:
                delta[:3, :3] = torch.tensor([[1, 0, 0], [0, c, -s], [0, s, c]])
        elif action != 'stay':
            raise ValueError('unknown action')
        for frame in range(3):
            if segment or frame:
                pose = pose@delta
            poses.append(pose.clone())
        poses = torch.stack(([old] if segment else [])+poses)
        relative = camera_module.compute_relative_poses(poses, framewise=True)
        if segment:
            relative = relative[1:]
        k = torch.tensor(meta['intrinsics']).repeat(3, 1).cuda()
        camera = camera_module.get_plucker_embeddings(relative.cuda(), k, h*8, w*8)
        camera = camera.reshape(3, h, 8, w, 8, 6).permute(0, 1, 3, 5, 2, 4).reshape(3, h, w, 384).bfloat16()
        record(f'{segment}.camera', read(f'{segment}.camera', (3, h, w, 384)), camera.float().cpu().numpy())
        latent = tensor(read(f'{segment}.latent.0', (3, h, w, 16)))
        kwargs = dict(context=[text], seq_len=n, y=[tensor(condition)],
            dit_cond_dict={'c2ws_plucker_emb': [camera.permute(3, 0, 1, 2)[None]]},
            kv_cache=cache, crossattn_cache=cross, current_start=segment*n,
            max_attention_size=n*6, frame_seqlen=h*w//4)
        for step, t in enumerate(timesteps):
            with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=not a.offload_reference):
                flow = model([latent], t.reshape(1).cuda(), cross_attn_first_call=segment==0 and step==0, **kwargs)[0]
            sigma = scheduler.sigmas[torch.argmin((scheduler.timesteps-t).abs())].double().cuda()
            clean = (latent.double()-sigma*flow.double()).to(flow.dtype)
            record(f'{segment}.clean.{step}', read(f'{segment}.clean.{step}', (3, h, w, 16)), thwc(clean[None]))
            if step+1 < len(timesteps):
                noise = tensor(read(f'{segment}.noise.{step}', (3, h, w, 16)))
                latent = scheduler.add_noise(clean, noise, timesteps[step+1])
        with torch.autocast('cuda', dtype=torch.bfloat16, cache_enabled=not a.offload_reference):
            model([clean], torch.zeros(1, device='cuda'), cross_attn_first_call=False, **kwargs)
        final_latents.append(clean.cpu())
    del model, cache, cross, kwargs, latent, clean, flow
    gc.collect()
    torch.cuda.empty_cache()
    vae_module = importlib.import_module('wan.modules.vae2_1')
    vae = vae_module.Wan2_1_VAE(vae_pth=str(a.model/'Wan2.1_VAE.pth'))
    decode_precision = meta.get('vae_decode_precision', 'fp32')
    if decode_precision not in ('bf16', 'fp32'):
        raise ValueError('invalid trace decode precision')
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=decode_precision == 'bf16'):
        pixels = vae.model.decode(torch.cat(final_latents, dim=1)[None].cuda(), vae.scale).float().clamp(-1, 1)
    pixels = thwc(pixels)
    start = 0
    for segment in range(len(meta['actions'])):
        frames = 9 if segment == 0 else 12
        reference = pixels[start:start+frames]
        record(f'{segment}.pixels', read(f'{segment}.pixels', reference.shape), reference)
        start += frames
    # Whole denoising gates remain separate from the more sensitive pixel decoder.
    component_ok = all(r['finite'] and r['relative_l2'] < .01 for name, r in results.items() if not name.endswith('.pixels'))
    pixels_ok = all(r['finite'] and r['relative_l2'] < .03 for name, r in results.items() if name.endswith('.pixels'))
    report = dict(decode_precision=decode_precision, schedule_exact=bool(schedule_ok), reference_weight_offload=a.offload_reference,
                  results=results, passed=bool(schedule_ok and component_ok and pixels_ok))
    a.output.write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('LingBot rollout precision gate failed')


if __name__ == '__main__':
    main()
