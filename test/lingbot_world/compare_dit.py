"""Compare native LingBot DiT with the pinned official forward implementation.

PyTorch is used only by this test. Runtime inputs and outputs cross a float32
C ABI; the production shared library does not link a Python tensor runtime.
"""
import argparse
import ctypes
import gc
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import torch
from safetensors.torch import load_file


def official_module(root):
    # Import just the actual model source, without the unrelated CLI pipeline.
    for name, path in [('wan', root / 'wan'), ('wan.modules', root / 'wan/modules')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    return importlib.import_module('wan.modules.model_fast')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--attention', choices=['flash', 'vllm-fa2'], default='flash')
    parser.add_argument('--height', type=int, default=8)
    parser.add_argument('--width', type=int, default=8)
    parser.add_argument('--segments', type=int, default=4)
    parser.add_argument('--timestep', type=float, default=650.)
    parser.add_argument('--inputs', type=Path, help='Official T5/VAE/camera NPZ from prepare_inputs.py')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(83)
    if args.attention == 'vllm-fa2':
        # vLLM packages the same FlashAttention-2 API under a different module
        # name. The official attention wrapper and its dtype casts stay intact.
        import vllm.vllm_flash_attn as flash_attn
        sys.modules['flash_attn'] = flash_attn
    module = official_module(args.official)
    with torch.device('meta'):
        model = module.WanModelFast(model_type='i2v', patch_size=(1, 2, 2),
            text_len=512, in_dim=36, dim=1536, ffn_dim=8960, freq_dim=256,
            text_dim=4096, out_dim=16, num_heads=12, num_layers=30,
            local_attn_size=9, sink_size=3, qk_norm=True, cross_attn_norm=True,
            eps=1e-6)
    index = json.loads((args.weights / 'model.safetensors.index.json').read_text())
    state = {}
    for shard in sorted(set(index['weight_map'].values())):
        state.update(load_file(str(args.weights / shard)))
    model.load_state_dict(state, strict=True, assign=True)
    del state
    # Preserve the official checkpoint's FP32 parameters: the source explicitly
    # disables BF16 autocast for time_embedding, time_projection and the head.
    # Main projections still run in the official outer BF16 autocast context.
    model = model.eval().cuda()
    model.freqs = torch.cat([module.rope_params(1024, 44),
        module.rope_params(1024, 42), module.rope_params(1024, 42)], dim=1).cuda()
    inputs = np.load(args.inputs) if args.inputs else None
    f, h, w, nt = 3, args.height, args.width, 17
    if inputs is not None:
        _, h, w, _ = inputs['condition'].shape
        nt = len(inputs['text'])
        if len(inputs['condition']) < args.segments*f:
            raise ValueError('not enough prepared condition frames')
    n = f * h * w // 4
    cache = [dict(k=torch.zeros(1, n*3, 12, 128, device='cuda', dtype=torch.bfloat16),
        v=torch.zeros(1, n*3, 12, 128, device='cuda', dtype=torch.bfloat16),
        global_end_index=torch.tensor(0, device='cuda'),
        local_end_index=torch.tensor(0, device='cuda')) for _ in range(30)]
    cross = [dict(k=torch.zeros(1, 512, 12, 128, device='cuda', dtype=torch.bfloat16),
        v=torch.zeros(1, 512, 12, 128, device='cuda', dtype=torch.bfloat16),
        is_init=torch.tensor(0, device='cuda')) for _ in range(30)]
    text = torch.randn(nt, 4096).bfloat16() if inputs is None else torch.from_numpy(inputs['text']).bfloat16()
    fixtures = []
    for segment in range(args.segments):
        latent = torch.randn(f, h, w, 16)
        condition = torch.randn(f, h, w, 20)
        camera = torch.randn(f, h, w, 384).bfloat16() * .1
        if inputs is not None:
            condition = torch.from_numpy(inputs['condition'][segment*f:(segment+1)*f])
            camera = torch.from_numpy(inputs['camera'][segment*f:(segment+1)*f]).bfloat16()
            if 'latent' in inputs:
                latent = torch.from_numpy(inputs['latent'][segment*f:(segment+1)*f])
        for timestep, commit in [(args.timestep, False), (0., True)]:
            with torch.autocast('cuda', dtype=torch.bfloat16):
                result = model([latent.permute(3, 0, 1, 2).cuda()],
                    torch.tensor([timestep], device='cuda'), [text.cuda()], n,
                    y=[condition.permute(3, 0, 1, 2).cuda()],
                    dit_cond_dict={'c2ws_plucker_emb': [camera.permute(3, 0, 1, 2)[None].cuda()]},
                    kv_cache=cache, crossattn_cache=cross,
                    current_start=segment*n, max_attention_size=3*n,
                    frame_seqlen=h*w//4, cross_attn_first_call=segment==0 and not commit)[0]
            fixtures.append(dict(segment=segment, timestep=timestep, commit=commit,
                latent=latent.numpy(), condition=condition.numpy(), camera=camera.float().numpy(),
                reference=result.permute(1, 2, 3, 0).float().cpu().numpy()))
            print('reference', segment, timestep, flush=True)
    del model, cache, cross, result
    gc.collect()
    torch.cuda.empty_cache()
    lib = ctypes.CDLL(str(args.library))
    pointer = ctypes.POINTER(ctypes.c_float)
    lib.fastllm_lingbot_error.restype = ctypes.c_char_p
    lib.fastllm_lingbot_dit_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.fastllm_lingbot_dit_create.restype = ctypes.c_void_p
    lib.fastllm_lingbot_dit_destroy.argtypes = [ctypes.c_void_p]
    lib.fastllm_lingbot_dit_reset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    lib.fastllm_lingbot_dit_forward.argtypes = [ctypes.c_void_p] + [ctypes.c_int]*4 + [pointer]*4 + [ctypes.c_float, ctypes.c_int, pointer]
    handle = lib.fastllm_lingbot_dit_create(str(args.weights).encode(), 0)
    if not handle:
        raise RuntimeError(lib.fastllm_lingbot_error().decode())
    results = []
    try:
        assert lib.fastllm_lingbot_dit_reset(handle, 9, 3) == 0
        text_array = np.ascontiguousarray(text.float().numpy())
        for fixture in fixtures:
            arrays = [np.ascontiguousarray(fixture[key]) for key in ['latent', 'condition']]
            arrays += [text_array, np.ascontiguousarray(fixture['camera'])]
            output = np.empty(fixture['reference'].shape, dtype=np.float32, order='C')
            ret = lib.fastllm_lingbot_dit_forward(handle, f, h, w, nt,
                *[array.ctypes.data_as(pointer) for array in arrays], fixture['timestep'],
                fixture['commit'], output.ctypes.data_as(pointer))
            if ret:
                raise RuntimeError(lib.fastllm_lingbot_error().decode())
            expected = fixture['reference'].astype(np.float64)
            delta = output.astype(np.float64) - expected
            relative_l2 = float(np.linalg.norm(delta) / max(np.linalg.norm(expected), 1e-12))
            results.append(dict(segment=fixture['segment'], timestep=fixture['timestep'],
                commit=fixture['commit'], relative_l2=relative_l2,
                max_abs=float(np.max(np.abs(delta))), finite=bool(np.isfinite(output).all()),
                bf16_output=bool(((output.view(np.uint32) & 0xffff) == 0).all())))
            print(results[-1], flush=True)
    finally:
        lib.fastllm_lingbot_dit_destroy(handle)
    report = dict(attention=args.attention, fp32_accumulation=True, inputs=str(args.inputs) if args.inputs else 'random',
        shape=[f, h, w], results=results,
        passed=all(r['finite'] and r['bf16_output'] and r['relative_l2'] < .01 for r in results))
    args.output.write_text(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit('DiT relative L2 exceeded 0.01')


if __name__ == '__main__':
    main()
