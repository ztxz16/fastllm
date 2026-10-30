"""Wan2.1 VAE differential and streaming tests against the official source.

Torch is confined to this reference test. The native library uses CUDA/cuBLAS.
"""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch


def thwc(x):
    return x.squeeze(0).permute(1, 2, 3, 0).float().cpu().contiguous().numpy()


def metrics(a, b):
    delta = a.astype(np.float64) - b.astype(np.float64)
    mse = np.mean(delta ** 2)
    return dict(relative_l2=float(np.linalg.norm(delta) / max(np.linalg.norm(b.astype(np.float64)), 1e-12)),
                max_abs=float(np.abs(delta).max()), finite=bool(np.isfinite(a).all()),
                psnr_db=float(10 * np.log10(4 / max(mse, 1e-30))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--condition-inputs', type=Path,
                        help='prepare_inputs NPZ: encode its first image followed by zero frames')
    parser.add_argument('--height', type=int, default=64)
    parser.add_argument('--width', type=int, default=64)
    parser.add_argument('--frames', type=int, default=9)
    args = parser.parse_args()
    if args.frames < 5 or (args.frames-1) % 4:
        parser.error('frames must be 1+4k, k>=1')
    if args.height % 8 or args.width % 8:
        parser.error('dimensions must be multiples of 8')
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    spec = importlib.util.spec_from_file_location('wan_vae_reference', args.official/'wan/modules/vae2_1.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    vae = module.Wan2_1_VAE(vae_pth=str(args.model/'Wan2.1_VAE.pth'))
    gen = torch.Generator(device='cuda').manual_seed(981)
    if args.condition_inputs:
        image = np.load(args.condition_inputs)['first_image']
        args.height, args.width = image.shape[:2]
        pixels = torch.zeros((1, 3, args.frames, args.height, args.width), device='cuda')
        pixels[0, :, 0] = torch.from_numpy(image).permute(2, 0, 1).cuda()
    else:
        pixels = torch.rand((1, 3, args.frames, args.height, args.width), generator=gen, device='cuda')*2-1
    encoded = vae.model.encode(pixels, vae.scale)
    decoded = vae.model.decode(encoded, vae.scale).clamp(-1, 1)
    pixels, encoded, decoded = map(thwc, [pixels, encoded, decoded])
    del vae
    torch.cuda.empty_cache()
    print('reference ready', pixels.shape, encoded.shape, flush=True)

    lib = ctypes.CDLL(str(args.library))
    ptr = ctypes.POINTER(ctypes.c_float)
    lib.fastllm_lingbot_error.restype = ctypes.c_char_p
    lib.fastllm_lingbot_vae_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.fastllm_lingbot_vae_create.restype = ctypes.c_void_p
    lib.fastllm_lingbot_vae_destroy.argtypes = [ctypes.c_void_p]
    lib.fastllm_lingbot_vae_reset.argtypes = [ctypes.c_void_p]
    for op in ['encode', 'decode']:
        getattr(lib, 'fastllm_lingbot_vae_'+op).argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ptr, ptr, ctypes.c_size_t]
    handle = lib.fastllm_lingbot_vae_create(str(args.model/'native/vae.safetensors').encode(), 0)
    if not handle:
        raise RuntimeError(lib.fastllm_lingbot_error().decode())
    def reset():
        if lib.fastllm_lingbot_vae_reset(handle):
            raise RuntimeError(lib.fastllm_lingbot_error().decode())
    def run(op, value, shape):
        out = np.empty(shape, dtype=np.float32)
        count = getattr(lib, 'fastllm_lingbot_vae_'+op)(handle, *value.shape[:3],
            value.ctypes.data_as(ptr), out.ctypes.data_as(ptr), out.size)
        if count != shape[0]:
            raise RuntimeError(f'{op}: expected {shape[0]} frames, got {count}: '+lib.fastllm_lingbot_error().decode())
        return out
    results = {}
    try:
        # Compare each component with identical inputs, then the complete round trip.
        native_encoded = run('encode', pixels, encoded.shape)
        native_decoded = run('decode', encoded, decoded.shape)
        results['encode'] = metrics(native_encoded, encoded)
        results['encode']['different_bf16_values'] = int((torch.from_numpy(native_encoded).bfloat16() !=
                                                        torch.from_numpy(encoded).bfloat16()).sum())
        results['decode'] = metrics(native_decoded, decoded)
        reset()
        roundtrip = run('decode', native_encoded, decoded.shape)
        results['roundtrip'] = metrics(roundtrip, decoded)
        reset()
        enc_chunks = [run('encode', pixels[:1], encoded[:1].shape)]
        for start in range(1, len(pixels), 4):
            enc_chunks.append(run('encode', pixels[start:start+4], encoded[:1].shape))
        results['stream_encode'] = metrics(np.concatenate(enc_chunks), encoded)
        results['encode_partition'] = metrics(np.concatenate(enc_chunks), native_encoded)
        dec_chunks = []
        for index in range(len(encoded)):
            frames = 1 if index == 0 else 4
            dec_chunks.append(run('decode', encoded[index:index+1], (frames, *decoded.shape[1:])))
        results['stream_decode'] = metrics(np.concatenate(dec_chunks), decoded)
        results['decode_partition'] = metrics(np.concatenate(dec_chunks), native_decoded)
    finally:
        lib.fastllm_lingbot_vae_destroy(handle)
    passed = all(v['finite'] and v['relative_l2'] < .001 for v in results.values())
    report = dict(shape=pixels.shape, results=results, passed=passed)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit('Wan VAE relative L2 exceeded 0.001')


if __name__ == '__main__':
    main()
