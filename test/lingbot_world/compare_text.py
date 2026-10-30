"""Official UMT5 differential test; dependencies are only for validation."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--prompt-file', type=Path, action='append', default=[],
                        help='Additional UTF-8 prompt files')
    parser.add_argument('--token-boundaries', action='store_true',
                        help='Exercise mask/reduction boundaries from 31 to 512 valid tokens')
    args = parser.parse_args()
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    for name, path in [('wan', args.official/'wan'), ('wan.modules', args.official/'wan/modules')]:
        package = types.ModuleType(name)
        package.__path__ = [str(path)]
        sys.modules[name] = package
    module = importlib.import_module('wan.modules.t5')
    model = module.umt5_xxl(encoder_only=True, return_tokenizer=False,
                           dtype=torch.bfloat16, device='meta')
    state = torch.load(args.model/'models_t5_umt5-xxl-enc-bf16.pth', map_location='cpu', weights_only=True, mmap=True)
    model.load_state_dict(state, strict=True, assign=True)
    del state
    model = model.eval().cuda()
    tokenizer = module.HuggingfaceTokenizer(str(args.model/'google/umt5-xxl'), seq_len=512, clean='whitespace')
    fixtures = []
    prompts = [p.read_text().strip() for p in args.prompt_file]
    prompts += ['A tree beside a lake.', '房间里有一张桌子，窗外是晴朗的天空。', '']
    cases = [(prompt, None) for prompt in prompts]
    if args.token_boundaries:
        cases += [(f'token_boundary_{count}', count)
                  for count in [31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512]]
    for prompt, length in cases:
        if length is None:
            ids, mask = tokenizer(prompt, return_mask=True, add_special_tokens=True)
        else:
            ids = torch.zeros(1, 512, dtype=torch.int64)
            pattern = torch.tensor([320, 37703, 157763, 289, 102544, 274])
            ids[0, :length-1] = pattern.repeat((length+5)//6)[:length-1]
            ids[0, length-1] = 1
            mask = torch.zeros_like(ids)
            mask[0, :length] = 1
        count = int(mask.sum())
        features = model(ids.cuda(), mask.cuda())[0, :count].float().cpu().numpy()
        fixtures.append((prompt, ids[0, :count].int().numpy(), features))
        print('reference', prompt, count, flush=True)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    lib = ctypes.CDLL(str(args.library))
    lib.fastllm_lingbot_error.restype = ctypes.c_char_p
    lib.fastllm_lingbot_text_create.argtypes = [ctypes.c_char_p, ctypes.c_int]
    lib.fastllm_lingbot_text_create.restype = ctypes.c_void_p
    lib.fastllm_lingbot_text_destroy.argtypes = [ctypes.c_void_p]
    lib.fastllm_lingbot_text_encode.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.POINTER(ctypes.c_float)]
    handle = lib.fastllm_lingbot_text_create(str(args.model/'native/t5.safetensors').encode(), 0)
    if not handle:
        raise RuntimeError(lib.fastllm_lingbot_error().decode())
    results = []
    try:
        for prompt, ids, features in fixtures:
            output = np.empty(features.shape, dtype=np.float32)
            ret = lib.fastllm_lingbot_text_encode(handle, ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                len(ids), output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)))
            if ret:
                raise RuntimeError(lib.fastllm_lingbot_error().decode())
            delta = output.astype(np.float64)-features.astype(np.float64)
            result = dict(prompt=prompt, tokens=ids.tolist(), relative_l2=float(np.linalg.norm(delta)/np.linalg.norm(features.astype(np.float64))),
                max_abs=float(np.abs(delta).max()), finite=bool(np.isfinite(output).all()))
            results.append(result)
            print(result, flush=True)
    finally:
        lib.fastllm_lingbot_text_destroy(handle)
    report = dict(results=results, passed=all(r['finite'] and r['relative_l2']<.01 for r in results))
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    if not report['passed']:
        raise SystemExit('T5 relative L2 exceeded 0.01')


if __name__ == '__main__':
    main()
