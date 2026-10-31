"""Compare native GPU noise with upstream CUDA randn and patchify.

Only this reference test imports torch. Build operator_bridge.cpp as documented
in compare_ops.py, then pass --library, --official and --report. --sglang may
point to SGLang's runtime/models/vaes/nomas.py for an additional source check.
"""
import argparse
import ast
import ctypes as C
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def function(path, name, owner=None):
    tree = ast.parse(path.read_text())
    scope = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == owner).body if owner else tree.body
    node = next(n for n in scope if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    # Execute the upstream body, without loading unrelated optional runtimes.
    env = dict(torch=torch, TUBELET=(4, 16, 16), TUBE_DIM=3072)
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), env)
    return env[name]


@torch.no_grad()
def main(args):
    torch.set_num_threads(2)
    lib = C.CDLL(str(args.library))
    lib.nomas_test_error.restype = C.c_char_p
    lib.nomas_test_create_precision.argtypes = [C.c_int]
    lib.nomas_test_create_precision.restype = C.c_void_p
    lib.nomas_test_destroy.argtypes = [C.c_void_p]
    fp = C.POINTER(C.c_float)
    lib.nomas_test_pixel_noise.argtypes = [C.c_void_p, C.c_uint64, C.c_int, C.c_int, C.c_int, C.c_int, fp]
    base = args.official / "hymm/models/vae/pixelumm_decoder"
    official = function(base / "autoencoder.py", "_make_noise", "PixelUMMDCAELatentDecoder")
    patchify = function(base / "model.py", "patchify")
    sglang = None
    if args.sglang:
        tree = ast.parse(args.sglang.read_text())
        owner = next(n.name for n in tree.body if isinstance(n, ast.ClassDef)
                     and any(isinstance(m, ast.FunctionDef) and m.name == "_noise" for m in n.body))
        sglang = function(args.sglang, "_noise", owner)
    report = dict(torch=torch.__version__, cuda=torch.version.cuda, cases=[], l2_limit=1e-6)
    cases = [(0, 0, 1, 96, 64), (42, 0, 5, 224, 384), (42, 0, 10, 224, 384),
             (42, 5, 10, 224, 384), (42, 45, 10, 224, 384), (7, 50, 10, 288, 512),
             (2**32+7, 123, 10, 512, 512), (17, 1000000, 3, 64, 96)]
    for fp32 in (False, True):
        ctx = lib.nomas_test_create_precision(int(fp32))
        if not ctx:
            raise RuntimeError(lib.nomas_test_error().decode())
        dtype = torch.float32 if fp32 else torch.bfloat16

        def native(seed, first, frames, height, width):
            actual = np.empty((frames*(height//16)*(width//16), 3072), dtype=np.float32)
            rc = lib.nomas_test_pixel_noise(ctx, seed, first, frames, height, width, actual.ctypes.data_as(fp))
            if rc:
                raise RuntimeError(lib.nomas_test_error().decode())
            return actual

        try:
            for seed, first, frames, height, width in cases:
                latent = torch.empty(1, 32, frames, height//32, width//32, device="cuda", dtype=dtype)
                noise = official(SimpleNamespace(seed=seed), latent, first, first+frames)
                if sglang:
                    state = SimpleNamespace(config=SimpleNamespace(arch_config=SimpleNamespace(seed=seed)))
                    assert torch.equal(noise, sglang(state, latent, first, first+frames))
                expected = patchify(noise.to(dtype)).float().cpu().numpy().reshape(-1, 3072)
                actual = native(seed, first, frames, height, width)
                delta = actual.astype(np.float64)-expected
                l2 = float(np.sqrt(np.sum(delta*delta)/np.sum(expected.astype(np.float64)**2)))
                row = dict(dtype=str(dtype), seed=seed, first=first, frames=frames, height=height, width=width,
                           relative_l2=l2, max_abs=float(np.max(np.abs(delta))),
                           different_bits=int(np.count_nonzero(actual.view(np.uint32)!=expected.view(np.uint32))),
                           finite=bool(np.isfinite(actual).all()), sglang_equal=bool(sglang),
                           passed=bool(np.isfinite(actual).all() and l2<=report["l2_limit"]))
                report["cases"].append(row)
                print(json.dumps(row), flush=True)
            whole = native(42, 0, 10, 224, 384)
            tail = native(42, 5, 5, 224, 384)
            assert np.array_equal(whole[len(tail):], tail), "overlapping noise changed"
            assert np.array_equal(tail, native(42, 5, 5, 224, 384)), "noise is not repeatable"
            assert not np.array_equal(tail, native(43, 5, 5, 224, 384)), "seed ignored"
            dummy = np.empty(1, dtype=np.float32)
            for first, frames, height, width in [(-1, 1, 64, 64), (0, 0, 64, 64), (0, 1, 63, 64),
                                                  (0, 65536, 64, 64), (0, 1, 2**30, 2**30)]:
                assert lib.nomas_test_pixel_noise(ctx, 42, first, frames, height, width, dummy.ctypes.data_as(fp)) != 0
        finally:
            lib.nomas_test_destroy(ctx)
    report["passed"] = all(row["passed"] for row in report["cases"])
    report["temporal_overlap_repeatability_invalid_inputs"] = "passed"
    args.report.write_text(json.dumps(report, indent=2)+"\n")
    if not report["passed"]:
        raise SystemExit("noise reference comparison failed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--sglang", type=Path)
    parser.add_argument("--report", type=Path, required=True)
    main(parser.parse_args())
