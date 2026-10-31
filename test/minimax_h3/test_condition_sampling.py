"""Native CPU audio resampling and FP16 visual posterior sampling parity."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import numpy as np
import torch
import torchaudio

p = argparse.ArgumentParser()
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 1
lib = ctypes.CDLL(a.library)
lib.h3_error.restype = ctypes.c_char_p
fp = ctypes.POINTER(ctypes.c_float)
lib.h3_resample_audio.argtypes = [fp, ctypes.c_int, ctypes.c_int, ctypes.c_int, fp]
lib.h3_sample_posterior.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, fp, fp]
report = {}


def compare(name, actual, expected, tolerance):
    delta = actual.astype(np.float64) - expected
    r = {
        "relative_l2": float(
            np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))
        ),
        "max_abs": float(np.abs(delta).max()),
    }
    report[name] = r
    assert r["relative_l2"] < tolerance, (name, r)


for rate, channels in ((32000, 1), (44100, 2), (48000, 1), (16000, 2)):
    n = rate * 2 + 13
    x = np.random.default_rng(42).normal(0, 0.1, (n, channels)).astype(np.float32)
    expected = torchaudio.transforms.Resample(rate, 32000)(torch.from_numpy(x.T.copy()))
    if channels == 1:
        expected = expected.expand(2, -1)
    expected = expected.T.numpy()
    actual = np.empty(expected.shape, dtype=np.float32)
    result = lib.h3_resample_audio(
        x.ctypes.data_as(fp), n, channels, rate, actual.ctypes.data_as(fp)
    )
    assert result == len(expected), lib.h3_error().decode()
    compare(f"audio_{rate}_{channels}", actual, expected, 1e-5)
for t, h, w in ((1, 4, 4), (2, 6, 8), (17, 8, 8)):
    moments = np.random.default_rng(42).normal(0, 0.2, (t, h, w, 48)).astype(np.float32)
    x = torch.from_numpy(moments).permute(3, 0, 1, 2).contiguous()
    mean, logvar = x.chunk(2, dim=0)
    noise = torch.randn(mean.shape, generator=torch.Generator().manual_seed(42))
    expected = (
        (mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise)
        .half()
        .float()
        .permute(1, 2, 3, 0)
        .numpy()
    )
    actual = np.empty(expected.shape, dtype=np.float32)
    assert (
        lib.h3_sample_posterior(
            t, h, w, moments.ctypes.data_as(fp), actual.ctypes.data_as(fp)
        )
        == 0
    ), lib.h3_error().decode()
    compare(f"posterior_{t}_{h}_{w}", actual, expected, 1e-5)
Path(a.output).write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
