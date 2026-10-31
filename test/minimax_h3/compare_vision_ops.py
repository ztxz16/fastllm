"""Feed each BF16 vision operator its native input to isolate arithmetic drift."""

import argparse
import ctypes
import json
import os
import torch
from pathlib import Path
import numpy as np

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model/text_encoder")
p.add_argument("--fixture", required=True)
p.add_argument("--library", required=True)
a = p.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 1
root = Path(a.fixture)
cfg = json.loads((root / "config.json").read_text())
lib = ctypes.CDLL(a.library)
lib.h3_error.restype = ctypes.c_char_p
fp = ctypes.POINTER(ctypes.c_float)
lib.h3_vision_operator.argtypes = [
    ctypes.c_char_p,
    ctypes.c_char_p,
    ctypes.c_int,
    ctypes.c_int,
    fp,
    ctypes.c_int,
    ctypes.c_int,
    fp,
]
report = {}
for name, source, columns in [
    ("norm1", "input.0", 1152),
    ("qkv", "norm1", 1152),
    ("attention", "qkv", 3456),
    ("projected", "attention", 1152),
    ("norm2", "residual", 1152),
    ("fc1", "norm2", 1152),
    ("gelu", "fc1", 4304),
    ("fc2", "gelu", 4304),
]:
    x = np.fromfile(root / f"vision.{source}.reference.f32", dtype=np.float32)
    expected = np.fromfile(root / f"vision.{name}.reference.f32", dtype=np.float32)
    actual = np.empty_like(expected)
    assert (
        lib.h3_vision_operator(
            a.model.encode(),
            name.encode(),
            len(x) // columns,
            columns,
            x.ctypes.data_as(fp),
            cfg["height"],
            cfg["width"],
            actual.ctypes.data_as(fp),
        )
        == 0
    ), lib.h3_error().decode()
    delta = actual.astype(np.float64) - expected
    report[name] = {
        "relative_l2": float(
            np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64))
        ),
        "max_abs": float(np.abs(delta).max()),
        "equal_fraction": float(np.mean(actual == expected)),
    }
    actual.tofile(root / f"isolated.{name}.f32")
(root / "operator-report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert max(x["relative_l2"] for x in report.values()) < 0.01, report
