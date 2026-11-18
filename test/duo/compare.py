"""Compare each native latent chunk with its reference, in float64."""

import argparse
import json
import math
from pathlib import Path

import numpy as np


def tolerance(value):
    limit = float(value)
    if not math.isfinite(limit) or limit < 0:
        raise argparse.ArgumentTypeError("expected a finite nonnegative tolerance")
    return limit


parser = argparse.ArgumentParser()
parser.add_argument("reference", type=Path)
parser.add_argument("native", type=Path)
parser.add_argument("--max-absolute", type=tolerance)
parser.add_argument("--max-relative-l2", "--max-relative-rms", type=tolerance)
args = parser.parse_args()

manifest = args.reference / "complete.json"
fp8 = (
    manifest.is_file()
    and json.loads(manifest.read_text())["format"] == "duo-release-modelopt-v1"
)
l2_limit = args.max_relative_l2
if l2_limit is None:
    l2_limit = 0.001 if fp8 else 0.0005
absolute_limit = args.max_absolute
if absolute_limit is None and not fp8:
    absolute_limit = 0.001

results = []
failures = []
for k in range(3):
    for kind in ("video", "audio"):
        reference = np.frombuffer(
            (args.reference / f"expected-{k}.{kind}.f32").read_bytes(), dtype="<f4"
        )
        native = np.frombuffer(
            (args.native / f"chunk-{k}.{kind}.f32").read_bytes(), dtype="<f4"
        )
        label = f"chunk {k} {kind}"
        if not reference.size or native.shape != reference.shape:
            raise SystemExit(f"{label}: empty or mismatched tensors")
        if not np.isfinite(reference).all() or not np.isfinite(native).all():
            raise SystemExit(f"{label}: nonfinite tensor")

        reference = reference.astype(np.float64)
        diff = native - reference
        absolute = float(np.abs(diff).max())
        norm = np.linalg.norm(reference)
        if norm == 0 and absolute != 0:
            raise SystemExit(f"{label}: nonzero output for an all-zero reference")
        l2 = float(np.linalg.norm(diff) / norm) if norm else 0.0
        # The ratio of RMS values equals relative L2 for equal-sized tensors.
        results.append(
            {
                "chunk": k,
                "kind": kind,
                "max_absolute": absolute,
                "relative_l2": l2,
                "relative_rms": l2,
            }
        )
        if l2 > l2_limit:
            failures.append(f"{label}: relative L2 {l2:.6g} exceeds {l2_limit:g}")
        if absolute_limit is not None and absolute > absolute_limit:
            failures.append(
                f"{label}: max absolute {absolute:.6g} exceeds {absolute_limit:g}"
            )

print(json.dumps(results, indent=2, allow_nan=False))
if failures:
    raise SystemExit("\n".join(failures))
