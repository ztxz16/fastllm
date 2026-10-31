"""CPU-only native/reference checks for condition layouts and random draws."""

import argparse
import ctypes
import json
from pathlib import Path
import numpy as np
import torch
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
)

p = argparse.ArgumentParser()
p.add_argument("--library", required=True)
p.add_argument("--output", required=True)
a = p.parse_args()
lib = ctypes.CDLL(a.library)
fp, ip = ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int)
lib.h3_error.restype = ctypes.c_char_p
lib.h3_condition_layout.argtypes = [
    ctypes.c_int,
    ip,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ip,
    fp,
    ip,
    ip,
    ip,
]
lib.h3_normal_noise.argtypes = [ctypes.c_uint64, ctypes.c_int, fp]
checks = 0
for frames, h, w in ((2, 4, 4), (37, 28, 48), (102, 48, 84)):
    for anchors in ([], [0], [-1], [0, -1]):
        tags = np.array([1, 1, 0, 0, 0, 1, 1], dtype=np.int32)
        refs = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.from_numpy(tags),
            frames,
            h,
            w,
            207,
            (1, 2, 2),
            2,
            2,
            0,
            tuple("first" if x == 0 else "last" for x in anchors),
        )
        positions = np.empty(refs[0].shape, dtype=np.float32)
        result_tags = np.empty(refs[1].shape, dtype=np.int32)
        video = np.empty(refs[2].shape, dtype=np.int32)
        audio = np.empty(refs[3].shape, dtype=np.int32)
        anchor = np.array(anchors, dtype=np.int32)
        result = lib.h3_condition_layout(
            len(tags),
            tags.ctypes.data_as(ip),
            frames,
            h,
            w,
            207,
            len(anchors),
            anchor.ctypes.data_as(ip),
            positions.ctypes.data_as(fp),
            result_tags.ctypes.data_as(ip),
            video.ctypes.data_as(ip),
            audio.ctypes.data_as(ip),
        )
        assert result == len(result_tags), lib.h3_error().decode()
        for actual, expected in zip((positions, result_tags, video, audio), refs[:4]):
            np.testing.assert_array_equal(actual, expected.numpy().astype(actual.dtype))
        checks += 1
noise = {}
for count in (16, 24, 384, 6144, 2193):
    for seed in (0, 42, 987654321):
        expected = torch.randn(
            count, generator=torch.Generator().manual_seed(seed)
        ).numpy()
        actual = np.empty(count, dtype=np.float32)
        assert lib.h3_normal_noise(seed, count, actual.ctypes.data_as(fp)) == 0
        delta = actual.astype(np.float64) - expected
        l2 = float(np.linalg.norm(delta) / np.linalg.norm(expected.astype(np.float64)))
        noise[f"{count}/{seed}"] = {
            "relative_l2": l2,
            "max_abs": float(np.abs(delta).max()),
        }
        assert l2 < 1e-6, noise
report = {"exact_layouts": checks, "noise": noise}
# Ref2VA packs heterogeneous blocks in request order, with a shared A/V clock.
from types import SimpleNamespace
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3Ref2VAPrepareLayoutStep,
)

lib.h3_reference_layout.argtypes = lib.h3_condition_layout.argtypes
for geometry in (
    [(0, 1, 128, 128, 0)],
    [(0, 1, 8, 16, 0), (0, 1, 16, 8, 0)],
    [(0, 1, 4, 4, 0), (2, 0, 0, 0, 113)],
    [(1, 17, 28, 48, 119), (0, 1, 128, 256, 0), (2, 0, 0, 0, 107)],
    [(2, 0, 0, 0, 105), (1, 37, 48, 28, 0), (1, 2, 4, 4, 37)],
):
    tags = np.array([1, 1, 0, 0, 1], dtype=np.int32)
    refs = [
        SimpleNamespace(kind=("image", "video", "audio")[k], has_audio=bool(at))
        for k, t, h, w, at in geometry
    ]
    visual = [
        torch.empty(1, 24, t, h, w, device="meta") for k, t, h, w, at in geometry if t
    ]
    sound = [
        torch.empty(at * 2, 32, device="meta") for k, t, h, w, at in geometry if at
    ]
    expected = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
        torch.from_numpy(tags), refs, visual, sound, 37, 28, 48, 207, (1, 2, 2), 2, 2, 0
    )
    positions = np.empty(expected[0].shape, dtype=np.float32)
    result_tags = np.empty(expected[1].shape, dtype=np.int32)
    video = np.empty(expected[2].shape, dtype=np.int32)
    audio = np.empty(expected[3].shape, dtype=np.int32)
    g = np.array(geometry, dtype=np.int32)
    n = lib.h3_reference_layout(
        len(tags),
        tags.ctypes.data_as(ip),
        37,
        28,
        48,
        207,
        len(g),
        g.ctypes.data_as(ip),
        positions.ctypes.data_as(fp),
        result_tags.ctypes.data_as(ip),
        video.ctypes.data_as(ip),
        audio.ctypes.data_as(ip),
    )
    assert n == len(result_tags), lib.h3_error().decode()
    for actual, ref in zip((positions, result_tags, video, audio), expected[:4]):
        np.testing.assert_array_equal(actual, ref.numpy().astype(actual.dtype))
report["exact_reference_layouts"] = 5

from PIL import Image

bp = ctypes.POINTER(ctypes.c_uint8)
lib.h3_resize_image.argtypes = [
    ctypes.c_int,
    ctypes.c_int,
    bp,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    bp,
]
for sh, sw, h, w in (
    (64, 64, 256, 256),
    (193, 333, 448, 768),
    (333, 193, 64, 64),
    (33, 145, 256, 1024),
):
    rgb = np.random.default_rng(42).integers(0, 256, (sh, sw, 3), dtype=np.uint8)
    rgba = np.concatenate((rgb, np.full((sh, sw, 1), 255, dtype=np.uint8)), axis=-1)
    for cubic in (0, 1, 2):
        target = np.empty((h, w, 4), dtype=np.uint8)
        assert (
            lib.h3_resize_image(
                sh, sw, rgba.ctypes.data_as(bp), h, w, cubic, target.ctypes.data_as(bp)
            )
            == 0
        )
        expected = np.asarray(
            Image.fromarray(rgb).resize(
                (w, h), Image.Resampling.BICUBIC if cubic else Image.Resampling.LANCZOS
            )
        )
        if cubic == 2:
            import torchvision.transforms.v2.functional as F
            from torchvision.transforms import InterpolationMode

            expected = (
                F.resize(
                    torch.from_numpy(rgb).permute(2, 0, 1),
                    [h, w],
                    InterpolationMode.BICUBIC,
                    antialias=True,
                )
                .permute(1, 2, 0)
                .numpy()
            )
        np.testing.assert_array_equal(target[:, :, :3], expected)
report["exact_resizes"] = 12
Path(a.output).write_text(json.dumps(report, indent=2))
print("Ref2VA layouts and image resize: exact")
