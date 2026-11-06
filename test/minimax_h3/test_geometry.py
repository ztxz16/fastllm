"""CPU-only check of packed multimodal coordinates and exact FP32 sigma grids."""

import argparse
import ctypes

import numpy as np
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
)

parser = argparse.ArgumentParser()
parser.add_argument("--library", required=True)
args = parser.parse_args()
lib = ctypes.CDLL(args.library)
fp = ctypes.POINTER(ctypes.c_float)
ip = ctypes.POINTER(ctypes.c_int)
lib.h3_schedule.argtypes = [ctypes.c_int, ctypes.c_float, fp]
lib.h3_layout.argtypes = [ctypes.c_int] * 5 + [fp, ip]
for steps in range(2, 101):
    for shift in (3.0, 12.0):
        expected = MiniMaxH3Scheduler(shift)
        expected.set_timesteps(steps)
        actual = np.empty(steps, dtype=np.float32)
        count = lib.h3_schedule(steps, shift, actual.ctypes.data_as(fp))
        np.testing.assert_array_equal(actual[:count], expected.sigmas.numpy())
for nt, frames, h, w, audio in [
    (1, 2, 2, 2, 9),
    (17, 37, 32, 48, 207),
    (63, 102, 48, 84, 575),
    (3, 12, 18, 4, 66),
]:
    n = nt + frames * h * w // 4 + audio * 2
    pos = np.empty((n, 3), dtype=np.float32)
    tags = np.empty(n, dtype=np.int32)
    count = lib.h3_layout(nt, frames, h, w, audio, pos.ctypes.data_as(fp), tags.ctypes.data_as(ip))
    assert count == n
    expected = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
        torch.ones(nt, dtype=torch.long), frames, h, w, audio, (1, 2, 2), 2, 2, 0
    )
    np.testing.assert_array_equal(pos, expected[0].float().numpy())
    np.testing.assert_array_equal(tags, expected[1].numpy())
print("198 schedules and 4 packed layouts match the reference exactly")
