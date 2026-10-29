"""Differential CPU codec test. Build image_codec.cpp + image.cpp, then pass its path.
Pillow and NumPy are test-only dependencies, never runtime dependencies.
"""

import argparse
import struct
import subprocess
import tempfile
import zlib
from pathlib import Path
import numpy as np
from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("binary")
a = p.parse_args()
with tempfile.TemporaryDirectory() as tmp:
    root = Path(tmp)
    rng = np.random.default_rng(1)
    for mode in ["RGBA", "RGB", "L", "LA", "P"]:
        image = Image.fromarray(rng.integers(0, 256, (43, 57, 4), dtype=np.uint8)).convert(mode)
        image.save(root / "input.png")
        for width, height in [(57, 43), (32, 32), (98, 128), (13, 21)]:
            subprocess.run(
                [a.binary, str(root / "input.png"), str(root / "output.raw"), str(width), str(height)],
                check=True,
            )
            actual = np.fromfile(root / "output.raw", np.uint8).reshape(height, width, 4)
            expected = np.array(image.convert("RGBA").resize((width, height), Image.Resampling.LANCZOS))
            np.testing.assert_array_equal(actual, expected)

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    # Exercise every DEFLATE block kind with exact raw RGBA bytes.
    raw = b"\0" + bytes(range(128)) * 2
    for level, strategy in [(0, zlib.Z_DEFAULT_STRATEGY), (9, zlib.Z_FIXED), (6, zlib.Z_DEFAULT_STRATEGY)]:
        codec = zlib.compressobj(level, zlib.DEFLATED, 15, 8, strategy)
        compressed = codec.compress(raw) + codec.flush()
        png = (
            b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", 64, 1, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", compressed)
            + chunk(b"IEND", b"")
        )
        (root / "input.png").write_bytes(png)
        subprocess.run([a.binary, str(root / "input.png"), str(root / "output.raw"), "64", "1"], check=True)
        assert (root / "output.raw").read_bytes() == raw[1:]
    for bad in [png[:20], png[:-15], png[:40] + b"X" + png[41:], b"not a png"]:
        (root / "input.png").write_bytes(bad)
        assert (
            subprocess.run(
                [a.binary, str(root / "input.png"), str(root / "output.raw"), "64", "1"], capture_output=True
            ).returncode
            != 0
        )
print("PASS: decode, RGBA Lanczos, stored/fixed/dynamic DEFLATE, malformed PNG rejection")
