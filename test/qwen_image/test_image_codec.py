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

    # Exact encoder bytes, including Adler reduction and DEFLATE boundaries.
    for width, height in [(1,1), (17,31), (1387,1), (1388,1), (16383,1), (16384,1), (576,320)]:
        rgba = rng.integers(0, 256, (height, width, 4), dtype=np.uint8).tobytes()
        (root / "input.raw").write_bytes(rgba)
        subprocess.run([a.binary, "--encode", str(root / "input.raw"), str(root / "encoded.png"),
                        str(width), str(height)], check=True)
        raw = b"".join(b"\0"+rgba[y*width*4:(y+1)*width*4] for y in range(height))
        stored = bytearray(b"\x78\x01")
        for start in range(0, len(raw), 65535):
            part = raw[start:start+65535]
            stored += struct.pack("<BHH", int(start+len(part)==len(raw)), len(part), len(part)^65535)+part
        stored += struct.pack(">I", zlib.adler32(raw))
        expected = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB",width,height,8,6,0,0,0))
                    + chunk(b"IDAT",stored) + chunk(b"IEND",b""))
        assert (root / "encoded.png").read_bytes() == expected
        assert np.array(Image.open(root / "encoded.png")).tobytes() == rgba

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
print("PASS: exact encode, decode, RGBA Lanczos, stored/fixed/dynamic DEFLATE, malformed PNG rejection")
