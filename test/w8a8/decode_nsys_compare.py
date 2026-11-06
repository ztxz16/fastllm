#!/usr/bin/env python3
"""W8A8模型的FastLLM/vLLM Decode Nsight Systems对比入口。"""

import sys
from pathlib import Path


REPO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_DIR / "test" / "nvfp4"))

import decode_nsys_compare as shared_decode_nsys_compare  # noqa: E402


if __name__ == "__main__":
    if "--quantization" not in sys.argv:
        sys.argv.extend(("--quantization", "w8a8"))
    shared_decode_nsys_compare.main()
