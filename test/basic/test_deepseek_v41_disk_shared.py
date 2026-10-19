#!/usr/bin/env python3
"""Run the shared-expert placement regression using the existing tiny fixture."""
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    try:
        from test_deepseek_v41_cpu_fixture import unpack
    except ModuleNotFoundError as error:
        if error.name != "numpy":
            raise
        print("SKIP: numpy is required to unpack the V4.1 fixture")
        return 77

    with tempfile.TemporaryDirectory(prefix="fastllm-v41-disk-shared-") as path:
        unpack(str(Path(__file__).with_name("deepseek_v41_fixture.npz")), path)
        result = subprocess.run([sys.argv[1], path], stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, check=True)
        print(result.stdout, end="")
        if "PASS disk shared placement" not in result.stdout:
            raise RuntimeError("shared placement regression did not finish")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
