"""Compile ordinary Linear and grouped MXFP4 from empty caches, without a server."""
import importlib.util
from pathlib import Path
import sys
import tempfile


def main():
    path = Path(__file__).resolve().parents[2] / "tools" / "fastllm_triton_server.py"
    spec = importlib.util.spec_from_file_location("fastllm_triton_server", path)
    server = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(server)
    if server.triton is None:
        print("SKIP: Triton compiler is unavailable")
        return 77
    # Offline compilation deliberately bypasses warm application caches, which
    # previously hid an undefined compiler option in the ordinary Linear path.
    with tempfile.TemporaryDirectory(prefix="fastllm-triton-cold-") as cache:
        for dtype in ("fp16", "bf16"):
            payload = dict(op="linear", arch=120, cache_dir=cache,
                           input_dtype=dtype, weight_dtype=dtype, output_dtype=dtype)
            result = server.handle_compile(payload)
            assert result["ok"] and Path(result["cubin"]).stat().st_size > 0
            assert server.handle_compile(payload) == result
            print(f"PASS: cold and warm Linear {dtype}")
        payload = dict(op="mxfp4_grouped", arch=120, cache_dir=cache)
        result = server.handle_compile(payload)
        assert result["ok"] and result["version"] == 2
        assert set(result["kernels"]) == {"gate", "activate", "down"}
        for kernel in result["kernels"].values():
            assert Path(kernel["cubin"]).stat().st_size > 0
        assert server.handle_compile(payload) == result
        print("PASS: cold and warm grouped MXFP4 compiler ABI")
    return 0


if __name__ == "__main__":
    sys.exit(main())
