"""Runtime, native DLL closure and real launcher HTTP checks for Windows bundles."""
from __future__ import annotations

import argparse
import ctypes
import importlib
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


def audit(root=ROOT):
    import pefile
    binaries = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".exe", ".dll", ".pyd", ".node"})
    local = {p.name.lower() for p in binaries}
    system = Path(os.environ["SystemRoot"]) / "System32"
    report, missing = [], []
    for path in binaries:
        pe = pefile.PE(str(path), fast_load=True)
        try:
            pe.parse_data_directories(directories=[1, 13])
            imports = [i.dll.decode("ascii").lower() for attribute in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT") for i in getattr(pe, attribute, [])]
            for dll in sorted(set(imports)):
                # Vendor/runtime DLLs are never satisfied by the build machine.
                vendor = re.match(r"(?:msvcp|vcruntime|vcomp|concrt|cudart|cublas|nvrtc|nccl)", dll)
                if dll in local:
                    provider = "bundled"
                elif dll == "nvcuda.dll":
                    provider = "NVIDIA system driver (GPU only)"
                elif not vendor and (dll.startswith(("api-ms-win-", "ext-ms-win-")) or (system / dll).is_file()):
                    provider = "Windows"
                else:
                    provider = "MISSING"
                    missing.append(f"{path.relative_to(root)} -> {dll}")
                report.append(f"{path.relative_to(root)} -> {dll}: {provider}")
        finally:
            pe.close()
    (root / "DLL-DEPENDENCIES.txt").write_text("\n".join(report) + "\n", encoding="utf-8")
    if missing:
        raise RuntimeError("Unbundled native dependencies:\n" + "\n".join(missing))
    print(f"[OK] PE dependency audit: {len(binaries)} binaries")


def smoke(model=None, device="cuda"):
    for args in (("--help",), ("launch", "--help"), ("webui", "--help")):
        subprocess.run([str(ROOT / "ftllm.exe"), *args], check=True, capture_output=True, timeout=45)
    for tool in ("aria2c", "rg", "fd"):
        subprocess.run([tool, "--version"], check=True, capture_output=True, timeout=20)
    print("[OK] Bundled downloader and Agent search executables")
    from ftllm_agent_runtime import PiAgentRuntime
    info = PiAgentRuntime("http://127.0.0.1:1/v1", "portable-check").info()
    assert info["pi_version"].endswith("0.84.4"), info
    print("[OK] Bundled Pi executable:", info["pi_version"])
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="ftllm-launch-check-") as directory:
        logfile = Path(directory) / "launcher.log"
        with logfile.open("wb") as log:
            process = subprocess.Popen([
                str(ROOT / "ftllm.exe"), "launch", "--no-browser", "--host", "127.0.0.1",
                "--port", str(port), "--config", str(Path(directory) / '配置 space.json'),
            ], stdout=log, stderr=subprocess.STDOUT, creationflags=subprocess.CREATE_NO_WINDOW)
            try:
                deadline = time.monotonic() + 60
                token = None
                while time.monotonic() < deadline:
                    output = logfile.read_text(encoding="utf-8", errors="replace")
                    match = re.search(r"\?token=([A-Za-z0-9_-]+)", output)
                    if match:
                        token = match[1]
                        try:
                            with urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2) as response:
                                assert response.status == 200
                                assert b"<html" in response.read().lower()
                            break
                        except OSError:
                            pass
                    if process.poll() is not None:
                        raise RuntimeError(output)
                    time.sleep(0.2)
                else:
                    raise RuntimeError("Launcher did not become ready:\n" + output)
                def request(endpoint, body=None):
                    req = urllib.request.Request(f"http://127.0.0.1:{port}{endpoint}",
                        data=json.dumps(body).encode() if body is not None else None,
                        headers={"x-ftllm-launcher-token": token, "Content-Type": "application/json"})
                    with urllib.request.urlopen(req, timeout=20) as response:
                        return json.load(response)
                assert "defaultProfile" in request("/api/bootstrap")
                hardware = request("/api/hardware")
                assert hardware["memory"]["total"] > 0, hardware
                if model:
                    with socket.socket() as model_socket:
                        model_socket.bind(("127.0.0.1", 0))
                        model_port = model_socket.getsockname()[1]
                    request("/api/runtime/start", {
                        "command": "server", "model": str(model.resolve()),
                        "model_name": "portable-test", "device": device,
                        "host": "127.0.0.1", "port": str(model_port),
                        "dtype": "auto", "threads": "4", "max_context_length": "2048",
                    })
                    deadline = time.monotonic() + 180
                    while time.monotonic() < deadline:
                        state = request("/api/runtime")
                        if state.get("ready"):
                            break
                        if state.get("phase") in {"failed", "error", "exited", "stopped"}:
                            raise RuntimeError(json.dumps(request("/api/logs")["entries"][-25:], ensure_ascii=False))
                        time.sleep(0.5)
                    else:
                        raise RuntimeError(json.dumps(request("/api/logs")["entries"][-25:], ensure_ascii=False))
                    payload = {"model": "portable-test", "messages": [{"role": "user", "content": "Say hello."}],
                               "max_tokens": 16, "temperature": 0, "stream": False}
                    completion_request = urllib.request.Request(
                        f"http://127.0.0.1:{model_port}/v1/chat/completions",
                        data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"})
                    with urllib.request.urlopen(completion_request, timeout=90) as response:
                        completion = json.load(response)
                    answer = completion["choices"][0]["message"]["content"]
                    assert answer.strip(), completion
                    request("/api/runtime/stop", {})
                    print(f"[OK] Launcher -> {device} model -> chat completion: {answer!r}")
                request("/api/shutdown", {})
                assert process.wait(timeout=20) == 0
                print("[OK] ftllm.exe launch: HTML, API, hardware detection, graceful shutdown")
            finally:
                if process.poll() is None:
                    subprocess.run([str(Path(os.environ["SystemRoot"]) / "System32/taskkill.exe"),
                                    "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                    process.wait(timeout=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--audit", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--model", type=Path, help="Also deploy and query a local model through the launcher")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args()
    assert sys.platform == "win32" and ctypes.sizeof(ctypes.c_void_p) == 8
    assert Path(sys.executable).resolve().is_relative_to(ROOT)
    assert all(Path(p).resolve().is_relative_to(ROOT) for p in sys.path if p), sys.path
    print("[OK] Isolated bundled Python:", sys.version.split()[0])
    for module in ("ssl", "sqlite3", "ctypes", "numpy", "transformers", "tokenizers", "sentencepiece",
                   "fastapi", "uvicorn", "openai", "modelscope", "aria2c", "pptx", "pypdf", "pandas",
                   "openpyxl", "xlsxwriter", "imageio", "imageio_ffmpeg", "ftllm.launcher", "ftllm.webui_server"):
        importlib.import_module(module)
    print("[OK] Python, launcher, WebUI, tokenizer, downloader and document imports")
    from ftllm import llm
    assert llm.has_device("cpu")
    print("[OK] FastLLM native library:", Path(llm.fastllm_lib._name).name)
    # Load the CPU DLL explicitly even on a CUDA developer machine.
    cpu = ROOT / "runtime/Lib/site-packages/ftllm/fastllm_tools-cpu.dll"
    if cpu.exists():
        lib = ctypes.CDLL(str(cpu))
        lib.has_device.argtypes = [ctypes.c_char_p]
        lib.has_device.restype = ctypes.c_bool
        assert lib.has_device(b"cpu") and not lib.has_device(b"cuda")
        print("[OK] Independent CPU fallback DLL")
    if args.require_cuda:
        assert llm.has_device("cuda"), "CUDA backend was not loaded"
        driver = ctypes.WinDLL("nvcuda.dll")
        assert driver.cuInit(0) == 0
        count = ctypes.c_int()
        assert driver.cuDeviceGetCount(ctypes.byref(count)) == 0 and count.value > 0
        print(f"[OK] CUDA driver: {count.value} GPU(s)")
    if args.audit:
        audit()
    if args.smoke:
        smoke(args.model, args.device)
    print("[OK] Portable checks passed")


if __name__ == "__main__":
    main()
