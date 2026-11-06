"""仅在Attention NCU脚本显式注入时安装CUDA Profiler信号控制。"""

import ctypes
import os
from pathlib import Path
import signal


def _load_cudart():
    errors = []
    for name in ("libcudart.so", "libcudart.so.13", "libcudart.so.12"):
        try:
            library = ctypes.CDLL(name)
            library.cudaProfilerStart.restype = ctypes.c_int
            library.cudaProfilerStop.restype = ctypes.c_int
            return library
        except OSError as error:
            errors.append(f"{name}: {error}")
    raise RuntimeError("无法加载CUDA Runtime：" + "; ".join(errors))


def _install():
    pid_dir = os.environ.get("FASTLLM_NCU_TARGET_PID_DIR")
    if not pid_dir:
        return

    # vLLM把EngineCore放在独立Python进程。sitecustomize会被API Server及其
    # Python子进程自动加载，使真正持有CUDA Context的进程也能响应采集信号。
    def profiler_start(_signum, _frame):
        try:
            status = _load_cudart().cudaProfilerStart()
            print(
                f"[ncu-control] pid={os.getpid()} "
                f"cudaProfilerStart status={status}", flush=True)
        except Exception as error:
            print(f"[ncu-control] pid={os.getpid()} start failed: {error}",
                  flush=True)

    def profiler_stop(_signum, _frame):
        try:
            status = _load_cudart().cudaProfilerStop()
            print(
                f"[ncu-control] pid={os.getpid()} "
                f"cudaProfilerStop status={status}", flush=True)
        except Exception as error:
            print(f"[ncu-control] pid={os.getpid()} stop failed: {error}",
                  flush=True)

    signal.signal(signal.SIGUSR1, profiler_start)
    signal.signal(signal.SIGUSR2, profiler_stop)
    directory = Path(pid_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{os.getpid()}.pid").write_text(
        str(os.getpid()), encoding="utf-8")


_install()
