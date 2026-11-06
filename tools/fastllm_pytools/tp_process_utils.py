"""Linux 进程组生命周期：正式文本入口和验收器共用，不依赖 PID 1 回收 worker。"""

from contextlib import contextmanager
import ctypes
import os
import signal
import subprocess
import time


def enable_child_subreaper():
    """接管协调器退出后的孤儿 worker；配置失败时拒绝启动，避免依赖容器 PID 1。"""
    libc = ctypes.CDLL(None, use_errno=True)
    prctl = libc.prctl
    prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    prctl.restype = ctypes.c_int
    if prctl(36, 1, 0, 0, 0) != 0:  # PR_SET_CHILD_SUBREAPER
        error = ctypes.get_errno()
        raise OSError(error, "cannot enable child subreaper: " + os.strerror(error))


def wait_case(process, timeout):
    """有界等待但不回收组长，保留 PID/PGID 所有权直至整组清理完成。"""
    deadline = time.monotonic() + timeout
    while True:
        result = os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT)
        if result is not None:
            return result.si_status if result.si_code == os.CLD_EXITED else -result.si_status
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        time.sleep(min(0.02, remaining))


def cleanup_process_group(process, timeout=5):
    """终止并回收该组所有后代；超时报错，不回收其他并发进程组的子进程。"""
    deadline = time.monotonic() + timeout
    try:
        # SIGKILL 对 SIGSTOP 中的 worker 同样生效；先杀整组，再回收组长。
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=max(0, deadline - time.monotonic()))

    # 只回收本 case 的进程组，不能使用 waitpid(-1) 抢走其他子进程的状态。
    while True:
        try:
            pid, _ = os.waitpid(-process.pid, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError(f"process group {process.pid} cleanup timed out")
            time.sleep(min(0.02, remaining))


@contextmanager
def isolated_process(command, **kwargs):
    """创建独立进程组；调用者须用 wait_case 保留组长 PID，退出时有界回收整组。"""
    enable_child_subreaper()
    process = subprocess.Popen(command, start_new_session=True, **kwargs)
    try:
        yield process
    finally:
        cleanup_process_group(process)
