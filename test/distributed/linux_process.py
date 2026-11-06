"""Linux pidfd 适配：兼容未暴露 pidfd API 的 Python 构建，不退回裸 PID 发信号。"""

import ctypes
import os
import signal
import sys


def _syscall(number, *arguments):
    # Linux x86_64 和 asm-generic（含 aarch64）的 pidfd 系统调用编号相同。
    # 其他 ABI 不猜测编号；要求使用带原生 pidfd API 的 Python。
    if (sys.platform != "linux" or os.uname().machine not in ("x86_64", "aarch64")
            or ctypes.sizeof(ctypes.c_void_p) != 8):
        raise RuntimeError("pidfd fallback requires Linux x86_64/aarch64; use Python with native pidfd APIs")
    libc = ctypes.CDLL(None, use_errno=True)
    libc.syscall.restype = ctypes.c_long
    result = libc.syscall(ctypes.c_long(number), *arguments)
    if result == -1:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return result


def _int32(value):
    if not isinstance(value, int) or not -(1 << 31) <= value < (1 << 31):
        raise ValueError("pidfd argument must be a signed 32-bit integer")
    return ctypes.c_int(value)


def open_pidfd(pid):
    native = getattr(os, "pidfd_open", None)
    if callable(native):
        return native(pid, 0)
    return _syscall(434, _int32(pid), ctypes.c_uint(0))


def send_pidfd_signal(fd, sig):
    native = getattr(signal, "pidfd_send_signal", None)
    if callable(native):
        native(fd, sig, None, 0)
        return
    _syscall(424, _int32(fd), _int32(sig), ctypes.c_void_p(None), ctypes.c_uint(0))


def check_pidfd_support():
    fd = open_pidfd(os.getpid())
    try:
        # 信号 0 仅检查能力和权限，不改变当前进程状态。
        send_pidfd_signal(fd, 0)
    finally:
        os.close(fd)
