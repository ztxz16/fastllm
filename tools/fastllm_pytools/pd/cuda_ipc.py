"""CUDA IPC 原生后端的薄入口；缺少原生模块时拒绝，不回退 Python 传输。"""

from . import _core


class CudaIPC:
    def __new__(cls):
        return _core.load().CudaIPC()
