"""测试子进程显存分配/读取夹具；不包含 KV 传输和完成判断实现。"""

import ctypes as C
import ctypes.util
import threading



class CudaError(RuntimeError):
    pass


class _Handle(C.Structure):
    _fields_ = [("data", C.c_ubyte * 64)]


class _UUID(C.Structure):
    _fields_ = [("data", C.c_ubyte * 16)]


class CudaFixture:
    """只用于测试所有者进程，提供 CUDA 原始分配/上传/读回与 UUID 查询。"""

    def __init__(self, library=None):
        self._library = library
        self._cuda = None
        self._thread = None
        self._devices = {}

    def _start(self):
        if self._cuda is not None:
            if self._thread != threading.get_ident():
                raise CudaError("CUDA IPC backend must use its owner thread")
            return
        path = self._library or ctypes.util.find_library("cudart") or "libcudart.so"
        self._cuda = C.CDLL(path)
        self._thread = threading.get_ident()
        pointer, size, integer = C.c_void_p, C.c_size_t, C.c_int
        signatures = {
            "cudaSetDevice": [integer],
            "cudaGetDeviceCount": [C.POINTER(integer)],
        }
        for name, args in signatures.items():
            function = getattr(self._cuda, name)
            function.argtypes, function.restype = args, integer
        self._cuda.cudaGetErrorString.argtypes = [integer]
        self._cuda.cudaGetErrorString.restype = C.c_char_p
        count = integer()
        self._call("cudaGetDeviceCount", C.byref(count))
        driver = C.CDLL("libcuda.so.1")
        driver.cuInit.argtypes, driver.cuInit.restype = [C.c_uint], integer
        driver.cuDeviceGet.argtypes, driver.cuDeviceGet.restype = [C.POINTER(integer), integer], integer
        uuid = getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)
        uuid.argtypes, uuid.restype = [C.POINTER(_UUID), integer], integer
        if driver.cuInit(0):
            raise CudaError("CUDA driver initialization failed")
        for ordinal in range(count.value):
            device = _UUID()
            handle = integer()
            if driver.cuDeviceGet(C.byref(handle), ordinal) or uuid(C.byref(device), handle):
                raise CudaError("CUDA driver GPU identity query failed")
            self._devices[bytes(device.data).hex()] = ordinal

    def _call(self, name, *args):
        result = getattr(self._cuda, name)(*args)
        if result:
            description = self._cuda.cudaGetErrorString(result)
            raise CudaError(f"{name}: {description.decode() if description else result}")
