"""单机 CUDA IPC 数据路径；没有 CPU 数据中转，也不触发任何编译。"""

import ctypes as C
import ctypes.util
from dataclasses import dataclass, field
import threading

from .protocol import ipc_handle


class CudaError(RuntimeError):
    pass


class _Handle(C.Structure):
    _fields_ = [("data", C.c_ubyte * 64)]


class _UUID(C.Structure):
    _fields_ = [("data", C.c_ubyte * 16)]


@dataclass
class Pending:
    device: int
    stream: object
    event: object
    start_event: object = None
    mappings: list = field(default_factory=list)
    done: bool = False
    error: str = ""
    bytes: int = 0
    gpu_ms: float = 0.0


class CudaIPC:
    """/**
     * 通过原生 CUDA Runtime 在服务进程内映射 worker 显存并提交异步拷贝。
     * 只接受已协商的句柄与区域；所有源和目标映射在实际完成后关闭。
     * GPU UUID 负责跨服务定位，不能把不同 CUDA_VISIBLE_DEVICES 下的编号直接互用。
     * 一个实例固定由传输后台线程调用；不跨线程复用其运行上下文。
     */"""

    def __init__(self, library=None):
        self._library = library
        self._cuda = None
        self._thread = None
        self._devices = {}
        self._pending = []
        self._quarantined = []
        self._mappings = {}

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
            "cudaDeviceCanAccessPeer": [C.POINTER(integer), integer, integer],
            "cudaIpcOpenMemHandle": [C.POINTER(pointer), _Handle, C.c_uint],
            "cudaIpcCloseMemHandle": [pointer],
            "cudaStreamCreateWithFlags": [C.POINTER(pointer), C.c_uint],
            "cudaStreamSynchronize": [pointer],
            "cudaStreamDestroy": [pointer],
            "cudaEventCreateWithFlags": [C.POINTER(pointer), C.c_uint],
            "cudaEventRecord": [pointer, pointer],
            "cudaEventQuery": [pointer],
            "cudaEventDestroy": [pointer],
            "cudaEventElapsedTime": [C.POINTER(C.c_float), pointer, pointer],
            "cudaMemcpy2DAsync": [pointer, size, pointer, size, size, size, integer, pointer],
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

    def submit(self, copies):
        """/**
         * 打开本批全部映射后提交拷贝并记录完成事件，不等待 GPU。
         * 部分提交发生错误时保留 Pending，仍须等待其 stream 排空；不能丢失句柄。
         * @return Pending；预检失败抛错且没有写入。
         */"""
        self._start()
        if not copies:
            raise ValueError("empty copy batch")
        devices = {copy.target.device for copy in copies}
        if len(devices) != 1 or next(iter(devices)) not in self._devices:
            raise CudaError("target GPU is not visible in the transfer service")
        ordinal = self._devices[next(iter(devices))]
        self._call("cudaSetDevice", ordinal)
        for copy in copies:
            source = self._devices.get(copy.source.device)
            if source is None:
                raise CudaError("source GPU must be visible for P2P validation")
            if source != ordinal:
                supported = C.c_int()
                self._call("cudaDeviceCanAccessPeer", C.byref(supported), ordinal, source)
                if not supported.value:
                    raise CudaError("GPU pair does not support P2P; CPU fallback is disabled")

        task = Pending(ordinal, C.c_void_p(), C.c_void_p(), C.c_void_p())
        mapped = {}
        try:
            for copy in copies:
                for region in (copy.source, copy.target):
                    if region.memory not in mapped:
                        key = (ordinal, region.memory)
                        if key in self._mappings:
                            address, refs = self._mappings[key]
                        else:
                            handle = _Handle.from_buffer_copy(ipc_handle(region.memory))
                            address, refs = C.c_void_p(), 0
                            self._call("cudaIpcOpenMemHandle", C.byref(address), handle, 1)
                        self._mappings[key] = (address, refs + 1)
                        mapped[region.memory] = address.value
                        task.mappings.append(key)
            self._call("cudaStreamCreateWithFlags", C.byref(task.stream), 1)
            self._call("cudaEventCreateWithFlags", C.byref(task.event), 0)
            self._call("cudaEventCreateWithFlags", C.byref(task.start_event), 0)
        except Exception:
            self._dispose(task)
            raise

        self._pending.append(task)
        try:
            self._call("cudaEventRecord", task.start_event, task.stream)
            for copy in copies:
                src, dst = copy.source, copy.target
                begin, heads = copy.token_begin, copy.head_end - copy.head_begin
                merge_heads = src.head_stride == dst.head_stride == copy.vector_bytes
                for head in ([copy.head_begin] if merge_heads else range(copy.head_begin, copy.head_end)):
                    width = copy.vector_bytes * (heads if merge_heads else 1)
                    task.bytes += width * (copy.token_end - begin)
                    self._call("cudaMemcpy2DAsync",
                               C.c_void_p(mapped[dst.memory] + dst.byte_offset(begin, head)), dst.token_stride,
                               C.c_void_p(mapped[src.memory] + src.byte_offset(begin, head)), src.token_stride,
                               width, copy.token_end - begin, 4, task.stream)
            self._call("cudaEventRecord", task.event, task.stream)
        except Exception as error:
            # 异常后事件可能未记录，query 必须改用 stream 排空；任务仍交给管理器持有。
            task.error = str(error)
        return task

    def query(self, task):
        """/** 事件就绪只表示实际访问结束；异步错误不能冒充成功。 */"""
        self._start()
        self._call("cudaSetDevice", task.device)
        if task.done:
            return True
        if task.error:
            self._call("cudaStreamSynchronize", task.stream)
            task.done = True
            return True
        result = self._cuda.cudaEventQuery(task.event)
        if result == 600:  # cudaErrorNotReady
            return False
        if result:
            task.error = f"cudaEventQuery failed: {result}"
            return False
        elapsed = C.c_float()
        self._call("cudaEventElapsedTime", C.byref(elapsed), task.start_event, task.event)
        task.gpu_ms = elapsed.value
        task.done = True
        return True

    def retire(self, task):
        """/** GPU 已排空后关闭 IPC 映射；失败仍报告给任务管理器。 */"""
        if not task.done:
            raise CudaError("cannot unmap memory still in use")
        self._call("cudaSetDevice", task.device)
        self._dispose(task)
        self._pending.remove(task)

    def _dispose(self, task):
        if task.start_event is not None and task.start_event.value:
            self._call("cudaEventDestroy", task.start_event)
            task.start_event = C.c_void_p()
        if task.event.value:
            self._call("cudaEventDestroy", task.event)
            task.event = C.c_void_p()
        if task.stream.value:
            self._call("cudaStreamDestroy", task.stream)
            task.stream = C.c_void_p()
        while task.mappings:
            key = task.mappings[-1]
            address, refs = self._mappings[key]
            if refs == 1:
                self._call("cudaIpcCloseMemHandle", address)
                del self._mappings[key]
            else:
                self._mappings[key] = (address, refs - 1)
            task.mappings.pop()

    def close(self):
        if self._pending or self._mappings:
            raise CudaError("IPC mappings still have outstanding transfers")
