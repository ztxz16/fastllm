"""正式算子薄适配；不在本文件导入时加载 CUDA。"""

import ctypes
import hashlib
from pathlib import Path


class Native:
    def __init__(self, path, device, stream, rank, group, mode=1):
        import torch.distributed as dist

        self.library = ctypes.CDLL(str(Path(path).resolve()))
        lib = self.library
        lib.ar_error.restype = ctypes.c_char_p
        lib.ar_build_arch.restype = ctypes.c_char_p
        try:
            for name in ("ar_source_sha", "ar_header_sha", "ar_build_git"):
                getattr(lib, name).restype = ctypes.c_char_p
        except AttributeError as exc:
            raise RuntimeError("原生适配库较旧，请在服务器重新执行 build_allreduce_adapter.sh") from exc
        lib.ar_id_size.restype = ctypes.c_size_t
        lib.ar_unique_id.argtypes = [ctypes.c_void_p]
        lib.ar_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                                  ctypes.c_void_p, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)]
        lib.ar_sum.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
        lib.ar_info.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong),
                               ctypes.POINTER(ctypes.c_ulonglong)]
        lib.ar_destroy.argtypes = [ctypes.c_void_p]
        lib.ar_versions.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int)]
        self.handle = ctypes.c_void_p()
        self.device, self.mode = device, mode
        uid = ctypes.create_string_buffer(lib.ar_id_size())
        if rank == 0:
            self.check(lib.ar_unique_id(uid))
        shared = [uid.raw if rank == 0 else None]
        dist.broadcast_object_list(shared, src=0, group=group)
        ctypes.memmove(uid, shared[0], len(shared[0]))
        self.check(lib.ar_create(device, dist.get_world_size(group), rank, mode, uid,
                                 stream.cuda_stream, ctypes.byref(self.handle)))

    def check(self, code):
        if code:
            raise RuntimeError(self.library.ar_error().decode())

    def __call__(self, tensor):
        import torch

        if tensor.dtype != torch.float32 or not tensor.is_contiguous() or tensor.device.index != self.device:
            raise ValueError("native adapter requires local contiguous FP32")
        self.check(self.library.ar_sum(self.handle, tensor.data_ptr(), tensor.numel()))
        return tensor

    def path(self, n):
        return "ft_custom" if self.mode == 1 and n <= 65536 else "nccl"

    def info(self):
        if self.mode == 0:
            return None
        custom, nccl = ctypes.c_ulonglong(), ctypes.c_ulonglong()
        self.check(self.library.ar_info(self.handle, ctypes.byref(custom), ctypes.byref(nccl)))
        return {"custom": custom.value, "nccl": nccl.value}

    def metadata(self):
        from .reporting import ROOT

        runtime, nccl = ctypes.c_int(), ctypes.c_int()
        self.check(self.library.ar_versions(ctypes.byref(runtime), ctypes.byref(nccl)))
        source = self.library.ar_source_sha().decode()
        header = self.library.ar_header_sha().decode()
        if source != hashlib.sha256((ROOT / "src/distributed/process_allreduce.cu").read_bytes()).hexdigest() or (
                header != hashlib.sha256((ROOT / "include/distributed/process_allreduce.h").read_bytes()).hexdigest()):
            raise RuntimeError("已编译算子与当前生产源码不一致，请在服务器重新构建")
        return {"cuda_runtime": runtime.value, "nccl_version": nccl.value,
                "compile_architecture": self.library.ar_build_arch().decode(),
                "build_git": self.library.ar_build_git().decode(), "source_sha256": source,
                "header_sha256": header,
                "library_sha256": hashlib.sha256(Path(self.library._name).read_bytes()).hexdigest()}

    def close(self):
        self.check(self.library.ar_destroy(self.handle))
        self.handle = ctypes.c_void_p()


class Vllm:
    def __init__(self, device, group, mode):
        import torch.distributed as dist
        from vllm.distributed.parallel_state import init_distributed_environment
        from vllm.distributed.device_communicators.custom_all_reduce import CustomAllreduce
        from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

        # 正式 P2P 检查依赖 vLLM world group；只创建分布式环境，不初始化模型或 TP 层。
        init_distributed_environment(world_size=dist.get_world_size(group), rank=dist.get_rank(group),
                                     local_rank=device, backend="gloo")
        self.custom = CustomAllreduce(group, device, register_graph_buffers=mode != "graph_copy")
        if self.custom.disabled:
            raise RuntimeError("vLLM 自定义算子未启用；严格比较不接受静默回退")
        self.fallback = PyNcclCommunicator(group, device)
        if self.fallback.disabled:
            raise RuntimeError("vLLM PyNccl fallback unavailable")
        self.last_path = None

    def __call__(self, tensor):
        result = self.custom.custom_all_reduce(tensor)
        if result is not None:
            self.last_path = "vllm_custom"
            return result
        self.last_path = "nccl"
        result = self.fallback.all_reduce(tensor)
        if result is None:
            raise RuntimeError("vLLM NCCL fallback returned no output")
        return result

    def path(self, n):
        return self.last_path

    def info(self):
        return None

    def metadata(self):
        import vllm
        root = Path(vllm.__file__).resolve().parent
        sources = [root / "distributed/device_communicators/custom_all_reduce.py",
                   root / "distributed/device_communicators/pynccl.py"]
        libraries = sorted({Path(line.split()[-1]) for line in Path("/proc/self/maps").read_text().splitlines()
                            if "/vllm/" in line and ".so" in line and line.split()[-1].startswith("/")})
        return {"vllm_version": vllm.__version__, "vllm_module": str(Path(vllm.__file__).resolve()),
                "max_size": self.custom.max_size, "registered_graph": self.custom._capture_registered,
                "nccl_version": self.fallback.nccl_version,
                "source_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
                "extension_hashes": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in libraries}}

    def close(self):
        self.custom.close()
        self.fallback.destroy()
