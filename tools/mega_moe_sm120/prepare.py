"""Build-time SM120 MXFP4 expert graph preparation; never imported by inference.

The kernels and Green Context graph implementation are provided by FlashInfer:
https://github.com/flashinfer-ai/flashinfer/tree/main/flashinfer/moe_ep/kernel_src/sm120/split_cutedsl_megakernel
(Apache-2.0; upstream kernel provenance is retained in that package.)
This adapter supplies same-process peer allocations. No NVSHMEM communicator is
needed. Python prepares graphs only during package export; native inference reconstructs
them from relocatable metadata.
"""
import concurrent.futures
import contextlib
import importlib
import math
import pathlib
import sys
import threading
import types

_context = threading.local()
_runtime = None


class _PeerArena:
    def __init__(self, devices):
        self.devices = devices
        self.condition = threading.Condition()
        self.allocations = {}
        self.pointers = {}
        self.counts = {}
        self.error = None
        self.heaps = {}
        self.offsets = {}
        self.sizes = {}

    def reserve(self, rank, bucket, sizes):
        import torch
        # Every peer-visible tensor must have the same offset from its rank's
        # base, because upstream uses one pointer mapper for all shared tensors.
        self.sizes[rank, bucket] = sizes
        total = sum((size + 255) // 256 * 256 for size in sizes)
        self.heaps[rank, bucket] = torch.zeros(total, dtype=torch.uint8, device=self.devices[rank])
        self.offsets[rank, bucket] = 0

    def abort(self, error):
        with self.condition:
            self.error = error
            self.condition.notify_all()

    def allocate(self, shape, dtype):
        import torch
        rank, bucket = _context.rank, _context.bucket
        with self.condition:
            serial = self.counts.get((rank, bucket), 0)
            size = math.prod(shape) * dtype.itemsize
            sizes = self.sizes[rank, bucket]
            if serial >= len(sizes) or sizes[serial] != size:
                raise RuntimeError('FlashInfer shared-memory layout differs from the prepared arena')
            offset = self.offsets[rank, bucket]
            value = self.heaps[rank, bucket][offset:offset+size].view(dtype).reshape(shape)
            self.offsets[rank, bucket] = offset + (size + 255) // 256 * 256
            self.counts[rank, bucket] = serial + 1
            key = (bucket, serial)
            group = self.allocations.setdefault(key, {})
            group[rank] = (value.data_ptr(), tuple(shape), dtype)
            self.pointers[value.data_ptr()] = key
            self.condition.notify_all()
        return value

    def peers(self, tensor, world):
        base = tensor.data_ptr()
        with self.condition:
            key = self.pointers[base]
            if not self.condition.wait_for(lambda: self.error is not None or len(self.allocations[key]) == world, timeout=300):
                raise RuntimeError('SM120 expert peer allocation rendezvous timed out')
            if self.error is not None:
                raise RuntimeError('SM120 expert preparation failed on a peer') from self.error
            group = self.allocations[key]
            expected = group[_context.rank][1:]
            if any(group[r][1:] != expected for r in range(world)):
                raise RuntimeError('SM120 expert peer allocations have unequal layouts')
            return base, tuple(group[r][0] - base for r in range(world))


def _load_runtime():
    global _runtime
    if _runtime is not None:
        return _runtime
    import torch
    import flashinfer.moe_ep.kernel_src.sm120.split_cutedsl_megakernel as upstream
    upstream.bootstrap_paths()
    # Load the unmodified upstream shim in a private namespace. The stock
    # FlashInfer allocator and other users' distributed runtimes are unaffected.
    name = __name__ + '._shim'
    package = types.ModuleType(name)
    package.__path__ = [str(pathlib.Path(upstream.__file__).parent / 'shim')]
    sys.modules[name] = package
    comm = types.ModuleType(name + '.comm')
    def ensure_not_capturing(operation):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(operation + ' requires eager warmup before capture')
    def sym_zeros(shape, dtype):
        return _context.arena.allocate(shape, dtype)
    def sym_byte_view(shape, dtype):
        root = sym_zeros((math.prod(shape) * dtype.itemsize,), torch.uint8)
        return root.view(dtype).reshape(shape), root
    comm.ensure_not_capturing = ensure_not_capturing
    comm.sym_zeros = sym_zeros
    comm.sym_byte_view = sym_byte_view
    comm.compute_peer_offsets = lambda t, n: _context.arena.peers(t, n)
    comm.free_sym_tensor = lambda tensor: None  # torch owns these plain allocations
    sys.modules[comm.__name__] = comm
    runtime = importlib.import_module(name + '.runtime')
    base = runtime.MegaMoESm120W4A8Frontend
    compile_lock = threading.Lock()
    class LocalFrontend(base):
        def _barrier(self):
            if not getattr(_context, 'compiling', False):
                _context.barrier.wait(timeout=300)

        def _ensure_compiled(self, inputs):
            if self._compiled is not None:
                return self._compiled
            # Peer allocation rendezvous must precede the compiler lock. The
            # CuTeDSL compiler uses process-global state; GPU execution does not.
            self._ensure_execution()
            self._barrier()
            with compile_lock:
                _context.compiling = True
                try:
                    result = super()._ensure_compiled(inputs)
                finally:
                    _context.compiling = False
            self._barrier()
            return result
    runtime.MegaMoESm120W4A8Frontend = LocalFrontend
    _runtime = (runtime, importlib.import_module(name + '.staging'),
                importlib.import_module(name + '.weights'))
    return _runtime


class _CudaBytes:
    def __init__(self, pointer, count):
        self.__cuda_array_interface__ = {
            'shape': (count,), 'typestr': '|u1', 'data': (pointer, False), 'version': 3}


def _cuda_bytes(pointer, count, device):
    import torch
    return torch.as_tensor(_CudaBytes(pointer, count), device=device)


class GraphProvider:
    def __init__(self, devices, layers, hidden=4096, intermediate=2048, experts=256, topk=8,
                 rank_local_combine=False, dispatch_rank_cache=False):
        import torch
        from cuda.bindings import runtime as cuda
        # cuda.core chooses its CUDA-major namespace lazily. Resolve it before
        # starting rank threads so no worker observes a partially imported API.
        from cuda.core import Device
        self.runtime, self.staging, self.weight_ops = _load_runtime()
        self.devices = list(devices)
        self.layers = layers
        self.hidden, self.intermediate, self.experts, self.topk = hidden, intermediate, experts, topk
        self.rank_local_combine = rank_local_combine
        self.dispatch_rank_cache = dispatch_rank_cache
        self.arena = _PeerArena(self.devices)
        self.barrier = threading.Barrier(len(self.devices))
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=len(self.devices))
        self.workspaces = [{} for _ in self.devices]
        self.weights = [{} for _ in self.devices]
        self.inputs = [{} for _ in self.devices]
        self.graphs = [{} for _ in self.devices]
        for device in self.devices:
            with torch.cuda.device(device):
                for peer in self.devices:
                    if device == peer:
                        continue
                    if not torch.cuda.can_device_access_peer(device, peer):
                        raise RuntimeError('SM120 EP requires direct CUDA peer access for every device pair')
                    result = cuda.cudaDeviceEnablePeerAccess(peer, 0)[0]
                    if int(result) not in (0, 704):
                        raise RuntimeError('Cannot enable CUDA peer access: ' + str(result))
                    cuda.cudaGetLastError()

    @contextlib.contextmanager
    def context(self, rank, bucket):
        import torch
        _context.rank, _context.bucket = rank, bucket
        _context.arena, _context.barrier = self.arena, self.barrier
        with torch.cuda.device(self.devices[rank]):
            yield

    def _weights(self, rank, layer):
        import torch
        if layer in self.weights[rank]:
            return self.weights[rank][layer]
        w13, w2, s13, s2 = [], [], [], []
        for gate, down in self.layers[rank][layer]:
            for pointer, n, k, data, scale in (
                    (gate, 2*self.intermediate, self.hidden, w13, s13),
                    (down, self.hidden, self.intermediate, w2, s2)):
                pointer = int(pointer, 0) if isinstance(pointer, str) else pointer
                size = n*k//2
                raw = _cuda_bytes(pointer, size+n*k//32, self.devices[rank])
                data.append(raw[:size].reshape(n, k//2))
                scale.append(raw[size:].reshape(n, k//32))
        transformed = self.weight_ops.transform_prequantized_weights(
            torch.stack(w13).view(torch.float4_e2m1fn_x2),
            torch.stack(w2).view(torch.float4_e2m1fn_x2),
            torch.stack(s13).view(torch.float8_e8m0fnu),
            torch.stack(s2).view(torch.float8_e8m0fnu),
            hidden=self.hidden, intermediate=self.intermediate)
        self.weights[rank][layer] = transformed
        return transformed

    def _prepare_rank(self, rank, layer, bucket):
        import torch
        with self.context(rank, bucket):
            runtime = self.runtime
            if bucket not in self.workspaces[rank]:
                cfg = runtime.MegaMoESm120W4A8Config(
                    rank=rank, world_size=len(self.devices), max_tokens_per_rank=bucket,
                    num_topk=self.topk, num_total_experts=self.experts, hidden=self.hidden,
                    # Every owner holds complete experts. EP size is not the
                    # tensor-parallel width of an expert's GEMMs.
                    intermediate=self.intermediate, tensor_parallel_size=1,
                    knobs={'rank_local_combine': self.rank_local_combine,
                           'dispatch_rank_cache': self.dispatch_rank_cache,
                           # Preserve the established K2 arithmetic/scheduling
                           # family while comparing communication strategies.
                           'k2_warps': 8, 'k2_tail_reclaim': False})
                plan_workspace = types.SimpleNamespace(config=cfg, _execution_plans={})
                planner = runtime.MegaMoESm120W4A8Frontend(plan_workspace, compile_bucket=bucket)
                spec, bundle = planner._ensure_plan(bucket)
                sizes = [bucket*self.hidden, bucket*((self.hidden//32+3)//4*4),
                         bucket*self.topk*8, bucket*self.topk*4, bucket*self.hidden*2,
                         bundle.shared_workspace_bytes, bucket*self.topk*self.hidden*2]
                if spec.kernel.rank_local_combine:
                    sizes.append((bucket+1)*len(self.devices)*4)
                self.arena.reserve(rank, bucket, sizes)
                self.workspaces[rank][bucket] = runtime.allocate_workspace(cfg)
                self.workspaces[rank][bucket]._execution_plans[bucket] = (spec, bundle)
                self.inputs[rank][bucket] = (
                    torch.zeros((bucket, self.hidden), dtype=torch.bfloat16, device='cuda'),
                    torch.full((bucket, self.topk), -1, dtype=torch.int64, device='cuda'),
                    torch.zeros((bucket, self.topk), dtype=torch.float32, device='cuda'),
                    torch.cuda.Stream())
            workspace = self.workspaces[rank][bucket]
            weight = self._weights(rank, layer)
            x, ids, scores, stream = self.inputs[rank][bucket]
            def run():
                if not self.rank_local_combine:
                    # Direct return folds routing scores into expert stores.
                    # Invalid routes perform no store; clear their old slots
                    # before the collective's start barrier and final reduction.
                    # The first eager call uses a newly zeroed allocation.
                    for frontend in workspace._frontends.values():
                        if frontend._compiled is not None:
                            frontend._compiled.execution.combine_output.view(torch.uint8).zero_()
                self.staging.stage_inputs(x, scores, ids, workspace.x, workspace.x_scale,
                                          workspace.topk_ids, workspace.topk_weights, quantize_input=True, scales=None)
                return runtime.run_split_mega_moe(workspace, weight)
            with torch.cuda.stream(stream):
                run()
                stream.synchronize()
                self.barrier.wait(timeout=300)
                graph = torch.cuda.CUDAGraph(keep_graph=True)
                with torch.cuda.graph(graph, stream=stream, capture_error_mode='thread_local'):
                    run()
                graph.instantiate()
            self.graphs[rank][layer, bucket] = graph
            return dict(device=self.devices[rank], capacity=bucket,
                        x=hex(x.data_ptr()), indices=hex(ids.data_ptr()), scores=hex(scores.data_ptr()),
                        output=hex(workspace.output.data_ptr()), graph=hex(graph.raw_cuda_graph()),
                        graph_exec=hex(graph.raw_cuda_graph_exec()))

    def prepare(self, layer, bucket):
        import torch
        for device in self.devices:
            torch.cuda.synchronize(device)
        def prepare_rank(rank):
            try:
                return self._prepare_rank(rank, layer, bucket)
            except BaseException as error:
                self.arena.abort(error)
                self.barrier.abort()
                raise
        futures = [self.pool.submit(prepare_rank, rank)
                   for rank in range(len(self.devices))]
        return [future.result() for future in futures]

    def close(self):
        import torch
        for rank, device in enumerate(self.devices):
            with torch.cuda.device(device):
                torch.cuda.synchronize()
                self.graphs[rank].clear()
                for workspace in self.workspaces[rank].values():
                    workspace.destroy()
                self.workspaces[rank].clear()
                self.inputs[rank].clear()
                self.weights[rank].clear()
        self.arena.heaps.clear()
        self.pool.shutdown()
