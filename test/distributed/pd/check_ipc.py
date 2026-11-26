"""真实单机跨进程 CUDA IPC 验证；走正式映射/异步任务/传输后端，不替代模型组合验收。"""

import argparse
import base64
import ctypes as C
from dataclasses import replace
import json
import multiprocessing as mp
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
from fastllm_pytools.pd.protocol import Peer, Identity, Layout, Region, Manifest, ProtocolError, token_digest
from fastllm_pytools.pd.mapping import plan_transfer
from fastllm_pytools.pd.cuda_ipc import CudaIPC, _Handle
from fastllm_pytools.pd.tasks import TransferManager, State
from report import run


def value(layer, component, token, head, dim):
    return float(layer * 400 + (component == "v") * 200 + token * 8 + head * 2 + dim % 2)


def owner(pipe, source, tp, paged, devices, mode):
    """/** 子进程持有真实 GPU 分配；父进程仅收到 IPC 元数据，结束通知后才释放。 */"""
    allocations = []
    try:
        backend = CudaIPC()
        backend._start()
        cuda = backend._cuda
        cuda.cudaMalloc.argtypes = [C.POINTER(C.c_void_p), C.c_size_t]
        cuda.cudaFree.argtypes = [C.c_void_p]
        cuda.cudaMemcpy.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.c_int]
        peers = Peer("p", "boot-p", "host"), Peer("d", "boot-d", "host")
        identity = Identity("request", "transfer", *peers, "p_to_d", mode)
        layout = Layout("test-model", 2, 4, 8, "float16", tp, "paged" if paged else "continuous", 8 if paged else 0)
        uuid = {ordinal: key for key, ordinal in backend._devices.items()}
        regions, expected = [], {}
        for rank in range(tp):
            device = devices[(rank + int(not source)) % len(devices)]
            backend._call("cudaSetDevice", device)
            hb, he = rank * 4 // tp, (rank + 1) * 4 // tp
            heads, tokens, dim = he - hb, 17, 8
            for layer in range(2):
                for component in ("k", "v"):
                    # 分页采用 NHD；连续测试采用 HND，以验证 stride 重排。
                    token_stride, head_stride = (heads * dim * 2, dim * 2) if paged else (dim * 2, tokens * dim * 2)
                    size = tokens * heads * dim * 2
                    guard = 32
                    data = bytearray(b"\xcd" * (size + guard * 2))
                    correct = bytearray(data)
                    for t in range(tokens):
                        for h in range(hb, he):
                            for d in range(dim):
                                offset = guard + t * token_stride + (h - hb) * head_stride + d * 2
                                correct[offset:offset + 2] = struct.pack("<e", value(layer, component, t, h, d))
                                data[offset:offset + 2] = correct[offset:offset + 2] if source else struct.pack("<e", -100)
                    address = C.c_void_p()
                    backend._call("cudaMalloc", C.byref(address), len(data))
                    allocations.append((device, address, len(data)))
                    upload = (C.c_ubyte * len(data)).from_buffer(data)
                    backend._call("cudaMemcpy", address, C.cast(upload, C.c_void_p), len(data), 1)
                    handle = _Handle()
                    cuda.cudaIpcGetMemHandle.argtypes = [C.POINTER(_Handle), C.c_void_p]
                    backend._call("cudaIpcGetMemHandle", C.byref(handle), address)
                    encoded = base64.b64encode(bytes(handle.data)).decode()
                    expected[encoded] = bytes(correct).hex()
                    for begin in range(0, tokens, 8 if paged else tokens):
                        end = min(tokens, begin + (8 if paged else tokens))
                        regions.append(Region(rank, uuid[device], layer, component, begin, end, hb, he,
                            encoded, len(data), guard + begin * token_stride, token_stride, head_stride))
        manifest = Manifest(1, identity, peers[0 if source else 1], layout, 0, 17,
                            token_digest(list(range(17))), tuple(regions))
        pipe.send(dict(manifest=manifest.to_json(), pid=os.getpid()))
        while True:
            action = pipe.recv()
            if action == "read":
                actual = {}
                for (device, address, size), handle in zip(allocations, expected):
                    backend._call("cudaSetDevice", device)
                    result = (C.c_ubyte * size)()
                    backend._call("cudaMemcpy", C.cast(result, C.c_void_p), address, size, 2)
                    actual[handle] = bytes(result).hex()
                pipe.send(dict(actual=actual, expected=expected))
            elif action == "close":
                for device, address, _ in allocations:
                    backend._call("cudaSetDevice", device)
                    backend._call("cudaFree", address)
                allocations.clear()
                pipe.send(dict(closed=True))
                return
            else:
                raise ValueError("unknown test owner command")
    except BaseException as error:
        pipe.send(dict(error=repr(error)))
        raise


def receive(pipe):
    if not pipe.poll(90):
        raise TimeoutError("GPU owner response timeout")
    result = pipe.recv()
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def ipc_case(devices, p, d, mode, fault=None):
    context = mp.get_context("spawn")
    children, connections = [], []
    manager = None
    drained = False
    try:
        descriptions = []
        for source, tp, paged in ((True, p, p % 2 == 0), (False, d, p % 2 != 0)):
            parent, child = context.Pipe()
            process = context.Process(target=owner, args=(child, source, tp, paged, devices, mode))
            # 致命 CUDA 故障时保留分配到本测试进程整体退出；不在未排空时发送 close 复用内存。
            process.daemon = True
            process.start()
            child.close()
            children.append(process); connections.append(parent)
            descriptions.append(receive(parent))
        source, target = (Manifest.from_json(item["manifest"]) for item in descriptions)
        backend = CudaIPC()
        released = []
        manager = TransferManager(target.owner if mode == "pull" else source.owner, backend)
        connections[1].send("read")
        before = receive(connections[1])["actual"]
        if fault and fault.startswith("reject_"):
            try:
                bad = target
                if fault == "reject_request":
                    bad = replace(target, identity=replace(target.identity, request="other"))
                elif fault == "reject_boot":
                    bad = replace(target, identity=replace(target.identity,
                                  source=replace(target.identity.source, boot="old")))
                elif fault == "reject_digest":
                    bad = replace(target, prefix_digest="0" * 64)
                elif fault == "reject_overlap":
                    bad = replace(target, regions=target.regions + (target.regions[0],))
                else:
                    raise AssertionError("unknown fault case")
                manager.register(source, bad, lambda: released.append(True))
            except ProtocolError:
                pass
            else:
                raise AssertionError("invalid transfer accepted")
            manager.close(timeout=90)
            drained = True
            connections[1].send("read")
            if receive(connections[1])["actual"] != before or released or manager.statistics()["copied_bytes"]:
                raise AssertionError("rejected handoff modified actual GPU destination or released ownership")
            return dict(rejected=True, actual_gpu_unchanged=True, mode=mode)
        manager.register(source, target, lambda: released.append(True))
        if fault == "cancel_before":
            manager.cancel(source.identity)
        layers = range(1 if fault == "cancel_after_layer" else source.layout.layers)
        for layer in layers:
            for rank in sorted({c.source.rank for c in plan_transfer(source, target) if c.source.layer == layer}):
                manager.publish_ready(source.identity, rank, layer)
                manager.publish_ready(source.identity, rank, layer)  # 重复就绪不得重复提交。
        if fault == "cancel_after_layer":
            deadline = time.monotonic() + 90
            while manager.statistics()["completed_batches"] == 0:
                if time.monotonic() >= deadline:
                    raise TimeoutError("first layer never completed")
                time.sleep(0.005)
            manager.cancel(source.identity)
            for rank in range(source.layout.tp_size):
                manager.publish_ready(source.identity, rank, 1)  # 取消后迟到通知不得继续写。
        manager.seal(source.identity)
        completion = manager.notifications.get(timeout=90)
        expected_state = State.CANCELLED if fault else State.READY
        if completion.state != expected_state or released != [True]:
            raise AssertionError(f"transfer failed or release repeated: {completion}")
        manager.close(timeout=90)
        drained = True
        connections[1].send("read")
        result = receive(connections[1])
        expected = result["expected"]
        if fault == "cancel_before":
            expected = before
        elif fault == "cancel_after_layer":
            expected = dict(before)
            for region in target.regions:
                if region.layer == 0:
                    expected[region.memory] = result["expected"][region.memory]
        if result["actual"] != expected:
            raise AssertionError("actual GPU KV content or guard bytes differ")
        return dict(exact="PASS", source_tp=p, target_tp=d, mode=mode,
                    owner_pids=[c.pid for c in children], **manager.statistics())
    finally:
        if manager is not None and not drained:
            manager.close(timeout=90)
            drained = True
        for pipe, process in zip(connections, children):
            if process.is_alive():
                pipe.send("close")
                if receive(pipe).get("closed") is not True:
                    raise RuntimeError("GPU owner did not release memory")
            process.join(90)
            pipe.close()
            if process.exitcode != 0:
                raise RuntimeError(f"GPU owner failed: {process.pid} exit={process.exitcode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", default="0,1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    devices = [int(x) for x in args.devices.split(",")]
    if len(set(devices)) < 2:
        parser.error("at least two visible GPUs required")
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,uuid,driver_version,compute_cap", "--format=csv,noheader"], text=True)
    cases = [(f"ipc_{mode}_p{p}_d{d}", lambda p=p, d=d, m=mode: ipc_case(devices, p, d, m))
             for p, d in ((1, 1), (1, 2), (2, 1), (2, 2), (2, 4), (4, 2)) for mode in ("pull", "push")]
    cases += [(f"ipc_{mode}_{fault}", lambda m=mode, f=fault: ipc_case(devices, 2, 1, m, f))
              for mode in ("pull", "push") for fault in ("reject_request", "reject_boot", "reject_digest",
                  "reject_overlap", "cancel_before", "cancel_after_layer")]
    return run(cases, args.output, backend="production TransferManager + CudaIPC", metadata_extra=dict(gpu=gpu),
        scope="24 个跨进程 GPU 内容、拒绝后无写入、取消及迟到通知用例；逻辑分片复用物理 GPU，不代表模型 TP4 部署验收")


if __name__ == "__main__":
    raise SystemExit(main())
