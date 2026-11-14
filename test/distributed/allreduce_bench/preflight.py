"""在独立进程检查已有 GPU 软件和六种卡对；不安装、不构建扩展。"""

import argparse
import ctypes
import hashlib
import itertools
import json
from pathlib import Path
import subprocess
import sys
import traceback


def loaded_cudart():
    paths = {line.split()[-1] for line in Path("/proc/self/maps").read_text().splitlines()
             if "libcudart.so" in line and line.split()[-1].startswith("/")}
    if len(paths) != 1:
        raise RuntimeError(f"需要唯一的已加载 CUDA runtime，实际为 {sorted(paths)}")
    runtime = ctypes.CDLL(paths.pop())
    runtime.cudaDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
    runtime.cudaDeviceGetP2PAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int,
                                                ctypes.c_int, ctypes.c_int]
    return runtime


def attribute(function, *args):
    value = ctypes.c_int()
    code = function(ctypes.byref(value), *args)
    if code != 0:
        raise RuntimeError(f"CUDA attribute query failed: {code}, args={args}")
    return value.value


def gpu_probe(device_ids, result):
    import torch

    result.update(torch=torch.__version__, torch_cuda=torch.version.cuda,
                  torch_arches=torch.cuda.get_arch_list(), gpus=[], pairs=[])
    if not torch.cuda.is_available():
        raise RuntimeError("当前 Python 环境没有可用的 CUDA PyTorch")
    if max(device_ids) >= torch.cuda.device_count():
        raise ValueError("请求的逻辑设备超出 CUDA_VISIBLE_DEVICES 可见范围")
    for device in device_ids:
        with torch.cuda.device(device):
            # 实际运行已有 CUDA kernel，不能只相信架构列表。
            value = torch.ones(4, device=f"cuda:{device}")
            if float((value + value).sum().item()) != 8.0:
                raise RuntimeError(f"device {device}: CUDA tensor smoke failed")
            prop = torch.cuda.get_device_properties(device)
            result["gpus"].append({"device": device, "name": prop.name,
                                   "uuid": str(getattr(prop, "uuid", "unknown")),
                                   "major": prop.major, "minor": prop.minor,
                                   "sm_count": prop.multi_processor_count,
                                   "total_memory": prop.total_memory})
    runtime = loaded_cudart()
    for gpu in result["gpus"]:
        # cudaDevAttrUnifiedAddressing 的稳定 ABI 编号为 41。
        gpu["uva"] = bool(attribute(runtime.cudaDeviceGetAttribute, 41, gpu["device"]))
    for first, second in itertools.combinations(device_ids, 2):
        access = [torch.cuda.can_device_access_peer(first, second),
                  torch.cuda.can_device_access_peer(second, first)]
        # cudaDevP2PAttrNativeAtomicSupported 的稳定 ABI 编号为 3。
        atomics = [attribute(runtime.cudaDeviceGetP2PAttribute, 3, a, b)
                   for a, b in ((first, second), (second, first))]
        properties = [gpu for gpu in result["gpus"] if gpu["device"] in (first, second)]
        ft_candidate = all(access) and all(atomics) and all(
            p["major"] >= 7 and p["sm_count"] >= 16 and p["uva"] for p in properties)
        result["pairs"].append({"devices": [first, second], "p2p": access,
                                "native_atomics": atomics, "ft_candidate": ft_candidate,
                                "vllm_p2p_candidate": all(access)})

    # 只加载现有扩展；真正的跨卡自定义调用在 smoke 阶段验证。
    import vllm
    from vllm import _custom_ops  # noqa: F401
    for op in ("init_custom_ar", "all_reduce", "register_buffer"):
        str(getattr(torch.ops._C_custom_ar, op).default._schema)
    root = Path(vllm.__file__).resolve().parent
    result["vllm"] = {"version": getattr(vllm, "__version__", "unknown"),
                      "module": str(root), "python_sources": {}}
    for name in ("distributed/device_communicators/custom_all_reduce.py",
                 "distributed/device_communicators/cuda_communicator.py"):
        path = root / name
        result["vllm"]["python_sources"][name] = hashlib.sha256(path.read_bytes()).hexdigest()
    result["nccl_version"] = torch.cuda.nccl.version()
    result["nccl_libraries"] = sorted({line.split()[-1]
        for line in Path("/proc/self/maps").read_text().splitlines() if "libnccl.so" in line})
    return result


def run(args, output, meta):
    from .plan import devices
    from .reporting import capture

    meta["nvidia_smi"] = capture(["nvidia-smi"])
    meta["topology"] = capture(["nvidia-smi", "topo", "-m"])
    meta["vllm_git"] = capture(["git", "-C", args.vllm_root, "rev-parse", "HEAD"])
    meta["vllm_dirty"] = capture(["git", "-C", args.vllm_root, "status", "--short"])
    probe_file = output / "probe.json"
    command = [args.python, str(Path(__file__).resolve()), "--devices", args.devices,
               "--output", str(probe_file)]
    meta["probe_command"] = command
    with (output / "probe.stdout.log").open("w") as stdout, (output / "probe.stderr.log").open("w") as stderr:
        result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=args.timeout)
    if probe_file.exists():
        meta["probe"] = json.loads(probe_file.read_text())
    if result.returncode:
        raise RuntimeError(f"GPU/vLLM 预检失败，见 {output / 'probe.stderr.log'}")
    probe = meta["probe"]
    if len(probe["gpus"]) != len(devices(args.devices)):
        raise RuntimeError("GPU 预检报告不完整")
    rows = []
    for pair in probe["pairs"]:
        rows.append(dict(case="pair_" + "_".join(map(str, pair["devices"])), stage="preflight",
                         devices=pair["devices"], status="PASS", **{k: v for k, v in pair.items() if k != "devices"},
                         reason="候选卡对；尚未证明自定义 kernel 能执行"))
    candidates = [pair["devices"] for pair in probe["pairs"] if pair["ft_candidate"]]
    meta["candidate_pairs"] = candidates
    meta["compile_architecture"] = "native adapter not built in preflight"
    if not candidates:
        rows.append(dict(case="no_common_pair", stage="preflight", status="FAIL",
                         reason="没有满足 FT 双向 P2P/native atomics 条件的卡对"))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--devices", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    data = {}
    try:
        gpu_probe([int(v) for v in args.devices.split(",")], data)
    except Exception as exc:
        data["error"] = str(exc)
        Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2))
        traceback.print_exc()
        raise SystemExit(1)
    Path(args.output).write_text(json.dumps(data, ensure_ascii=False, indent=2))
