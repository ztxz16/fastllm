"""Read-only hardware service shared by Launcher and UI plugins."""
import os
import platform
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List


def _read_text(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as file:
            return file.read().strip()
    except OSError:
        return ""


def _memory_info() -> Dict[str, int]:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
                (name, ctypes.c_ulonglong) for name in (
                    "total", "available", "page_total", "page_available",
                    "virtual_total", "virtual_available", "extended_available",
                )
            ]

        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return {"total": status.total, "available": status.available}
        return {"total": 0, "available": 0}
    values = {}
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as file:
            for line in file:
                key, raw = line.split(":", 1)
                number = raw.strip().split()[0]
                values[key] = int(number) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return {
        "total": values.get("MemTotal", 0),
        "available": values.get("MemAvailable", 0),
    }


def _gpu_info() -> List[Dict[str, Any]]:
    executable = shutil.which("nvidia-smi")
    if not executable:
        return []
    query = (
        "index,name,memory.total,memory.free,utilization.gpu,temperature.gpu,"
        "driver_version"
    )
    try:
        result = subprocess.run(
            [
                executable,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=4,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    output = []
    for line in result.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 7:
            continue
        output.append({
            "index": parts[0],
            "name": parts[1],
            "memoryTotalMiB": parts[2],
            "memoryFreeMiB": parts[3],
            "utilization": parts[4],
            "temperature": parts[5],
            "driver": parts[6],
        })
    return output


def detect_hardware(model_path: str = "") -> Dict[str, Any]:
    cpu_model = ""
    try:
        with open("/proc/cpuinfo", "r", encoding="utf-8") as file:
            for line in file:
                if line.lower().startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except (OSError, IndexError):
        pass
    try:
        affinity = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        affinity = os.cpu_count() or 1

    numa_nodes = []
    node_root = Path("/sys/devices/system/node")
    if node_root.is_dir():
        for node in sorted(node_root.glob("node[0-9]*")):
            numa_nodes.append({
                "name": node.name,
                "cpus": _read_text(str(node / "cpulist")),
                "memory": _read_text(str(node / "meminfo")).splitlines()[:1],
            })

    disk_target = os.path.expanduser(model_path) if model_path else os.getcwd()
    if not os.path.exists(disk_target):
        disk_target = os.path.dirname(disk_target) or os.getcwd()
    try:
        disk = shutil.disk_usage(disk_target)
        disk_info = {"path": disk_target, "total": disk.total, "free": disk.free}
    except OSError:
        disk_info = {"path": disk_target, "total": 0, "free": 0}

    try:
        from .env import env
        build = dict(env.build_info)
    except Exception:
        build = {}
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu": {
            "model": cpu_model or platform.processor() or "Unknown CPU",
            "logical": os.cpu_count() or 1,
            "available": affinity,
        },
        "memory": _memory_info(),
        "gpus": _gpu_info(),
        "numa": numa_nodes,
        "disk": disk_info,
        "build": build,
    }
