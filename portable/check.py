#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


MIN_GLIBC = (2, 35)
MIN_DRIVER = (525, 60, 13)
MIN_COMPUTE_CAPABILITY = (6, 0)


def parse_version(value: str, width: int = 3) -> tuple[int, ...]:
    parts: list[int] = []
    for piece in value.strip().split("."):
        digits = "".join(character for character in piece if character.isdigit())
        if not digits:
            break
        parts.append(int(digits))
    return tuple((parts + [0] * width)[:width])


def glibc_version() -> tuple[int, int]:
    try:
        libc = ctypes.CDLL(None)
        libc.gnu_get_libc_version.restype = ctypes.c_char_p
        value = libc.gnu_get_libc_version().decode("ascii")
        major, minor = value.split(".", 1)
        return int(major), int(minor)
    except Exception:
        _, value = platform.libc_ver()
        parsed = parse_version(value, 2)
        return parsed[0], parsed[1]


def query_nvidia_smi_driver() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    versions = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return versions[0] if versions else None


def query_cuda_devices() -> tuple[int, list[tuple[str, int, int]]]:
    driver = ctypes.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ctypes.c_uint]
    driver.cuInit.restype = ctypes.c_int
    driver.cuDriverGetVersion.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDriverGetVersion.restype = ctypes.c_int
    driver.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
    driver.cuDeviceGetCount.restype = ctypes.c_int
    driver.cuDeviceGetName.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    driver.cuDeviceGetName.restype = ctypes.c_int
    driver.cuDeviceComputeCapability.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_int,
    ]
    driver.cuDeviceComputeCapability.restype = ctypes.c_int

    result = driver.cuInit(0)
    if result != 0:
        raise RuntimeError(f"cuInit 返回错误码 {result}")

    api_version = ctypes.c_int()
    if driver.cuDriverGetVersion(ctypes.byref(api_version)) != 0:
        api_version.value = 0

    count = ctypes.c_int()
    result = driver.cuDeviceGetCount(ctypes.byref(count))
    if result != 0:
        raise RuntimeError(f"cuDeviceGetCount 返回错误码 {result}")

    devices: list[tuple[str, int, int]] = []
    for index in range(count.value):
        name_buffer = ctypes.create_string_buffer(256)
        major = ctypes.c_int()
        minor = ctypes.c_int()
        if driver.cuDeviceGetName(name_buffer, len(name_buffer), index) != 0:
            name = f"GPU {index}"
        else:
            name = name_buffer.value.decode("utf-8", errors="replace")
        result = driver.cuDeviceComputeCapability(
            ctypes.byref(major), ctypes.byref(minor), index
        )
        if result != 0:
            raise RuntimeError(
                f"cuDeviceComputeCapability({index}) 返回错误码 {result}"
            )
        devices.append((name, major.value, minor.value))

    return api_version.value, devices


def main() -> int:
    parser = argparse.ArgumentParser(description="检查 ftllm 绿色包运行环境")
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="没有可用 CUDA 驱动/GPU 或 GPU 低于 SM60 时返回失败",
    )
    args = parser.parse_args()

    failures: list[str] = []
    warnings: list[str] = []

    def ok(message: str) -> None:
        print(f"[ OK ] {message}")

    def warn(message: str) -> None:
        warnings.append(message)
        print(f"[WARN] {message}")

    def fail(message: str) -> None:
        failures.append(message)
        print(f"[FAIL] {message}")

    bundle_root = Path(__file__).resolve().parent.parent
    info_path = bundle_root / "BUILD-INFO.json"
    try:
        build_info = json.loads(info_path.read_text(encoding="utf-8"))
        package = build_info["package"]
        ok(f"绿色包：{package['name']} {package['version']}")
    except Exception as error:
        fail(f"无法读取 BUILD-INFO.json：{error}")

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        fail(f"系统架构不支持：{platform.system()} {platform.machine()}")
    else:
        ok(f"系统架构：Linux {platform.machine()}")

    current_glibc = glibc_version()
    if current_glibc < MIN_GLIBC:
        fail(
            f"glibc {current_glibc[0]}.{current_glibc[1]} 过低，"
            f"至少需要 {MIN_GLIBC[0]}.{MIN_GLIBC[1]}"
        )
    else:
        ok(f"glibc {current_glibc[0]}.{current_glibc[1]}")

    required_distributions = (
        "ftllm",
        "transformers",
        "fastapi",
        "nvidia-cuda-runtime-cu12",
        "nvidia-cublas-cu12",
        "nvidia-nccl-cu12",
    )
    for distribution_name in required_distributions:
        try:
            version = importlib.metadata.version(distribution_name)
            ok(f"Python 依赖：{distribution_name} {version}")
        except importlib.metadata.PackageNotFoundError:
            fail(f"缺少 Python 依赖：{distribution_name}")

    driver_branch = query_nvidia_smi_driver()
    cuda_available = False
    try:
        cuda_api_version, devices = query_cuda_devices()
        cuda_available = bool(devices)
        api_major = cuda_api_version // 1000
        api_minor = (cuda_api_version % 1000) // 10
        ok(f"CUDA Driver API：{api_major}.{api_minor}")
        if not devices:
            message = "CUDA 驱动未枚举到 GPU"
            fail(message) if args.require_cuda else warn(message)
        for index, (name, major, minor) in enumerate(devices):
            capability = (major, minor)
            message = f"GPU {index}：{name}，SM{major}{minor}"
            if capability < MIN_COMPUTE_CAPABILITY:
                fail(f"{message}，低于最低 SM60")
            else:
                ok(message)
    except (OSError, RuntimeError) as error:
        message = f"CUDA 驱动不可用：{error}"
        fail(message) if args.require_cuda else warn(message)

    if driver_branch is None:
        message = "无法通过 nvidia-smi 读取 NVIDIA 驱动分支版本"
        fail(message) if args.require_cuda else warn(message)
    elif parse_version(driver_branch) < MIN_DRIVER:
        fail(f"NVIDIA 驱动 {driver_branch} 过低，至少需要 525.60.13")
    else:
        ok(f"NVIDIA 驱动：{driver_branch}")

    try:
        from ftllm import llm

        loaded_library = os.path.basename(str(llm.fastllm_lib._name))
        ok(f"FastLLM 原生库：{loaded_library}")
        if args.require_cuda and "-cpu" in loaded_library:
            fail("只加载了 CPU fallback，没有加载 CUDA 原生库")
        elif cuda_available and not llm.has_device("cuda"):
            fail("FastLLM 原生库没有注册 CUDA 后端")
    except SystemExit as error:
        fail(f"FastLLM 原生库加载时退出：{error}")
    except BaseException as error:
        fail(f"FastLLM 原生库加载失败：{error}")

    if failures:
        print(f"\n检查失败：{len(failures)} 项失败，{len(warnings)} 项警告。")
        return 1
    print(f"\n检查通过：{len(warnings)} 项警告。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
