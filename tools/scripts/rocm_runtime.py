"""Installed only into ftllm-rocm wheels as ftllm._rocm_init."""
import ctypes
import json
from pathlib import Path


def initialize():
    package = Path(__file__).resolve().parent
    info = json.loads((package / "rocm_build_info.json").read_text())
    version = info['sdk_version']
    # Minimal Linux installations may not provide libnuma.
    numa = package / 'libnuma.so.1'
    if numa.exists():
        ctypes.CDLL(str(numa), mode=ctypes.RTLD_GLOBAL)
    try:
        import rocm_sdk
        rocm_sdk.initialize_process(
            preload_shortnames=['amdhip64', 'hipblas', 'hipblaslt', 'rccl'],
            check_version=version,
            fail_on_version_mismatch=True,
            env_override=False,
        )
    except (ImportError, OSError, RuntimeError) as error:
        raise RuntimeError(
            f'ftllm-rocm requires the ROCm {version} Python runtime packages. '
            f'Install rocm[libraries]=={version} from '
            'https://stable.repo.amd.com/rocm/whl-next/ and a matching '
            'rocm-sdk-device-<gfx target> package. '
            f'Runtime initialization failed: {error}'
        ) from error
