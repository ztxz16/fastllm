__all__ = ["llm"]

from importlib.metadata import version, PackageNotFoundError

for _pkg in ("ftllm", "ftllm-nightly", "ftllm-rocm"):
    try:
        __version__ = version(_pkg)
        break
    except PackageNotFoundError:
        continue
else:
    __version__ = "unknown"