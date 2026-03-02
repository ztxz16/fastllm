import os
import sys

TOOLS_DIR = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "..", "tools"
))
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)


def get_cache_path(repo_id: str) -> str:
    try:
        from fastllm_pytools.util import get_fastllm_cache_path
        return get_fastllm_cache_path(repo_id)
    except ImportError:
        home = os.path.expanduser("~")
        return os.path.join(home, ".cache", "fastllm", repo_id)


def download_model(repo_id: str, local_dir: str = "") -> str:
    if not local_dir:
        local_dir = get_cache_path(repo_id)

    try:
        from fastllm_pytools.download import HFDNormalDownloader
        downloader = HFDNormalDownloader(repo_id, local_dir=local_dir)
        downloader.run()
        return str(downloader.local_dir)
    except ImportError:
        raise RuntimeError(
            "fastllm_pytools not found. Please ensure the fastllm tools directory is available."
        )
