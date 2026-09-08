"""Bundle-local DLL and subprocess paths, also applied to launcher children."""
import os
import sys
from pathlib import Path

_runtime = Path(sys.executable).resolve().parent
_root = _runtime.parent
_dll_paths = [_runtime, _runtime / "DLLs", _runtime / "Lib/site-packages/ftllm"]
_dll_paths += list((_runtime / "Lib/site-packages/nvidia").glob("*/bin"))
_handles = [os.add_dll_directory(str(p)) for p in _dll_paths if p.is_dir()]
os.environ["PATH"] = os.pathsep.join(
    [str(p) for p in [_root, _root / "tools", _runtime,
                      _runtime / "Lib/site-packages/aria2c/bin", _runtime / "Scripts",
                      Path(os.environ["SystemRoot"]) / "System32/WindowsPowerShell/v1.0",
                      *_dll_paths] if p.is_dir()]
    + [os.environ.get("PATH", "")]
)
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["PYTHONNOUSERSITE"] = "1"
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ.setdefault("XDG_CONFIG_HOME", str(_root / "data/config"))
os.environ.setdefault("HF_HOME", str(_root / "data/huggingface"))
os.environ.setdefault("MODELSCOPE_CACHE", str(_root / "data/modelscope"))
os.environ.setdefault("FASTLLM_CACHE_DIR", str(_root / "data/cache"))
