"""Build a small wheel to catch regressions in native-file installation layout."""

from pathlib import Path
import platform
import shutil
import subprocess
import sys
import zipfile

import pytest


@pytest.mark.skipif(
    not (sys.platform.startswith("linux") or sys.platform == "win32")
    or platform.machine().lower() not in {"x86_64", "amd64"},
    reason="the runtime wheel targets Linux and Windows x86-64",
)
def test_wheel_installs_binaries_in_platlib_without_cpython_abi(tmp_path):
    source = Path(__file__).resolve().parents[1]
    for name in ("setup.py", "pyproject.toml", "README.md", "LICENSE"):
        shutil.copy2(source / name, tmp_path / name)
    package = tmp_path / "src/ftllm_agent_runtime"
    shutil.copytree(
        source / "src/ftllm_agent_runtime", package,
        ignore=shutil.ignore_patterns("bin", "__pycache__"),
    )
    # Placeholder payloads keep this layout test offline and small; the release
    # smoke test exercises the actual upstream executables separately.
    payloads = (
        "bin/pi", "bin/rg", "bin/fd", "bin/package.json", "bin/photon_rs_bg.wasm",
        "bin/theme/dark.json", "bin/theme/light.json", "bin/theme/theme-schema.json",
        "licenses/agent-tools/manifest.json", "licenses/agent-tools/rg/COPYING",
        "licenses/agent-tools/rg/LICENSE-MIT", "licenses/agent-tools/rg/UNLICENSE",
        "licenses/agent-tools/fd/LICENSE-APACHE", "licenses/agent-tools/fd/LICENSE-MIT",
    )
    executables = ("pi", "rg", "fd")
    if sys.platform == "win32":
        payloads = tuple(name for name in payloads if name not in {"bin/pi", "bin/rg", "bin/fd"}
                         and not name.startswith("licenses/agent-tools/"))
        payloads += ("bin/pi.exe", "bin/node_modules/test/native/clipboard.node")
        executables = ("pi.exe",)
    for name in payloads:
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("test payload\n", encoding="utf-8")
    for name in executables:
        (package / "bin" / name).chmod(0o755)

    result = subprocess.run(
        [sys.executable, "setup.py", "bdist_wheel"], cwd=tmp_path,
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = (tmp_path / "dist").glob("*.whl")
    tag = "win_amd64" if sys.platform == "win32" else "linux_x86_64"
    assert wheel.name.endswith(f"-py3-none-{tag}.whl")
    with zipfile.ZipFile(wheel) as archive:
        assert not any(".data/purelib/" in name for name in archive.namelist())
        metadata, = (name for name in archive.namelist() if name.endswith("/WHEEL"))
        assert "Root-Is-Purelib: false" in archive.read(metadata).decode()
        for name in executables:
            info = archive.getinfo(f"ftllm_agent_runtime/bin/{name}")
            if sys.platform != "win32":
                assert (info.external_attr >> 16) & 0o111 == 0o111
        if sys.platform == "win32":
            assert archive.read("ftllm_agent_runtime/bin/node_modules/test/native/clipboard.node")

    (package / "bin" / executables[0]).unlink()
    result = subprocess.run([sys.executable, "setup.py", "bdist_wheel"], cwd=tmp_path,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode != 0
    assert "Incomplete Pi runtime" in result.stderr + result.stdout
