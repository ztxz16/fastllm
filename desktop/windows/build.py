"""Assemble and test the Windows Electron application, then archive it."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[2]
ELECTRON = json.loads((REPO / "portable/windows/runtime-lock.json").read_text(encoding="utf-8"))["electron"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("runtime", "electron", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--smoke-model", type=Path)
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Windows is required")
    helpers = runpy.run_path(str(REPO / "portable/windows/build.py"))
    sha256, run = helpers["sha256"], helpers["run"]
    if sha256(args.electron) != ELECTRON["sha256"]:
        raise RuntimeError("Electron SHA256 mismatch")
    info = json.loads((args.runtime / "BUILD-INFO.json").read_text(encoding="utf-8"))
    version = info["package"]["version"]
    backend = "cpu"
    if info["features"].get("USE_CUDA"):
        backend = "cu12-sm" + "_".join(info["cuda_architectures"]).replace("-real", "").replace("-virtual", "")
    archive = args.output / f"FastLLM-Launcher-{version}-windows-x64-{backend}-{info['source_revision']}.zip"
    final = args.output / "FastLLM"
    if final.exists() or archive.exists():
        raise RuntimeError("Desktop output already exists; choose another output directory")
    bundle = args.runtime.parent / "测试 a" / "FastLLM"
    bundle.mkdir(parents=True)
    with zipfile.ZipFile(args.electron) as source:
        source.extractall(bundle)
    (bundle / "electron.exe").rename(bundle / "FastLLM-Launcher.exe")
    # This is the stock Electron demo, not part of this application.
    (bundle / "resources/default_app.asar").unlink()
    app = bundle / "resources/app"
    shutil.copytree(REPO / "desktop/app", app, ignore=shutil.ignore_patterns("__pycache__"))
    shutil.copy2(REPO / "tools/fastllm_pytools/launcher_assets/launcher-icon.png", app / "icon.png")
    package = json.loads((app / "package.json").read_text(encoding="utf-8"))
    package["version"] = version
    helpers["write_json"](app / "package.json", package)
    args.runtime.rename(bundle / "ftllm")
    # Desktop users have exactly one GUI entrypoint; the internal runtime keeps
    # its CLI/check commands, but must not advertise a browser-launch shortcut.
    (bundle / "ftllm/Launch.cmd").unlink()
    helpers["copy_vc_runtime"](bundle)
    shutil.copy2(REPO / "desktop/windows/README.txt", bundle / "README.txt")
    shutil.copy2(REPO / "LICENSE", bundle / "LICENSE-FastLLM")
    info["desktop"] = {"electron_version": ELECTRON["version"], "electron_sha256": ELECTRON["sha256"],
                       "entrypoint": "FastLLM-Launcher.exe", "platform": "win32-x64"}
    info["desktop"]["application_sha256"] = {
        p.relative_to(app).as_posix(): sha256(p) for p in sorted(app.rglob("*")) if p.is_file()
    }
    helpers["write_json"](bundle / "BUILD-INFO.json", info)
    python = bundle / "ftllm/runtime/python.exe"
    # Audit Electron AND all embedded Python/Pi/compute libraries, including
    # delay-load DLLs. VC runtime must be bundled, never borrowed from the host.
    run(python, "-I", "-B", "-X", "utf8", "-c",
        "import runpy,sys; from pathlib import Path; "
        "runpy.run_path(sys.argv[1])['audit'](Path(sys.argv[2]))",
        REPO / "portable/windows/check.py", bundle)
    if not args.skip_tests:
        node_env = dict(os.environ, ELECTRON_RUN_AS_NODE="1")
        executable = bundle / "FastLLM-Launcher.exe"
        def node_test(*arguments):
            # A GUI-subsystem EXE has no attached console on Windows. Pipes are
            # necessary to reliably collect Node test diagnostics under Python.
            result = subprocess.run([str(executable), *(str(arg) for arg in arguments)],
                                    env=node_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    encoding="utf-8", errors="replace")
            print(result.stdout, flush=True)
            result.check_returncode()
        node_test("--test", REPO / "desktop/tests/runtime.test.js")
        node_test(REPO / "desktop/tests/smoke_windows.js", bundle, args.output / "electron-test",
                  *([args.smoke_model] if args.smoke_model else []))
        info["desktop"]["smoke_tests"] = "real Electron renderer, embedded Studio, isolated PATH, window close + process cleanup, relocated Unicode path"
        info["desktop"]["model_tested"] = args.smoke_model.name if args.smoke_model else None
    else:
        info["desktop"]["smoke_tests"] = "skipped"
    helpers["write_json"](bundle / "BUILD-INFO.json", info)
    bundle.rename(final)
    helpers["archive_bundle"](final, archive)
    run(sys.executable, "-I", "-B", "-X", "utf8", REPO / "desktop/tests/verify_windows_archive.py", archive,
        "--report", args.output / "archive-verification.json")


if __name__ == "__main__":
    main()
