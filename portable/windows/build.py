"""Build a complete, relocatable Windows runtime from a Windows ftllm wheel."""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile

ASSETS = Path(__file__).resolve().parent
REPO = ASSETS.parents[1]
RUNTIME_LOCK = json.loads((ASSETS / "runtime-lock.json").read_text(encoding="utf-8"))
PYTHON = RUNTIME_LOCK["python"]


def run(*args, **kwargs):
    print("[portable]", subprocess.list2cmdline([str(x) for x in args]), flush=True)
    return subprocess.run([str(x) for x in args], check=True, **kwargs)


def sha256(path):
    with open(path, "rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def archive_bundle(final, archive):
    files = sorted(p for p in final.rglob("*") if p.is_file() and p.name != "MANIFEST.sha256")
    (final / "MANIFEST.sha256").write_text("".join(f"{sha256(p)}  {p.relative_to(final).as_posix()}\n" for p in files), encoding="utf-8")
    print(f"[portable] Compressing {final}", flush=True)
    partial = archive.with_suffix(".zip.part")
    with zipfile.ZipFile(partial, "w", zipfile.ZIP_DEFLATED, compresslevel=6, allowZip64=True) as zip_out:
        for path in sorted(final.rglob("*")):
            if path.is_file():
                zip_out.write(path, path.relative_to(final.parent).as_posix())
    partial.replace(archive)
    archive.with_suffix(".zip.sha256").write_text(f"{sha256(archive)}  {archive.name}\n", encoding="ascii")
    print(f"[portable] Complete: {archive}\n[portable] Extracted directory: {final}", flush=True)


def copy_vc_runtime(runtime):
    vswhere = Path(os.environ["ProgramFiles(x86)"]) / "Microsoft Visual Studio/Installer/vswhere.exe"
    vs = Path(subprocess.check_output([
        str(vswhere), "-latest", "-products", "*", "-requires",
        "Microsoft.VisualStudio.Component.VC.Tools.x86.x64", "-property", "installationPath",
    ], text=True).strip())
    redists = sorted((vs / "VC/Redist/MSVC").glob("14.*"), key=lambda p: tuple(map(int, p.name.split("."))))
    if not redists:
        raise RuntimeError("Visual Studio x64 redistributable DLLs not found")
    dlls = list((redists[-1] / "x64").glob("Microsoft.VC*.CRT/*.dll"))
    dlls += list((redists[-1] / "x64").glob("Microsoft.VC*.OpenMP/*.dll"))
    if not any(p.name.lower() == "msvcp140.dll" for p in dlls):
        raise RuntimeError("Missing redistributable msvcp140.dll")
    for path in dlls:
        shutil.copy2(path, runtime / path.name)
    return {p.name: sha256(p) for p in dlls}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("wheel", "output", "cache", "constraints", "launcher"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--no-archive", action="store_true", help="Prepare runtime directory for the Electron packager")
    args = parser.parse_args()
    if sys.platform != "win32":
        parser.error("Windows x64 is required")
    inspect_wheel = runpy.run_path(str(REPO / "tools/windows/verify_wheel.py"))["inspect_wheel"]
    with zipfile.ZipFile(args.wheel) as wheel:
        metadata, features = inspect_wheel(wheel)
        if features.get("USE_NCCL"):
            raise RuntimeError("Portable Windows builds require USE_NCCL=OFF")
    revision = subprocess.check_output(["git", "rev-parse", "--short=8", "HEAD"], cwd=REPO, text=True).strip()
    cuda_label = "cu12-sm" + args.cuda_arch.replace("-real", "").replace("-virtual", "").replace(";", "_") if features.get("USE_CUDA") else "cpu"
    name = f"{metadata['Name']}-{metadata['Version']}-windows-x64-portable-{cuda_label}-{revision}"
    if not re.fullmatch(r"[A-Za-z0-9._+-]+", name):
        raise RuntimeError("Unsafe bundle filename")
    # Keep the extracted top-level folder short for Windows Explorer/MAX_PATH.
    # Full version/architecture provenance belongs in the ZIP filename/manifest.
    final = args.output / "ftllm"
    archive = args.output / (name + ".zip")
    if final.exists() or archive.exists():
        raise RuntimeError(f"Output already exists; choose another output directory: {final}")
    # Retain failed staging trees for diagnosis. Never delete user-selected paths.
    staging = Path(tempfile.mkdtemp(prefix=".p-", dir=args.output))
    bundle = staging / "ftllm"
    bundle.mkdir()
    python_archive = args.cache / PYTHON["filename"]
    if sha256(python_archive) != PYTHON["sha256"]:
        raise RuntimeError("Python archive SHA256 mismatch")
    with tarfile.open(python_archive) as tar:
        tar.extractall(bundle, filter="data")
    runtime = bundle / "runtime"
    (bundle / "python").rename(runtime)
    python = runtime / "python.exe"
    site = runtime / "Lib/site-packages"
    wheelhouse = args.cache / "wheelhouse-cp311-win_amd64"
    wheelhouse.mkdir(exist_ok=True)
    # Explicit extras cover optional tokenizers, document and video readers,
    # plus DLL auditing; none require a compiler on the target computer.
    dependencies = [str(args.wheel), "sentencepiece", "protobuf", "pypdf>=4", "openpyxl>=3.1",
                    "XlsxWriter>=3.1", "python-pptx>=1", "pefile==2024.8.26"]
    constraints = ["--constraint", str(args.constraints)]
    if not args.offline:
        run(python, "-I", "-m", "pip", "download", "--only-binary=:all:",
            "--dest", wheelhouse, "--find-links", wheelhouse, *constraints, *dependencies)
    run(python, "-I", "-m", "pip", "install", "--no-index", "--find-links", wheelhouse,
        "--no-compile", "--no-warn-script-location", *constraints, *dependencies)
    # Build the companion wheel in a staging copy, leaving the source tree clean.
    agent = staging / "agent-source"
    shutil.copytree(REPO / "tools/ftllm_agent_runtime", agent,
                    ignore=shutil.ignore_patterns("__pycache__", "build", "dist", "*.egg-info", "bin"))
    run(sys.executable, "-I", agent / "scripts/fetch_pi.py", "--archive", args.cache / RUNTIME_LOCK["pi"]["filename"],
        *(["--offline"] if args.offline else []))
    # Wheel adds a long .data/purelib prefix; keep its scratch directory short
    # so Pi's nested node_modules also build under longer output directories.
    run(sys.executable, "-I", "setup.py", "bdist_wheel", "--bdist-dir", staging / "w", cwd=agent)
    run(python, "-I", "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
        next((agent / "dist").glob("*.whl")))
    vc_runtime = copy_vc_runtime(runtime)
    # Pi is a separate process and needs its own app-local CRT search root.
    pi_bin = site / "ftllm_agent_runtime/bin"
    for dll in vc_runtime:
        shutil.copy2(runtime / dll, pi_bin / dll)
    shutil.copy2(ASSETS / "sitecustomize.py", site / "sitecustomize.py")
    # _pth prevents registry/user-site/PYTHONPATH leakage, including in children
    # launched directly through sys.executable without -I.
    (runtime / "python311._pth").write_text(".\nDLLs\nLib\nLib/site-packages\nimport site\n", encoding="ascii")
    shutil.copy2(args.launcher, bundle / "ftllm.exe")
    for asset in ("Launch.cmd", "python.cmd", "ftllm-check.cmd", "README.txt"):
        shutil.copy2(ASSETS / asset, bundle / asset)
    (bundle / "libexec").mkdir()
    shutil.copy2(ASSETS / "check.py", bundle / "libexec/check.py")
    (bundle / "THIRD-PARTY").mkdir()
    shutil.copy2(REPO / "LICENSE", bundle / "THIRD-PARTY/FastLLM-LICENSE")
    (bundle / "tools").mkdir()
    for tool in ("rg", "fd"):
        asset = RUNTIME_LOCK[tool]
        tool_archive = args.cache / asset["filename"]
        if sha256(tool_archive) != asset["sha256"]:
            raise RuntimeError(f"{tool} SHA256 mismatch")
        with zipfile.ZipFile(tool_archive) as tool_zip:
            for member in tool_zip.namelist():
                leaf = Path(member).name
                if leaf == tool + ".exe":
                    (bundle / "tools" / leaf).write_bytes(tool_zip.read(member))
                elif leaf.startswith(("LICENSE", "COPYING", "UNLICENSE")):
                    (bundle / "THIRD-PARTY" / (tool + "-" + leaf)).write_bytes(tool_zip.read(member))
    (bundle / "THIRD-PARTY/SOURCES.txt").write_text(
        "".join(f"{name}: {RUNTIME_LOCK[name]['source']}\n" for name in ("python", "pi", "rg", "fd"))
        + "Microsoft VC runtime: https://learn.microsoft.com/cpp/windows/redistributing-visual-cpp-files\n"
        "Python and NVIDIA package licenses: runtime/Lib/site-packages/*dist-info/\n", encoding="utf-8")
    lock = run(python, "-I", "-m", "pip", "list", "--format=freeze", capture_output=True, text=True).stdout
    (bundle / "requirements-lock.txt").write_text(lock, encoding="utf-8")
    run(python, "-I", "-m", "pip", "check")
    write_json(bundle / "BUILD-INFO.json", {
        "format_version": 1, "package": {"name": metadata["Name"], "version": metadata["Version"]},
        "source_revision": revision, "source_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO)),
        "python": {key: PYTHON[key] for key in ("version", "build", "sha256")},
        "wheel": {"filename": args.wheel.name, "sha256": sha256(args.wheel)},
        "features": features, "cuda_architectures": args.cuda_arch.split(";") if features.get("USE_CUDA") else [],
        "vc_runtime": vc_runtime, "constraints_sha256": sha256(args.constraints),
        "built_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "smoke_tests": "skipped" if args.skip_tests else "clean-environment + relocated path",
    })
    # Move before checking: this catches absolute paths in generated executables.
    # Deliberately include spaces and non-ASCII characters in the test path.
    relocated_parent = staging / "测试 a"
    relocated_parent.mkdir()
    relocated = relocated_parent / "ftllm"
    bundle.rename(relocated)
    # pip's generated EXEs contain absolute interpreter paths. Remove these
    # before testing so the audit and smoke checks cover exactly what ships.
    # aria2's real binary stays in site-packages/aria2c/bin on PATH.
    for path in (relocated / "runtime/Scripts").glob("*.exe"):
        path.unlink()
    for path in (relocated / "runtime/Lib/site-packages").glob("*.dist-info/direct_url.json"):
        path.unlink()
    if not args.skip_tests:
        clean_env = {k: v for k, v in os.environ.items() if not k.upper().startswith(("PYTHON", "CUDA", "CONDA", "VIRTUAL_ENV", "PIP_"))}
        clean_env["PATH"] = str(Path(os.environ["SystemRoot"]) / "System32")
        check_args = ["--smoke", "--audit"] + (["--require-cuda"] if args.require_cuda else [])
        run(relocated / "runtime/python.exe", "-I", "-B", "-X", "utf8", relocated / "libexec/check.py",
            *check_args, env=clean_env, cwd=staging)
    # Remove generated bytecode before hashing the final tree.
    for path in relocated.rglob("__pycache__"):
        shutil.rmtree(path)
    relocated.rename(final)
    if args.no_archive:
        print(f"[portable] Runtime prepared: {final}", flush=True)
    else:
        archive_bundle(final, archive)


if __name__ == "__main__":
    main()
