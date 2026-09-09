"""Verify a Windows ftllm wheel, optionally against source and CUDA architectures."""
from __future__ import annotations

import argparse
import email
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def stream_hash(stream):
    digest = hashlib.sha256()
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
    return digest.hexdigest()


def sha256(path):
    with path.open("rb") as stream:
        return stream_hash(stream)


def member_hash(archive, name):
    with archive.open(name) as stream:
        return stream_hash(stream)


def check_architectures(requested, elf, ptx):
    real = {re.match(r"\d+", arch)[0] for arch in elf}
    virtual = {re.match(r"\d+", arch)[0] for arch in ptx}
    require(bool(real or virtual), "No embedded CUDA architectures found")
    for item in requested.split(";"):
        if item in {"native", "all", "all-major"}:
            continue  # Resolve these on the build host; report the actual images.
        match = re.fullmatch(r"(\d+)[af]?(?:-(real|virtual))?", item)
        require(match is not None, f"Invalid CUDA architecture: {item}")
        number, kind = match.groups()
        if kind != "virtual":
            require(number in real, f"Missing CUDA machine code for SM{number}: {elf}")
        if kind != "real":
            require(number in virtual, f"Missing CUDA PTX for SM{number}: {ptx}")


def find_cuobjdump(requested):
    if requested:
        return str(Path(requested).resolve()) if Path(requested).is_file() else requested
    command = shutil.which("cuobjdump")
    if command:
        return command
    if os.environ.get("CUDA_PATH"):
        candidate = Path(os.environ["CUDA_PATH"]) / "bin/cuobjdump.exe"
        if candidate.is_file():
            return str(candidate)
    raise ValueError("cuobjdump not found; add CUDA bin to PATH or use --cuobjdump")


def inspect_wheel(archive, *, version=None, backend=None):
    """Shared metadata and payload checks for wheel and portable builders."""
    names = archive.namelist()
    require(len(names) == len(set(names)), "Duplicate wheel members")
    metadata_names = [name for name in names if name.endswith(".dist-info/METADATA")]
    require(len(metadata_names) == 1, "Expected exactly one wheel METADATA file")
    prefix = metadata_names[0].rsplit("/", 1)[0]
    metadata = email.message_from_bytes(archive.read(metadata_names[0]))
    require(metadata["Name"] in {"ftllm", "ftllm-nightly"}, "Not an ftllm wheel")
    require(bool(metadata["Version"]), "Missing wheel version")
    if version:
        require(metadata["Version"] == version, f"Expected {version}, found {metadata['Version']}")
    tags = email.message_from_bytes(archive.read(prefix + "/WHEEL")).get_all("Tag")
    require(tags == ["py3-none-win_amd64"], f"Unexpected Windows wheel tags: {tags}")
    features = json.loads(archive.read("ftllm/build_info.json"))
    actual_backend = "cuda" if features.get("USE_CUDA") else "cpu"
    if backend:
        require(actual_backend == backend, f"Expected {backend} wheel, found {actual_backend}")
    required = ["__init__.py", "cli.py", "llm.py", "launcher.py", "fastllm_tools.dll",
                "fastllm_triton_server.py", "launcher_assets/index.html",
                "launcher_assets/locales/zh-CN.json", "ui_plugins/studio/app.js",
                "ui_plugins/studio/template.html", "ui_plugins/models/plugin.json", "plugin_assets/host.js"]
    if actual_backend == "cuda":
        required.append("fastllm_tools-cpu.dll")
    for name in required:
        require("ftllm/" + name in names, f"Missing wheel member: ftllm/{name}")
    entry_points = archive.read(prefix + "/entry_points.txt").decode().replace(" ", "")
    require("ftllm=ftllm.cli:main" in entry_points.splitlines(), "Missing ftllm CLI entrypoint")
    return metadata, features


def verify(wheel, *, version=None, backend=None, source_root=None, native_root=None,
           cuda_arch=None, cuobjdump=None, report_dir=None):
    with zipfile.ZipFile(wheel) as archive:
        metadata, features = inspect_wheel(archive, version=version, backend=backend)
        require(archive.testzip() is None, "Wheel CRC verification failed")
        actual_backend = "cuda" if features.get("USE_CUDA") else "cpu"
        if source_root:
            sources = source_root / "tools/fastllm_pytools"
            require(sources.is_dir(), f"Source directory not found: {sources}")
            for source in sorted(sources.rglob("*")):
                if source.is_file() and source.suffix != ".pyc" and "__pycache__" not in source.parts:
                    name = "ftllm/" + source.relative_to(sources).as_posix()
                    require(member_hash(archive, name) == sha256(source), f"Source asset mismatch: {name}")
            service = "fastllm_triton_server.py"
            require(member_hash(archive, "ftllm/" + service) == sha256(source_root / "tools" / service),
                    f"Source asset mismatch: ftllm/{service}")
        native_hashes = {"fastllm_tools.dll": member_hash(archive, "ftllm/fastllm_tools.dll")}
        if actual_backend == "cuda":
            native_hashes["fastllm_tools-cpu.dll"] = member_hash(archive, "ftllm/fastllm_tools-cpu.dll")
        if native_root:
            require(native_hashes["fastllm_tools.dll"] == sha256(
                native_root / actual_backend / "tools/ftllm/fastllm_tools.dll"), "Native DLL mismatch")
            if actual_backend == "cuda":
                require(native_hashes["fastllm_tools-cpu.dll"] == sha256(
                    native_root / "cpu/tools/ftllm/fastllm_tools.dll"), "CPU fallback DLL mismatch")
        elf, ptx = [], []
        if actual_backend == "cuda" and cuda_arch:
            command = find_cuobjdump(cuobjdump)
            with tempfile.TemporaryDirectory(prefix="ftllm-wheel-check-") as temp:
                dll = Path(temp) / "fastllm_tools.dll"
                with archive.open("ftllm/fastllm_tools.dll") as source, dll.open("wb") as target:
                    shutil.copyfileobj(source, target)
                for kind, result in (("elf", elf), ("ptx", ptx)):
                    output = subprocess.check_output([command, "--list-" + kind, str(dll)],
                                                     encoding="utf-8", errors="replace")
                    if report_dir:
                        (report_dir / f"cuda-{kind}.txt").write_text(output, encoding="utf-8")
                    result.extend(sorted(set(re.findall(r"\bsm_(\d+[af]?)\b", output)),
                                         key=lambda value: (int(re.match(r"\d+", value)[0]), value)))
            check_architectures(cuda_arch, elf, ptx)
    return {
        "passed": True, "package": metadata["Name"], "version": metadata["Version"],
        "wheel": wheel.name, "wheel_sha256": sha256(wheel), "wheel_bytes": wheel.stat().st_size,
        "features": features, "native_sha256": native_hashes,
        "cuda_architectures_checked": bool(actual_backend == "cuda" and cuda_arch),
        "cuda_elf_architectures": elf, "cuda_ptx_architectures": ptx,
        "source_assets_checked": source_root is not None, "native_build_checked": native_root is not None,
        "requires_dist": metadata.get_all("Requires-Dist", []),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version", help="Require this version; does not change the package version")
    parser.add_argument("--backend", choices=("cpu", "cuda"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--native-root", type=Path)
    parser.add_argument("--cuda-arch", help="Check actual machine code/PTX using the CMake architecture list")
    parser.add_argument("--cuobjdump", help="CUDA cuobjdump executable (otherwise PATH/CUDA_PATH)")
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    options = vars(args).copy()
    report = options.pop("report")
    if report:
        report.parent.mkdir(parents=True, exist_ok=True)
    result = verify(**options, report_dir=report.parent if report else None)
    output = json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    if report:
        report.write_text(output, encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
