"""Collect the verified wheel and Electron archive into one release directory."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import shutil


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def collect(wheel, output):
    def read(name):
        return json.loads((output / name).read_text(encoding="utf-8"))

    wheel_info = read("wheel-verification.json")
    archive_info = read("archive-verification.json")
    info = read("FastLLM/BUILD-INFO.json")
    require(wheel_info["passed"] and archive_info["passed"], "Release verification did not pass")
    require(info["package"]["version"] == wheel_info["version"], "Wheel/desktop version mismatch")
    require(info["smoke_tests"] != "skipped" and info["desktop"]["smoke_tests"] != "skipped",
            "Release requires runtime and desktop smoke tests")
    require(archive_info["build_info"] == info, "Archive and extracted BUILD-INFO differ")
    wheel_sha = sha256(wheel)
    require(wheel_sha == wheel_info["wheel_sha256"] == info["wheel"]["sha256"], "Wheel SHA256 mismatch")
    require(wheel.name == info["wheel"]["filename"] == wheel_info["wheel"], "Wheel filename mismatch")
    if info["features"].get("USE_CUDA"):
        require(wheel_info["cuda_architectures_checked"], "CUDA architectures were not verified")
    archive_name = archive_info["archive"]
    require(Path(archive_name).name == archive_name, "Invalid archive filename")
    archive = output / archive_name
    archive_sha = sha256(archive)
    require(archive_sha == archive_info["sha256"], "Archive changed after verification")
    reports = sorted((output / "electron-test").glob("run-*/RESULT.json"), key=lambda path: path.stat().st_mtime)
    require(bool(reports), "Missing Electron test result")
    desktop = json.loads(reports[-1].read_text(encoding="utf-8"))
    require(all(desktop[key] for key in ("passed", "nativeWindowClose", "managedPortsClosed")),
            "Electron startup/shutdown checks did not pass")
    require((Path(desktop["model"]).name if desktop.get("model") else None) == info["desktop"]["model_tested"],
            "Model test result does not match the packaged build")
    destination = output / wheel.name
    if destination.exists():
        require(sha256(destination) == wheel_sha, f"Refusing to overwrite different wheel: {destination}")
    else:
        shutil.copy2(wheel, destination)
    require(sha256(destination) == wheel_sha, "Copied wheel SHA256 mismatch")
    artifacts = [{"file": path.name, "bytes": path.stat().st_size, "sha256": digest}
                 for path, digest in ((destination, wheel_sha), (archive, archive_sha))]
    for item in artifacts:
        (output / (item["file"] + ".sha256")).write_text(f"{item['sha256']}  {item['file']}\n", encoding="ascii")
    (output / "SHA256SUMS.txt").write_text(
        "".join(f"{item['sha256']}  {item['file']}\n" for item in artifacts), encoding="ascii")
    result = {
        "passed": True, "version": wheel_info["version"],
        "completed_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source_revision": info["source_revision"], "source_dirty": info["source_dirty"],
        "artifacts": artifacts, "wheel_checks": wheel_info, "desktop_checks": desktop,
        "archive_file_count": archive_info["file_count"],
        "gpu_inference_tested": bool(desktop.get("model")),
    }
    (output / "release-verification.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "README.md").write_text(
        f"# FastLLM {result['version']} Windows release\n\n"
        f"- Wheel: [{destination.name}]({destination.name})\n"
        f"- Electron portable ZIP: [{archive.name}]({archive.name})\n"
        "- Extract the entire ZIP and run `FastLLM/FastLLM-Launcher.exe`.\n"
        "- The extracted `FastLLM/` directory is also ready to run.\n\n"
        "Python, Electron, dependencies and user-space runtime libraries are bundled. "
        "Windows x64 and an AVX2 CPU are required; GPU mode requires an NVIDIA driver. "
        "Model weights are supplied separately.\n\n"
        "Checksums: `SHA256SUMS.txt`. Results: `release-verification.json`. "
        "Logs: `release-*.log`; Electron diagnostics: `electron-test/`.\n"
        f"GPU model inference tested: {result['gpu_inference_tested']}.\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(collect(args.wheel.resolve(), args.output.resolve()), ensure_ascii=False, indent=2))
