"""Check the final ZIP checksum, every manifest entry and desktop payload."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def require(condition, message):
    if not condition:
        raise ValueError(message)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--report", type=Path, help="Write a machine-readable verification result")
    args = parser.parse_args()
    with args.archive.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    require(digest == args.archive.with_suffix(".zip.sha256").read_text(encoding="ascii").split()[0], "ZIP SHA256 mismatch")
    with zipfile.ZipFile(args.archive) as bundle:
        names = bundle.namelist()
        require(len(names) == len(set(names)), "Duplicate ZIP entries")
        entries = [line.split("  ", 1)[::-1] for line in
                   bundle.read("FastLLM/support/MANIFEST.sha256").decode("utf-8").splitlines()]
        manifest = dict(entries)
        require(len(entries) == len(manifest), "Duplicate manifest entries")
        require(set(names) == {"FastLLM/" + name for name in manifest} | {"FastLLM/support/MANIFEST.sha256"}, "ZIP/manifest members differ")
        require({name.split("/")[1] for name in names} == {
            "FastLLM-Launcher.exe", "ftllm-launch-webui.exe", "ftllm.exe", "README.html", "support"
        }, "Unexpected files in desktop root")
        for index, (name, expected) in enumerate(manifest.items(), 1):
            require(not Path(name).is_absolute() and ".." not in Path(name).parts, f"Unsafe archive member: {name}")
            with bundle.open("FastLLM/" + name) as stream:
                require(hashlib.file_digest(stream, "sha256").hexdigest() == expected, f"File hash mismatch: {name}")
            if index % 5000 == 0:
                print(f"[verify] {index}/{len(manifest)} files", flush=True)
        for name in ("FastLLM-Launcher.exe", "ftllm-launch-webui.exe", "ftllm.exe", "README.html",
                     "support/FastLLM-Launcher.exe", "support/resources/app/main.js", "support/runtime/python.exe"):
            require("FastLLM/" + name in names, f"Missing desktop payload: {name}")
        require("FastLLM/support/resources/default_app.asar" not in names, "Stock Electron demo in archive")
        require("FastLLM/support/Launch.cmd" not in names, "Redundant browser shortcut in desktop archive")
        require(not any(name.lower().endswith("/nvcuda.dll") for name in names), "NVIDIA driver must not be bundled")
        info = json.loads(bundle.read("FastLLM/support/BUILD-INFO.json"))
        for name, expected in info["desktop"]["application_sha256"].items():
            require(manifest["support/resources/app/" + name] == expected, f"Desktop source hash mismatch: {name}")
    print(f"[OK] ZIP SHA256 + CRC + {len(manifest)} file hashes + Electron payload\nSHA256 {digest}")
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps({
            "passed": True, "archive": args.archive.name, "sha256": digest,
            "file_count": len(manifest), "build_info": info,
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
