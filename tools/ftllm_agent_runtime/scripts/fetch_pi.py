#!/usr/bin/env python3
"""Fetch and verify the pinned Pi Linux x86-64 standalone executable."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import platform
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path


PI_VERSION = "0.84.4"
ARCHIVE_NAME = "pi-linux-x64.tar.gz"
ARCHIVE_URL = (
    "https://github.com/earendil-works/pi/releases/download/"
    f"v{PI_VERSION}/{ARCHIVE_NAME}"
)
ARCHIVE_SHA256 = "c2f3c3e6a1850bd87654cc3ca8811013272397c3d042a4e2a64c43ee1b423972"
LICENSE_URL = (
    "https://raw.githubusercontent.com/earendil-works/pi/"
    f"v{PI_VERSION}/LICENSE"
)
LICENSE_SHA256 = "0457f5bcec3b3b211605dfb5d1a49042fd638f3686a410fe099c24a25af13c48"

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = ROOT / "src" / "ftllm_agent_runtime"
BINARY_PATH = PACKAGE_ROOT / "bin" / "pi"
PACKAGE_JSON_PATH = PACKAGE_ROOT / "bin" / "package.json"
PHOTON_WASM_PATH = PACKAGE_ROOT / "bin" / "photon_rs_bg.wasm"
THEME_ROOT = PACKAGE_ROOT / "bin" / "theme"
LICENSE_PATH = PACKAGE_ROOT / "licenses" / "PI_LICENSE"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(url: str) -> bytes:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": f"ftllm-agent-runtime/{PI_VERSION}"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return response.read()


def verified(data: bytes, expected: str, label: str) -> bytes:
    actual = sha256(data)
    if actual != expected:
        raise RuntimeError(
            f"{label} SHA-256 mismatch: expected {expected}, got {actual}"
        )
    return data


def extract_member(archive: bytes, name: str) -> bytes:
    with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as bundle:
        member = bundle.getmember(name)
        if not member.isfile():
            raise RuntimeError(f"Pinned Pi archive does not contain a regular {name} file")
        source = bundle.extractfile(member)
        if source is None:
            raise RuntimeError(f"Could not read {name} from the pinned Pi archive")
        return source.read()


def atomic_write(path: Path, data: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "wb") as temporary:
            temporary.write(data)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.chmod(temporary_name, mode)
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def cached_archive(url: str, digest: str, cached: Path | None, offline: bool) -> bytes:
    if cached is not None and cached.is_file():
        data = cached.read_bytes()
        if sha256(data) == digest:
            print(f"Using verified archive cache: {cached}", flush=True)
            return data
    if offline:
        raise RuntimeError(f"Offline build requires a valid cached archive: {cached or url}")
    print(f"Downloading {url}", flush=True)
    data = verified(fetch(url), digest, url.rsplit("/", 1)[-1])
    if cached is not None:
        atomic_write(cached, data, 0o644)
    return data


def load_archive(archive_path: Path | None, cache_dir: Path | None, offline: bool) -> bytes:
    if archive_path is not None:
        return verified(archive_path.expanduser().read_bytes(), ARCHIVE_SHA256, ARCHIVE_NAME)
    cached = cache_dir / f"pi-{PI_VERSION}-linux-x64.tar.gz" if cache_dir else None
    return cached_archive(ARCHIVE_URL, ARCHIVE_SHA256, cached, offline)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--archive",
        type=Path,
        help="Use an already-downloaded pi-linux-x64.tar.gz archive",
    )
    parser.add_argument("--cache-dir", type=Path, help="Cache the verified pinned Pi archive")
    parser.add_argument("--offline", action="store_true", help="Use local files only")
    args = parser.parse_args()

    machine = platform.machine().lower()
    if not sys.platform.startswith("linux") or machine not in {"x86_64", "amd64"}:
        parser.error("this prototype supports only Linux x86-64")

    archive = load_archive(args.archive, args.cache_dir, args.offline)

    binary = extract_member(archive, "pi/pi")
    package_json = extract_member(archive, "pi/package.json")
    photon_wasm = extract_member(archive, "pi/photon_rs_bg.wasm")
    atomic_write(BINARY_PATH, binary, 0o755)
    atomic_write(PACKAGE_JSON_PATH, package_json, 0o644)
    atomic_write(PHOTON_WASM_PATH, photon_wasm, 0o644)
    for theme_name in ("dark.json", "light.json", "theme-schema.json"):
        atomic_write(
            THEME_ROOT / theme_name,
            extract_member(archive, f"pi/theme/{theme_name}"),
            0o644,
        )

    license_text = b""
    if LICENSE_PATH.is_file():
        cached_license = LICENSE_PATH.read_bytes()
        if sha256(cached_license) == LICENSE_SHA256:
            license_text = cached_license
            print(f"Using verified cached Pi license: {LICENSE_PATH}", flush=True)
    if not license_text:
        if args.offline:
            raise RuntimeError("Offline build requires the verified cached Pi license")
        print(f"Downloading {LICENSE_URL}", flush=True)
        license_text = verified(fetch(LICENSE_URL), LICENSE_SHA256, "Pi LICENSE")
    atomic_write(LICENSE_PATH, license_text, 0o644)

    print(f"Pi {PI_VERSION}: {BINARY_PATH} ({len(binary)} bytes)")
    print(f"Pi package metadata: {PACKAGE_JSON_PATH}")
    print(f"Pi license: {LICENSE_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
