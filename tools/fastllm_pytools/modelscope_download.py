#!/usr/bin/env python3
"""ModelScope download worker used by the browser launcher."""

import argparse
import fnmatch
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from urllib.error import HTTPError
from urllib.parse import quote
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import NamedTuple

if __package__:
    from .model_catalog import DOWNLOAD_BUNDLES
else:  # The launcher runs this file directly, also in wheel installations.
    from model_catalog import DOWNLOAD_BUNDLES


PROGRESS_PREFIX = "FASTLLM_MODELSCOPE_PROGRESS "


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Download one complete ModelScope model with aggregate progress."
    )
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--bundle", choices=tuple(DOWNLOAD_BUNDLES))
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only download files matching this glob; repeat for multiple patterns.",
    )
    return parser.parse_args(argv)


def emit_progress(payload):
    print(
        PROGRESS_PREFIX
        + json.dumps(payload, ensure_ascii=False, separators=(",", ":")),
        flush=True,
    )


@dataclass
class _FileProgress:
    size: int = 0
    downloaded: int = 0
    finished: bool = False


class AggregateProgress:
    """Report one byte-weighted total for concurrent ModelScope downloads."""

    def __init__(self, file_sizes):
        self._lock = threading.Lock()
        self._files = {
            str(name): _FileProgress(size=max(0, int(size)))
            for name, size in file_sizes.items()
        }
        self._stage = ""
        self._component = ""
        self._last_ratio = -1.0
        self._last_report_time = 0.0
        with self._lock:
            self._report_locked("download.plan", force=True)

    def stage(self, stage, component=""):
        with self._lock:
            self._stage = stage
            self._component = component
            self._report_locked("download.progress", force=True)

    def callback_type(self, prefix=""):
        tracker = self

        class FileProgress:
            def __init__(self, filename, file_size):
                self.filename = prefix + str(filename)
                tracker.register(self.filename, file_size)

            def update(self, size):
                tracker.update(self.filename, size)

            def end(self):
                tracker.complete(self.filename)

        return FileProgress

    def register(self, filename, file_size):
        with self._lock:
            if filename in self._files:
                return
            self._files[filename] = _FileProgress(
                size=max(0, int(file_size or 0))
            )
            self._report_locked("download.plan", force=True)

    def update(self, filename, size):
        try:
            increment = max(0, int(size))
        except (TypeError, ValueError):
            return
        if increment == 0:
            return
        with self._lock:
            entry = self._files.get(filename)
            if entry is None:
                entry = _FileProgress()
                self._files[filename] = entry
            entry.downloaded = min(
                entry.size, entry.downloaded + increment
            )
            self._report_locked("download.progress")

    def complete(self, filename):
        with self._lock:
            entry = self._files.get(filename)
            if entry is None:
                entry = _FileProgress(finished=True)
                self._files[filename] = entry
            else:
                entry.downloaded = entry.size
                entry.finished = True
            self._report_locked("download.progress", force=True)

    def finish(self, filenames=None):
        # Cached files may not receive callbacks. Count them only after the SDK
        # has successfully verified/downloaded the complete snapshot.
        with self._lock:
            for filename in self._files if filenames is None else filenames:
                entry = self._files[filename]
                entry.downloaded = entry.size
                entry.finished = True
            self._report_locked("download.progress", force=True)

    def _snapshot_locked(self, event_type):
        entries = tuple(self._files.values())
        total_bytes = sum(entry.size for entry in entries)
        downloaded_bytes = min(
            total_bytes, sum(entry.downloaded for entry in entries)
        )
        total_files = len(entries)
        completed_files = sum(entry.finished for entry in entries)
        if total_bytes > 0:
            ratio = downloaded_bytes / total_bytes
        elif total_files > 0:
            ratio = completed_files / total_files
        else:
            ratio = 0.0
        return {
            "version": 1,
            "type": event_type,
            "stage": self._stage,
            "component": self._component,
            "downloadedBytes": downloaded_bytes,
            "totalBytes": total_bytes,
            "completedFiles": completed_files,
            "totalFiles": total_files,
        }, ratio

    def _report_locked(self, event_type, force=False):
        payload, ratio = self._snapshot_locked(event_type)
        now = time.monotonic()
        if (
            not force
            and ratio < self._last_ratio + 0.001
            and now < self._last_report_time + 0.25
        ):
            return
        self._last_ratio = max(self._last_ratio, ratio)
        self._last_report_time = now
        emit_progress(payload)


def _normalize_patterns(patterns):
    if isinstance(patterns, str):
        patterns = patterns.replace(";", ",").replace("\n", ",").split(",")
    return tuple(str(pattern).strip() for pattern in (patterns or ()) if str(pattern).strip())


def _filter_file_sizes(file_sizes, patterns):
    patterns = _normalize_patterns(patterns)
    if not patterns:
        return dict(file_sizes)
    return {
        filename: size
        for filename, size in file_sizes.items()
        if any(fnmatch.fnmatchcase(filename, pattern) for pattern in patterns)
    }


def _download_plan(model_id, revision, token, include=None, *, checksums=None):
    from modelscope.hub.api import HubApi
    from modelscope.utils.constant import DEFAULT_MODEL_REVISION, REPO_TYPE_MODEL

    requested_revision = revision or DEFAULT_MODEL_REVISION
    try:
        from modelscope_hub import HubApi as MetadataApi
    except ImportError:
        MetadataApi = None
    if MetadataApi is not None:
        # Newer ModelScope's legacy shim drops the SHA-256 and directory flag.
        entries = MetadataApi(token=token).list_repo_files(model_id, "model", revision=requested_revision)
        sizes = {item.path: item.size for item in entries if not item.is_dir}
        if checksums is not None:
            checksums.update((item.path, item.sha256 or "") for item in entries if not item.is_dir)
        return _filter_file_sizes(sizes, include), requested_revision, None

    api = HubApi(token=token)
    endpoint = api.get_endpoint_for_read(
        repo_id=model_id,
        repo_type=REPO_TYPE_MODEL,
    )
    cookies = api.get_cookies(access_token=token)
    revision_detail = api.get_valid_revision_detail(
        model_id,
        revision=requested_revision,
        cookies=cookies,
        endpoint=endpoint,
    )
    resolved_revision = revision_detail["Revision"]
    repo_files = api.get_model_files(
        model_id=model_id,
        revision=resolved_revision,
        recursive=True,
        use_cookies=False if cookies is None else cookies,
        endpoint=endpoint,
    )
    file_sizes = {}
    for repo_file in repo_files:
        if repo_file.get("Type") == "tree":
            continue
        filename = str(repo_file.get("Path") or repo_file.get("Name") or "")
        if filename:
            if checksums is not None:
                checksums[filename] = repo_file.get("Sha256") or ""
            file_sizes[filename] = max(0, int(repo_file.get("Size") or 0))
    return _filter_file_sizes(file_sizes, include), resolved_revision, cookies


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _safe_path(name):
    path = PurePosixPath(name)
    if (not name or name != path.as_posix() or name == "." or ".." in path.parts or path.is_absolute()
            or "\\" in name or PureWindowsPath(name).drive or ":" in name or "\x00" in name):
        raise ValueError(f"Invalid repository file path: {name!r}")
    return path


def preparation_error(recipe):
    if not uses_modelscope(recipe) and not importlib.util.find_spec("filelock"):
        return "Component downloads require filelock."
    if recipe.get("prepare") == "lingbot":
        if not all(importlib.util.find_spec(mod) for mod in ("torch", "safetensors")):
            return "LingBot asset preparation requires torch and safetensors in the launcher environment."
    return ""


def uses_modelscope(recipe):
    return not recipe or any(src.get("hub", "ModelScope") == "ModelScope" for src in recipe["sources"])


def _lingbot_converter():
    package = Path(__file__).resolve().parent
    for script in (package / "lingbot_world_assets/convert_assets.py", package.parent / "lingbot_world/convert_assets.py"):
        if script.is_file():
            return script
    raise ValueError("LingBot asset converter is missing from this installation")


class ComponentPlan(NamedTuple):
    repo: str
    revision: str
    cookies: object
    hashes: dict
    paths: dict
    cache_key: str
    hub: str


def _public_file_plan(src):
    revision = src["revision"]
    if "sha256" in src:
        name, = src["files"]
        return {name: src["bytes"]}, {name: src["sha256"]}
    url = f"https://huggingface.co/api/models/{src['repo']}/revision/{revision}?blobs=true"
    with urllib.request.urlopen(url, timeout=30) as response:
        metadata = json.load(response)
    if metadata["sha"] != revision:
        raise ValueError(f"{src['repo']}: unexpected Hugging Face revision")
    sizes, hashes = {}, {}
    for item in metadata["siblings"]:
        name = item["rfilename"]
        if any(fnmatch.fnmatchcase(name, pattern) for pattern in src["files"]):
            sizes[name] = item["size"]
            hashes[name] = item["lfs"]["sha256"] if item.get("lfs") else "git:" + item["blobId"]
    return sizes, hashes


def _bundle_plan(recipe, token):
    repositories = {}
    components = []
    destination_sizes = {}
    for src in recipe["sources"]:
        repo = src["repo"]
        hashes = {}
        hub = src.get("hub", "ModelScope")
        if hub == "Hugging Face":
            revision, cookies = src["revision"], None
            sizes, hashes = _public_file_plan(src)
        else:
            if repo not in repositories:
                plan = _download_plan(repo, "master", token, checksums=hashes)
                repositories[repo] = (*plan, hashes)
            all_sizes, revision, cookies, hashes = repositories[repo]
            sizes = _filter_file_sizes(all_sizes, src["files"])
        for pattern in (*src["files"], *src["required"]):
            if not any(fnmatch.fnmatchcase(name, pattern) for name in sizes):
                raise ValueError(f"{repo}: missing component {pattern}")
        hashes = {name: hashes[name] for name in sizes}
        if any(len(checksum) != 64 and not (hub == "Hugging Face" and checksum.startswith("git:")
                                            and len(checksum) == 44) for checksum in hashes.values()):
            raise ValueError(f"{repo}: missing checksum in repository manifest")
        fingerprint = _digest(sorted(hashes.items()))
        if src.get("checksum") and fingerprint != src["checksum"]:
            raise ValueError(f"{repo}: published files differ from the base required by the adapter")
        paths = {}
        for name, size in sizes.items():
            _safe_path(name)
            dest = (PurePosixPath(src["into"]) / _safe_path(src["rename"].get(name, name))).as_posix()
            _safe_path(dest)
            if dest in destination_sizes or dest.startswith(".ftllm-"):
                raise ValueError(f"Overlapping bundle path: {dest}")
            destination_sizes[dest] = size
            paths[name] = dest
        components.append(ComponentPlan(repo, revision, cookies, hashes, paths, _digest([hub, repo, fingerprint]), hub))
    return components, destination_sizes


def _file_stamp(path):
    stat = path.stat()
    return [stat.st_size, stat.st_mtime_ns]


def _matches_checksum(path, expected):
    if expected.startswith("git:"):
        digest = hashlib.sha1(f"blob {path.stat().st_size}\0".encode())
        expected = expected[4:]
    else:
        digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == expected


def _copy_component_file(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    # Settings and tokenizers can be edited independently in each assembled model.
    if target.suffix in (".safetensors", ".gguf", ".pth", ".bin"):
        try:
            os.link(source, target)
            return
        except OSError:
            pass  # Filesystems without hardlinks can still produce a portable bundle.
    shutil.copy2(source, target)


def _reuse_installed_files(parent, cache, repo, hashes, paths, sizes, verified):
    # Look only in the two conventional sibling directories. Verify contents before reuse.
    candidates = (parent / repo.split("/")[-1], parent / repo.replace("/", "--"))
    remaining = []
    for name, dest in paths.items():
        cached = cache / "files" / name
        if cached.is_file() and verified.get(name) == _file_stamp(cached):
            continue
        for root in candidates:
            existing = root / name
            if not existing.is_file():
                continue
            before = _file_stamp(existing)
            if before[0] == sizes[dest] and _matches_checksum(existing, hashes[name]):
                if cached.exists():
                    cached.unlink()
                _copy_component_file(existing.resolve(), cached)
                if before == _file_stamp(existing):
                    verified[name] = _file_stamp(cached)
                else:
                    verified.pop(name, None)
                break
        else:
            remaining.append(name)
    return remaining


def _verify_component(cache, hashes, sizes, paths, previous):
    # Verify each newly downloaded byte once. Subsequent bundles can reuse a verified base.
    stamp = cache / "verified.json"
    current = {}
    for name, checksum in hashes.items():
        path = cache / "files" / name
        current[name] = _file_stamp(path)
        if current[name][0] != sizes[paths[name]] or (
                previous.get(name) != current[name] and not _matches_checksum(path, checksum)):
            # Discard only the downloader's private entry, so the SDK can refetch it on retry.
            path.unlink()
            raise ValueError(f"Incomplete or corrupt component: {name}")
    temp = stamp.with_suffix(".tmp")
    temp.write_text(json.dumps(current))
    temp.replace(stamp)


def _fetch_public_file(url, output, size, progress):
    """Resume a pinned public object. The caller verifies its catalog SHA-256 before publication."""
    output.parent.mkdir(parents=True, exist_ok=True)
    part = output.with_suffix(output.suffix + ".part")
    offset = part.stat().st_size if part.exists() else 0
    if offset > size:
        part.unlink()
        offset = 0
    if offset < size:
        headers = {"Accept-Encoding": "identity"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60) as response:
            if response.status == 206:
                if response.headers.get("Content-Range") != f"bytes {offset}-{size - 1}/{size}":
                    raise ValueError(f"Invalid resume range for {output.name}")
            elif response.status == 200:
                offset = 0  # The server ignored Range. Restart instead of appending.
            else:
                raise ValueError(f"Unexpected HTTP {response.status} for {output.name}")
            progress.update(offset)
            with part.open("ab" if offset else "wb") as stream:
                for block in iter(lambda: response.read(4 << 20), b""):
                    offset += len(block)
                    if offset > size:
                        raise ValueError(f"Remote file exceeds its manifest size: {output.name}")
                    stream.write(block)
                    progress.update(len(block))
    if part.stat().st_size != size:
        raise ValueError(f"Incomplete transfer: {output.name}")
    part.replace(output)
    progress.end()


def _check_shards(root, files):
    for filename in files:
        if not filename.endswith(".safetensors.index.json"):
            continue
        path = root / filename
        weights = json.loads(path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weights, dict) or not weights:
            raise ValueError(f"Empty shard index: {filename}")
        for shard in set(weights.values()):
            if (PurePosixPath(filename).parent / _safe_path(shard)).as_posix() not in files:
                raise ValueError(f"{filename}: missing shard {shard}")


def bundle_receipt(target, preset):
    if ".ftllm-downloads" in target.resolve().parts:
        raise ValueError("The component cache cannot be used as a model destination")
    if not preset:
        if (target / ".ftllm-bundle.json").exists():
            raise ValueError(f"Destination already contains a managed bundle: {target}")
        return None
    if target.is_symlink() or (target.parent / ".ftllm-downloads").is_symlink():
        raise ValueError("Choose a dedicated model destination directory")
    if target.exists() and any(target.iterdir()):
        marker = target / ".ftllm-bundle.json"
        if marker.is_file():
            receipt = json.loads(marker.read_text())
            if isinstance(receipt, dict) and receipt.get("preset") == preset:
                return receipt
        raise ValueError(f"Destination already contains another model. Choose an empty directory: {target}")
    return None


def download_bundle(recipe, target, workers, token, snapshot):
    error = preparation_error(recipe)
    if error:
        raise ValueError(error)
    from filelock import FileLock
    converter = _lingbot_converter() if recipe.get("prepare") == "lingbot" else None
    cache_root = target.parent / ".ftllm-downloads"
    bundle_receipt(target, recipe["preset"])
    cache_root.mkdir(parents=True, exist_ok=True)
    target_key = _digest(os.path.normcase(str(target)))
    marker = ".ftllm-bundle.json"
    with FileLock(str(cache_root / (target_key + ".lock"))):
        receipt = bundle_receipt(target, recipe["preset"])
        components, sizes = _bundle_plan(recipe, token)
        signature = _digest([[(c.cache_key, c.paths) for c in components], recipe.get("manifest"),
                             recipe.get("prepare"), recipe.get("entry")])
        progress = AggregateProgress({f"{i}/{name}": sizes[dest] for i, component in enumerate(components)
                                      for name, dest in component.paths.items()})
        if receipt:
            if receipt.get("signature") == signature:
                # Recheck modified files even when an edit preserves the byte length.
                hashes = {c.paths[name]: sha for c in components for name, sha in c.hashes.items()}
                expected = sizes.keys() | receipt.get("generated", {}).keys()
                if all((target / name).is_file() and (
                        _file_stamp(target / name) == receipt.get("files", {}).get(name) or
                        name in hashes and _matches_checksum(target / name, hashes[name])) for name in expected):
                    progress.finish()
                    return str(target)
            raise ValueError(f"Existing bundle is incomplete or has a different revision. Choose a new directory: {target}")

        # A cancelled worker may leave this private stage. Rebuild it from immutable component caches.
        stage = cache_root / (target_key + ".stage")
        if stage.exists():
            shutil.rmtree(stage)
        stage.mkdir()
        try:
            for i, (repo, revision, cookies, hashes, paths, key, hub) in enumerate(components):
                cache = cache_root / key
                cache.mkdir(exist_ok=True)
                with FileLock(str(cache / "download.lock")):
                    context = f"{i + 1}/{len(components)} · {repo}"
                    progress.stage("fetch", context)
                    stamp = cache / "verified.json"
                    previous = json.loads(stamp.read_text()) if stamp.is_file() else {}
                    remaining = _reuse_installed_files(target.parent, cache, repo, hashes, paths, sizes, previous)
                    if remaining:
                        callback = progress.callback_type(f"{i}/")
                        if hub == "Hugging Face":
                            def fetch(name):
                                url = f"https://huggingface.co/{repo}/resolve/{revision}/{quote(name)}"
                                size = sizes[paths[name]]
                                _fetch_public_file(url, cache / "files" / name, size, callback(name, size))
                            with ThreadPoolExecutor(max_workers=workers) as executor:
                                list(executor.map(fetch, remaining))
                        else:
                            snapshot(model_id=repo, revision=revision, cookies=cookies, token=token,
                                     local_dir=str(cache / "files"), allow_patterns=remaining,
                                     max_workers=workers, progress_callbacks=[callback])
                    progress.stage("verify", context)
                    _verify_component(cache, hashes, sizes, paths, previous)
                    for name, dest in paths.items():
                        _copy_component_file(cache / "files" / name, stage / dest)
                progress.finish(f"{i}/{name}" for name in paths)
            _check_shards(stage, sizes)
            generated = {}
            if recipe.get("manifest"):
                path = stage / "h3_quantization.json"
                path.write_text(json.dumps(recipe["manifest"], indent=2))
                generated[path.name] = path.stat().st_size
            if converter:
                progress.stage("convert")
                subprocess.run([sys.executable, str(converter), str(stage)], check=True)
                for name in ("native/t5.safetensors", "native/vae.safetensors"):
                    generated[name] = (stage / name).stat().st_size
                    if not generated[name]:
                        raise ValueError(f"LingBot conversion produced an empty {name}")
            stamps = {name: _file_stamp(stage / name) for name in sizes.keys() | generated.keys()}
            (stage / marker).write_text(json.dumps({"preset": recipe["preset"], "signature": signature,
                                                    "files": stamps, "generated": generated}))
            # rename only replaces an empty directory on POSIX. Windows requires its removal first.
            if os.name == "nt" and target.exists():
                target.rmdir()
            stage.rename(target)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    return str(target)


def download(args, token):
    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    snapshot_download = None
    if uses_modelscope(DOWNLOAD_BUNDLES.get(getattr(args, "bundle", None))):
        from modelscope.hub.snapshot_download import snapshot_download
    if getattr(args, "bundle", None):
        recipe = DOWNLOAD_BUNDLES[args.bundle]
        if args.model != recipe["id"] or args.include or args.revision not in (None, "master"):
            raise ValueError("Bundle source, revision and file selection are fixed by the catalog")
        return download_bundle(recipe, Path(local_dir), args.max_workers, token, snapshot_download)
    bundle_receipt(Path(local_dir), "")
    include = _normalize_patterns(getattr(args, "include", ()))
    file_sizes, revision, cookies = _download_plan(
        args.model,
        args.revision,
        token,
        include,
    )
    if include and not file_sizes:
        raise ValueError("No repository files match the requested download patterns")
    aggregate = AggregateProgress(file_sizes)
    snapshot_kwargs = dict(
        model_id=args.model,
        revision=revision,
        local_dir=local_dir,
        max_workers=args.max_workers,
        cookies=cookies,
        token=token,
        progress_callbacks=[aggregate.callback_type()],
    )
    if include:
        snapshot_kwargs["allow_patterns"] = list(include)
    destination = snapshot_download(**snapshot_kwargs)
    aggregate.finish()
    return destination


def main(argv=None):
    args = parse_arguments(argv)
    if not 1 <= args.max_workers <= 64:
        print("--max-workers must be between 1 and 64", file=sys.stderr)
        return 2
    include = _normalize_patterns(args.include)
    if len(include) > 64 or any(len(pattern) > 512 or pattern.startswith("-") for pattern in include):
        print("--include accepts at most 64 patterns of up to 512 characters", file=sys.stderr)
        return 2
    token = os.environ.get("MODELSCOPE_API_TOKEN") or None
    try:
        destination = download(args, token)
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("modelscope"):
            print(
                "ModelScope is not installed. Run: "
                "python -m pip install 'modelscope>=1.34.0,<2'",
                file=sys.stderr,
            )
            return 2
        raise
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        # HTTP exceptions can include expiring signed URLs. Only expose their status.
        detail = f"HTTP {error.code}: {error.reason}" if isinstance(error, HTTPError) else str(error)
        if token:
            detail = detail.replace(token, "[redacted]")
        emit_progress({"version": 1, "type": "download.error", "error": detail[:1000]})
        return 1
    print(f"Model downloaded to: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
