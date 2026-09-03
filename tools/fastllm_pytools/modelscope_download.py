#!/usr/bin/env python3
"""ModelScope download worker used by the browser launcher."""

import argparse
import json
import os
import sys
import threading
import time
from dataclasses import dataclass


PROGRESS_PREFIX = "FASTLLM_MODELSCOPE_PROGRESS "


def parse_arguments(argv=None):
    parser = argparse.ArgumentParser(
        description="Download one complete ModelScope model with aggregate progress."
    )
    parser.add_argument("command", choices=["download"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--local_dir", required=True)
    parser.add_argument("--revision", default=None)
    parser.add_argument("--max-workers", type=int, default=4)
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
        self._last_ratio = -1.0
        self._last_report_time = 0.0
        with self._lock:
            self._report_locked("download.plan", force=True)

    def callback_type(self):
        tracker = self

        class FileProgress:
            def __init__(self, filename, file_size):
                self.filename = str(filename)
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


def _download_plan(model_id, revision, local_dir, token):
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import create_temporary_directory_and_cache
    from modelscope.utils.constant import DEFAULT_MODEL_REVISION, REPO_TYPE_MODEL

    requested_revision = revision or DEFAULT_MODEL_REVISION
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
    _, cache = create_temporary_directory_and_cache(
        model_id,
        local_dir=local_dir,
        repo_type=REPO_TYPE_MODEL,
    )
    file_sizes = {}
    for repo_file in repo_files:
        if repo_file.get("Type") == "tree":
            continue
        try:
            already_downloaded = cache.exists(repo_file)
        except Exception:
            already_downloaded = False
        if already_downloaded:
            continue
        filename = str(repo_file.get("Path") or repo_file.get("Name") or "")
        if filename:
            file_sizes[filename] = max(0, int(repo_file.get("Size") or 0))
    return file_sizes, resolved_revision, cookies


def download(args, token):
    from modelscope.hub.snapshot_download import snapshot_download

    local_dir = os.path.abspath(os.path.expanduser(args.local_dir))
    file_sizes, revision, cookies = _download_plan(
        args.model,
        args.revision,
        local_dir,
        token,
    )
    aggregate = AggregateProgress(file_sizes)
    return snapshot_download(
        model_id=args.model,
        revision=revision,
        local_dir=local_dir,
        max_workers=args.max_workers,
        cookies=cookies,
        token=token,
        progress_callbacks=[aggregate.callback_type()],
    )


def main(argv=None):
    args = parse_arguments(argv)
    if not 1 <= args.max_workers <= 64:
        print("--max-workers must be between 1 and 64", file=sys.stderr)
        return 2
    try:
        destination = download(args, os.environ.get("MODELSCOPE_API_TOKEN") or None)
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("modelscope"):
            print(
                "ModelScope is not installed. Run: "
                "python -m pip install 'modelscope>=1.34.0,<2'",
                file=sys.stderr,
            )
            return 2
        raise
    print(f"Model downloaded to: {destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
