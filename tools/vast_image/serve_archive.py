#!/usr/bin/env python3
"""通过 SSH 转发提供单个归档的分段下载；仅监听本机，不发布目录。"""

import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import re


def fingerprint(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def byte_range(header, size):
    match = re.fullmatch(r"bytes=(\d*)-(\d*)", header)
    if not match or not any(match.groups()):
        raise ValueError("需要单个 bytes 范围")
    first, last = match.groups()
    if not first:
        length = int(last)
        if length == 0:
            raise ValueError("后缀长度必须大于零")
        return max(0, size - length), size - 1
    start = int(first)
    end = min(int(last), size - 1) if last else size - 1
    if start > end:
        raise ValueError("请求范围超出文件")
    return start, end


def make_server(archive, port):
    archive = Path(archive).resolve(strict=True)
    if not archive.is_file():
        raise ValueError("下载目标必须是普通文件")
    digest = hashlib.sha256()
    with archive.open("rb") as source:
        original = fingerprint(os.fstat(source.fileno()))
        if original[2] == 0:
            raise ValueError("归档不能为空")
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
        if fingerprint(os.fstat(source.fileno())) != original:
            raise ValueError("归档在计算 SHA256 时发生变化，请重新启动")

    metadata = json.dumps({"name": archive.name, "size": original[2],
                           "sha256": digest.hexdigest()}).encode("utf-8")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_HEAD(self):
            self.respond(False)

        def do_GET(self):
            self.respond(True)

        def respond(self, body):
            if self.path not in ("/archive.tar", "/metadata"):
                self.send_error(404)
                return

            try:
                source = archive.open("rb")
            except OSError:
                self.send_error(409, "Archive unavailable; restart the server")
                return
            with source:
                if fingerprint(os.fstat(source.fileno())) != original:
                    self.send_error(409, "Archive changed; restart the server")
                    return

                try:
                    if self.path == "/metadata":
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.send_header("Content-Length", str(len(metadata)))
                        self.end_headers()
                        if body:
                            self.wfile.write(metadata)
                        return

                    start, end = 0, original[2] - 1
                    range_header = self.headers.get("Range") if body else None
                    if range_header is not None:
                        try:
                            start, end = byte_range(range_header, original[2])
                        except ValueError:
                            self.send_response(416)
                            self.send_header("Content-Range", f"bytes */{original[2]}")
                            self.send_header("Content-Length", "0")
                            self.end_headers()
                            return

                    self.send_response(206 if range_header is not None else 200)
                    self.send_header("Accept-Ranges", "bytes")
                    self.send_header("Content-Type", "application/octet-stream")
                    self.send_header("Content-Length", str(end - start + 1))
                    if range_header is not None:
                        self.send_header("Content-Range", f"bytes {start}-{end}/{original[2]}")
                    self.end_headers()
                    if body:
                        source.seek(start)
                        remaining = end - start + 1
                        while remaining:
                            chunk = source.read(min(1024 * 1024, remaining))
                            if not chunk:
                                break
                            self.wfile.write(chunk)
                            remaining -= len(chunk)
                except (BrokenPipeError, ConnectionResetError):
                    pass

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", type=Path)
    parser.add_argument("--port", type=int, default=18765)
    args = parser.parse_args()
    print("正在计算归档 SHA256……", flush=True)
    with make_server(args.archive, args.port) as server:
        print(f"下载服务已启动：127.0.0.1:{server.server_port}，按 Ctrl+C 停止。", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
