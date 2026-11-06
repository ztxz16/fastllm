"""Scoped generation bridge and a stdlib-only CLI usable by Pi's bash tool.

Run this file directly with --help. It does not need the ftllm Python package
in the agent's environment and never receives the launcher's control token or
the model service API keys.
"""
import argparse
import hmac
import json
import secrets
import shlex
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


class GenerationBridge:
    def __init__(self, services, on_download=None):
        self.services = services
        self.token = secrets.token_urlsafe(32)
        self.created = set()
        self.downloaded = set()
        self.lock = threading.Lock()
        self.closed = False
        self.cancel_requested = False
        self.on_download = on_download
        bridge = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                bridge.handle(self)

            def do_POST(self):
                bridge.handle(self)

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @property
    def url(self):
        return "http://127.0.0.1:" + str(self.server.server_port)

    def instructions(self):
        command = shlex.join([sys.executable, str(Path(__file__).resolve()),
                              "--bridge", self.url, "--token", self.token])
        return ("\n\n你可以调用 Launcher 中已启动的生成模型。用户要求生成视频时，请实际调用，"
                "不要只提供提示词。服务目录按 capabilities 区分模型能力。工具命令前缀：\n" + command +
                "\n先运行 `前缀 catalog` 获取最新服务 ID、能力和默认参数。"
                "生成：`前缀 generate --service ID --prompt '描述' --size 640x384 --seconds 5 "
                "--output videos/result.mp4 --wait 45`。长任务会返回 pending 和任务 id；"
                "继续执行 `前缀 wait --service ID --id JOB_ID --output videos/result.mp4 --wait 45`，"
                "直到 completed。不要重复创建同一个任务。状态：`前缀 status --service ID --id JOB_ID`；"
                "取消：`前缀 cancel --service ID --id JOB_ID`。输出必须是工作区中的相对路径，"
                "不要覆盖现有文件；完成后告诉用户保存路径。成功下载的视频会自动显示在会话中。"
                "若没有匹配服务，告诉用户先启动相应模型，不要虚构结果。"
                "调用凭据仅在本轮有效，不要把命令前缀或凭据写进文件、最终回答或发送到外部。")

    def handle(self, handler):
        if not hmac.compare_digest(handler.headers.get("Authorization", ""), "Bearer " + self.token):
            handler.send_error(403)
            return
        try:
            with self.lock:
                closed = self.closed
            if closed:
                handler.send_error(410)
                return
            if handler.path == "/catalog" and handler.command == "GET":
                status, mime, data = 200, "application/json", json.dumps(self.services.catalog()).encode()
            else:
                service_id, path = handler.path.lstrip("/").split("/", 1)
                length = int(handler.headers.get("Content-Length", "0"))
                if not 0 <= length <= 1024 * 1024:
                    raise ValueError("请求过大")
                body = handler.rfile.read(length) if length else None
                status, mime, data = self.services.media_request(service_id, handler.command, path, body)
                if 200 <= status < 300:
                    if handler.command == "POST" and path == "videos":
                        key = (service_id, json.loads(data)["id"])
                        with self.lock:
                            self.created.add(key)
                            cancel = self.cancel_requested
                        # The native POST can finish after close() snapshots
                        # existing jobs. Cancel that late arrival as well.
                        if cancel:
                            self._cancel_job(*key)
                    if handler.command == "GET" and path.endswith("/content"):
                        key = (service_id, path.split("/")[1])
                        with self.lock:
                            if not self.closed and key not in self.downloaded and self.on_download:
                                self.on_download(key[1] + ".mp4", data)
                            self.downloaded.add(key)
            handler.send_response(status)
            handler.send_header("Content-Type", mime)
            handler.send_header("Content-Length", str(len(data)))
            handler.end_headers()
            handler.wfile.write(data)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except Exception as error:
            handler.send_error(400, str(error).encode("ascii", "replace").decode())

    def close(self, cancel=False):
        with self.lock:
            closed = self.closed
            self.closed = True
            self.cancel_requested |= cancel
            created = tuple(self.created)
        if not closed:
            self.server.shutdown()
            self.server.server_close()
            self.thread.join(timeout=2)
        if cancel:
            for service_id, job_id in created:
                self._cancel_job(service_id, job_id)

    def _cancel_job(self, service_id, job_id):
        try:
            self.services.media_request(service_id, "POST", "videos/" + job_id + "/cancel")
        except Exception:
            pass


def main(argv=None):
    parser = argparse.ArgumentParser(description="Call the launcher's local generation services")
    parser.add_argument("--bridge", required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("action", choices=("catalog", "generate", "status", "wait", "cancel"))
    parser.add_argument("--service")
    parser.add_argument("--id")
    parser.add_argument("--prompt")
    parser.add_argument("--size", default="640x384")
    parser.add_argument("--seconds", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--output")
    parser.add_argument("--wait", type=int, default=45)
    args = parser.parse_args(argv)
    # This credential is valid only for a loopback bridge, never a remote URL.
    from urllib.parse import urlparse
    parsed = urlparse(args.bridge)
    if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.path not in ("", "/"):
        parser.error("bridge must be a local HTTP URL")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, body=None, raw=False, post=False):
        data = json.dumps(body).encode() if body is not None else (b"" if post else None)
        req = urllib.request.Request(args.bridge.rstrip("/") + "/" + path, data=data,
            headers={"Authorization": "Bearer " + args.token, "Content-Type": "application/json"})
        with opener.open(req, timeout=120) as response:
            result = response.read()
            return result if raw else json.loads(result)

    if args.action == "catalog":
        print(json.dumps(request("catalog"), ensure_ascii=False))
        return 0
    import re
    if not args.service or not re.fullmatch(r"[a-f0-9]{32}", args.service):
        parser.error("--service must be an ID from catalog")
    path = args.service + "/videos"
    output = None
    if args.output:
        output = (Path.cwd() / args.output).resolve()
        if Path(args.output).is_absolute() or Path.cwd().resolve() not in output.parents:
            parser.error("--output must be a relative path inside the workspace")
        if output.exists():
            parser.error("output already exists; choose a new filename")
    if args.action == "generate":
        if not args.prompt or not output:
            parser.error("generate requires --prompt and --output")
        body = {"prompt": args.prompt, "size": args.size, "seconds": args.seconds}
        if args.seed is not None:
            body["seed"] = args.seed
        job = request(path, body)
    else:
        if not args.id or not re.fullmatch(r"[A-Za-z0-9_-]+", args.id):
            parser.error("--id is required")
        job = request(path + "/" + args.id + ("/cancel" if args.action == "cancel" else ""),
                      post=args.action == "cancel")
    if args.action in {"generate", "wait"}:
        if not output:
            parser.error("--output is required")
        deadline = time.monotonic() + max(0, min(45, args.wait))
        while job["status"] in {"queued", "in_progress"} and time.monotonic() < deadline:
            time.sleep(2)
            job = request(path + "/" + job["id"])
        if job["status"] == "completed":
            content = request(path + "/" + job["id"] + "/content", raw=True)
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("xb") as file:
                file.write(content)
            job["saved_to"] = str(output)
        elif job["status"] in {"queued", "in_progress"}:
            job["pending"] = True
    print(json.dumps(job, ensure_ascii=False))
    return 1 if job["status"] in {"failed", "cancelled"} else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as error:
        print(json.dumps({"error": str(error)}, ensure_ascii=False), file=sys.stderr)
        sys.exit(1)
