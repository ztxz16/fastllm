"""Authenticated, separate-origin HTTP/SSE/WebSocket entries for native agents."""

import asyncio
import base64
import hmac
import html
import json
import re
import secrets
import socket
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit


def browser_address(browser_origin, name="OpenCode"):
    origin = urlsplit(browser_origin)
    if (origin.scheme != "http" or not origin.hostname or ":" in origin.hostname
            or origin.username is not None or origin.password is not None):
        raise RuntimeError(f"{name} embedding requires an HTTP Launcher address with an IPv4 address or hostname.")
    return origin


def create_harness_proxy(launch_url, bind_host, browser_origin):
    import httpx
    address = urlsplit(launch_url)
    if address.scheme != "http" or address.hostname not in {"127.0.0.1", "localhost"} or not address.port:
        raise RuntimeError("Harness did not provide a loopback launch address.")
    upstream = f"http://127.0.0.1:{address.port}"
    # The native cookie is bound to the loopback authority. Keep it in the
    # backend; each browser uses the proxy's own authenticated entry instead.
    with httpx.Client(trust_env=False, timeout=5, follow_redirects=False) as client:
        response = client.get(upstream + "/?" + address.query)
        if response.status_code != 303 or not response.cookies:
            raise RuntimeError("Harness authentication failed. Retry opening it.")
        cookie = "; ".join(f"{key}={value}" for key, value in response.cookies.items())
    return AgentProxy(upstream, None, bind_host, browser_origin, agent="harness", upstream_cookie=cookie)


class AgentProxy:
    def __init__(self, upstream, password, bind_host, browser_origin, *, agent="opencode", upstream_cookie=""):
        import uvicorn
        self.name = {"opencode":"OpenCode", "harness":"DeepSeek Harness"}[agent]
        self.agent = agent
        self.upstream = upstream.rstrip("/")
        self.authorization = "Basic " + base64.b64encode(f"opencode:{password}".encode()).decode() if password is not None else ""
        self.upstream_cookie = upstream_cookie
        self.token = secrets.token_urlsafe(32)
        self.cookie = f"ftllm-{agent}-" + secrets.token_hex(8)
        self._lock = threading.Lock()
        self._parents = {}
        self.listener = socket.socket()
        self.listener.bind((bind_host, 0))
        self.url = self.url_for(browser_origin)
        self.origin = self.url.split("/_ftllm/open", 1)[0]
        self.server = uvicorn.Server(uvicorn.Config(self._app(), log_level="error", access_log=False,
            ws="websockets", timeout_graceful_shutdown=1))
        self.thread = threading.Thread(target=self.server.run, kwargs={"sockets": [self.listener]}, daemon=True)

    def url_for(self, browser_origin):
        # Only authenticated Launcher requests register entry addresses. Keep
        # each browser on its own hostname without restarting the native app.
        parent = browser_address(browser_origin, self.name)
        host = f"{parent.hostname}:{self.listener.getsockname()[1]}"
        with self._lock:
            self._parents.setdefault(host, set()).add(f"http://{parent.netloc}")
        return f"http://{host}/_ftllm/open?token={self.token}"

    def _parent_origins(self, request):
        with self._lock:
            return sorted(self._parents.get(request.headers.get("host"), ()))

    def _authorized(self, request):
        origin = "http://" + request.headers.get("host", "")
        return (bool(self._parent_origins(request))
                and hmac.compare_digest(request.cookies.get(self.cookie, ""), self.token)
                and request.headers.get("origin", origin) == origin)

    def _headers(self, headers):
        # Inject only the native service's credentials, including on terminal
        # upgrades; browser and Launcher credentials never reach the app.
        blocked = {"host", "cookie", "authorization", "origin", "connection", "upgrade",
                   "transfer-encoding", "content-length", "accept-encoding"}
        result = {key: value for key, value in headers.items()
                  if key.lower() not in blocked and not key.lower().startswith(("sec-websocket-", "x-ftllm-"))}
        if self.upstream_cookie:
            result["cookie"] = self.upstream_cookie
        if self.authorization:
            result["authorization"] = self.authorization
        result["accept-encoding"] = "identity"
        return result

    def _app(self):
        import httpx
        from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
        from starlette.responses import FileResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
        from websockets.legacy.client import connect
        from websockets.exceptions import ConnectionClosed

        @asynccontextmanager
        async def lifespan(app):
            async with httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(10, read=None),
                                         follow_redirects=False) as client:
                app.state.client = client
                yield

        app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

        @app.get("/_ftllm/opencode/{filename}")
        async def appearance(request: Request, filename: str):
            if self.agent != "opencode" or not self._authorized(request):
                return JSONResponse({"error": "Open OpenCode from Launcher."}, status_code=403)
            if filename not in {"appearance.js", "appearance.css"}:
                return JSONResponse({"error": "Not found"}, status_code=404)
            return FileResponse(Path(__file__).with_name("ui_plugins") / "opencode" / filename,
                                headers={"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})

        @app.get("/_ftllm/open")
        async def open_page(request: Request):
            if (not self._parent_origins(request)
                    or not hmac.compare_digest(request.query_params.get("token", ""), self.token)):
                return JSONResponse({"error": f"Open {self.name} from Launcher."}, status_code=403)
            response = RedirectResponse("/", status_code=303)
            response.set_cookie(self.cookie, self.token, httponly=True, samesite="strict")
            response.headers.update({"Cache-Control": "no-store", "Referrer-Policy": "no-referrer"})
            return response

        def target(scope):
            # Concatenation (not urljoin) keeps // and absolute-form paths on
            # the fixed loopback upstream. Preserve encoded paths and queries.
            path = scope.get("raw_path", scope["path"].encode()).decode("ascii")
            query = scope.get("query_string", b"").decode("ascii")
            return self.upstream + path + ("?" + query if query else "")

        @app.api_route("/{path:path}", methods=["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
        async def proxy(request: Request, path: str):
            if not self._authorized(request):
                return JSONResponse({"error": f"Open {self.name} from Launcher."}, status_code=403)
            try:
                upstream_request = app.state.client.build_request(request.method, target(request.scope),
                    headers=self._headers(request.headers), content=request.stream())
                response = await app.state.client.send(upstream_request, stream=True)
            except httpx.HTTPError:
                return JSONResponse({"error": f"{self.name} is unavailable. Reopen it from Launcher."}, status_code=502)
            headers = {key: value for key, value in response.headers.items()
                       if key not in {"connection", "transfer-encoding", "content-length", "content-encoding",
                                      "set-cookie", "x-frame-options", "access-control-allow-origin"}}
            csp = headers.get("content-security-policy", "")
            headers["content-security-policy"] = csp + "; frame-ancestors " + " ".join(self._parent_origins(request))
            headers.update({"Referrer-Policy": "no-referrer", "Cache-Control": "no-store"})
            if (self.agent == "opencode" and request.method == "GET" and response.status_code == 200
                    and response.headers.get("content-type", "").split(";", 1)[0].strip() == "text/html"):
                # Keep native scripts/API untouched. Only the document gets the
                # fixed appearance adapter; JS, SSE and terminal streams stay streamed.
                try:
                    document = await response.aread()
                finally:
                    await response.aclose()
                parents = html.escape(json.dumps(self._parent_origins(request)), quote=True)
                adapter = ('<link rel="stylesheet" href="/_ftllm/opencode/appearance.css">'
                           '<script defer src="/_ftllm/opencode/appearance.js" '
                           f'data-parents="{parents}"></script>')
                document = re.sub(b"</head\\s*>", lambda match: adapter.encode() + match.group(),
                                  document, count=1, flags=re.IGNORECASE)
                headers.pop("etag", None)
                return Response(document, status_code=response.status_code, headers=headers)
            async def stream():
                try:
                    async for chunk in response.aiter_bytes():
                        yield chunk
                except httpx.HTTPError:
                    # End an interrupted stream (including on process shutdown)
                    # so the original UI can reconnect its SSE subscription.
                    pass
                finally:
                    await response.aclose()
            return StreamingResponse(stream(), status_code=response.status_code, headers=headers)

        @app.websocket("/{path:path}")
        async def proxy_socket(websocket: WebSocket, path: str):
            if not self._authorized(websocket):
                await websocket.close(code=1008)
                return
            try:
                async with connect(target(websocket.scope).replace("http://", "ws://", 1),
                                   extra_headers=self._headers(websocket.headers), max_size=None,
                                   open_timeout=10, close_timeout=1) as upstream:
                    await websocket.accept()

                    async def upload():
                        while True:
                            message = await websocket.receive()
                            if message["type"] == "websocket.disconnect":
                                break
                            await upstream.send(message.get("text") if message.get("text") is not None else message["bytes"])

                    async def download():
                        async for message in upstream:
                            if isinstance(message, str):
                                await websocket.send_text(message)
                            else:
                                await websocket.send_bytes(message)

                    tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
                    try:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
            except (OSError, ConnectionClosed, WebSocketDisconnect):
                pass
            finally:
                try:
                    await websocket.close()
                except RuntimeError:
                    pass

        return app

    def start(self, cancelled):
        self.thread.start()
        deadline = time.monotonic() + 10
        while not self.server.started:
            if cancelled.wait(.02) or not self.thread.is_alive() or time.monotonic() > deadline:
                self.stop()
                raise RuntimeError(f"{self.name} embedded server could not start.")

    def stop(self):
        self.server.should_exit = True
        if self.thread.is_alive():
            self.thread.join(timeout=5)
        self.listener.close()
