"""Serve bundled interactive players and stream their session-scoped native APIs."""
import re

from .launcher import LauncherError
from .service_backends import PACKAGE, QwenImageBackend, media_assets


WORLD = r"v1/worlds/world_[0-9a-f]{32}"
VIDEO = r"v1/videos/video_[0-9a-f]{32}"
SESSION = r"sessions/[0-9a-f]{32}"
ROUTES = {
    "qwen_image": QwenImageBackend.demo_routes,
    "lingbot_world": {
        "GET": r"health|v1/worlds/world_[0-9a-f]{24}|v1/jobs/job_[0-9a-f]{24}|v1/worlds/files/job_[0-9a-f]{24}_[0-9]+\.png",
        "POST": r"v1/worlds|v1/worlds/world_[0-9a-f]{24}/actions",
        "DELETE": r"v1/worlds/world_[0-9a-f]{24}",
    },
    "hy_nomas": {
        "GET": rf"health|openapi\.json|{WORLD}|{VIDEO}(?:/content|/events|/frames/[0-9]+)?",
        "POST": rf"v1/worlds|v1/videos|{WORLD}/actions|{VIDEO}/cancel",
        "PATCH": WORLD, "DELETE": WORLD,
    },
    "duo": {
        "GET": rf"resolutions|{SESSION}/(?:events|turn-[0-9]+/avc)|media/[0-9a-f]{{32}}/turn-[0-9]+/(?:chunk-[0-9]+\.(?:mp4|dav)|reply\.mp4)",
        "POST": rf"sessions(?:/custom)?|{SESSION}/turns", "DELETE": SESSION,
    },
}


def install_demo_routes(app, pool, cookie, cookie_token):
    import httpx
    from anyio import CancelScope
    from fastapi import Request
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

    def service(session_id):
        state = pool.service(session_id).state()
        if (state["sessionId"] != session_id or state.get("backend") not in ROUTES
                or state["phase"] != "running" or not state["ready"]):
            raise LauncherError("请先启动对应的交互服务。")
        return state

    @app.post("/api/services/{session_id}/demo")
    def open_demo(request: Request, session_id: str):
        service(session_id)
        prefix = f"/demos/{session_id}/"
        response = JSONResponse({"url": prefix})
        response.set_cookie(cookie, cookie_token, path=prefix, httponly=True,
                            samesite="strict", secure=request.url.scheme == "https")
        return response

    @app.api_route("/demos/{session_id}/{path:path}", methods=["GET", "POST", "PATCH", "DELETE"])
    async def demo(request: Request, session_id: str, path: str):
        state = service(session_id)
        backend = state["backend"]
        # Only serve executable content from the installed Launcher package.
        if request.method == "GET":
            if not path:
                page = {"hy_nomas": PACKAGE / "nomas_demo.html", "duo": media_assets("duo") / "live.html",
                        "qwen_image": media_assets("qwen_image") / "demo.html",
                        "lingbot_world": media_assets("lingbot_world") / "demo.html"}[backend]
                return FileResponse(page, media_type="text/html", headers={"Content-Security-Policy":
                    "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; "
                    "img-src 'self' blob: data:; media-src 'self' blob:; object-src 'none'; frame-ancestors 'self'"})
            if path == "nomas-cafe.webp" and backend == "hy_nomas":
                return FileResponse(PACKAGE / "ui_plugins/studio/nomas-cafe.webp", media_type="image/webp")
            if path == "avc-player.js" and backend == "duo":
                return FileResponse(media_assets("duo") / path, media_type="text/javascript")
        pattern = ROUTES[backend].get(request.method)
        if not pattern or not re.fullmatch(pattern, path):
            return JSONResponse({"detail": "Unknown demo route."}, status_code=404)

        limit = (64 if backend == "duo" and path.endswith("/turns") else 32) << 20
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > limit:
                return JSONResponse({"detail": "Request body is too large."}, status_code=413)
            body.extend(chunk)
        # The service may have stopped or its port been reused during the upload.
        item = pool.service(session_id)
        with item._lock:
            state = item.state()
            if state["sessionId"] != session_id or state["phase"] != "running" or not state["ready"]:
                raise LauncherError("交互服务已退出或更换。")
            url = state["endpoint"].rstrip("/") + "/" + path
            headers = {key: request.headers[key] for key in ("content-type", "range") if key in request.headers}
            if item._service_api_key:
                headers["Authorization"] = "Bearer " + item._service_api_key
        client = httpx.AsyncClient(trust_env=False, timeout=httpx.Timeout(10, read=None))
        try:
            outgoing = client.build_request(request.method, url, params=request.query_params,
                                            headers=headers, content=bytes(body) if body else None)
            upstream = await client.send(outgoing, stream=True)
        except httpx.HTTPError:
            await client.aclose()
            return JSONResponse({"detail": "交互服务连接失败。"}, status_code=502)
        except BaseException:
            with CancelScope(shield=True):
                await client.aclose()
            raise

        async def stream():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                with CancelScope(shield=True):
                    await upstream.aclose()
                    await client.aclose()

        # Treat unexpected upstream documents as inert data on the launcher origin.
        mime = upstream.headers.get("content-type", "").partition(";")[0].lower()
        if mime not in {"application/json", "image/png", "text/event-stream", "video/mp4", "application/octet-stream"}:
            mime = "application/octet-stream"
        headers = {key: upstream.headers[key] for key in ("content-range", "accept-ranges") if key in upstream.headers}
        return StreamingResponse(stream(), status_code=upstream.status_code, media_type=mime, headers=headers)
