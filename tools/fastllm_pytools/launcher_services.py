"""Multiple independently managed services, with the legacy selected-service API."""
import threading
import urllib.error
import urllib.request

from .launcher import LauncherError, LauncherRuntime
from .service_backends import BACKENDS, backend_catalog


ACTIVE = {"starting", "running", "stopping"}
NATIVE_AGENTS = ("harness", "opencode", "codex", "claude")
MEDIA_BODY_LIMIT = 32 * 1024 * 1024
MEDIA_RESULT_LIMIT = 256 * 1024 * 1024


class LauncherServicePool:
    # Configuration, downloads, installer and plugin installation belong to the
    # launcher. Agents, chat sessions and process state belong to each service.
    SHARED = {"config_path", "profiles", "default_profile", "save_profile", "delete_profile",
              "download_defaults", "download_catalog", "download_state", "preview_download",
              "start_download", "stop_download", "plugins", "_agent_installer",
              "set_shutdown_callback", "request_shutdown", "logs", "clear_logs"}

    def __init__(self, runtime):
        self.root = runtime
        self.current = runtime
        self.members = [runtime]
        self.lock = threading.RLock()
        self._closed = False
        # Embedded LLMs share history. Reserve a conversation across services,
        # while each WebUI retains ownership of its own cancellation controls.
        self.generation_lock = threading.RLock()
        self.generation_controls = {}
        runtime.media_services = self
        runtime._agent_installer._on_installed = self._enable_installed_agent
        runtime.plugins._on_enabled = self._plugin_enabled_changed
        runtime.plugins._runtimes = {name: lambda name=name: self._managed_agent(name).state()
                                    for name in NATIVE_AGENTS}
        runtime.plugins._runtime_actions = {name: lambda operation, name=name: self.manage_agent(name, operation)
                                           for name in NATIVE_AGENTS}
        # One log cursor, with service labels on child entries.
        self._original_append_log = runtime._append_log

    def _enable_installed_agent(self):
        for item in tuple(self.members):
            item._enable_installed_agent()

    def _plugin_enabled_changed(self, plugin_id):
        for item in tuple(self.members):
            item._plugin_enabled_changed(plugin_id)

    def __getattr__(self, name):
        return getattr(self.root if name in self.SHARED else self.current, name)

    def _managed_agent(self, name):
        # Installation is shared. Its status must remain visible when the
        # selected model changes while an operation is running.
        agents = [getattr(item, name) for item in tuple(self.members)]
        return next((agent for agent in agents if agent.state()["phase"] not in {"stopped", "failed"}),
                    getattr(self.current, name))

    def _stop_agent_peers(self, name):
        for item in self.members:
            agent = getattr(item, name)
            if agent.state()["phase"] in {"installing", "upgrading", "removing", "starting"}:
                raise LauncherError("Agent 正在启动或更新，请等待完成或先取消。")
        # Native agents share their installation and workspace directories;
        # preserve the single active instance of each adapter across models.
        for item in self.members:
            if item is not self.current:
                agent = getattr(item, name)
                agent.stop()
                if agent._thread and agent._thread.is_alive():
                    raise LauncherError("Agent 正在停止，请稍后重试。")

    def open_agent(self, agent_id, launcher_host, browser_origin, *, install=False):
        if agent_id not in NATIVE_AGENTS:
            raise LauncherError("Unknown agent.")
        with self.lock:
            state = self.current.state()
            if self._closed or state.get("kind", "llm") != "llm" or not state["ready"]:
                raise LauncherError("请先选择已就绪的 LLM 服务。")
            self._stop_agent_peers(agent_id)
            return self.current.open_agent(agent_id, launcher_host, browser_origin, install=install)

    def open_harness(self, launcher_host, browser_origin, *, install=False):
        return self.open_agent("harness", launcher_host, browser_origin, install=install)

    def manage_agent(self, agent_id, operation):
        with self.lock:
            if self._closed:
                raise LauncherError("Launcher 已关闭。")
            if operation == "cancel":
                for item in self.members:
                    getattr(item, agent_id).stop()
                return getattr(self.current, agent_id).state()
            self._stop_agent_peers(agent_id)
            return getattr(self.current, agent_id).manage(operation)

    def states(self):
        with self.lock:
            states = [item.state() for item in self.members]
            return [state for state in states if state.get("sessionId")]

    def state(self):
        with self.lock:
            return {**self.current.state(), "services": self.states(), "backends": backend_catalog()}

    def service(self, session_id):
        with self.lock:
            for item in self.members:
                if session_id and item.state()["sessionId"] == session_id:
                    return item
        raise LauncherError("模型服务已退出或更换，请刷新服务列表。")

    def select(self, session_id):
        with self.lock:
            self.current = self.service(session_id)
            return self.state()

    def start(self, payload):
        if not isinstance(payload, dict):
            raise LauncherError("Invalid configuration format.")
        try:
            port = int(payload.get("port", "8080"))
        except (TypeError, ValueError):
            raise LauncherError("端口必须在 1-65535 之间。") from None
        with self.lock:
            if self._closed:
                raise LauncherError("Launcher 已关闭。")
            # Check before spawning: a starting child may not have bound yet.
            for item in self.members:
                state = item.state()
                if state["phase"] in ACTIVE and int(state["endpoint"].rsplit(":", 1)[-1]) == port:
                    raise LauncherError("该端口已有模型服务，请为并行服务选择不同端口。")
            idle = next((item for item in self.members if item.state()["phase"] not in ACTIVE), None)
            if idle is None:
                if len(self.members) >= 16:
                    raise LauncherError("同时运行的模型服务不能超过 16 个。")
                root = self.root
                idle = LauncherRuntime(root.config_path, popen_factory=root._popen,
                    webui_history_dir=root._webui_history_dir,
                    agent_workspace_root=root._agent_workspace_root,
                    allow_remote_workspace_agent=root._allow_remote_workspace_agent,
                    disable_workspace_agent=root._disable_workspace_agent)
                idle.plugins = root.plugins
                idle.media_services = self
                idle._append_log = lambda source, level, message, item=idle: self._original_append_log(
                    source, level, "[" + (item.state().get("profileName") or item.state().get("modelName") or "service") + "] " + str(message))
                self.members.append(idle)
            idle.start(payload)
            self.current = idle
            return self.state()

    def stop(self, session_id=None):
        with self.lock:
            item = self.service(session_id) if session_id else self.current
            item.stop()
            return self.state()

    def embedded_webui(self, session_id, launcher_host):
        return self.service(session_id).embedded_webui(session_id, launcher_host)

    def catalog(self):
        # Credentials stay in the owning process manager, never in discovery.
        return [{"id": state["sessionId"], "name": state["profileName"] or state["modelName"],
                 "model": state["modelName"], **BACKENDS[state["backend"]].api_contract(state)}
                for state in self.states() if state["ready"] and state["phase"] == "running"
                and state.get("kind", "llm") != "llm"]

    def media_request(self, service_id, method, path, data=None, content_type="application/json"):
        """Bounded, authenticated requests allowed by a registered backend.

        Deliberately not a URL proxy: neither a model nor a browser can choose
        an arbitrary host/path, forward a credential, or follow a redirect.
        """
        if data is not None and len(data) > MEDIA_BODY_LIMIT:
            raise LauncherError("生成请求不能超过 32 MiB。")
        item = self.service(service_id)
        with item._lock:
            state = item.state()
            if state["sessionId"] != service_id:
                raise LauncherError("模型服务已退出或更换，请刷新服务列表。")
            if not state["ready"] or state["phase"] != "running":
                raise LauncherError("生成服务尚未就绪或已停止。")
            backend = BACKENDS[state.get("backend", "llm")]
            route = getattr(backend, "media_path", lambda *_: None)(method, path)
            if route is None:
                raise LauncherError("不支持的生成 API 操作。")
            headers = {"Content-Type": content_type}
            if item._service_api_key:
                headers["Authorization"] = "Bearer " + item._service_api_key
            url = state["endpoint"].rstrip("/") + route
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        request = urllib.request.Request(url, data=data, headers=headers, method=method)
        try:
            response = opener.open(request, timeout=120)
        except urllib.error.HTTPError as error:
            response = error
        except (OSError, urllib.error.URLError) as error:
            raise LauncherError("无法连接视频服务：" + str(error)) from error
        with response:
            body = response.read(MEDIA_RESULT_LIMIT + 1)
            if len(body) > MEDIA_RESULT_LIMIT:
                raise LauncherError("生成结果超过 256 MiB 下载限制。")
            return response.code, response.headers.get("Content-Type", "application/octet-stream"), body

    def close(self):
        with self.lock:
            self._closed = True
            for item in self.members:
                item.close()


def install_service_routes(app, pool):
    from fastapi import Request
    from fastapi.responses import Response
    from starlette.concurrency import run_in_threadpool

    @app.post("/api/services/{service_id}/select")
    def select(service_id: str):
        return pool.select(service_id)

    @app.post("/api/services/{service_id}/stop")
    def stop(service_id: str):
        return pool.stop(service_id)

    @app.get("/api/services")
    def services():
        return {"services": pool.states(), "backends": backend_catalog()}

    @app.api_route("/api/media/{service_id}/{path:path}", methods=["GET", "POST"])
    async def media(service_id: str, path: str, request: Request):
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MEDIA_BODY_LIMIT:
                return Response(status_code=413)
        status, content_type, body = await run_in_threadpool(pool.media_request, service_id,
            request.method, path, bytes(data) if data else None,
            request.headers.get("content-type", "application/json"))
        return Response(body, status_code=status, media_type=content_type)
