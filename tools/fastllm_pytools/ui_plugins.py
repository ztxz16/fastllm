"""Versioned UI plugins. No plugin Python or model-generated commands are run.

Bundled modules are release code. Editable plugins are HTML/CSS/JavaScript in
opaque-origin frames; the broker below is their only access to host services.
"""
import hashlib
import json
import mimetypes
import os
import re
import shutil
import secrets
import tempfile
import threading
import time
from contextlib import contextmanager
from pathlib import Path, PurePosixPath


BUNDLED_PLUGINS = Path(__file__).with_name("ui_plugins")
CORE_ASSETS = Path(__file__).with_name("plugin_assets")
PLUGIN_ID = re.compile(r"[a-z][a-z0-9-]{0,47}\Z")
CAPABILITIES = {"hardware.read", "runtime.read", "model.chat", "studio.context", "studio.insert"}
SHELL_SLOTS = {"topbar", "sidebar", "statusbar"}
SLOTS = {"page", "studio", "theme"} | SHELL_SLOTS
THEME_COLORS = {"background", "surface", "surfaceMuted", "text", "mutedText", "border",
                "primary", "primaryHover", "primaryText", "primarySoft", "sidebar", "topbar"}
REPLACES = {"", "launch", "download", "logs", "hardware", "webui"}
EXTENSIONS = {".html", ".js", ".css", ".json", ".svg", ".md"}
MAX_PLUGIN_BYTES = 1024 * 1024
MAX_FILES = 64


class PluginError(ValueError):
    pass


class PluginConflict(PluginError):
    pass


def default_plugins_dir():
    return str(Path.home() / ".fastllm" / "plugins")


def mount_studio_assets(app):
    from fastapi import HTTPException
    from fastapi.responses import FileResponse

    @app.get("/assets/webui/{filename}")
    def studio_asset(filename: str):
        directory = (BUNDLED_PLUGINS / "studio" if filename in {"app.js", "styles.css", "template.html"}
                     else Path(__file__).with_name("webui_assets"))
        path = directory / filename
        if filename.startswith(".") or not path.is_file():
            raise HTTPException(404)
        return FileResponse(path)


def _id(value):
    if not isinstance(value, str) or not PLUGIN_ID.fullmatch(value):
        raise PluginError("插件 ID 只允许小写字母、数字和连字符，以字母开头，最多 48 字符")
    return value


def _relative(value):
    if not isinstance(value, str) or "\\" in value or ":" in value or "\x00" in value:
        raise PluginError("插件文件路径无效")
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or str(path) != value
            or any(part in (".", "..") or part.startswith(".") for part in path.parts)
            or path.suffix.lower() not in EXTENSIONS):
        raise PluginError("只能读写插件内的 HTML、JS、CSS、JSON、SVG 和 Markdown 文件")
    return path


def _bounded_int(value, minimum, maximum):
    return type(value) is int and minimum <= value <= maximum


def _theme(value):
    if not isinstance(value, dict) or not value or value.keys() - {"light", "dark", "layout"}:
        raise PluginError("皮肤只支持 light、dark 配色和 layout 布局")
    for mode in ("light", "dark"):
        colors = value.get(mode, {})
        if (not isinstance(colors, dict) or colors.keys() - THEME_COLORS
                or any(not isinstance(c, str) or not re.fullmatch(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?", c)
                       for c in colors.values())):
            raise PluginError("皮肤颜色必须使用支持的名称和 #RGB 或 #RRGGBB，不允许 CSS 代码")
    layout = value.get("layout", {})
    checks = {"sidebarSide": lambda v: v in ("left", "right"),
              "sidebarWidth": lambda v: _bounded_int(v, 180, 320),
              "contentWidth": lambda v: _bounded_int(v, 720, 1800),
              "spacing": lambda v: v in ("compact", "comfortable"),
              "font": lambda v: v in ("system", "serif", "mono"),
              "radius": lambda v: _bounded_int(v, 0, 24)}
    if (not isinstance(layout, dict) or layout.keys() - checks.keys()
            or any(not checks[k](v) for k, v in layout.items())):
        raise PluginError("皮肤布局字段或范围无效")
    return value


def validate_bundle(plugin_id, files):
    _id(plugin_id)
    if not isinstance(files, dict) or not 1 <= len(files) <= MAX_FILES:
        raise PluginError("插件文件数量无效（最多 64 个）")
    size = 0
    for name, content in files.items():
        _relative(name)
        if not isinstance(content, str):
            raise PluginError("插件文件必须为 UTF-8 文本")
        size += len(content.encode("utf-8"))
    if size > MAX_PLUGIN_BYTES:
        raise PluginError("插件总大小不能超过 1 MiB")
    try:
        manifest = json.loads(files["plugin.json"])
    except (KeyError, ValueError) as error:
        raise PluginError("缺少有效的 plugin.json") from error
    if not isinstance(manifest, dict) or manifest.get("id") != plugin_id or manifest.get("apiVersion") != 1:
        raise PluginError("plugin.json 的 id 或 apiVersion 无效")
    if not isinstance(manifest.get("name"), str) or not 1 <= len(manifest["name"]) <= 80:
        raise PluginError("插件名称无效")
    if (not isinstance(manifest.get("slot", "page"), str) or manifest.get("slot", "page") not in SLOTS
            or not isinstance(manifest.get("replaces", ""), str) or manifest.get("replaces", "") not in REPLACES):
        raise PluginError("插件挂载位置无效")
    slot = manifest.get("slot", "page")
    if slot != "page" and manifest.get("replaces"):
        raise PluginError("只有页面插件可以替换启动器页面")
    capabilities = manifest.get("capabilities", [])
    if (not isinstance(capabilities, list) or any(not isinstance(c, str) or c not in CAPABILITIES for c in capabilities)):
        raise PluginError("插件申请了不支持的能力")
    extras = {}
    if slot == "theme":
        if capabilities or "entry" in manifest or any(PurePosixPath(n).suffix not in {".json", ".md"} for n in files):
            raise PluginError("皮肤插件只使用 JSON 配置，不需要 HTML、脚本或服务能力")
        entry = ""
        extras["theme"] = _theme(manifest.get("theme"))
    else:
        if "theme" in manifest:
            raise PluginError("请使用独立的 theme 插件修改主界面皮肤")
        entry = manifest.get("entry", "index.html")
        _relative(entry)
        if entry not in files or not entry.endswith(".html"):
            raise PluginError("插件入口必须是存在的 HTML 文件")
    if "size" in manifest:
        size = manifest["size"]
        if (slot not in SHELL_SLOTS or not isinstance(size, dict) or size.keys() - {"width", "height"}
                or not _bounded_int(size.get("width", 240), 80, 640)
                or not _bounded_int(size.get("height", 44), 24, 160)):
            raise PluginError("主界面组件 size 只支持 width（80–640）和 height（24–160）")
        extras["size"] = size
    # Native modules can only come from the read-only distribution directory.
    if "native" in manifest:
        raise PluginError("用户插件不能声明 native")
    return {"id": plugin_id, "name": manifest["name"], "apiVersion": 1,
            "slot": slot, "replaces": manifest.get("replaces", ""),
            "entry": entry, "capabilities": sorted(set(capabilities)), **extras}


def _revision(files):
    return hashlib.sha256(json.dumps(files, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:20]


class PluginRegistry:
    def __init__(self, directory=None, *, on_enabled=None, runtimes=None, runtime_actions=None):
        self.directory = Path(directory or default_plugins_dir()).expanduser().absolute()
        # Never turn an application/package directory into an editable workspace.
        package = Path(__file__).resolve().parent
        resolved = self.directory.resolve()
        if resolved == package or package in resolved.parents or resolved in package.parents:
            raise PluginError("插件目录必须与 FastLLM 核心代码目录分开")
        self._lock = threading.RLock()
        self._last_good = {}
        self._previews = {}
        # Only application code can register native process hooks. Manifests
        # and editable plugins cannot provide callbacks or execution commands.
        self._on_enabled = on_enabled
        self._runtimes = runtimes or {}
        self._runtime_actions = runtime_actions or {}

    def manage_runtime(self, plugin_id, operation):
        _id(plugin_id)
        action = self._runtime_actions.get(plugin_id)
        if not action or operation not in {"install", "upgrade", "remove", "cancel"}:
            raise PluginError("此插件不支持该运行环境管理操作")
        # Do not hold the registry lock while joining runtime workers.
        try:
            return action(operation)
        except RuntimeError as error:
            raise PluginConflict(str(error)) from error

    def preview(self, plugin_id, files):
        manifest = validate_bundle(plugin_id, files)
        with self._lock:
            token = secrets.token_urlsafe(24)
            self._previews = {k: v for k, v in self._previews.items() if v[0] > time.monotonic()}
            while len(self._previews) >= 8:
                del self._previews[next(iter(self._previews))]
            self._previews[token] = (time.monotonic() + 600, manifest, dict(files))
            return {"token": token, "plugin": {**manifest, "enabled": True, "builtin": False,
                    "revision": _revision(files), "previewToken": token, "error": ""}}

    def preview_files(self, token, plugin_id):
        with self._lock:
            item = self._previews.get(token)
            if not item or item[0] <= time.monotonic() or item[1]["id"] != plugin_id:
                raise PluginError("预览已过期，请重新预览")
            return item[2]

    def _root(self):
        if self.directory.is_symlink() or self.directory.resolve() != self.directory:
            raise PluginError("插件目录不能经过符号链接")
        return self.directory

    def _read_files(self, directory):
        files = {}
        size = 0
        if directory.is_symlink():
            raise PluginError("插件不能使用符号链接")
        for path in sorted(directory.rglob("*")):
            if path.is_symlink():
                raise PluginError("插件不能使用符号链接")
            if not path.is_file():
                continue
            name = path.relative_to(directory).as_posix()
            _relative(name)
            size += path.stat().st_size
            if size > MAX_PLUGIN_BYTES or len(files) >= MAX_FILES:
                raise PluginError("插件超过文件数量或大小限制")
            files[name] = path.read_text(encoding="utf-8")
        return files

    @contextmanager
    def _write_lock(self):
        """Serialize publishers, including a standalone WebUI in another process."""
        root = self._root()
        root.mkdir(parents=True, exist_ok=True)
        path = root / ".lock"
        if path.is_symlink():
            raise PluginError("插件锁不能是符号链接")
        fd = os.open(path, os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0), 0o600)
        try:
            if os.name == "nt":
                import msvcrt
                if os.fstat(fd).st_size == 0:
                    os.write(fd, b"0")
                os.lseek(fd, 0, os.SEEK_SET)
                msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
            else:
                import fcntl
                fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            os.close(fd)

    def _state(self):
        path = self._root() / ".state.json"
        if path.is_symlink():
            raise PluginError("插件状态文件不能是符号链接")
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            return value if isinstance(value, dict) else {}
        except (FileNotFoundError, ValueError):
            return {}

    def _save_state(self, value):
        root = self._root()
        root.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(prefix=".state-", dir=root)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                json.dump(value, stream, ensure_ascii=False)
            os.replace(name, root / ".state.json")
        finally:
            if os.path.exists(name):
                os.unlink(name)

    def list(self):
        with self._lock:
            state = self._state()
            records = []
            for path in sorted(BUNDLED_PLUGINS.glob("*/plugin.json")):
                manifest = json.loads(path.read_text(encoding="utf-8"))
                record = {**manifest, "builtin": True, "enabled": state.get(manifest["id"], True),
                          "revision": "builtin", "error": ""}
                runtime = self._runtimes.get(manifest["id"])
                if runtime:
                    status = runtime()
                    record["runtime"] = {key: status.get(key) for key in
                        ("installed", "phase", "managed", "source", "version", "targetVersion", "stage", "done", "total", "error")}
                    record["runtime"]["manageable"] = manifest["id"] in self._runtime_actions
                records.append(record)
            reserved = {item["id"] for item in records}
            live = set()
            for path in sorted(self._root().glob("*")):
                if path.name.startswith(".") or not path.is_dir():
                    continue
                plugin_id = path.name
                live.add(plugin_id)
                error = ""
                try:
                    if plugin_id in reserved:
                        raise PluginError("用户插件不能覆盖内置插件 ID，请使用新 ID 和 replaces")
                    files = self._read_files(path)
                    manifest = validate_bundle(plugin_id, files)
                    record = {**manifest, "builtin": False, "revision": _revision(files)}
                    self._last_good[plugin_id] = (record, files)
                except (PluginError, OSError, UnicodeError, ValueError) as problem:
                    error = str(problem)
                    previous = self._last_good.get(plugin_id)
                    if not previous:
                        records.append({"id": plugin_id, "name": plugin_id, "builtin": False,
                                        "enabled": False, "revision": "", "error": error})
                        continue
                    record = previous[0]
                records.append({**record, "enabled": state.get(plugin_id, True), "error": error})
            for plugin_id in self._last_good.keys() - live:
                del self._last_good[plugin_id]
            return {"directory": str(self.directory), "plugins": records}

    def get(self, plugin_id):
        _id(plugin_id)
        for item in self.list()["plugins"]:
            if item["id"] == plugin_id:
                return item
        raise PluginError("插件不存在")

    def files(self, plugin_id):
        with self._lock:
            item = self.get(plugin_id)
            if item["builtin"]:
                return {p.name: p.read_text(encoding="utf-8") for p in (BUNDLED_PLUGINS / plugin_id).iterdir()
                        if p.is_file() and p.suffix in EXTENSIONS}
            if plugin_id not in self._last_good:
                raise PluginError(item["error"])
            return dict(self._last_good[plugin_id][1])

    def set_enabled(self, plugin_id, enabled):
        with self._lock, self._write_lock():
            plugin = self.get(plugin_id)
            if not isinstance(enabled, bool):
                raise PluginError("enabled 必须为布尔值")
            state = self._state()
            if enabled and plugin.get("slot") == "theme":
                self._disable_themes(state)
            state[plugin_id] = enabled
            self._save_state(state)
            result = self.get(plugin_id)
        # Process shutdown may join worker threads: never hold registry locks
        # while notifying the owner, which also reads the enabled state.
        if self._on_enabled:
            self._on_enabled(plugin_id)
        return result

    def delete(self, plugin_id, expected_revision):
        with self._lock, self._write_lock():
            plugin = self.get(plugin_id)
            if plugin["builtin"]:
                raise PluginError("内置插件不能删除")
            if expected_revision != plugin["revision"]:
                raise PluginConflict("插件已被其他操作修改，请重新读取后再删除")
            root = self._root()
            directory, backup = root / plugin_id, root / ".history" / (plugin_id + ".json")
            if directory.is_symlink() or backup.is_symlink() or backup.parent.is_symlink():
                raise PluginError("插件及历史目录不能使用符号链接")
            state = self._state()
            shutil.rmtree(directory)
            backup.unlink(missing_ok=True)
            state.pop(plugin_id, None)
            self._save_state(state)
            self._last_good.pop(plugin_id, None)
            return {"deleted": plugin_id}

    def _disable_themes(self, state):
        for plugin in self.list()["plugins"]:
            if plugin.get("slot") == "theme":
                state[plugin["id"]] = False

    def reset_theme(self):
        with self._lock, self._write_lock():
            state = self._state()
            self._disable_themes(state)
            self._save_state(state)
            return self.list()

    def apply(self, plugin_id, files, expected_revision):
        manifest = validate_bundle(plugin_id, files)
        with self._lock, self._write_lock():
            current = next((p for p in self.list()["plugins"] if p["id"] == plugin_id), None)
            if current and current["builtin"]:
                raise PluginError("内置插件只读，请创建定制插件")
            if expected_revision != (current["revision"] if current else ""):
                raise PluginConflict("插件已被其他操作修改，请重新读取后再应用")
            root = self._root()
            root.mkdir(parents=True, exist_ok=True)
            destination = root / plugin_id
            if destination.is_symlink():
                raise PluginError("插件不能使用符号链接")
            # Keep the last valid version for one-click recovery, outside discovery.
            history = root / ".history"
            if history.is_symlink():
                raise PluginError("插件历史目录不能是符号链接")
            history.mkdir(exist_ok=True)
            backup = history / (plugin_id + ".json")
            if backup.is_symlink():
                raise PluginError("插件历史文件不能是符号链接")
            old = self.files(plugin_id) if current and current["revision"] else None
            staged = Path(tempfile.mkdtemp(prefix=".staging-", dir=root))
            moved = root / (staged.name + "-old")
            try:
                for name, content in files.items():
                    target = staged / name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_text(content, encoding="utf-8")
                if destination.exists():
                    os.replace(destination, moved)
                try:
                    os.replace(staged, destination)
                except OSError:
                    if moved.exists():
                        os.replace(moved, destination)
                    raise
                if old is not None:
                    backup.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
            finally:
                shutil.rmtree(staged, ignore_errors=True)
                shutil.rmtree(moved, ignore_errors=True)
            self._last_good[plugin_id] = ({**manifest, "builtin": False, "revision": _revision(files)}, dict(files))
            state = self._state()
            if manifest["slot"] == "theme" and state.get(plugin_id, True):
                self._disable_themes(state)
                state[plugin_id] = True
                self._save_state(state)
            return self.get(plugin_id)

    def rollback(self, plugin_id, expected_revision):
        with self._lock:
            _id(plugin_id)
            path = self._root() / ".history" / (plugin_id + ".json")
            if path.is_symlink() or path.parent.is_symlink():
                raise PluginError("插件历史不能使用符号链接")
            try:
                files = json.loads(path.read_text(encoding="utf-8"))
            except FileNotFoundError as error:
                raise PluginError("没有可恢复的上一版本") from error
            return self.apply(plugin_id, files, expected_revision)


EDITOR_INSTRUCTIONS = """你是 FastLLM 界面插件开发者。只返回 JSON 对象，不要 Markdown：
{"summary":"修改说明","files":{"plugin.json":"JSON文本","index.html":"HTML文本",...}}。
files 是插件的完整文件集合，不是补丁。保留用户未要求改变的行为和外观。
这是连续的界面编辑对话。结合之前的用户要求和修改摘要理解本轮要求；本轮 reference 是最新文件，优先在其基础上修改。
summary 用简短自然语言回复用户，说明本轮修改结果，不要罗列代码或生成过程。
plugin.json: {"apiVersion":1,"id":"指定ID","name":"名称","entry":"index.html",
"slot":"page/studio/topbar/sidebar/statusbar/theme","replaces":"仅page可选launch/download/logs/hardware/webui",
"capabilities":["hardware.read","runtime.read","model.chat","studio.context","studio.insert"]}。
用户请求主界面右上角组件时使用 topbar，侧栏使用 sidebar，底栏使用 statusbar；这些组件切换页面时常驻。
主界面组件可用 size:{"width":240,"height":44}，宽80–640、高24–160像素；布局要紧凑并适配窄屏。
换肤使用独立 slot:theme 插件，只需 plugin.json，不要 entry、HTML、JS 或 capabilities。
theme:{"light":{配色},"dark":{配色},"layout":{布局}}。
配色可用 background/surface/surfaceMuted/text/mutedText/border/primary/primaryHover/primaryText/primarySoft/sidebar/topbar，
值只能是 #RGB 或 #RRGGBB。布局可用 sidebarSide:left/right、sidebarWidth:180–320、contentWidth:720–1800、
spacing:compact/comfortable、font:system/serif/mono、radius:0–24。未指定字段保留默认值，不允许任意 CSS。
皮肤会同步主界面与工作室；侧栏位置和宽度、内容宽度、间距只调整启动器布局。
只能使用实际需要的 capabilities。插件是浏览器 HTML/CSS/JS，不能运行 Python、shell、安装依赖，
不能访问核心、父窗口 DOM、cookie、localStorage、网络或任意文件。不要请求这些权限。
核心自动注入 SDK：await ftllm.call("hardware.read") 返回硬件信息；runtime.read 返回模型服务状态；
model.chat({messages:[{role:"user",content:"..."}],max_tokens:1024}) 返回 {content,reasoning}；
studio.context 返回当前对话副本；studio.insert({text:"..."}) 在工作室输入框插入文本。
硬件返回 {memory:{total,available},gpus:[{name,memoryTotalMiB,memoryFreeMiB,utilization,temperature}]}，内存单位字节，GPU数值可能是字符串。
接收主题事件必须读取 e.detail：window.addEventListener("ftllm-context", e => {
  const {theme, locale, palette} = e.detail; /* 用这些值更新插件样式 */
}); palette 为当前皮肤配色；theme 为 light/dark。不要从 e.theme 或 e.palette 读取。
使用相对路径加载插件自己的 JS/CSS；不使用 CDN。用户未要求换肤时，保持现有浅色/深色、绿色主色和紧凑布局。
界面文案不暴露实现细节。错误用可读消息显示，异步调用使用 try/catch。
"""


def _proposal_request(registry, payload):
    plugin_id = _id(payload.get("id"))
    instruction = payload.get("instruction", "")
    if not isinstance(instruction, str) or not 1 <= len(instruction.strip()) <= 12000:
        raise PluginError("请输入修改要求（最多 12000 字符）")
    history = payload.get("history", [])
    if (not isinstance(history, list)
            or any(not isinstance(message, dict) or message.get("role") not in ("user", "assistant")
                   or not isinstance(message.get("content"), str) for message in history)):
        raise PluginError("编辑对话历史格式无效")
    if len(history) > 128 or sum(len(message["content"]) for message in history) > 120000:
        raise PluginError("编辑对话过长（最多 128 条消息、120000 字符）；未截断历史，请新建对话后重试")
    current = next((p for p in registry.list()["plugins"] if p["id"] == plugin_id), None)
    if current and current["builtin"]:
        raise PluginError("请为定制插件使用新 ID")
    reference = payload.get("reference", "")
    target = payload.get("target", "")
    if not isinstance(target, str) or target and target not in SLOTS:
        raise PluginError("定制目标无效")
    source = registry.files(plugin_id) if current else registry.files(reference) if reference else {}
    if "draft" in payload:
        validate_bundle(plugin_id, payload["draft"])
        if payload.get("expectedRevision") != (current["revision"] if current else ""):
            raise PluginConflict("当前界面已被其他操作修改，请重新读取后再生成")
        source = payload["draft"]
    request = {"id": plugin_id, "instruction": instruction, "reference": source,
               "target": target or "根据修改要求选择挂载位置；修改已有插件时保留位置，除非用户要求移动"}
    messages = [{"role": "system", "content": EDITOR_INSTRUCTIONS},
                *({"role": message["role"], "content": message["content"]} for message in history),
                {"role": "user", "content": json.dumps(request, ensure_ascii=False)}]
    return messages, current["revision"] if current else ""


def _proposal_result(payload, content, revision):
    plugin_id, target = payload["id"], payload.get("target", "")
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        proposal = json.loads(text)
        files = proposal["files"]
    except (ValueError, KeyError, TypeError) as error:
        raise PluginError("模型未返回完整的插件 JSON；未修改任何文件，请重试") from error
    manifest = validate_bundle(plugin_id, files)
    if target and manifest["slot"] != target:
        raise PluginError("模型返回的挂载位置与选择的定制目标不一致；未修改文件，请重试")
    return {"id": plugin_id, "summary": str(proposal.get("summary", ""))[:4000],
            "manifest": manifest, "files": files, "expectedRevision": revision}


def propose_plugin(registry, client, payload):
    messages, revision = _proposal_request(registry, payload)
    try:
        content, _ = client.complete(
            messages,
            max_tokens=16384, thinking_level="关闭", temperature=0.2)
    except (RuntimeError, OSError) as error:
        raise PluginError(str(error)) from error
    return _proposal_result(payload, content, revision)


def stream_proposal(registry, client, payload, control):
    yield {"stage": "reading"}
    messages, revision = _proposal_request(registry, payload)
    yield {"stage": "generating", "characters": 0}
    content = []
    characters = 0
    for delta, _reasoning in client.stream(messages, max_tokens=16384, thinking_level="关闭",
                                           temperature=0.2, top_p=1.0, top_k=1,
                                           repeat_penalty=1.0, control=control):
        control.check()
        characters += len(delta)
        if characters > MAX_PLUGIN_BYTES * 2:
            raise PluginError("生成内容超过大小限制；未修改文件")
        content.append(delta)
        yield {"stage": "generating", "characters": characters}
    yield {"stage": "validating", "characters": characters}
    yield {"stage": "ready", "proposal": _proposal_result(payload, "".join(content), revision)}


def install_plugin_routes(app, registry, model_client, hardware=None, runtime_state=None):
    from fastapi import HTTPException, Request
    from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse

    @app.exception_handler(PluginError)
    async def plugin_error(_request, error):
        return JSONResponse({"error": str(error), "detail": str(error)},
                            status_code=409 if isinstance(error, PluginConflict) else 400)

    def mutation(request):
        # Also protects standalone WebUI, whose existing chat API has no token.
        origin = request.headers.get("origin")
        if request.headers.get("x-ftllm-plugin-request") != "1" or (
                origin and origin != str(request.base_url).rstrip("/")):
            raise HTTPException(403, "Invalid plugin request origin")

    async def body(request):
        mutation(request)
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > MAX_PLUGIN_BYTES * 2:
                raise HTTPException(413, "Plugin request is too large")
        try:
            payload = json.loads(data)
        except (ValueError, UnicodeError) as error:
            raise PluginError("插件请求必须为 JSON 对象") from error
        if not isinstance(payload, dict):
            raise PluginError("插件请求必须为 JSON 对象")
        return payload

    @app.get("/api/plugins")
    def plugins():
        return registry.list()

    @app.get("/api/plugins/{plugin_id}/files")
    def plugin_files(plugin_id: str):
        return {"plugin": registry.get(plugin_id), "files": registry.files(plugin_id)}

    @app.post("/api/plugins/{plugin_id}/enabled")
    async def enabled(plugin_id: str, request: Request):
        from starlette.concurrency import run_in_threadpool
        return await run_in_threadpool(registry.set_enabled, plugin_id, (await body(request)).get("enabled"))

    @app.delete("/api/plugins/{plugin_id}")
    async def delete(plugin_id: str, request: Request):
        return registry.delete(plugin_id, (await body(request)).get("expectedRevision"))

    @app.post("/api/plugins/{plugin_id}/runtime/{operation}")
    async def runtime_operation(plugin_id: str, operation: str, request: Request):
        from starlette.concurrency import run_in_threadpool
        await body(request)
        return await run_in_threadpool(registry.manage_runtime, plugin_id, operation)

    @app.post("/api/plugins/{plugin_id}/apply")
    async def apply(plugin_id: str, request: Request):
        payload = await body(request)
        return registry.apply(plugin_id, payload.get("files"), payload.get("expectedRevision"))

    @app.post("/api/plugins/{plugin_id}/rollback")
    async def rollback(plugin_id: str, request: Request):
        return registry.rollback(plugin_id, (await body(request)).get("expectedRevision"))

    @app.post("/api/plugins/propose")
    async def propose(request: Request):
        from starlette.concurrency import run_in_threadpool
        payload = await body(request)
        return await run_in_threadpool(propose_plugin, registry, model_client(), payload)

    @app.post("/api/plugins/propose-stream")
    async def propose_stream(request: Request):
        import asyncio
        try:
            from .webui_server import GenerationControl
        except ImportError:
            from webui_server import GenerationControl
        payload = await body(request)
        client, control = model_client(), GenerationControl()
        async def events():
            iterator = stream_proposal(registry, client, payload, control)
            try:
                while True:
                    event = await asyncio.to_thread(next, iterator, None)
                    if event is None:
                        break
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except (PluginError, RuntimeError, OSError, ValueError) as error:
                yield json.dumps({"stage": "error", "error": str(error)}, ensure_ascii=False) + "\n"
            finally:
                control.cancel()
        return StreamingResponse(events(), media_type="application/x-ndjson", headers={"Cache-Control": "no-store"})

    @app.post("/api/plugins/preview")
    async def preview(request: Request):
        payload = await body(request)
        return registry.preview(payload.get("id"), payload.get("files"))

    @app.post("/api/plugins/preview/call")
    async def preview_call(request: Request):
        from starlette.concurrency import run_in_threadpool
        capability = (await body(request)).get("capability")
        if capability == "hardware.read" and hardware:
            return await run_in_threadpool(hardware)
        if capability == "runtime.read" and runtime_state:
            return runtime_state()
        raise HTTPException(403, "预览仅提供状态读取；请应用修改后使用交互功能")

    @app.post("/api/plugins/reset-theme")
    async def reset_theme(request: Request):
        await body(request)
        return registry.reset_theme()

    @app.post("/api/plugins/{plugin_id}/call")
    async def call(plugin_id: str, request: Request):
        from starlette.concurrency import run_in_threadpool
        plugin = registry.get(plugin_id)
        payload = await body(request)
        capability = payload.get("capability")
        if not plugin["enabled"] or plugin["builtin"] or capability not in plugin.get("capabilities", []):
            raise HTTPException(403, "Plugin capability is not granted")
        if payload.get("revision") != plugin["revision"]:
            raise HTTPException(409, "Plugin changed; reload it before calling services")
        if capability == "hardware.read" and hardware:
            return await run_in_threadpool(hardware)
        if capability == "runtime.read" and runtime_state:
            return runtime_state()
        if capability == "model.chat":
            arguments = payload.get("arguments") or {}
            if not isinstance(arguments, dict):
                raise PluginError("模型调用参数必须为对象")
            messages = arguments.get("messages")
            if (not isinstance(messages, list) or not 1 <= len(messages) <= 128
                    or any(not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant")
                           or not isinstance(m.get("content"), str) for m in messages)
                    or sum(len(m["content"]) for m in messages) > 120000):
                raise PluginError("模型消息格式或长度无效")
            try:
                max_tokens = min(8192, max(1, int(arguments.get("max_tokens", 1024))))
            except (ValueError, TypeError, OverflowError) as error:
                raise PluginError("max_tokens 无效") from error
            try:
                content, reasoning = await run_in_threadpool(
                    model_client().complete, [{"role": m["role"], "content": m["content"]} for m in messages],
                    max_tokens=max_tokens, thinking_level="关闭")
            except (RuntimeError, OSError) as error:
                raise PluginError(str(error)) from error
            return {"content": content, "reasoning": reasoning}
        raise HTTPException(400, "Capability is unavailable in this context")

    @app.get("/plugin-core/{filename}")
    def core_asset(filename: str):
        if filename not in {"host.js", "sdk.js", "manager.js", "runtime-manager.js", "conversations.js", "styles.css", "appearance.js", "appearance.css", "preview.js", "native-agent.js", "native-agent.css", "session-agent.js", "session-agent.css"}:
            raise HTTPException(404)
        return FileResponse(CORE_ASSETS / filename, headers={"Cache-Control": "no-cache"})

    @app.get("/plugin-runtime/{plugin_id}/{filename:path}")
    def plugin_asset(plugin_id: str, filename: str, request: Request):
        plugin = registry.get(plugin_id)
        if plugin["builtin"] or not plugin["enabled"]:
            raise HTTPException(404)
        files = registry.files(plugin_id)
        return asset_response(files, filename, request, "/plugin-runtime/" + plugin_id)

    @app.get("/plugin-preview/{token}/{plugin_id}/{filename:path}")
    def preview_asset(token: str, plugin_id: str, filename: str, request: Request):
        return asset_response(registry.preview_files(token, plugin_id), filename, request,
                              "/plugin-preview/" + token + "/" + plugin_id)

    def asset_response(files, filename, request, resource_path):
        _relative(filename)
        if filename not in files:
            raise HTTPException(404)
        content = files[filename]
        origin = str(request.base_url).rstrip("/")
        base = str(request.scope.get("root_path", "")).rstrip("/")
        if filename.endswith(".html"):
            # This executes before any plugin code, including code in <head>.
            sdk = f'<script src="{base}/plugin-core/sdk.js"></script>'
            head = re.search(r"<head\b[^>]*>", content, re.IGNORECASE)
            if head:
                content = content[:head.end()] + sdk + content[head.end():]
            else:
                content = "<!doctype html>" + sdk + re.sub(r"<!doctype[^>]*>", "", content, flags=re.IGNORECASE)
        return Response(content, media_type=mimetypes.guess_type(filename)[0] or "text/plain", headers={
            "Content-Security-Policy": (
                f"default-src 'none'; script-src 'unsafe-inline' {origin}{base}{resource_path}/ "
                f"{origin}{base}/plugin-core/sdk.js; style-src 'unsafe-inline' {origin}{base}{resource_path}/; "
                f"img-src data: blob: {origin}{base}{resource_path}/; "
                "font-src data:; connect-src 'none'; object-src 'none'; "
                "frame-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'self'; "
                "sandbox allow-scripts allow-downloads"),
            "Access-Control-Allow-Origin": "*", "Cache-Control": "no-store",
            "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer",
        })
