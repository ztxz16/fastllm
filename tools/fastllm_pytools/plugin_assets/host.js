import {mountManager} from "./manager.js";
import {mountAppearance} from "./appearance.js";

export async function mountPluginHost({root = document, basePath = "", request, slot = "page",
  navigation, container, context = () => ({}), studioCall, navigate, signal, preview = false, catalog} = {}) {
  const lifecycle = new AbortController();
  const abort = () => destroy();
  signal?.addEventListener("abort", abort, {once:true});
  const frames = new Map(), nativeViews = new Map();
  let records = [], loading = false, timer, manager;
  const shellSlots = new Map();
  if (slot === "page") {
    for (const [name, selector] of [["topbar", ".topbar"], ["sidebar", ".sidebar"], ["statusbar", ".workspace"]]) {
      const parent = root.querySelector(selector);
      if (!parent) continue;
      const region = document.createElement("section");
      region.className = `plugin-shell-slot plugin-shell-${name}`; region.hidden = true;
      region.setAttribute("aria-label", {topbar:"顶部扩展", sidebar:"侧栏扩展", statusbar:"底部扩展"}[name]);
      if (name === "sidebar") parent.querySelector(".sidebar-footer").before(region);
      else parent.append(region);
      shellSlots.set(name, region);
    }
  }
  const appearance = mountAppearance({root, basePath, studio:slot === "studio", styles:!preview, changed:() => {
    for (const frame of frames.values()) sendContext(frame);
  }});
  const style = document.createElement("link");
  style.rel = "stylesheet"; style.href = basePath + "/plugin-core/styles.css";
  if (!preview) (root.head || root).append(style);
  const api = (path, options = {}) => request(path, {...options, signal:options.signal || lifecycle.signal,
    headers:{...options.headers, "X-FTLLM-Plugin-Request":"1"}});
  const button = document.createElement("button");
  button.type = "button"; button.className = "plugin-manager-button";
  button.title = "自定义界面";
  button.innerHTML = `<span class="plugin-manager-emblem" aria-hidden="true"><svg viewBox="0 0 24 24">
    <path d="M11 5H6a3 3 0 0 0-3 3v10a3 3 0 0 0 3 3h12a3 3 0 0 0 3-3v-6M3 10h8M8 10v11"/>
    <path d="m17 2 1.3 3.7L22 7l-3.7 1.3L17 12l-1.3-3.7L12 7l3.7-1.3L17 2Z" class="plugin-manager-spark"/>
    <path d="m12 17 4-4m-5 5 1-1"/></svg></span><span>自定义界面</span>`;
  if (!preview) {
    const embedded = root.host?.hasAttribute("data-embedded");
    if (!embedded) manager = mountManager({root, basePath, request:api, refresh, getRecords:() => records, context});
    if (slot === "page") {
      (navigation || container).append(button);
      button.addEventListener("click", () => manager.open(), {signal:lifecycle.signal});
    }
  }
  window.addEventListener("message", event => {
    if (event.origin !== "null" || event.data?.type !== "ftllm:connect" || !event.ports[0]) return;
    const frame = [...frames.values()].find(item => item.iframe.contentWindow === event.source);
    if (!frame) return;
    if (frame.port) { event.ports[0].close(); return; }
    frame.connect(event.ports[0]);
  }, {signal:lifecycle.signal});

  function sendContext(frame) { frame.port?.postMessage({type:"context", value:{...context(), palette:appearance.palette()}}); }
  function removeFrame(id) {
    const frame = frames.get(id);
    if (!frame) return;
    frame.controller.abort(); frame.port?.close(); clearTimeout(frame.timeout);
    frame.panel.remove(); frame.button?.remove(); frames.delete(id);
  }
  function createFrame(plugin, parent, customNavigation = false) {
    const panel = document.createElement("section");
    panel.className = customNavigation ? "view plugin-page" : "plugin-page";
    panel.id = `view-plugin-${plugin.id}`;
    if (shellSlots.has(plugin.slot)) {
      panel.style.setProperty("--plugin-width", (plugin.size?.width || 240) + "px");
      panel.style.setProperty("--plugin-height", (plugin.size?.height || (plugin.slot === "sidebar" ? 96 : plugin.slot === "statusbar" ? 32 : 44)) + "px");
    }
    const status = document.createElement("p"); status.className = "plugin-status";
    status.textContent = "正在加载…";
    const iframe = document.createElement("iframe");
    iframe.title = plugin.name; iframe.className = "plugin-frame";
    iframe.setAttribute("sandbox", "allow-scripts allow-downloads");
    iframe.referrerPolicy = "no-referrer";
    const runtimePath = plugin.previewToken ? `/plugin-preview/${plugin.previewToken}` : "/plugin-runtime";
    iframe.src = `${basePath}${runtimePath}/${encodeURIComponent(plugin.id)}/${plugin.entry}?revision=${plugin.revision}`;
    panel.append(status, iframe); parent.append(panel);
    const controller = new AbortController();
    const frame = {panel, iframe, controller, revision:plugin.revision, calls:0};
    frames.set(plugin.id, frame);
    if (customNavigation) {
      const nav = document.createElement("button"); nav.type = "button"; nav.className = "nav-button";
      nav.dataset.viewButton = `plugin-${plugin.id}`; nav.title = plugin.name;
      nav.innerHTML = '<svg class="icon" aria-hidden="true"><use href="#icon-grid"/></svg><span class="plugin-navigation-label"><strong></strong></span>';
      nav.querySelector("strong").textContent = plugin.name;
      navigation.insertBefore(nav, preview ? null : button); frame.button = nav;
    }
    frame.timeout = setTimeout(() => { status.hidden = false; status.textContent = "组件未能加载，可以在自定义界面中停用或恢复上一版。"; }, 15000);
    frame.connect = port => {
      frame.port = port;
      port.onmessage = async ({data}) => {
        if (controller.signal.aborted || !data || typeof data !== "object") return;
        if (data.type === "ready") { clearTimeout(frame.timeout); status.hidden = !frame.failed; sendContext(frame); return; }
        if (data.type === "error") { frame.failed = true; status.hidden = false; status.textContent = `组件错误：${String(data.message).slice(0,1000)}`; return; }
        if (!Number.isSafeInteger(data.id)) return;
        let value, error;
        try {
          if (!plugin.capabilities.includes(data.capability)) throw new Error("插件未申请此能力");
          if (preview && !["hardware.read", "runtime.read"].includes(data.capability)) throw new Error("预览仅提供状态读取，请应用后使用交互功能");
          if (frame.calls >= 4 || JSON.stringify(data.arguments).length > 160000) throw new Error("插件请求过多或过大");
          frame.calls++;
          try {
            if (data.capability.startsWith("studio.")) {
              if (!studioCall) throw new Error("请先打开工作室");
              value = await studioCall(data.capability, data.arguments);
            } else {
              value = await api(`/api/plugins/${plugin.id}/call`, {method:"POST", signal:controller.signal,
                body:JSON.stringify({revision:plugin.revision, capability:data.capability, arguments:data.arguments})});
            }
          } finally { frame.calls--; }
        } catch (problem) { error = String(problem.message || problem); }
        if (!controller.signal.aborted) port.postMessage({id:data.id, value, error});
      };
      port.start(); port.postMessage({type:"connected"});
    };
    return frame;
  }
  function updateFrame(plugin, parent, customNavigation = false) {
    const previous = frames.get(plugin.id);
    if (previous?.revision === plugin.revision) return;
    const wasActive = previous?.button && previous.panel.classList.contains("active");
    removeFrame(plugin.id); createFrame(plugin, parent, customNavigation);
    if (wasActive) navigate?.(customNavigation ? `plugin-${plugin.id}` : plugin.replaces || "launch");
  }
  function reconcile() {
    appearance.update(records);
    const desired = new Set();
    if (slot === "page") {
      for (const plugin of records.filter(p => p.builtin && p.view)) {
        const panel = root.querySelector(`#view-${plugin.view}`);
        if (!panel) continue;
        let entry = nativeViews.get(plugin.view);
        const replacement = records.find(p => !p.builtin && p.enabled && p.revision && p.replaces === plugin.view);
        if (!entry && (replacement || !plugin.enabled)) {
          const native = document.createElement("div"); native.className = "plugin-native";
          native.append(...panel.childNodes); panel.append(native);
          const status = document.createElement("p"); status.className = "plugin-status"; status.textContent = "此页面已停用，可在自定义界面中重新启用。";
          panel.append(status); entry = {panel, native, status}; nativeViews.set(plugin.view, entry);
        }
        if (entry) {
          entry.native.hidden = Boolean(replacement || !plugin.enabled);
          entry.status.hidden = Boolean(replacement || plugin.enabled);
        }
        if (replacement) {
          desired.add(replacement.id);
          updateFrame(replacement, panel);
        }
      }
    }
    for (const plugin of records.filter(p => !p.builtin && p.enabled && p.revision && (p.slot === slot || shellSlots.has(p.slot)) && !p.replaces)) {
      desired.add(plugin.id);
      updateFrame(plugin, shellSlots.get(plugin.slot) || container, plugin.slot === "page");
    }
    for (const [id, frame] of frames) {
      if (!desired.has(id)) {
        const active = frame.panel.classList.contains("active"); removeFrame(id);
        if (active) navigate?.("launch");
      } else sendContext(frame);
    }
    if (slot === "studio") container.hidden = !frames.size;
    for (const region of shellSlots.values()) region.hidden = !region.childElementCount;
  }
  async function refresh() {
    if (loading || lifecycle.signal.aborted) return;
    loading = true;
    try {
      const result = catalog ? {plugins:catalog()} : await api("/api/plugins");
      if (lifecycle.signal.aborted) return;
      records = result.plugins; reconcile(); manager?.update(result);
    } finally { loading = false; }
  }
  function destroy() {
    if (lifecycle.signal.aborted) return;
    lifecycle.abort(); clearInterval(timer); signal?.removeEventListener("abort", abort);
    for (const id of [...frames.keys()]) removeFrame(id);
    for (const {panel, native, status} of nativeViews.values()) { native.replaceWith(...native.childNodes); status.remove(); }
    manager?.destroy(); button.remove(); style.remove(); appearance.destroy();
    for (const region of shellSlots.values()) region.remove();
  }
  await refresh().catch(error => { button.title = error.message; });
  if (!preview) timer = setInterval(() => refresh().catch(() => {}), 2000);
  if (signal?.aborted) destroy();
  return {destroy, refresh, open:() => manager?.open(), title:view => records.find(p => `plugin-${p.id}` === view)?.name};
}
