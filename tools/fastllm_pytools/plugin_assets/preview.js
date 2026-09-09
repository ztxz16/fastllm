import {mountPluginHost} from "./host.js";

// Copy the rendered application into an isolated style scope. Native scripts
// and handlers are never copied; editable code still runs in sandboxed frames.
export function mountPreview({container, source, basePath, request, context, getRecords}) {
  const launcher = !source.host;
  let hosts = [], lifecycle, version = 0, draft = null, mode = context().theme || "light";
  let view = launcher ? source.querySelector(".view.active")?.id.replace("view-", "") || "launch" : "webui";
  let scope, frameHost, studioRoots = [];
  const ignored = "script,style,link,meta,base,iframe,object,embed,.plugin-manager,.plugin-manager-button,.plugin-shell-slot,.plugin-studio-slot,.plugin-page,.plugin-status,[data-view-button^='plugin-']";
  function sheets(root, main = false) {
    const result = [];
    const styles = root.styleSheets || [...root.querySelectorAll('link[rel="stylesheet"]')].map(n => n.sheet).filter(Boolean);
    for (const style of styles) {
      // All application styles come from installed assets, never editable CSS.
      if (!style.href || !new URL(style.href).pathname.match(/\/(?:assets|plugin-core)\//)) continue;
      const css = [...style.cssRules].filter(rule => !main || !rule.cssText.includes(":host")).map(rule => rule.cssText).join("\n");
      const sheet = new CSSStyleSheet();
      sheet.replaceSync(main ? css.replace(/:root((?:\[[^\]]+\])*)/g, (_, attrs) => attrs ? `:host(${attrs})` : ":host")
        .replace(/@media\s*(\((?:max|min)-width:[^{}]+\))/g, "@container $1") : css);
      result.push(sheet);
    }
    return result;
  }
  function copy(node, studios) {
    if (node.nodeType === Node.TEXT_NODE) return document.createTextNode(node.textContent);
    if (node.nodeType !== Node.ELEMENT_NODE || node.matches(ignored)) return null;
    if (node.classList.contains("plugin-native")) {
      const fragment = document.createDocumentFragment();
      for (const child of node.childNodes) { const next = copy(child, studios); if (next) fragment.append(next); }
      return fragment;
    }
    const next = node.cloneNode(false);
    if (next instanceof HTMLDialogElement) next.removeAttribute("open");
    for (const attr of [...next.attributes]) {
      if (/^on/i.test(attr.name) || ["style", "autofocus", "inert", "srcdoc", "formaction"].includes(attr.name)
          || attr.name.startsWith("data-plugin-") || (attr.name === "href" && next.tagName === "A")) next.removeAttribute(attr.name);
    }
    // Copy CSSOM values, avoiding executable attributes and inline style parsing.
    if (!node.shadowRoot) for (const name of node.style) next.style.setProperty(name, node.style.getPropertyValue(name));
    if (node instanceof HTMLInputElement || node instanceof HTMLTextAreaElement || node instanceof HTMLSelectElement) {
      if (!["password", "file"].includes(node.type)) next.value = node.value;
      else { next.value = ""; next.removeAttribute("value"); }
      if ("checked" in node) next.checked = node.checked;
    }
    for (const child of node.childNodes) { const cloned = copy(child, studios); if (cloned) next.append(cloned); }
    if (node instanceof HTMLSelectElement) next.value = node.value;
    if (node.shadowRoot) {
      const shadow = next.attachShadow({mode:"open"});
      shadow.adoptedStyleSheets = sheets(node.shadowRoot);
      for (const child of node.shadowRoot.childNodes) { const cloned = copy(child, studios); if (cloned) shadow.append(cloned); }
      studios.push(shadow);
    }
    return next;
  }
  function records() {
    let items = getRecords();
    if (draft) {
      items = items.filter(p => p.id !== draft.id).map(p => ({...p,
        enabled:draft.slot === "theme" && p.slot === "theme" ? false : p.enabled}));
      items.push(draft);
    }
    return items;
  }
  function navigate(value) {
    view = value;
    if (!scope || !launcher) return;
    for (const panel of scope.querySelectorAll(".view")) panel.classList.toggle("active", panel.id === `view-${value}`);
    for (const button of scope.querySelectorAll("[data-view-button]")) {
      const active = button.dataset.viewButton === value;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page"); else button.removeAttribute("aria-current");
    }
    scope.querySelector(".app-shell")?.classList.toggle("webui-active", value === "webui");
    scope.querySelector("#open-webui")?.classList.toggle("hidden", value === "webui");
    const title = scope.querySelector("#current-view-title");
    if (title) title.textContent = scope.querySelector(`[data-view-button="${CSS.escape(value)}"]`)?.textContent || value;
  }
  function theme(value) {
    mode = value;
    frameHost.dataset.theme = mode;
    for (const root of studioRoots) root.host.dataset.theme = mode;
    const picker = scope.querySelector("#theme-select");
    if (picker) picker.value = mode;
  }
  function interact(root, signal) {
    const $ = selector => root.querySelector(selector);
    const actions = {
      "new-profile":() => $("#profile-editor-modal")?.classList.remove("hidden"),
      "close-profile-editor":() => $("#profile-editor-modal")?.classList.add("hidden"),
      mobileMenu:() => sidebar(!$("#sidebar")?.classList.contains("open")),
      sidebarBackdrop:() => sidebar(false),
      topSettings:() => openDialog("settingsDialog"),
      closeSettings:() => closeDialog(), cancelSettings:() => closeDialog(), saveSettings:() => closeDialog(),
      agentButton:() => openDialog("agentDialog"), closeAgent:() => closeDialog()
    };
    const clickable = `[data-view-button],[data-open-view],[data-editor-section],.agent-card,${Object.keys(actions).map(id => `#${id}`).join(",")}`;
    let modal;
    function sidebar(open) { for (const node of root.querySelectorAll("#sidebar,#sidebarBackdrop")) node.classList.toggle("open", open); }
    function closeDialog() {
      if (!modal) return;
      const {dialog, placeholder, overlay, opener} = modal;
      dialog.close(); placeholder.replaceWith(dialog); overlay.remove(); modal = null; opener?.focus();
    }
    function openDialog(id) {
      closeDialog();
      const dialog = $("#" + id);
      if (!dialog) return;
      const overlay = document.createElement("div"), placeholder = document.createComment("preview dialog");
      overlay.className = "customizer-dialog-overlay";
      modal = {dialog, placeholder, overlay, opener:root.activeElement};
      dialog.replaceWith(placeholder); overlay.append(dialog); root.append(overlay);
      // Non-modal show keeps the dialog inside the scaled preview, out of the
      // document's top layer. The surrounding customization editor stays usable.
      dialog.show();
    }
    for (const node of root.querySelectorAll("button,input,select,textarea")) {
      if ((node.tagName === "BUTTON" && !node.matches(clickable))
          || node.matches('input[type=file],input[type=password],input[type=submit],input[type=image],#language-select')) {
        if (!node.disabled) node.classList.add("customizer-disabled");
        node.disabled = true;
        node.title = "此操作在预览中不可用";
      }
    }
    // These handlers only touch cloned DOM. No native application handlers,
    // persistent settings, chat requests or service actions run in the preview.
    root.addEventListener("click", event => {
      event.stopPropagation();
      const target = event.target;
      const button = target.closest?.(clickable);
      if (target.closest?.("a")) event.preventDefault();
      if (target.classList?.contains("customizer-dialog-overlay")) closeDialog();
      if (target.id === "profile-editor-modal") actions["close-profile-editor"]();
      if (!button || button.disabled) return;
      event.preventDefault();
      if (button.dataset.viewButton || button.dataset.openView) navigate(button.dataset.viewButton || button.dataset.openView);
      else if (button.dataset.editorSection) $("#editor-" + button.dataset.editorSection)?.scrollIntoView({block:"start"});
      else if (button.matches(".agent-card")) {
        for (const card of root.querySelectorAll(".agent-card")) card.classList.toggle("selected", card === button);
        $("#agentButton")?.classList.add("active");
        closeDialog();
      } else actions[button.id]?.();
    }, {signal});
    for (const name of ["submit", "input", "change", "keydown"]) root.addEventListener(name, event => {
      event.stopPropagation();
      if (name === "submit") event.preventDefault();
      if (name === "change") {
        if (event.target.id === "theme-select") theme(event.target.value);
        if (event.target.matches("[data-config-mode]")) {
          const simple = event.target.value !== "custom";
          for (const node of root.querySelectorAll("#profile-parameters,#automatic-config-actions")) node.classList.toggle("hidden", simple);
        }
      }
      if (name === "keydown" && event.key === "Escape") {
        event.preventDefault(); closeDialog(); sidebar(false); actions["close-profile-editor"]();
      }
    }, {signal});
  }
  async function build(candidate) {
    const attempt = ++version;
    lifecycle?.abort();
    for (const host of hosts) host.destroy();
    hosts = []; lifecycle = new AbortController();
    const signal = lifecycle.signal;
    const changedTarget = candidate?.id !== draft?.id || candidate?.slot !== draft?.slot || candidate?.replaces !== draft?.replaces;
    if (candidate !== undefined) draft = candidate;
    const studios = [];
    frameHost = document.createElement("div"); frameHost.className = "customizer-preview-page";
    frameHost.inert = true;
    scope = frameHost.attachShadow({mode:"open"});
    const layout = new CSSStyleSheet();
    const controls = new CSSStyleSheet(); controls.replaceSync(`
      :host{position:relative}
      .customizer-disabled{cursor:default!important;opacity:1!important}
      .customizer-dialog-overlay{position:absolute;inset:0;z-index:100;display:grid;place-items:center;background:#0006;padding:20px}
      .customizer-dialog-overlay>dialog[open]{position:relative;inset:auto;margin:0;max-width:100%;max-height:100%;box-sizing:border-box}
    `);
    layout.replaceSync(":host{display:block;width:100%;height:100%;overflow:hidden;background:var(--bg,#fff);color:var(--ink,#25352e);font-size:13px}");
    scope.adoptedStyleSheets = [...(launcher ? sheets(source, true) : []), layout, controls];
    frameHost.dataset.theme = mode;
    if (launcher) {
      for (const node of source.querySelectorAll("body > .icon-definitions,body > .app-shell")) scope.append(copy(node, studios));
    } else scope.append(copy(source.host, studios));
    container.replaceChildren(frameHost);
    // A Studio that has not been opened yet can still preview its release
    // template. It does not boot chat, create a conversation, or contact a model.
    const studioContainer = launcher && scope.querySelector("#webui-content");
    if (studioContainer && !studios.length) {
      const [templateResponse, styleResponse] = await Promise.all([
        fetch(basePath + "/assets/webui/template.html", {signal}),
        fetch(basePath + "/assets/webui/styles.css", {signal})
      ]);
      if (!templateResponse.ok || !styleResponse.ok) throw new Error("无法加载工作室预览");
      const template = document.createElement("template");
      template.innerHTML = (await templateResponse.text()).replaceAll("__WEBUI_BASE_PATH__", basePath);
      const host = document.createElement("div"); host.setAttribute("data-embedded", "");
      const shadow = host.attachShadow({mode:"open"}), stylesheet = new CSSStyleSheet();
      stylesheet.replaceSync(await styleResponse.text());
      shadow.adoptedStyleSheets = [stylesheet];
      for (const node of template.content.childNodes) { const cloned = copy(node, studios); if (cloned) shadow.append(cloned); }
      // Appearance and extension styles are trusted release assets as well.
      for (const name of ["styles.css", "appearance.css"]) {
        const response = await fetch(basePath + "/plugin-core/" + name, {signal});
        if (!response.ok) throw new Error("无法加载预览样式");
        const sheet = new CSSStyleSheet(); sheet.replaceSync(await response.text());
        shadow.adoptedStyleSheets = [...shadow.adoptedStyleSheets, sheet];
      }
      studioContainer.classList.remove("hidden"); studioContainer.append(host);
      scope.querySelector("#webui-placeholder")?.classList.add("hidden"); studios.push(shadow);
    }
    if (signal.aborted) return;
    const readOnlyRequest = (path, options) => {
      if (!path.endsWith("/call")) throw new Error("预览不允许此操作");
      return request("/api/plugins/preview/call", options);
    };
    const attach = async (root, slot, navigation, target) => {
      const host = await mountPluginHost({root, slot, basePath, navigation, container:target,
        preview:true, catalog:records, signal, request:readOnlyRequest,
        context:() => ({...context(), theme:mode}), navigate});
      if (signal.aborted) host.destroy(); else hosts.push(host);
    };
    if (launcher) await attach(scope, "page", scope.querySelector(".navigation"), scope.querySelector(".page-scroll"));
    for (const root of studios) {
      root.adoptedStyleSheets = [...root.adoptedStyleSheets, controls];
      root.host.dataset.theme = mode;
      const extensions = document.createElement("section"); extensions.className = "plugin-studio-slot";
      root.querySelector("#messages")?.before(extensions);
      await attach(root, "studio", root.querySelector(".top-actions"), extensions);
    }
    if (attempt !== version) return;
    if (changedTarget && draft?.slot === "page") navigate(draft.replaces || `plugin-${draft.id}`);
    else if (changedTarget && draft?.slot === "studio") navigate("webui");
    else navigate(view);
    studioRoots = studios;
    theme(mode);
    for (const root of [scope, ...studios]) interact(root, signal);
    frameHost.inert = false;
  }
  return {
    build:async candidate => { try { await build(candidate); } catch (error) { if (error.name !== "AbortError") throw error; } },
    destroy() { version++; lifecycle?.abort(); for (const host of hosts) host.destroy(); hosts = []; container.replaceChildren(); }
  };
}
