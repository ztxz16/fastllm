// Shared by ftllm webui and the Launcher conversation pane.
let localesRetries = 0;
function loadLocales(signal) {
  if (window.FASTLLM_LOCALES) return Promise.resolve();
  return new Promise((resolve, reject) => {
    const script = document.createElement("script");
    const url = new URL("../webui_locales.js", import.meta.url);
    if (localesRetries) url.searchParams.set("retry", localesRetries);
    script.src = url.href;
    const cleanup = () => { script.remove(); script.onload = script.onerror = null; signal.removeEventListener("abort", abort); };
    const fail = error => { cleanup(); localesRetries += 1; reject(error); };
    const abort = () => fail(new DOMException("WebUI closed", "AbortError"));
    script.onload = () => { cleanup(); resolve(); };
    script.onerror = () => fail(new Error("Unable to load WebUI. Try reopening it."));
    if (signal.aborted) { abort(); return; }
    signal.addEventListener("abort", abort, {once:true});
    document.head.append(script);
  });
}

function abortable(promise, signal) {
  return new Promise((resolve, reject) => {
    const abort = () => reject(new DOMException("WebUI closed", "AbortError"));
    if (signal.aborted) { abort(); return; }
    signal.addEventListener("abort", abort, {once:true});
    promise.then(resolve, reject).finally(() => signal.removeEventListener("abort", abort));
  });
}

export async function mountWebUI(host, {basePath = "", embedded = false, locale = "", iconUrl = "", signal} = {}) {
  const base = new URL(basePath || "/", location.origin);
  if (base.origin !== location.origin) throw new Error("WebUI must use the current origin.");
  basePath = base.pathname.replace(/\/$/, "");
  const lifecycle = new AbortController();
  const abort = () => lifecycle.abort();
  if (signal?.aborted) abort();
  signal?.addEventListener("abort", abort, {once:true});
  const root = host.attachShadow({mode:"open"});
  host.toggleAttribute("data-embedded", embedded);
  let observer;
  function destroy() {
    lifecycle.abort();
    signal?.removeEventListener("abort", abort);
    observer?.disconnect();
    root.querySelectorAll("dialog[open]").forEach(dialog => dialog.close());
    host.remove();
  }
  try {
    const response = await fetch(new URL("./template.html", import.meta.url), {signal:lifecycle.signal});
    if (!response.ok) throw new Error("Unable to load WebUI. Try reopening it.");
    const template = document.createElement("template");
    const escapedBase = basePath.replaceAll("&", "&amp;").replaceAll('"', "&quot;").replaceAll("<", "&lt;");
    template.innerHTML = (await response.text()).replaceAll("__WEBUI_BASE_PATH__", escapedBase);
    const stylesheet = document.createElement("link");
    stylesheet.rel = "stylesheet";
    stylesheet.href = new URL("./styles.css", import.meta.url).href;
    const styled = new Promise((resolve, reject) => {
      stylesheet.onload = resolve;
      stylesheet.onerror = () => reject(new Error("Unable to load WebUI. Try reopening it."));
    });
    root.append(stylesheet, template.content.cloneNode(true));
    await abortable(Promise.all([styled, loadLocales(lifecycle.signal)]), lifecycle.signal);
    function guard(handler) {
      return function (...args) {
        if (lifecycle.signal.aborted) return;
        const failed = error => { if (!lifecycle.signal.aborted) toast(error.message || String(error)); };
        try { const result = handler.apply(this, args); result?.catch?.(failed); return result; }
        catch (error) { failed(error); }
      };
    }
    function listen(type, handler) { root.addEventListener(type, guard(handler), {signal:lifecycle.signal}); }
    const $ = (selector) => root.querySelector(selector);
    const localeResources = window.FASTLLM_LOCALES || {};
    const preferredLocale = localeResources[locale] ? locale : (() => { try { const saved = localStorage.getItem("fastllm.locale"); if (localeResources[saved]) return saved; } catch (_) {} return "zh-CN"; })();
    const state = { config: {}, conversations: [], activeId: null, record: null, pending: [], uploading: 0, tasks: new Map(), stopping: new Set(), loadSequence: 0, agent: "chat", locale: preferredLocale, menuConversationId: null, renameConversationId: null, deleteConversationId: null, workspaceDirectory: null, workspaceLoading: false };
    const icon = `<img src="${new URL(iconUrl || localUrl("/assets/fastllm_icon.svg"), location.origin).href.replaceAll('"', "&quot;")}" alt="" draggable="false">`;
    const thinkingOptions = [["关闭","thinking.off"],["低","thinking.low"],["中","thinking.medium"],["高","thinking.high"]];
    const conversationOptions = [["关闭","mode.chat"],["快速搜索","mode.web_fast"],["深度浏览","mode.web_deep"]];
    const agentLabelKeys = {knowledge:"agent.knowledge",data:"agent.data",ppt:"agent.ppt",code:"agent.code",workspace:"agent.workspace"};
    const toolLabelKeys = {
      runtime_info:"tool.runtime_info", list_project_files:"tool.list_project_files",
      read_project_file:"tool.read_project_file", search_project_files:"tool.search_project_files",
      web_search:"tool.web_search", read_web_page:"tool.read_web_page", read:"tool.read",
      bash:"tool.bash", edit:"tool.edit", write:"tool.write", grep:"tool.grep",
      find:"tool.find", ls:"tool.ls"
    };
    const workspaceVideoExtensions = new Set(["avi","mkv","mov","mp4","webm"]);

    function t(key, params = {}) {
      const fallback = localeResources["zh-CN"] || {}; const active = localeResources[state.locale] || fallback; let value = active[key] ?? fallback[key] ?? key;
      for (const [name,replacement] of Object.entries(params)) value = value.replaceAll(`{${name}}`,String(replacement));
      return value;
    }

    function eventMessage(event) {
      return event.message_key ? t(event.message_key,event.message_params || {}) : event.message;
    }

    function renderLanguageMenu() {
      const menu = $("#languageMenu"); menu.replaceChildren();
      for (const locale of Object.keys(localeResources)) { const button = document.createElement("button"); button.className = `language-option${locale === state.locale ? " selected" : ""}`; button.textContent = localeResources[locale]["locale.name"] || locale; button.onclick = guard(() => setLocale(locale)); menu.append(button); }
    }

    function applyLocale() {
      host.lang = state.locale; if (!embedded) document.documentElement.lang = state.locale; root.querySelectorAll("[data-i18n]").forEach(node => node.textContent = t(node.dataset.i18n)); root.querySelectorAll("[data-i18n-placeholder]").forEach(node => node.placeholder = t(node.dataset.i18nPlaceholder)); root.querySelectorAll("[data-i18n-title]").forEach(node => node.title = t(node.dataset.i18nTitle)); root.querySelectorAll("[data-i18n-aria]").forEach(node => node.setAttribute("aria-label",t(node.dataset.i18nAria)));
      $("#languageLabel").textContent = t("locale.short"); renderLanguageMenu(); if (state.config.model) $("#modelName").textContent = state.config.model; renderSidebar(); renderPending(); if (state.record) { renderMessages(); renderModes(); } if ($("#workspaceDialog").open) renderWorkspaceDirectory(); syncGenerationUI();
    }

    function setLocale(locale) {
      if (!localeResources[locale]) return; state.locale = locale; try { localStorage.setItem("fastllm.locale",locale); } catch (_) {} $("#languageMenu").classList.remove("open"); applyLocale();
    }

    function localUrl(path) {
      const value = String(path || "");
      return value.startsWith("/") && !value.startsWith("//") ? basePath + value : value;
    }

    async function api(path, options = {}) {
      const response = await fetch(localUrl(path), { ...options, signal: lifecycle.signal, headers: { "Content-Type": "application/json", ...(options.headers || {}) } });
      if (!response.ok) {
        let message = `${response.status} ${response.statusText}`;
        try { message = (await response.json()).detail || message; } catch (_) {}
        throw new Error(message);
      }
      return response;
    }

    function toast(message) {
      const node = document.createElement("div"); node.className = "toast"; node.textContent = message;
      $("#toasts").append(node); setTimeout(() => node.remove(), 4200);
    }

    function setSidebar(open) { $("#sidebar").classList.toggle("open", open); $("#sidebarBackdrop").classList.toggle("open", open); }

    function closeConversationMenu() {
      $("#conversationActionMenu").classList.remove("open"); root.querySelectorAll(".conversation-more.menu-open").forEach(button => button.classList.remove("menu-open")); state.menuConversationId = null;
    }

    function openConversationMenu(conversationId, trigger) {
      const menu = $("#conversationActionMenu"); const reopening = state.menuConversationId === conversationId && menu.classList.contains("open"); closeConversationMenu(); if (reopening) return;
      state.menuConversationId = conversationId; trigger.classList.add("menu-open"); menu.classList.add("open"); const bounds = host.getBoundingClientRect(); const rect = trigger.getBoundingClientRect(); const triggerRect = {right:rect.right-bounds.left,top:rect.top-bounds.top,bottom:rect.bottom-bounds.top}; const menuRect = menu.getBoundingClientRect(); const left = Math.min(host.clientWidth - menuRect.width - 8,Math.max(8,triggerRect.right - menuRect.width)); let top = triggerRect.bottom + 5; if (top + menuRect.height > host.clientHeight - 8) top = triggerRect.top - menuRect.height - 5; menu.style.left = `${left}px`; menu.style.top = `${Math.max(8,top)}px`;
    }

    function isConversationGenerating(id = state.activeId) {
      if (!id) return false;
      if (state.tasks.has(id)) return true;
      return Boolean(state.conversations.find(item => item.id === id)?.generating);
    }

    function renderSidebar() {
      closeConversationMenu(); const list = $("#conversationList"); list.replaceChildren();
      for (const conversation of state.conversations) {
        const running = isConversationGenerating(conversation.id); const row = document.createElement("div"); row.className = `conversation${conversation.id === state.activeId ? " active" : ""}${running ? " running" : ""}`;
        const button = document.createElement("button"); button.className = "conversation-main"; const label = document.createElement("span"); label.className = "conversation-title"; label.textContent = conversation.title; button.append(label); if (running) { const status = document.createElement("span"); status.className = "conversation-running"; status.textContent = t("nav.running"); button.append(status); } button.onclick = guard(() => loadConversation(conversation.id));
        const more = document.createElement("button"); more.className = "conversation-more"; more.setAttribute("aria-label",t("nav.more")); more.innerHTML = `<svg viewBox="0 0 24 24" fill="currentColor"><circle cx="5" cy="12" r="1.6"/><circle cx="12" cy="12" r="1.6"/><circle cx="19" cy="12" r="1.6"/></svg>`; more.onclick = guard(event => { event.stopPropagation(); openConversationMenu(conversation.id,more); });
        row.append(button,more); list.append(row);
      }
    }

    function openRenameDialog() {
      const conversation = state.conversations.find(item => item.id === state.menuConversationId); if (!conversation || isConversationGenerating(conversation.id)) return; state.renameConversationId = conversation.id; closeConversationMenu(); $("#renameTitle").value = conversation.title; $("#renameDialog").showModal(); $("#renameTitle").select();
    }

    async function renameConversation() {
      const id = state.renameConversationId; const title = $("#renameTitle").value.trim(); if (!id) return; if (!title) { toast(t("rename.empty")); return; }
      try { const updated = await (await api(`/api/conversations/${id}`,{method:"PATCH",body:JSON.stringify({title})})).json(); if (id === state.activeId) state.record = updated; $("#renameDialog").close(); state.renameConversationId = null; await refreshConversations(); }
      catch (error) { toast(error.message); }
    }

    function openDeleteDialog() {
      const conversation = state.conversations.find(item => item.id === state.menuConversationId); if (!conversation || isConversationGenerating(conversation.id)) return; state.deleteConversationId = conversation.id; closeConversationMenu(); $("#deleteDialog").showModal();
    }

    async function deleteConversation() {
      const id = state.deleteConversationId; if (!id) return; const wasActive = id === state.activeId;
      try { await api(`/api/conversations/${id}`,{method:"DELETE"}); $("#deleteDialog").close(); state.deleteConversationId = null; await refreshConversations(); if (wasActive) { if (!state.conversations.length) await newConversation(); else await loadConversation(state.conversations[0].id); } }
      catch (error) { toast(error.message); }
    }

    function appendInline(parent, text) {
      const pattern = /(\*\*[^*]+\*\*|`[^`]+`)/g; let cursor = 0;
      for (const match of text.matchAll(pattern)) {
        parent.append(document.createTextNode(text.slice(cursor, match.index)));
        const token = match[0]; const node = document.createElement(token.startsWith("**") ? "strong" : "code");
        node.textContent = token.startsWith("**") ? token.slice(2,-2) : token.slice(1,-1); parent.append(node);
        cursor = match.index + token.length;
      }
      parent.append(document.createTextNode(text.slice(cursor)));
    }

    function appendTextBlock(parent, text) {
      const lines = text.split("\n"); let paragraph = []; let list = null;
      const flush = () => { if (!paragraph.length) return; const p = document.createElement("p"); appendInline(p, paragraph.join("\n")); parent.append(p); paragraph = []; };
      for (const line of lines) {
        const heading = line.match(/^(#{1,3})\s+(.+)$/); const item = line.match(/^\s*[-*]\s+(.+)$/);
        if (heading) { flush(); list = null; const h = document.createElement(`h${heading[1].length}`); appendInline(h, heading[2]); parent.append(h); }
        else if (item) { flush(); if (!list) { list = document.createElement("ul"); parent.append(list); } const li = document.createElement("li"); appendInline(li,item[1]); list.append(li); }
        else if (!line.trim()) { flush(); list = null; }
        else { list = null; paragraph.push(line); }
      }
      flush();
    }

    function renderMarkdown(node, text) {
      node.replaceChildren(); const source = String(text || ""); const fence = /```([^\n]*)\n?([\s\S]*?)```/g; let cursor = 0;
      for (const match of source.matchAll(fence)) {
        appendTextBlock(node, source.slice(cursor, match.index));
        const block = document.createElement("div"); block.className = "code-block";
        const head = document.createElement("div"); head.className = "code-head"; const language = document.createElement("span"); language.textContent = match[1].trim() || "code";
        const copy = document.createElement("button"); copy.className = "copy-code"; copy.textContent = t("common.copy"); copy.onclick = guard(async () => { await navigator.clipboard.writeText(match[2]); copy.textContent = t("common.copied"); setTimeout(() => copy.textContent = t("common.copy"),1200); });
        head.append(language,copy); const pre = document.createElement("pre"); const code = document.createElement("code"); code.textContent = match[2]; pre.append(code); block.append(head,pre); node.append(block);
        cursor = match.index + match[0].length;
      }
      appendTextBlock(node, source.slice(cursor));
    }

    function renderAttachments(parent, attachments) {
      if (!attachments?.length) return; const grid = document.createElement("div"); grid.className = "attachments";
      for (const attachment of attachments) {
        if (attachment.kind === "document") {
          let url; try { url = new URL(localUrl(attachment.url),location.origin); if (url.origin !== location.origin || !url.pathname.startsWith(localUrl("/api/conversations/"))) continue; } catch (_) { continue; }
          const card = document.createElement("a"); card.className = "attachment-card document"; card.href = url.href; card.target = "_blank"; card.rel = "noopener noreferrer";
          const iconBox = document.createElement("span"); iconBox.className = "document-icon"; iconBox.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 3h10l4 4v14H5zM15 3v5h5M8 12h8M8 16h6"/></svg>`;
          const copy = document.createElement("span"); copy.className = "document-copy"; const name = document.createElement("span"); name.className = "attachment-name"; name.textContent = attachment.name || t("common.document"); const meta = document.createElement("small"); meta.textContent = `${(attachment.name?.split(".").pop() || "FILE").toUpperCase()} · ${attachment.size >= 1048576 ? `${(attachment.size/1048576).toFixed(1)} MB` : `${Math.max(1,Math.round((attachment.size || 0)/1024))} KB`}`; copy.append(name,meta); card.append(iconBox,copy); grid.append(card); continue;
        }
        const card = document.createElement("div"); card.className = "attachment-card"; const media = document.createElement(attachment.kind === "video" ? "video" : "img"); media.src = localUrl(attachment.url); media.alt = attachment.name || t("common.attachment"); if (attachment.kind === "video") media.controls = true; const name = document.createElement("div"); name.className = "attachment-name"; name.textContent = attachment.name || t("common.attachment"); card.append(media,name); grid.append(card);
      }
      parent.append(grid);
    }

    function deckGrid(slides) {
      const grid = document.createElement("div"); grid.className = "deck-grid";
      for (const slide of slides || []) {
        const card = document.createElement("div"); card.className = "deck-slide";
        const index = document.createElement("span"); index.textContent = `SLIDE ${String(slide.index || "").padStart(2,"0")}`;
        const title = document.createElement("strong"); title.textContent = slide.title || t("artifact.untitled_slide");
        card.append(index,title); grid.append(card);
      }
      return grid;
    }

    function renderArtifacts(parent, artifacts) {
      for (const artifact of artifacts || []) {
        if (artifact.kind === "chart") {
          let url; try { url = new URL(localUrl(artifact.url),location.origin); if (url.origin !== location.origin || !url.pathname.startsWith(localUrl("/api/conversations/"))) continue; } catch (_) { continue; }
          const chart = document.createElement("a"); chart.className = "analysis-chart"; chart.href = url.href; chart.target = "_blank"; chart.rel = "noopener noreferrer"; const image = document.createElement("img"); image.src = url.href; image.alt = artifact.title || artifact.name || t("artifact.chart"); image.loading = "lazy"; const label = document.createElement("span"); label.textContent = artifact.title || artifact.name || t("artifact.chart"); chart.append(image,label); parent.append(chart); continue;
        }
        if (artifact.kind === "analysis_report") {
          let url; try { url = new URL(localUrl(artifact.url),location.origin); if (url.origin !== location.origin || !url.pathname.startsWith(localUrl("/api/conversations/"))) continue; } catch (_) { continue; }
          const card = document.createElement("div"); card.className = "artifact-card"; const main = document.createElement("div"); main.className = "artifact-main"; const iconBox = document.createElement("div"); iconBox.className = "artifact-icon data"; iconBox.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 3h10l4 4v14H5zM15 3v5h5M8 12h8M8 16h8"/></svg>`; const copy = document.createElement("div"); copy.className = "artifact-copy"; const name = document.createElement("strong"); name.textContent = artifact.name || t("artifact.analysis_report"); const meta = document.createElement("small"); meta.textContent = t("artifact.analysis_meta",{datasets:artifact.datasets || 0,analyses:artifact.analyses || 0,size:Math.max(1,Math.round((artifact.size || 0)/1024))}); copy.append(name,meta); const download = document.createElement("a"); download.className = "artifact-download"; download.href = url.href; download.download = artifact.name || t("artifact.analysis_report"); download.textContent = t("artifact.download_excel"); main.append(iconBox,copy,download); card.append(main); parent.append(card); continue;
        }
        if (artifact.kind === "code_patch") {
          let url; try { url = new URL(localUrl(artifact.url),location.origin); if (url.origin !== location.origin || !url.pathname.startsWith(localUrl("/api/conversations/"))) continue; } catch (_) { continue; }
          const card = document.createElement("div"); card.className = "artifact-card"; const main = document.createElement("div"); main.className = "artifact-main"; const iconBox = document.createElement("div"); iconBox.className = "artifact-icon code"; iconBox.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="m8 9-4 3 4 3M16 9l4 3-4 3M14 5l-4 14"/></svg>`; const copy = document.createElement("div"); copy.className = "artifact-copy"; const name = document.createElement("strong"); name.textContent = artifact.name || t("artifact.code_patch"); const meta = document.createElement("small"); meta.textContent = t("artifact.code_meta",{files:artifact.files || 0,additions:artifact.additions || 0,deletions:artifact.deletions || 0,size:Math.max(1,Math.round((artifact.size || 0)/1024))}); copy.append(name,meta); const download = document.createElement("a"); download.className = "artifact-download"; download.href = url.href; download.download = artifact.name || t("artifact.code_patch"); download.textContent = t("artifact.download_patch"); main.append(iconBox,copy,download); card.append(main); parent.append(card); continue;
        }
        if (artifact.kind !== "presentation") continue;
        let url; try { url = new URL(localUrl(artifact.url),location.origin); if (url.origin !== location.origin || !url.pathname.startsWith(localUrl("/api/conversations/"))) continue; } catch (_) { continue; }
        const card = document.createElement("div"); card.className = "artifact-card";
        const main = document.createElement("div"); main.className = "artifact-main";
        const iconBox = document.createElement("div"); iconBox.className = "artifact-icon"; iconBox.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M5 3h10l4 4v14H5zM15 3v5h5M8 12h8M8 16h6"/></svg>`;
        const copy = document.createElement("div"); copy.className = "artifact-copy";
        const name = document.createElement("strong"); name.textContent = artifact.name || "presentation.pptx";
        const meta = document.createElement("small"); const size = artifact.size ? ` · ${(artifact.size/1024).toFixed(0)} KB` : ""; meta.textContent = t("artifact.presentation_meta",{slides:artifact.slides || 0,style:artifact.style || "PPTX",size});
        copy.append(name,meta);
        const download = document.createElement("a"); download.className = "artifact-download"; download.href = url.href; download.download = artifact.name || "presentation.pptx"; download.textContent = t("artifact.download_pptx");
        main.append(iconBox,copy,download); card.append(main);
        if (artifact.preview?.length) { const details = document.createElement("details"); details.className = "deck-outline"; const summary = document.createElement("summary"); summary.textContent = t("artifact.view_outline",{count:artifact.preview.length}); details.append(summary,deckGrid(artifact.preview)); card.append(details); }
        parent.append(card);
      }
    }

    function renderSources(parent, sources) {
      if (!sources?.length) return; const list = document.createElement("div"); list.className = "source-list";
      for (const source of sources) {
        let url; try { url = new URL(localUrl(source.url),location.origin); if (!/^https?:$/.test(url.protocol)) continue; if (url.origin === location.origin && !url.pathname.startsWith(localUrl("/api/conversations/"))) continue; } catch (_) { continue; }
        const localSource = ["document","data","code"].includes(source.kind); const prefix = t(source.kind === "data" ? "source.data" : source.kind === "code" ? "source.code" : "source.document"); const link = document.createElement("a"); link.className = `source${localSource ? " document" : ""}`; link.href = url.href; link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = localSource ? `[${prefix}${source.index}] ${source.title}${source.location ? ` · ${source.location}` : ""}` : `[${source.index}] ${source.title || url.hostname}`; link.title = source.snippet || link.textContent; list.append(link);
      }
      parent.append(list);
    }

    function toolLabel(name) { return toolLabelKeys[name] ? t(toolLabelKeys[name]) : (name || t("tool.unknown")); }

    function toolDetailText(value) {
      if (typeof value === "string") return value;
      try { return JSON.stringify(value ?? {},null,2); } catch (_) { return String(value ?? ""); }
    }

    function toolPreview(call) {
      const args = call?.arguments;
      let value = "";
      if (args && typeof args === "object" && !Array.isArray(args)) {
        for (const key of ["command","path","file","pattern","query","url","source"]) {
          if (args[key] !== undefined && args[key] !== null && String(args[key])) { value = String(args[key]); break; }
        }
      } else if (args !== undefined && args !== null) value = String(args);
      return value.replace(/\s+/g," ").trim().slice(0,140);
    }

    function renderToolTrace(holder, toolCalls, live = false) {
      holder.replaceChildren(); const calls = Array.isArray(toolCalls) ? toolCalls : []; if (!calls.length) return;
      const trace = document.createElement("details"); trace.className = "tool-trace"; trace.open = live;
      const running = calls.some(call => (call.status || "done") === "running"); const summary = document.createElement("summary"); summary.textContent = t(running ? "tool.trace_live" : "tool.trace_done",{count:calls.length}); trace.append(summary);
      const list = document.createElement("div"); list.className = "tool-list";
      calls.forEach((call,index) => {
        const status = ["running","done","error","cancelled"].includes(call.status) ? call.status : "done";
        const step = document.createElement("details"); step.className = `tool-step ${status}`; step.open = status === "running" || status === "error";
        const head = document.createElement("summary"); const dot = document.createElement("span"); dot.className = "tool-state-dot";
        const name = document.createElement("span"); name.className = "tool-name"; name.textContent = `${index + 1}. ${toolLabel(call.name)}`;
        const preview = document.createElement("span"); preview.className = "tool-preview"; preview.textContent = toolPreview(call); preview.title = preview.textContent;
        const stateLabel = document.createElement("span"); stateLabel.className = "tool-status"; stateLabel.textContent = t(`tool.status_${status}`); head.append(dot,name,preview,stateLabel); step.append(head);
        const detail = document.createElement("div"); detail.className = "tool-detail";
        const argumentsText = toolDetailText(call.arguments); if (argumentsText && argumentsText !== "{}") { const label = document.createElement("span"); label.className = "tool-detail-label"; label.textContent = t("tool.parameters"); const pre = document.createElement("pre"); pre.textContent = argumentsText; detail.append(label,pre); }
        if (Object.prototype.hasOwnProperty.call(call,"result")) { const label = document.createElement("span"); label.className = "tool-detail-label"; label.textContent = t("tool.result"); const pre = document.createElement("pre"); pre.textContent = call.result || t("tool.no_output"); detail.append(label,pre); }
        if (call.arguments_truncated || call.result_truncated) { const note = document.createElement("div"); note.className = "tool-detail-note"; note.textContent = t("tool.truncated"); detail.append(note); }
        if (detail.childNodes.length) step.append(detail); list.append(step);
      });
      trace.append(list); holder.append(trace);
    }

    function createMessage(message, live = false) {
      const wrapper = document.createElement("article"); wrapper.className = `message ${message.role || "assistant"}`;
      if (message.role !== "user") { const mark = document.createElement("div"); mark.className = "assistant-mark"; mark.innerHTML = icon; wrapper.append(mark); }
      const body = document.createElement("div"); body.className = "message-body"; renderAttachments(body,message.attachments);
      const toolTrace = document.createElement("div"); toolTrace.className = "tool-trace-holder"; renderToolTrace(toolTrace,message.tool_calls,live); body.append(toolTrace);
      let reasoningDetails = null, reasoningContent = null;
      if (message.reasoning || live) { reasoningDetails = document.createElement("details"); reasoningDetails.className = "reasoning"; reasoningDetails.hidden = !message.reasoning; const summary = document.createElement("summary"); summary.textContent = t(live ? "chat.reasoning_live" : "chat.reasoning_done"); reasoningContent = document.createElement("div"); reasoningContent.className = "reasoning-content"; reasoningContent.textContent = message.reasoning || ""; reasoningDetails.append(summary,reasoningContent); body.append(reasoningDetails); }
      const status = document.createElement("div"); status.className = "status-line"; status.hidden = !message.cancelled; status.textContent = message.cancelled ? t("status.generation_stopped") : ""; body.append(status);
      const text = document.createElement("div"); text.className = "message-text"; renderMarkdown(text,message.content || ""); body.append(text); renderArtifacts(body,message.artifacts); renderSources(body,message.sources); wrapper.append(body);
      return { wrapper,body,text,status,toolTrace,reasoningDetails,reasoningContent };
    }

    function updateTaskView(task) {
      const live = task.live; if (!live || state.activeId !== task.id || !live.wrapper.isConnected) return;
      live.status.hidden = !task.showStatus; live.status.textContent = task.status || "";
      live.text.replaceChildren(); if (task.content) renderMarkdown(live.text,task.content); else live.text.innerHTML = `<span class="typing"><i></i><i></i><i></i></span>`;
      if (live.toolCallsVersion !== task.toolCallsVersion) { renderToolTrace(live.toolTrace,task.toolCalls,true); live.toolCallsVersion = task.toolCallsVersion; }
      if (task.reasoning) { live.reasoningDetails.hidden = false; live.reasoningContent.textContent = task.reasoning; }
      else live.reasoningDetails.hidden = true;
      live.planHolder.replaceChildren();
      if (task.plan?.type === "data") renderDataPlan(live.planHolder,task.plan.title,task.plan.analyses);
      else if (task.plan?.type === "ppt") { renderMarkdown(live.text,t("ppt.outline_done",{title:task.plan.title})); live.planHolder.append(deckGrid(task.plan.slides)); }
      $("#messages").scrollTop = $("#messages").scrollHeight;
    }

    function mountTask(task) {
      const live = createMessage({role:"assistant",content:""},true); live.toolCallsVersion = -1; live.planHolder = document.createElement("div"); live.body.append(live.planHolder); task.live = live; $("#messages").append(live.wrapper); updateTaskView(task);
    }

    function beginTask(id,userMessage,kind) {
      const task = { id, kind, userMessage, content:"", reasoning:"", status:"", showStatus:false, toolCalls:[], toolCallsVersion:0, plan:null, stopping:false, completed:false, live:null };
      state.tasks.set(id,task); if (state.activeId === id && state.record?.id === id) state.record.messages.push(userMessage); renderSidebar(); renderMessages(); syncGenerationUI(); return task;
    }

    function setTaskStatus(task,message) { task.status = message || ""; task.showStatus = true; updateTaskView(task); }
    function setTaskProgress(task,event) { task.content = event.content || ""; task.reasoning = event.reasoning || ""; task.showStatus = false; updateTaskView(task); }
    function setTaskToolCall(task,event) {
      const incoming = event.tool_call || {}; const callId = String(incoming.id || ""); let index = callId ? task.toolCalls.findIndex(call => String(call.id || "") === callId) : -1;
      if (index < 0 && event.type !== "tool_start") { for (let cursor = task.toolCalls.length - 1; cursor >= 0; cursor -= 1) { const call = task.toolCalls[cursor]; if (call.name === incoming.name && call.status === "running") { index = cursor; break; } } }
      if (index < 0) task.toolCalls.push({...incoming}); else task.toolCalls[index] = {...task.toolCalls[index],...incoming};
      task.toolCallsVersion += 1; task.showStatus = false; updateTaskView(task);
    }

    function completeTask(task,message) {
      task.completed = true;
      if (state.activeId !== task.id || !task.live?.wrapper.isConnected) return;
      const final = createMessage(message); task.live.wrapper.replaceWith(final.wrapper); task.live = null; if (state.record?.id === task.id) state.record.messages.push(message);
    }

    async function settleTask(task) {
      try { await refreshConversations(); } catch (error) { toast(error.message); }
      state.tasks.delete(task.id); renderSidebar(); syncGenerationUI();
      if (state.activeId === task.id) {
        try { const record = await (await api(`/api/conversations/${task.id}`)).json(); if (state.activeId === task.id) { state.record = record; const savedAgent = record.settings.agent_mode; state.agent = agentLabelKeys[savedAgent] ? savedAgent : "chat"; renderMessages(); renderModes(); syncGenerationUI(); } }
        catch (error) { toast(error.message); }
      }
      if (!isConversationGenerating() && state.activeId === task.id) $("#prompt").focus();
    }

    function renderHero() {
      const hero = document.createElement("div"); hero.className = "hero";
      if (state.agent === "workspace") {
        hero.innerHTML = `<div class="hero-mark">${icon}</div><div class="hero-kicker">PI AGENT</div><h1>${t("workspace.hero_title")}<br><span>${t("workspace.hero_accent")}</span></h1><p>${t("workspace.hero_description")}</p>`;
        const path = document.createElement("div"); path.className = "workspace-hero-path"; path.textContent = state.record?.settings.agent_workspace || ""; path.title = path.textContent; hero.append(path); $("#messages").append(hero); return;
      }
      hero.innerHTML = embedded ? `<h1>${t("nav.new_chat")}</h1><p>${t("composer.placeholder")}</p>` : `<div class="hero-mark">${icon}</div><div class="hero-kicker">FASTLLM INTELLIGENCE</div><h1>${t("hero.title_before")}<br><span>${t("hero.title_accent")}</span></h1><p>${t("hero.description")}</p><div class="feature-row"><span>${t("hero.feature_knowledge")}</span><span>${t("hero.feature_thinking")}</span><span>${t("hero.feature_vision")}</span><span>${t("hero.feature_web")}</span></div>`;
      const suggestions = document.createElement("div"); suggestions.className = "suggestions";
      const values = [["files","hero.read_files","hero.read_files_sub","M5 3h10l4 4v14H5zM15 3v5h5"],["search","hero.search","hero.search_sub","M11 4a7 7 0 1 0 0 14 7 7 0 0 0 0-14Zm5 12 4 4"],["vision","hero.vision","hero.vision_sub","M4 5h16v14H4zM7 15l3-3 3 3 2-2 3 3"]];
      for (const [action,titleKey,subKey,path] of values) { const button = document.createElement("button"); button.className = "suggestion"; button.innerHTML = `<span class="suggestion-icon"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="${path}"/></svg></span><span><strong>${t(titleKey)}</strong><small>${t(subKey)}</small></span>`; button.onclick = guard(async () => { if (action === "files") { $("#prompt").value = t("prompt.read_files"); await chooseAgent("knowledge"); } else if (action === "search") { $("#prompt").value = t("prompt.search_news"); await chooseConversationMode("快速搜索"); } else { await chooseConversationMode("关闭"); $("#prompt").value = t("prompt.vision"); $("#fileInput").click(); } resizePrompt(); $("#prompt").focus(); }); suggestions.append(button); }
      hero.append(suggestions); $("#messages").append(hero);
    }

    function renderMessages() {
      const container = $("#messages"); container.replaceChildren();
      if (!state.record?.messages?.length) renderHero();
      else for (const message of state.record.messages) container.append(createMessage(message).wrapper);
      const task = state.tasks.get(state.activeId); if (task && !task.completed) mountTask(task);
      requestAnimationFrame(() => { container.scrollTop = container.scrollHeight; });
    }

    async function refreshConversations() { state.conversations = await (await api("/api/conversations")).json(); renderSidebar(); syncGenerationUI(); }

    async function loadConversation(id) {
      if (!id) return; const sequence = ++state.loadSequence; const record = await (await api(`/api/conversations/${id}`)).json(); if (sequence !== state.loadSequence) return; state.activeId = id; state.record = record;
      const savedAgent = state.record.settings.agent_mode; state.agent = agentLabelKeys[savedAgent] ? savedAgent : "chat";
      if (!embedded) history.replaceState(null,"",`?chat=${encodeURIComponent(id)}`); try { localStorage.setItem("fastllm.webui.activeChat",id); } catch (_) {} state.pending = []; renderPending(); renderSidebar(); renderMessages(); renderModes(); syncGenerationUI(); setSidebar(false);
    }

    async function newConversation() {
      const record = await (await api("/api/conversations",{method:"POST",body:"{}"})).json(); await refreshConversations(); await loadConversation(record.id); $("#prompt").focus();
    }

    function renderWorkspaceDirectory() {
      const list = $("#workspaceList"); list.replaceChildren(); const current = state.workspaceDirectory;
      $("#workspacePath").value = current?.path || state.config.agent_workspace_root || ""; $("#workspaceUp").disabled = state.workspaceLoading || !current?.parent; $("#createWorkspace").disabled = state.workspaceLoading || !current?.writable;
      if (state.workspaceLoading) { const loading = document.createElement("div"); loading.className = "workspace-empty"; loading.textContent = t("workspace.loading"); list.append(loading); return; }
      if (!current?.directories?.length) { const empty = document.createElement("div"); empty.className = "workspace-empty"; empty.textContent = t("workspace.empty"); list.append(empty); return; }
      for (const directory of current.directories) {
        const button = document.createElement("button"); button.className = "workspace-directory"; button.title = directory.path; button.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8"><path d="M3 6h7l2 2h9v11H3z"/></svg>';
        const name = document.createElement("span"); name.textContent = directory.name; button.append(name); if (!directory.writable) { const access = document.createElement("small"); access.textContent = t("workspace.read_only"); button.append(access); } button.onclick = guard(() => loadWorkspaceDirectory(directory.path)); list.append(button);
      }
    }

    async function loadWorkspaceDirectory(path = "") {
      state.workspaceLoading = true; renderWorkspaceDirectory();
      try { state.workspaceDirectory = await (await api(`/api/agent/directories?path=${encodeURIComponent(path)}`)).json(); }
      catch (error) { toast(error.message); }
      finally { state.workspaceLoading = false; renderWorkspaceDirectory(); }
    }

    async function openWorkspaceDialog() {
      if (!state.config.workspace_agent_enabled || !state.config.pi_agent?.available) { toast(t("workspace.unavailable")); return; }
      state.workspaceDirectory = null; $("#workspaceDialog").showModal(); await loadWorkspaceDirectory(state.config.agent_workspace_root || ""); $("#workspacePath").focus();
    }

    async function createWorkspaceAgent() {
      const workspace = $("#workspacePath").value.trim(); if (!workspace) { toast(t("workspace.required")); return; }
      const button = $("#createWorkspace"); button.disabled = true;
      try {
        const record = await (await api("/api/conversations",{method:"POST",body:JSON.stringify({settings:{agent_mode:"workspace",agent_workspace:workspace,web_mode:"关闭"}})})).json(); $("#workspaceDialog").close(); await refreshConversations(); await loadConversation(record.id); $("#prompt").focus();
      } catch (error) { toast(error.message); }
      finally { button.disabled = Boolean(state.workspaceDirectory && !state.workspaceDirectory.writable); }
    }

    function renderMenu(menu, options, current, kind) {
      menu.replaceChildren(); for (const [value,labelKey] of options) { const button = document.createElement("button"); button.className = `mode-option${value === current ? " selected" : ""}`; button.textContent = t(labelKey); button.onclick = guard(async () => { if (isConversationGenerating()) return; state.record.settings[kind] = value; menu.classList.remove("open"); renderModes(); try { await saveRecordSettings(); } catch (error) { toast(error.message); } }); menu.append(button); }
    }

    function renderModes() {
      if (!state.record) return; const thinking = state.record.settings.thinking_level || "中"; const webMode = state.record.settings.web_mode || "关闭";
      $("#thinkingLabel").textContent = t(Object.fromEntries(thinkingOptions)[thinking] || "thinking.medium"); $("#thinkingButton").classList.toggle("active",thinking !== "关闭"); renderMenu($("#thinkingMenu"),thinkingOptions,thinking,"thinking_level");
      $("#webLabel").textContent = t(Object.fromEntries(conversationOptions)[webMode] || "mode.chat"); $("#webButton").classList.toggle("active",webMode !== "关闭"); renderConversationMenu(webMode); renderAgentSelection();
    }

    function renderAgentSelection() {
      const selectedAgentKey = agentLabelKeys[state.agent]; const selectedAgent = selectedAgentKey ? t(selectedAgentKey) : ""; $("#agentButton").classList.toggle("active",Boolean(selectedAgent)); $("#agentButton").title = selectedAgent ? t("composer.agent_current",{name:selectedAgent}) : t("composer.agent_select"); root.querySelectorAll(".agent-card").forEach(card => card.classList.toggle("selected",card.dataset.agent === state.agent));
      const workspace = state.record?.settings.agent_workspace || ""; const workspaceName = workspace.split("/").filter(Boolean).pop() || workspace; const workspaceMode = state.agent === "workspace"; $("#workspaceContext").hidden = !workspaceMode; $("#workspaceContextPath").textContent = workspace; $("#workspaceContextPath").title = workspace;
      const webMode = state.record?.settings.web_mode || "关闭"; const placeholders = {knowledge:"composer.placeholder_knowledge",data:"composer.placeholder_data",ppt:"composer.placeholder_ppt",code:"composer.placeholder_code"}; $("#prompt").placeholder = workspaceMode ? t("composer.placeholder_workspace",{name:workspaceName}) : t(placeholders[state.agent] || (webMode === "快速搜索" ? "composer.placeholder_web_fast" : webMode === "深度浏览" ? "composer.placeholder_web_deep" : "composer.placeholder"));
    }

    function renderConversationMenu(current) {
      const menu = $("#webMenu"); menu.replaceChildren(); for (const [value,labelKey] of conversationOptions) { const button = document.createElement("button"); button.className = `mode-option${value === current ? " selected" : ""}`; button.textContent = t(labelKey); button.onclick = guard(() => chooseConversationMode(value)); menu.append(button); }
    }

    async function chooseConversationMode(webMode) {
      if (!Object.fromEntries(conversationOptions)[webMode] || !state.record || isConversationGenerating()) return; if (state.agent !== "workspace") { state.agent = "chat"; state.record.settings.agent_mode = "chat"; } state.record.settings.web_mode = webMode; $("#webMenu").classList.remove("open"); renderModes();
      try { await saveRecordSettings(); } catch (error) { toast(error.message); }
      $("#prompt").focus();
    }

    function hasKnowledgeFiles() {
      const contains = item => item?.kind === "document"; return state.pending.some(contains) || (state.record?.messages || []).some(message => (message.attachments || []).some(contains));
    }

    function hasConfiguredFiles(extensionKey, filenameKey = "") {
      const extensions = state.config[extensionKey] || []; const filenames = new Set(filenameKey ? state.config[filenameKey] || [] : []);
      const matches = item => { const name = String(item?.name || "").toLowerCase(); return filenames.has(name) || extensions.some(extension => name.endsWith(extension)); };
      return state.pending.some(matches) || (state.record?.messages || []).some(message => (message.attachments || []).some(matches));
    }

    function hasCodeFiles() { return hasConfiguredFiles("code_extensions","code_filenames"); }
    function hasDataFiles() { return hasConfiguredFiles("data_extensions"); }

    async function chooseAgent(agent) {
      if (!agentLabelKeys[agent] || !state.record || isConversationGenerating()) return; state.agent = agent; state.record.settings.web_mode = "关闭"; state.record.settings.agent_mode = agent; renderModes(); $("#agentDialog").close();
      try { await saveRecordSettings(); } catch (error) { toast(error.message); }
      if (agent === "knowledge" && !hasKnowledgeFiles()) { $("#prompt").value = $("#prompt").value.trim() || t("prompt.read_files"); resizePrompt(); $("#fileInput").click(); }
      else if (agent === "data") { if (hasDataFiles()) openDataDialog(); else { toast(t("data.file_choose")); $("#fileInput").click(); } }
      else if (agent === "ppt") openPptDialog();
      else if (agent === "code" && !hasCodeFiles()) { $("#prompt").value = $("#prompt").value.trim() || t("prompt.review_code"); resizePrompt(); $("#fileInput").click(); }
      else $("#prompt").focus();
    }

    async function saveRecordSettings(extra = {}) { if (!state.record || isConversationGenerating()) return; const id = state.activeId; const payload = { title: extra.title ?? state.record.title, settings: { ...state.record.settings, ...(extra.settings || {}) } }; const record = await (await api(`/api/conversations/${id}`,{method:"PATCH",body:JSON.stringify(payload)})).json(); if (state.activeId === id) state.record = record; await refreshConversations(); }

    function renderPending() {
      const list = $("#pendingFiles"); list.replaceChildren(); list.classList.toggle("visible",state.pending.length > 0 || state.uploading > 0);
      for (const [index,file] of state.pending.entries()) { const item = document.createElement("div"); item.className = "pending-file"; const label = document.createElement("span"); label.textContent = file.name; const remove = document.createElement("button"); remove.textContent = "×"; remove.onclick = guard(() => { state.pending.splice(index,1); renderPending(); }); item.append(label,remove); list.append(item); }
      if (state.uploading) { const item = document.createElement("div"); item.className = "pending-file"; item.textContent = t("upload.uploading",{count:state.uploading}); list.append(item); }
    }

    async function uploadFiles(files) {
      if (!state.activeId || isConversationGenerating()) return;
      const conversationId = state.activeId;
      state.uploading += files.length; renderPending();
      for (const file of files) {
        try {
          const extension = (file.name.split(".").pop() || "").toLowerCase();
          const workspaceVideo = file.type.startsWith("video/") || workspaceVideoExtensions.has(extension);
          if (state.agent === "workspace" && workspaceVideo) throw new Error(t("workspace.video_unsupported"));
          if (file.size > state.config.max_upload_mb * 1024 * 1024) throw new Error(t("upload.too_large",{name:file.name}));
          const response = await fetch(localUrl(`/api/conversations/${conversationId}/attachments`),{method:"POST",headers:{"Content-Type":file.type || "application/octet-stream","X-Filename":encodeURIComponent(file.name)},body:file,signal:lifecycle.signal});
          if (!response.ok) { let detail = t("upload.failed"); try { detail = (await response.json()).detail || detail; } catch (_) {} throw new Error(detail); }
          const attachment = await response.json(); if (state.activeId === conversationId) state.pending.push(attachment);
        } catch (error) { toast(error.message); }
        finally { state.uploading -= 1; renderPending(); }
      }
      $("#fileInput").value = "";
    }

    function syncGenerationUI() {
      const running = isConversationGenerating(); const task = state.tasks.get(state.activeId); const stopping = Boolean(task?.stopping || state.stopping.has(state.activeId));
      $("#prompt").disabled = running; $("#sendButton").hidden = running; $("#sendButton").disabled = running; $("#stopButton").hidden = !running; $("#stopButton").disabled = stopping; $("#stopButton").title = t(stopping ? "composer.stopping" : "composer.stop"); $("#stopButton").setAttribute("aria-label",t(stopping ? "composer.stopping" : "composer.stop"));
      $("#attachButton").disabled = running; for (const selector of ["#agentButton","#webButton","#thinkingButton","#topSettings"]) $(selector).disabled = running;
    }

    async function stopActiveGeneration() {
      const id = state.activeId; if (!id || !isConversationGenerating(id)) return; const task = state.tasks.get(id); if (task) task.stopping = true; else state.stopping.add(id); syncGenerationUI();
      try {
        const result = await (await api(`/api/conversations/${id}/cancel`,{method:"POST",body:"{}"})).json();
        if (!task) { for (let attempt = 0; attempt < 80; attempt += 1) { await new Promise(resolve => setTimeout(resolve,250)); await refreshConversations(); if (!isConversationGenerating(id)) break; } if (state.activeId === id) await loadConversation(id); }
        else if (!result.cancelled) { await refreshConversations(); if (state.activeId === id) await loadConversation(id); }
      } catch (error) { if (task) task.stopping = false; toast(error.message); }
      finally { state.stopping.delete(id); syncGenerationUI(); }
    }

    async function readEventStream(response,onEvent) {
      const reader = response.body.getReader(), decoder = new TextDecoder(); let buffer = "";
      while (true) {
        const {value,done} = await reader.read(); buffer += decoder.decode(value || new Uint8Array(),{stream:!done}); const lines = buffer.split("\n"); buffer = lines.pop() || "";
        for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
        if (done) { if (buffer.trim()) onEvent(JSON.parse(buffer)); break; }
      }
    }

    async function sendMessage() {
      if (isConversationGenerating() || state.uploading || !state.record) return; if (state.agent === "data") { openDataDialog(); return; } if (state.agent === "ppt") { openPptDialog(); return; } if (state.agent === "knowledge" && !hasKnowledgeFiles()) { toast(t("chat.knowledge_required")); $("#fileInput").click(); return; } if (state.agent === "code" && !hasCodeFiles()) { toast(t("chat.code_required")); $("#fileInput").click(); return; }
      const prompt = $("#prompt").value.trim(); if (!prompt && !state.pending.length) return;
      const conversationId = state.activeId;
      const publicAttachments = state.pending.map(item => ({...item})); const userMessage = {role:"user",content:prompt || t("prompt.attachments"),attachments:publicAttachments};
      const attachments = state.pending.map(({token,name}) => ({token,name})); state.pending = []; renderPending(); $("#prompt").value = ""; resizePrompt(); const task = beginTask(conversationId,userMessage,"chat");
      try {
        const response = await api(`/api/conversations/${conversationId}/chat`,{method:"POST",body:JSON.stringify({prompt,attachments})});
        await readEventStream(response,event => {
          if (event.type === "status") setTaskStatus(task,eventMessage(event));
          else if (["tool_start","tool_update","tool_end"].includes(event.type)) setTaskToolCall(task,event);
          else if (event.type === "progress") setTaskProgress(task,event);
          else if (event.type === "knowledge") { setTaskStatus(task,t("chat.knowledge_count",{count:event.sources.length})); for (const warning of event.warnings || []) toast(warning); }
          else if (event.type === "code") { setTaskStatus(task,t("chat.code_count",{files:event.files || 0,count:event.sources.length})); for (const warning of event.warnings || []) toast(warning); }
          else if (event.type === "web") { setTaskStatus(task,t("chat.web_count",{count:event.sources.length})); for (const warning of event.warnings || []) toast(warning); }
          else if (event.type === "warning") toast(eventMessage(event));
          else if (event.type === "error") toast(t("chat.failed",{message:event.message}));
          else if (event.type === "cancelled") setTaskStatus(task,eventMessage(event));
          else if (event.type === "done") completeTask(task,event.message);
        });
      } catch (error) { if (!task.stopping) toast(error.message); }
      finally { await settleTask(task); }
    }

    function openPptDialog() {
      if (!state.record || isConversationGenerating()) return;
      $("#pptTopic").value = $("#prompt").value.trim(); $("#pptAudience").value = ""; $("#pptSlides").value = 8; $("#pptStyle").value = "tech"; $("#pptWeb").checked = state.record.settings.web_mode !== "关闭"; $("#pptDialog").showModal(); $("#pptTopic").focus();
    }

    function openDataDialog() {
      if (!state.record || isConversationGenerating()) return;
      if (!hasDataFiles()) { toast(t("data.file_required")); $("#fileInput").click(); return; }
      $("#dataQuestion").value = $("#prompt").value.trim() || t("data.default_question"); $("#dataDialog").showModal(); $("#dataQuestion").focus();
    }

    function renderDataPlan(holder,title,analyses) {
      holder.replaceChildren(); const card = document.createElement("div"); card.className = "data-plan"; const heading = document.createElement("strong"); heading.textContent = title || t("data.plan"); card.append(heading); const labels = {groupby:"data.groupby",trend:"data.trend",correlation:"data.correlation",top:"data.top"}; const text = (analyses || []).map((item,index) => `${index + 1}. ${labels[item.operation] ? t(labels[item.operation]) : item.operation} · ${item.dataset}`).join("\n"); card.append(document.createTextNode(text || t("data.overview"))); holder.append(card);
    }

    async function createDataAnalysis() {
      if (!state.record || isConversationGenerating()) return; const conversationId = state.activeId;
      const question = $("#dataQuestion").value.trim() || t("data.default_question");
      const publicAttachments = state.pending.map(item => ({...item})); const userMessage = {role:"user",content:t("data.user_message",{question}),attachments:publicAttachments}; const attachments = state.pending.map(({token,name}) => ({token,name})); state.pending = []; renderPending(); $("#dataDialog").close(); $("#prompt").value = ""; resizePrompt();
      const task = beginTask(conversationId,userMessage,"data");
      try {
        const response = await api(`/api/conversations/${conversationId}/analyses`,{method:"POST",body:JSON.stringify({question,attachments})});
        await readEventStream(response,event => {
          if (event.type === "status") setTaskStatus(task,eventMessage(event));
          else if (event.type === "data") setTaskStatus(task,t("status.data_loaded",{datasets:event.datasets.length,rows:event.datasets.reduce((sum,item) => sum + item.rows,0)}));
          else if (event.type === "data_plan") { task.plan = {type:"data",title:event.title,analyses:event.analyses}; updateTaskView(task); }
          else if (event.type === "progress") setTaskProgress(task,event);
          else if (event.type === "warning") toast(eventMessage(event));
          else if (event.type === "error") toast(t("data.failed",{message:event.message}));
          else if (event.type === "cancelled") setTaskStatus(task,eventMessage(event));
          else if (event.type === "done") completeTask(task,event.message);
        });
      } catch (error) { if (!task.stopping) toast(error.message); }
      finally { await settleTask(task); }
    }

    async function createPresentation() {
      if (!state.record || isConversationGenerating()) return; const conversationId = state.activeId;
      const topic = $("#pptTopic").value.trim(); if (!topic) { toast(t("ppt.topic_required")); return; }
      const audience = $("#pptAudience").value.trim(); const slideCount = Math.min(20,Math.max(4,Number($("#pptSlides").value) || 8)); const style = $("#pptStyle").value;
      const styleNames = {tech:"ppt.style_tech",business:"ppt.style_business",nature:"ppt.style_nature",premium:"ppt.style_premium"}; const currentWeb = state.record.settings.web_mode || "关闭"; const webMode = $("#pptWeb").checked ? (currentWeb === "关闭" ? "深度浏览" : currentWeb) : "关闭";
      const userMessage = {role:"user",content:t("ppt.user_message",{topic,audience:audience || t("ppt.default_audience"),slides:slideCount,style:t(styleNames[style])})};
      $("#pptDialog").close(); const task = beginTask(conversationId,userMessage,"ppt");
      try {
        const response = await api(`/api/conversations/${conversationId}/presentations`,{method:"POST",body:JSON.stringify({topic,audience,slide_count:slideCount,style,web_mode:webMode})});
        await readEventStream(response,event => {
          if (event.type === "status") setTaskStatus(task,eventMessage(event));
          else if (event.type === "knowledge") { setTaskStatus(task,t("status.knowledge",{count:event.sources.length})); for (const warning of event.warnings || []) toast(warning); }
          else if (event.type === "web") { setTaskStatus(task,t("status.ppt_web",{count:event.sources.length})); for (const warning of event.warnings || []) toast(warning); }
          else if (event.type === "warning") toast(eventMessage(event));
          else if (event.type === "plan") { task.plan = {type:"ppt",title:event.title,slides:event.slides}; updateTaskView(task); }
          else if (event.type === "error") toast(t("ppt.failed",{message:event.message}));
          else if (event.type === "cancelled") setTaskStatus(task,eventMessage(event));
          else if (event.type === "done") completeTask(task,event.message);
        });
      } catch (error) { if (!task.stopping) toast(error.message); }
      finally { await settleTask(task); }
    }

    function resizePrompt() { const input = $("#prompt"); input.style.height = "auto"; input.style.height = `${Math.min(input.scrollHeight,180)}px`; }

    function openSettings() {
      if (!state.record || isConversationGenerating()) return; const s = state.record.settings || {};
      const maximum = Number(s.max_new_tokens ?? state.config.max_token);
      $("#settingSystem").value = s.system_prompt || ""; $("#settingTokens").value = Number.isFinite(maximum) && maximum > 0 ? maximum : 0; $("#settingTemperature").value = s.temperature ?? 1; $("#settingTopP").value = s.top_p ?? .8; $("#settingTopK").value = s.top_k ?? 1; $("#settingPenalty").value = s.repeat_penalty ?? 1; $("#settingsDialog").showModal();
    }

    async function saveSettings() {
      const number = (selector,fallback) => { const value = Number($(selector).value); return Number.isFinite(value) ? value : fallback; };
      try { await saveRecordSettings({settings:{system_prompt:$("#settingSystem").value,max_new_tokens:Math.max(0,number("#settingTokens",0)),temperature:number("#settingTemperature",1),top_p:number("#settingTopP",.8),top_k:number("#settingTopK",1),repeat_penalty:number("#settingPenalty",1)}}); $("#settingsDialog").close(); toast(t("settings.saved")); }
      catch (error) { toast(error.message); }
    }

    async function boot() {
      state.config = await (await api("/api/config")).json();
      if (!embedded) document.title = state.config.title || "FastLLM";
      $("#modelName").textContent = state.config.model;
      const workspaceAvailable = state.config.workspace_agent_enabled && state.config.pi_agent?.available;
      $("#newAgent").disabled = !workspaceAvailable;
      $("#newAgent").title = workspaceAvailable ? "" : t("workspace.unavailable");
      $("#fileInput").accept = ["image/*","video/*","text/*",...(state.config.upload_extensions || [])].join(",");
      await refreshConversations();
      if (!state.conversations.length) await newConversation();
      let requested = !embedded && new URLSearchParams(location.search).get("chat");
      try { requested ||= localStorage.getItem("fastllm.webui.activeChat"); } catch (_) {}
      const target = state.conversations.find(item => item.id === requested)?.id || state.conversations[0].id;
      await loadConversation(target);
    }

    $("#newChat").onclick = guard(newConversation); $("#newAgent").onclick = guard(openWorkspaceDialog); $("#attachButton").onclick = guard(() => $("#fileInput").click()); $("#fileInput").onchange = guard(event => uploadFiles([...event.target.files])); $("#agentButton").onclick = guard(() => { renderAgentSelection(); $("#agentDialog").showModal(); });
    $("#sendButton").onclick = guard(sendMessage); $("#stopButton").onclick = guard(stopActiveGeneration); $("#prompt").oninput = guard(resizePrompt); $("#prompt").onkeydown = guard(event => { if (event.key === "Enter" && !event.shiftKey && !event.isComposing) { event.preventDefault(); sendMessage(); } });
    for (const [button,menu] of [["#webButton","#webMenu"],["#thinkingButton","#thinkingMenu"]]) $(button).onclick = guard(event => { event.stopPropagation(); const target = $(menu); root.querySelectorAll(".mode-menu.open").forEach(item => item !== target && item.classList.remove("open")); target.classList.toggle("open"); });
    listen("click",() => { root.querySelectorAll(".mode-menu.open").forEach(item => item.classList.remove("open")); $("#languageMenu").classList.remove("open"); closeConversationMenu(); });
    root.querySelectorAll(".agent-card").forEach(card => card.onclick = guard(() => chooseAgent(card.dataset.agent))); $("#closeAgent").onclick = guard(() => $("#agentDialog").close());
    $("#topSettings").onclick = guard(openSettings); $("#closeSettings").onclick = guard($("#cancelSettings").onclick = guard(() => $("#settingsDialog").close())); $("#saveSettings").onclick = guard(saveSettings);
    $("#languageButton").onclick = guard(event => { event.stopPropagation(); $("#languageMenu").classList.toggle("open"); closeConversationMenu(); });
    $("#renameConversationAction").onclick = guard(openRenameDialog); $("#deleteConversationAction").onclick = guard(openDeleteDialog);
    $("#workspaceUp").onclick = guard(() => state.workspaceDirectory?.parent && loadWorkspaceDirectory(state.workspaceDirectory.parent)); $("#workspacePath").onkeydown = guard(event => { if (event.key === "Enter" && !event.isComposing) { event.preventDefault(); loadWorkspaceDirectory(event.currentTarget.value.trim()); } }); $("#closeWorkspace").onclick = guard($("#cancelWorkspace").onclick = guard(() => $("#workspaceDialog").close())); $("#createWorkspace").onclick = guard(createWorkspaceAgent);
    $("#closeRename").onclick = guard($("#cancelRename").onclick = guard(() => { state.renameConversationId = null; $("#renameDialog").close(); })); $("#saveRename").onclick = guard(renameConversation); $("#renameTitle").onkeydown = guard(event => { if (event.key === "Enter" && !event.isComposing) { event.preventDefault(); renameConversation(); } });
    $("#closeDelete").onclick = guard($("#cancelDelete").onclick = guard(() => { state.deleteConversationId = null; $("#deleteDialog").close(); })); $("#confirmDelete").onclick = guard(deleteConversation);
    $("#closePpt").onclick = guard($("#cancelPpt").onclick = guard(() => $("#pptDialog").close())); $("#createPpt").onclick = guard(createPresentation);
    $("#closeData").onclick = guard($("#cancelData").onclick = guard(() => $("#dataDialog").close())); $("#createDataAnalysis").onclick = guard(createDataAnalysis);
    $("#mobileMenu").onclick = guard(() => setSidebar(true)); $("#sidebarBackdrop").onclick = guard(() => setSidebar(false));
    listen("dragover",event => { if ([...event.dataTransfer.types].includes("Files")) event.preventDefault(); }); listen("drop",event => { if (![...event.dataTransfer.types].includes("Files")) return; event.preventDefault(); uploadFiles([...event.dataTransfer.files]); });
    $("#conversationList").onscroll = guard(closeConversationMenu); window.addEventListener("resize",closeConversationMenu,{signal:lifecycle.signal}); applyLocale(); await boot();

    if (lifecycle.signal.aborted) throw new DOMException("WebUI closed", "AbortError");
    observer = new ResizeObserver(() => { closeConversationMenu(); resizePrompt(); });
    observer.observe(host);
    signal?.removeEventListener("abort", abort);
    return {destroy, setLocale: guard(setLocale)};
  } catch (error) {
    destroy();
    throw error;
  }
}
