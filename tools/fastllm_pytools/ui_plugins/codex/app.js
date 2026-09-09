import {mountNativeAgent} from "../../plugin-core/native-agent.js";

export function mountCodex(options) {
  const {request, t, chooseDirectory} = options;
  const $ = suffix => document.getElementById(`codex-${suffix}`);
  const rpc = (method, params = {}) => request("/api/agents/codex/rpc", {method:"POST", body:JSON.stringify({method, params})});
  const report = error => { $("chat-error").textContent = error?.message || ""; $("chat-error").classList.toggle("hidden", !error); };
  let running = false, active = false, epoch = "", cursor = 0, snapshotCursor = 0;
  let selected = "", title = "", sessions = [], nextCursor, polling = false, loading = false, sending = false;
  let draftWorkspace = "", selectedWorkspace = "", selectedReady = false, defaultWorkspace = "";
  let modelName = "", effortOptions = [], defaultEffort = "";
  let generation = 0, selection = 0, searchVersion = 0, turns = {}, approvals = [], approvalSignature = "";
  const items = new Map(), nodes = new Map(), drafts = new Map(), efforts = new Map();
  const collapsedProjects = new Set();
  const markdownTypes = new Set(["userMessage", "agentMessage", "plan", "reasoning"]);
  let markdown, markdownLoading = false, markdownAttempt = 0, markdownError = "", disposed = false;
  const storageKey = "ftllm.codex.sessions";
  try {
    const saved = JSON.parse(localStorage.getItem(storageKey) || "{}");
    selected = typeof saved.selected === "string" ? saved.selected : "";
    if (typeof saved.workspace === "string") draftWorkspace = saved.workspace;
    for (const [key, value] of Object.entries(saved.drafts || {})) if (typeof value === "string") drafts.set(key, value);
    for (const [key, value] of Object.entries(saved.efforts || {})) if (typeof value === "string") efforts.set(key, value);
    for (const path of Array.isArray(saved.collapsedProjects) ? saved.collapsedProjects : [])
      if (typeof path === "string") collapsedProjects.add(path);
  } catch (_) { /* Storage may be unavailable; keep the current input in memory. */ }
  const draftKey = () => selected || `workspace:${draftWorkspace}`;
  const effortKey = (key = draftKey()) => JSON.stringify([modelName, key]);
  const currentEffort = () => effortOptions.includes(efforts.get(effortKey())) ? efforts.get(effortKey()) : defaultEffort;
  // Migrate the former shared new-session draft once; keep later drafts separate by directory.
  if (drafts.has("") && !drafts.has(`workspace:${draftWorkspace}`)) drafts.set(`workspace:${draftWorkspace}`, drafts.get(""));
  drafts.delete("");
  $("workspace").value = draftWorkspace;
  $("prompt").value = drafts.get(draftKey()) || "";
  function save() {
    drafts.set(draftKey(), $("prompt").value);
    try { localStorage.setItem(storageKey, JSON.stringify({selected, workspace:draftWorkspace,
      drafts:Object.fromEntries(drafts), efforts:Object.fromEntries(efforts), collapsedProjects:[...collapsedProjects]})); }
    catch (_) { report(new Error(t("Could not save the input in this browser. Keep this page open."))); }
  }
  $("prompt").addEventListener("input", save);
  $("effort").addEventListener("change", () => { efforts.set(effortKey(), $("effort").value); save(); });
  $("workspace").addEventListener("input", () => {
    if (selected) return;
    save(); draftWorkspace = $("workspace").value;
    // Restore an existing directory's draft, including an explicitly empty
    // one. Keep the current input when choosing a directory without a draft.
    $("prompt").value = drafts.get(draftKey()) ?? $("prompt").value;
    save(); controls();
  });
  $("browse-workspace").addEventListener("click", () => chooseDirectory?.($("workspace"), $("browse-workspace")));
  async function loadMarkdown() {
    if (markdown || markdownLoading || disposed) return;
    markdownLoading = true;
    try {
      const url = new URL("../../assets/webui/markdown.js", import.meta.url);
      url.searchParams.set("codex", String(++markdownAttempt));
      const module = await import(url.href);
      if (disposed) return;
      markdown = module.renderMarkdown;
      if (markdownError && $("chat-error").textContent === markdownError) report(null);
      markdownError = "";
      for (const item of items.values()) renderItem(item);
    } catch (_) {
      if (!disposed) {
        markdownError = t("Could not load Markdown. Reopen the Codex page to retry.");
        report(new Error(markdownError));
      }
    } finally { markdownLoading = false; }
  }
  async function copyCode(code, button) {
    try { await navigator.clipboard.writeText(code); button.textContent = t("Copied"); }
    catch (_) { report(new Error(t("Could not copy to the clipboard."))); }
  }
  function controls() {
    const busy = Boolean(turns[selected]);
    $("effort").value = currentEffort();
    $("effort").disabled = !running || loading || sending || busy || !effortOptions.length;
    $("effort").title = t(effortOptions.length ? "Reasoning effort for this conversation"
      : "This model does not advertise adjustable reasoning effort.");
    $("send").disabled = !running || loading || sending || busy || (selected ? !selectedReady : !draftWorkspace.trim());
    $("cancel").classList.toggle("hidden", !busy); $("cancel").disabled = !running;
    $("new").disabled = !running || loading || sending;
    $("workspace").disabled = Boolean(selected) || !running || loading || sending;
    $("browse-workspace").disabled = Boolean(selected) || !running || loading || sending || !chooseDirectory;
    $("workspace-setup").classList.toggle("hidden", Boolean(selected));
    $("rename").disabled = !running || !selected || loading;
    $("archive").disabled = !running || !selected || busy || loading;
    $("title").textContent = title || t("New conversation");
    $("session-workspace").textContent = selected
      ? selectedWorkspace || t("Working directory unavailable") : draftWorkspace;
    $("session-workspace").title = $("session-workspace").textContent;
    $("turn-status").textContent = loading ? t("Loading conversation…") : busy ? t("Codex is working…")
      : t("Changes and commands run in this conversation's workspace.");
    for (const button of $("session-list").querySelectorAll(".codex-project-new"))
      button.disabled = !running || loading || sending;
  }
  function setSidebar(open) {
    $("content").classList.toggle("sessions-open", open);
    $("toggle-sessions").setAttribute("aria-expanded", String(open));
  }
  function clearMessages() { items.clear(); nodes.clear(); $("messages").replaceChildren(); }
  function beginDraft(workspace = selectedWorkspace || draftWorkspace || defaultWorkspace) {
    if (sending) return;
    save(); selection++; loading = false; selected = ""; selectedReady = false; selectedWorkspace = ""; title = "";
    draftWorkspace = workspace; $("workspace").value = workspace;
    $("prompt").value = drafts.get(draftKey()) || "";
    clearMessages(); report(null); approvalSignature = ""; renderApprovals();
    save(); renderSessions(); setSidebar(false); $("prompt").focus();
  }
  function rememberThread(thread) {
    const {turns:history, ...summary} = thread;
    const index = sessions.findIndex(session => session.id === thread.id);
    if (index < 0) sessions.unshift(summary);
    else sessions[index] = {...sessions[index], ...summary};
    return sessions.find(session => session.id === thread.id);
  }
  function renderSessions() {
    $("session-list").replaceChildren();
    const groups = new Map();
    for (const session of sessions) {
      const workspace = typeof session.cwd === "string" ? session.cwd : "";
      if (!groups.has(workspace)) groups.set(workspace, []);
      groups.get(workspace).push(session);
    }
    for (const [workspace, threads] of groups) {
      const group = document.createElement("section"); group.className = "codex-project";
      group.dataset.workspace = workspace;
      const heading = document.createElement("header"); heading.className = "codex-project-heading";
      const toggle = document.createElement("button"); toggle.type = "button"; toggle.className = "codex-project-toggle";
      const name = document.createElement("strong");
      name.textContent = workspace.replace(/[\\/]+$/, "").split(/[\\/]/).pop() || workspace || t("Working directory unavailable");
      const path = document.createElement("small"); path.textContent = workspace; toggle.title = workspace;
      toggle.append(name, path);
      const list = document.createElement("div"); list.className = "codex-project-sessions";
      list.hidden = collapsedProjects.has(workspace);
      toggle.setAttribute("aria-expanded", String(!list.hidden));
      toggle.addEventListener("click", () => {
        list.hidden = !list.hidden; toggle.setAttribute("aria-expanded", String(!list.hidden));
        if (list.hidden) collapsedProjects.add(workspace); else collapsedProjects.delete(workspace);
        save();
      });
      heading.append(toggle);
      if (workspace) {
        const add = document.createElement("button"); add.type = "button"; add.className = "codex-project-new small-button";
        add.textContent = "+"; add.title = t("New conversation in {workspace}", {workspace});
        add.setAttribute("aria-label", add.title);
        add.addEventListener("click", () => beginDraft(workspace)); heading.append(add);
      }
      group.append(heading, list); $("session-list").append(group);
      for (const session of threads) {
        const button = document.createElement("button"); button.type = "button";
        button.className = "codex-session"; button.classList.toggle("active", session.id === selected);
        button.dataset.threadId = session.id;
        const label = document.createElement("span"); label.textContent = session.name || session.preview || t("New conversation");
        button.append(label); button.title = label.textContent;
        const date = document.createElement("small"); date.textContent = new Date(session.updatedAt * 1000).toLocaleString();
        button.append(date); button.addEventListener("click", () => select(session.id).catch(report));
        button.setAttribute("aria-current", session.id === selected ? "page" : "false");
        list.append(button);
      }
    }
    $("more").classList.toggle("hidden", !nextCursor);
    const current = sessions.find(thread => thread.id === selected);
    if (current) { title = current.name || current.preview || ""; }
    controls();
  }
  async function list(more = false) {
    if (!running) return;
    const attempt = ++searchVersion, instance = generation;
    const params = {searchTerm:$("search").value};
    if (more && nextCursor) params.cursor = nextCursor;
    const result = await rpc("thread/list", params);
    if (attempt !== searchVersion || instance !== generation) return;
    sessions = more ? [...new Map([...sessions, ...result.data].map(session => [session.id, session])).values()] : result.data;
    nextCursor = result.nextCursor; renderSessions();
  }
  function content(item) {
    if (item.type === "userMessage") return (item.content || []).map(x => x.text || x.path || "").join("\n");
    if (["agentMessage", "plan"].includes(item.type)) return item.text || "";
    if (item.type === "reasoning") return [...(item.summary || []), ...(item.content || [])].join("\n");
    if (item.type === "commandExecution") return [item.command, item.cwd, item.aggregatedOutput,
      item.exitCode != null ? t("Exit code: {code}", {code:item.exitCode}) : ""].filter(Boolean).join("\n");
    if (item.type === "fileChange") return (item.changes || []).map(c => `${c.path}\n${c.diff || ""}`).join("\n\n");
    return JSON.stringify(item, null, 2);
  }
  function renderItem(item) {
    const scroller = $("messages"), bottom = scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 80;
    const labels = {userMessage:"You", agentMessage:"Codex", reasoning:"Thinking", commandExecution:"Command",
      fileChange:"File changes", plan:"Plan", mcpToolCall:"Tool", contextCompaction:"Context compaction"};
    let node = nodes.get(item.id);
    if (!node) {
      const message = ["userMessage", "agentMessage"].includes(item.type);
      node = document.createElement(message ? "article" : "details"); node.className = `codex-message ${item.type}`;
      node.dataset.itemId = item.id;
      const body = document.createElement(markdownTypes.has(item.type) ? "div" : "pre");
      if (markdownTypes.has(item.type)) body.className = "codex-markdown";
      node.append(document.createElement(message ? "strong" : "summary"), body);
      nodes.set(item.id, node); scroller.append(node);
    }
    node.firstChild.textContent = t(labels[item.type] || item.type) + (item.status ? ` · ${item.status}` : "");
    if (markdownTypes.has(item.type) && markdown) {
      node.lastChild.classList.remove("markdown-pending");
      markdown(node.lastChild, content(item), {t:key => key === "common.copy" ? t("Copy") : t(key), onCopy:copyCode});
    } else {
      node.lastChild.classList.toggle("markdown-pending", markdownTypes.has(item.type));
      node.lastChild.textContent = content(item);
    }
    if (bottom) scroller.scrollTop = scroller.scrollHeight;
  }
  function snapshot(thread, eventCursor) {
    const current = rememberThread(thread);
    selectedWorkspace = typeof current.cwd === "string" ? current.cwd : ""; selectedReady = true;
    collapsedProjects.delete(selectedWorkspace);
    title = thread.name || thread.preview || "";
    snapshotCursor = eventCursor || 0; clearMessages();
    for (const turn of thread.turns || []) {
      for (const item of turn.items || []) { items.set(item.id, item); renderItem(item); }
      if (turn.status === "inProgress") turns[selected] = turn.id;
      if (turn.error) report(new Error(turn.error.message));
    }
    controls(); renderSessions();
  }
  async function select(id, resume = true) {
    save(); const attempt = ++selection, instance = generation;
    loading = true; selectedReady = false; report(null); selected = id; title = "";
    selectedWorkspace = sessions.find(session => session.id === id)?.cwd || "";
    clearMessages(); approvalSignature = ""; renderApprovals(); setSidebar(false);
    $("prompt").value = drafts.get(selected) || ""; save(); controls();
    try {
      const result = await rpc(resume ? "thread/resume" : "thread/read", {threadId:id});
      if (attempt !== selection || instance !== generation) return;
      if (!efforts.has(effortKey()) && effortOptions.includes(result.reasoningEffort)) {
        efforts.set(effortKey(), result.reasoningEffort); save();
      }
      snapshot(result.thread, result._eventCursor);
      approvalSignature = ""; renderApprovals();
    } catch (error) {
      if (attempt === selection && instance === generation) report(error);
    } finally { if (attempt === selection && instance === generation) { loading = false; controls(); } }
  }
  async function create(carryDraft = false) {
    save(); const instance = generation, attempt = selection, previous = draftKey();
    const effort = currentEffort();
    const result = await rpc("thread/start", {cwd:draftWorkspace});
    if (instance !== generation || attempt !== selection) return;
    const draft = carryDraft ? $("prompt").value : drafts.get(result.thread.id) || "";
    if (carryDraft) drafts.delete(previous);
    selected = result.thread.id; selection++; $("prompt").value = draft;
    if (effort) efforts.set(effortKey(), effort);
    if (carryDraft) efforts.delete(effortKey(previous));
    save(); snapshot(result.thread, result._eventCursor); await list();
    if (instance !== generation) return;
    if (!sessions.some(thread => thread.id === result.thread.id)) { sessions.unshift(result.thread); renderSessions(); }
    return result.thread.id;
  }
  function applyEvent(event) {
    const p = event.params || {}, method = event.method;
    if (p.threadId !== selected || event.sequence <= snapshotCursor) return;
    if (method === "item/started" || method === "item/completed") {
      items.set(p.item.id, p.item); renderItem(p.item);
    } else if (method.endsWith("/delta") || method.endsWith("/outputDelta")
               || method.endsWith("/textDelta") || method.endsWith("/summaryTextDelta")) {
      const item = items.get(p.itemId);
      if (!item) return;
      if (method === "item/agentMessage/delta" || method === "item/plan/delta") item.text = (item.text || "") + p.delta;
      else if (method === "item/commandExecution/outputDelta") item.aggregatedOutput = (item.aggregatedOutput || "") + p.delta;
      else if (method.startsWith("item/reasoning/")) {
        const field = method.includes("summary") ? "summary" : "content";
        const index = p.summaryIndex ?? p.contentIndex ?? 0;
        item[field] ||= []; item[field][index] = (item[field][index] || "") + p.delta;
      }
      renderItem(item);
    } else if (method === "turn/completed") {
      if (p.turn.error) report(new Error(p.turn.error.message));
      list().catch(report);
    } else if (method === "error") report(new Error(p.error?.message || p.message || t("Codex request failed.")));
    else if (method === "thread/name/updated") { title = p.threadName; list().catch(report); }
  }
  async function respond(pending, result) {
    await request("/api/agents/codex/respond", {method:"POST", body:JSON.stringify({id:pending.id, result})});
    approvals = approvals.filter(item => item.id !== pending.id); approvalSignature = ""; renderApprovals();
  }
  function renderApprovals() {
    const current = approvals.filter(item => item.params?.threadId === selected);
    const signature = JSON.stringify(current.map(item => item.id));
    if (signature === approvalSignature) return;
    approvalSignature = signature; $("approvals").replaceChildren();
    for (const pending of current) {
      const box = document.createElement("div"); box.className = "codex-approval";
      const heading = document.createElement("strong"); heading.textContent = t("Codex needs your response"); box.append(heading);
      const description = document.createElement("pre");
      description.textContent = [pending.params.reason, pending.params.command, pending.params.cwd].filter(Boolean).join("\n")
        || JSON.stringify(pending.params, null, 2); box.append(description);
      function button(label, result) {
        const node = document.createElement("button"); node.type = "button"; node.className = "small-button"; node.textContent = t(label);
        node.addEventListener("click", async () => {
          for (const button of box.querySelectorAll("button")) button.disabled = true;
          try { await respond(pending, typeof result === "function" ? result() : result); }
          catch (error) { report(error); for (const button of box.querySelectorAll("button")) button.disabled = false; }
        }); box.append(node);
      }
      if (["item/commandExecution/requestApproval", "item/fileChange/requestApproval"].includes(pending.method)) {
        const choices = pending.params.availableDecisions || ["accept", "decline", "cancel"];
        for (const [label, decision] of [["Allow once", "accept"], ["Decline", "decline"], ["Cancel turn", "cancel"]])
          if (choices.includes(decision)) button(label, {decision});
      } else if (pending.method === "item/tool/requestUserInput") {
        const inputs = new Map();
        for (const question of pending.params.questions) {
          const label = document.createElement("label"); label.textContent = question.question;
          const input = document.createElement("input"); input.type = question.isSecret ? "password" : "text";
          input.setAttribute("aria-label", question.question); label.append(input);
          if (question.options?.length) {
            const select = document.createElement("select"); select.add(new Option(t("Choose or enter an answer"), ""));
            for (const option of question.options) select.add(new Option(`${option.label} — ${option.description || ""}`, option.label));
            select.addEventListener("change", () => { input.value = select.value; }); label.prepend(select);
          }
          box.append(label); inputs.set(question.id, input);
        }
        button("Submit answers", () => ({answers:Object.fromEntries([...inputs].map(([id, input]) => [id, {answers:[input.value]}]))}));
      } else if (pending.method === "item/permissions/requestApproval") {
        button("Allow this turn", {permissions:pending.params.permissions || {}, scope:"turn"});
        button("Decline", {permissions:{}, scope:"turn"});
      } else {
        const note = document.createElement("p"); note.textContent = t("This request is unsupported. Cancel the turn to continue."); box.append(note);
      }
      $("approvals").append(box);
    }
  }
  async function poll() {
    if (!running || !active || polling || loading) return;
    polling = true; const instance = generation;
    try {
      const result = await request(`/api/agents/codex/events?after=${cursor}&epoch=${encodeURIComponent(epoch)}`);
      if (instance !== generation) return;
      epoch = result.epoch; cursor = result.cursor; turns = result.turns; approvals = result.requests;
      if (result.reset && selected) await select(selected, false);
      else for (const event of result.events) applyEvent(event);
      renderApprovals(); controls();
    } catch (error) { if (instance === generation) report(error); }
    finally { polling = false; }
  }
  $("new").addEventListener("click", () => beginDraft());
  $("toggle-sessions").addEventListener("click", () => setSidebar(!$("content").classList.contains("sessions-open")));
  $("more").addEventListener("click", () => list(true).catch(report));
  let searchTimer;
  $("search").addEventListener("input", () => { clearTimeout(searchTimer); searchTimer = setTimeout(() => list().catch(report), 250); });
  $("rename").addEventListener("click", async () => {
    const name = window.prompt(t("Conversation name"), title);
    if (!name?.trim()) return;
    const id = selected, instance = generation;
    try {
      await rpc("thread/name/set", {threadId:id, name:name.trim()});
      if (instance !== generation) return;
      if (selected === id) { title = name.trim(); controls(); }
      await list();
    }
    catch (error) { report(error); }
  });
  $("archive").addEventListener("click", async () => {
    const id = selected, workspace = selectedWorkspace, attempt = selection, instance = generation;
    try {
      await rpc("thread/archive", {threadId:id});
      if (instance !== generation) return;
      sessions = sessions.filter(session => session.id !== id);
      if (selected === id && attempt === selection) beginDraft(workspace || defaultWorkspace);
      await list();
    } catch (error) { report(error); }
  });
  $("cancel").addEventListener("click", () => rpc("turn/interrupt", {threadId:selected, turnId:turns[selected]}).catch(report));
  $("compose").addEventListener("submit", async event => {
    event.preventDefault(); const text = $("prompt").value;
    if (!running || loading || sending || turns[selected] || !text.trim()
        || (selected ? !selectedReady : !draftWorkspace.trim())) return;
    const instance = generation, effort = currentEffort();
    sending = true; controls(); report(null);
    try {
      const id = selected || await create(true);
      if (!id || instance !== generation) return;
      const result = await rpc("turn/start", {threadId:id, text, ...(effort ? {effort} : {})});
      if (instance !== generation) return;
      turns[id] = result.turn.id;
      // A slow acknowledgement must not erase edits made while it was pending,
      // including drafts transferred from the first, not-yet-created session.
      if (drafts.get(id) === text) drafts.set(id, "");
      if (selected === id && $("prompt").value === text) $("prompt").value = "";
      save();
      await list();
    } catch (error) { report(error); }
    finally { sending = false; controls(); }
  });
  let composing = false;
  $("prompt").addEventListener("compositionstart", () => { composing = true; });
  $("prompt").addEventListener("compositionend", () => { composing = false; });
  $("prompt").addEventListener("keydown", event => {
    if (event.key !== "Enter" || event.shiftKey || event.isComposing || composing || event.keyCode === 229) return;
    event.preventDefault();
    if (!event.repeat && !$("send").disabled) $("compose").requestSubmit();
  });
  const timer = setInterval(poll, 500);
  return mountNativeAgent({...options, id:"codex", name:"Codex",
    onState(info, visible) {
      const becameVisible = visible && !active;
      active = visible;
      const connected = info.phase === "running";
      if (connected && (!running || becameVisible)) loadMarkdown();
      if (connected && (!running || epoch !== info.epoch)) {
        running = true; epoch = info.epoch; cursor = 0; generation++; turns = {}; approvals = []; selectedReady = false;
        modelName = info.modelName || "";
        effortOptions = Array.isArray(info.reasoningEfforts) ? info.reasoningEfforts : [];
        defaultEffort = effortOptions.includes(info.defaultReasoningEffort) ? info.defaultReasoningEffort : effortOptions.at(-1) || "";
        const labels = {minimal:"Minimal", low:"Low", medium:"Medium", high:"High", xhigh:"Extra high", max:"Maximum"};
        $("effort").replaceChildren(...(effortOptions.length ? effortOptions : [""]).map(value => {
          const option = document.createElement("option"); option.value = value;
          option.textContent = value ? `${t(labels[value] || value)} (${value})` : t("Model default");
          return option;
        }));
        defaultWorkspace = info.workspace || "";
        if (!draftWorkspace) { draftWorkspace = defaultWorkspace; $("workspace").value = draftWorkspace; save(); }
        const instance = generation, attempt = selection;
        loading = Boolean(selected);
        list().then(() => instance === generation && attempt === selection && selected ? select(selected) : undefined)
          .catch(error => { if (instance === generation) { loading = false; report(error); controls(); } });
      } else if (!connected && running) {
        running = false; generation++; selection++; loading = false; selectedReady = false; turns = {}; approvals = [];
        approvalSignature = ""; renderApprovals();
      }
      controls();
    },
    destroyContent() { disposed = true; running = false; generation++; clearInterval(timer); clearTimeout(searchTimer); save(); }
  });
}
