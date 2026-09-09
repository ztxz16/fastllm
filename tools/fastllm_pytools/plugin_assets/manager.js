import {conversationStore} from "./conversations.js";

// The customization subpage and recovery controls belong to the core.
export function mountManager({root, basePath, request, refresh, getRecords, context, openRuntime}) {
  const slots = {page:"独立页面", studio:"工作室扩展", topbar:"主界面右上角", sidebar:"主界面侧栏", statusbar:"主界面底栏", theme:"全局皮肤与布局"};
  const page = document.createElement("main"); page.className = "plugin-manager"; page.hidden = true;
  page.setAttribute("aria-label", "自定义界面");
  page.innerHTML = `<header class="plugin-heading"><button type="button" aria-label="关闭">← 返回</button><h1>自定义界面</h1><span>预览满意后，再应用到当前界面</span></header>
    <div class="customizer-layout"><aside class="customizer-controls">
      <header class="customizer-chat-heading"><button type="button" class="customizer-toggle-sessions" aria-label="对话列表" aria-expanded="false">☰</button><h2 class="customizer-session-title">新对话</h2><button type="button" class="customizer-new-chat">新建对话</button></header>
      <section class="customizer-sessions" aria-label="编辑对话管理" hidden><input class="customizer-session-search" type="search" placeholder="搜索对话" aria-label="搜索编辑对话">
        <div class="customizer-session-list" role="list" aria-label="编辑对话列表"></div><small>对话与草稿保存在当前浏览器</small></section>
      <p class="customizer-storage-error" role="alert" hidden></p>
      <div class="customizer-discussion"><div class="customizer-chat" role="log" aria-label="界面编辑对话" aria-live="polite">
        <p class="customizer-empty">告诉我你想怎样修改界面。每轮修改后都可以在右侧预览，再继续提出要求。</p></div>
      <section class="plugin-preview" hidden aria-label="待应用的修改"><div class="customizer-review"><small class="plugin-permissions"></small>
        <button type="button" class="plugin-apply plugin-primary">应用修改</button></div>
        <details class="customizer-code"><summary>查看与编辑文件</summary><select class="plugin-files" aria-label="预览文件"></select>
          <label>新版本<textarea class="plugin-after" rows="12" spellcheck="false" aria-label="编辑预览文件"></textarea></label><details><summary>当前版本</summary><pre class="plugin-before"></pre></details></details></section>
      <details class="customizer-library"><summary>已有自定义项与恢复</summary><button type="button" class="plugin-reset-theme">恢复默认皮肤</button><div class="plugin-list"></div><small class="plugin-directory"></small></details>
      </div><form class="plugin-editor">
        <div class="customizer-progress" aria-label="修改进度"><p class="plugin-result" role="status">描述修改要求，开始对话。</p><small class="plugin-timing"></small></div>
        <label>修改要求<textarea name="instruction" required rows="3" maxlength="12000" placeholder="例如：在右上角增加硬件状态栏。也可以继续说：再紧凑一点。"></textarea></label>
        <details class="customizer-options"><summary>修改选项</summary>
          <label>定制位置<select name="target"><option value="">根据修改要求选择</option>${Object.entries(slots).map(([value,label]) => `<option value="${value}">${label}</option>`).join("")}</select></label>
          <label>参考界面<select name="reference"><option value="">新建自定义项</option></select></label>
          <label>保存名称<input name="id" pattern="[a-z][a-z0-9-]{0,47}" placeholder="留空自动命名，例如 hardware-monitor" maxlength="48"></label></details>
        <div class="customizer-actions"><button class="plugin-primary" type="submit">发送</button><button class="plugin-cancel" type="button" hidden>取消生成</button></div>
      </form>
    </aside><section class="customizer-canvas" aria-label="整页预览"><div class="customizer-preview-toolbar"><strong>整页实时预览</strong>
      <select class="customizer-size" aria-label="预览尺寸"><option value="1280">桌面</option><option value="390">手机</option></select>
      <button type="button" class="customizer-refresh">刷新预览</button></div>
      <p class="customizer-preview-note">直接点击预览中的导航、主题和界面控件；操作仅在预览中生效，启动/停止服务和发送对话不可用。</p>
      <div class="customizer-viewport"><div class="customizer-screen"></div></div></section></div>`;
  (root.body || root).append(page);
  const $ = selector => page.querySelector(selector);
  const editor = $(".plugin-editor"), reference = editor.elements.reference, result = $(".plugin-result");
  const controller = new AbortController();
  let proposal = null, before = {}, busy = false, listSignature = "", preview, candidate;
  let run, elapsedTimer, editTimer, previewVersion = 0, started = 0, characters = 0, returnFocus, pushed = false;
  let turns = [], activeTurn = null, proposalTurn = null, previewProposal = null;
  let conversations = [], conversation, initialized = false, saveTimer, sessionEditor = null, storageAvailable = true;
  const storage = conversationStore(basePath || "/");
  function storageError(error) {
    $(".customizer-storage-error").hidden = false;
    $(".customizer-storage-error").textContent = `编辑记录无法读取或保存，请暂勿刷新页面：${error.message || error}`;
  }
  const ready = storage.load().then(saved => {
    if (saved) {
      if (saved.version !== 1 || !Array.isArray(saved.conversations)) throw new Error("编辑记录格式无法读取");
      const validProposal = value => !value || (typeof value.id === "string" && value.manifest
        && value.files && typeof value.files === "object" && !Array.isArray(value.files));
      const ids = new Set();
      for (const item of saved.conversations) {
        if (!item || typeof item.id !== "string" || ids.has(item.id) || typeof item.title !== "string"
            || !item.fields || typeof item.fields !== "object" || !Array.isArray(item.turns)
            || item.turns.some(turn => !turn || typeof turn.instruction !== "string" || typeof turn.summary !== "string"
              || typeof turn.status !== "string" || !Array.isArray(turn.steps)
              || turn.steps.some(step => !step || typeof step.name !== "string" || typeof step.message !== "string"))
            || !validProposal(item.proposal) || !validProposal(item.previewProposal)) throw new Error("编辑记录格式无法读取");
        ids.add(item.id);
      }
      conversations = saved.conversations;
      conversation = conversations.find(item => item.id === saved.active);
      for (const item of conversations) for (const turn of item.turns) if (turn.pending) {
        turn.pending = false; turn.summary = ""; turn.status = "上次生成已中断，可继续修改。";
      }
    }
  }).catch(error => { conversations = []; conversation = null; storageAvailable = false; storageError(error); }).then(() => {
    if (!conversations.length) conversations.push(newConversation());
    conversation ||= conversations[0];
  });
  const application = root.host ? root.querySelector(".app") : root.querySelector(".app-shell");
  const emptyConversation = $(".customizer-empty");
  function newConversation() {
    return {id:crypto.randomUUID?.() || `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}`,
      title:"新对话", customTitle:false, updatedAt:Date.now(), turns:[], fields:{}};
  }
  function checkpoint() {
    if (!initialized) return;
    Object.assign(conversation, {
      turns:turns.map(({node, ...turn}) => turn), proposal, before, previewProposal,
      proposalTurn:turns.indexOf(proposalTurn), updatedAt:Date.now(),
      fields:Object.fromEntries(["instruction", "id", "reference", "target"].map(name => [name, editor.elements[name].value])),
      result:result.textContent, timing:$(".plugin-timing").textContent, stage:page.dataset.stage,
      file:$(".plugin-files").value, codeOpen:$(".customizer-code").open, optionsOpen:$(".customizer-options").open,
      scrollTop:$(".customizer-discussion").scrollTop
    });
  }
  async function persist() {
    clearTimeout(saveTimer);
    if (!initialized || !storageAvailable) return;
    checkpoint();
    try {
      await storage.save({version:1, active:conversation.id, conversations});
      $(".customizer-storage-error").hidden = true;
    } catch (error) { storageError(error); }
  }
  function saveSoon() { clearTimeout(saveTimer); saveTimer = setTimeout(persist, 300); }
  function renderSessions() {
    $(".customizer-session-title").textContent = conversation?.title || "新对话";
    $(".customizer-session-title").title = conversation?.title || "新对话";
    const query = $(".customizer-session-search").value.trim().toLocaleLowerCase();
    const list = $(".customizer-session-list"); list.replaceChildren();
    for (const item of [...conversations].sort((a, b) => b.updatedAt - a.updatedAt)) {
      if (query && !item.title.toLocaleLowerCase().includes(query)) continue;
      const row = document.createElement("div"); row.className = "customizer-session"; row.dataset.session = item.id;
      row.setAttribute("role", "listitem"); row.classList.toggle("active", item === conversation);
      row.innerHTML = `<div class="customizer-session-row"><button type="button" data-session-action="switch"><strong></strong><small></small></button>
        <button type="button" data-session-action="rename" aria-label="重命名对话" title="重命名">✎</button><button type="button" data-session-action="delete" aria-label="删除对话" title="删除">×</button></div>`;
      row.querySelector("strong").textContent = item.title;
      row.querySelector("small").textContent = `${item.turns.length} 轮${item.proposal ? " · 有草稿" : ""}`;
      row.querySelector("[data-session-action=switch]").setAttribute("aria-current", item === conversation ? "true" : "false");
      if (sessionEditor?.id === item.id) {
        const edit = document.createElement("div"); edit.className = "customizer-session-edit";
        if (sessionEditor.mode === "rename") {
          edit.innerHTML = `<input aria-label="对话名称" maxlength="80"><button type="button" data-session-action="save-name">保存</button><button type="button" data-session-action="cancel">取消</button>`;
          edit.querySelector("input").value = item.title;
        } else edit.innerHTML = `<p>删除此对话及其未应用草稿？已应用的界面会保留。</p><button type="button" data-session-action="confirm-delete">删除此对话</button><button type="button" data-session-action="cancel">取消</button>`;
        row.append(edit);
      }
      list.append(row);
    }
    if (!list.childElementCount) { const note = document.createElement("p"); note.textContent = "没有匹配的对话"; list.append(note); }
  }
  function hydrateConversation() {
    turns = conversation.turns.map(turn => ({...turn, steps:[...(turn.steps || [])]}));
    proposal = conversation.proposal || null; before = conversation.before || {};
    previewProposal = conversation.previewProposal || null; candidate = null;
    proposalTurn = turns[conversation.proposalTurn] || null;
    $(".customizer-chat").replaceChildren(emptyConversation, ...turns.map(renderTurn));
    emptyConversation.hidden = !!turns.length;
    for (const name of ["instruction", "id", "reference", "target"]) editor.elements[name].value = conversation.fields[name] || "";
    $(".plugin-preview").hidden = !proposal; $(".plugin-apply").disabled = true;
    if (proposal) {
      showProposal();
      if (conversation.file in proposal.files) { $(".plugin-files").value = conversation.file; showFile(); }
    }
    $(".customizer-code").open = !!conversation.codeOpen; $(".customizer-options").open = !!conversation.optionsOpen;
    page.dataset.stage = conversation.stage || "";
    report(conversation.result || "描述修改要求，开始对话。"); $(".plugin-timing").textContent = conversation.timing || "";
    $(".customizer-discussion").scrollTop = conversation.scrollTop || 0;
    renderSessions();
  }
  async function restorePreview() {
    const fallback = previewProposal, session = conversation;
    const version = previewVersion + (proposal ? 1 : 0);
    if (proposal && await validatePreview()) return;
    if (session !== conversation || version !== previewVersion || page.hidden || controller.signal.aborted) return;
    if (fallback) {
      const checked = await request("/api/plugins/preview", {method:"POST", body:JSON.stringify(fallback)});
      if (version !== previewVersion || controller.signal.aborted) return;
      candidate = checked.plugin;
    }
    await renderPreview();
  }
  async function selectConversation(next) {
    if (busy) return;
    clearTimeout(editTimer); clearTimeout(saveTimer); previewVersion++;
    checkpoint(); setBusy(true);
    preview?.destroy(); preview = null;
    conversation = next; sessionEditor = null; initialized = true;
    hydrateConversation();
    await persist();
    try { await restorePreview(); }
    catch (error) { report(error); }
    finally { setBusy(false); renderSessions(); if (!page.hidden) editor.elements.instruction.focus(); }
  }
  function fitPreview() {
    const width = Number($(".customizer-size").value), height = width === 390 ? 844 : 800;
    const area = $(".customizer-viewport"), screen = $(".customizer-screen");
    const scale = Math.max(0, Math.min(area.clientWidth / width, area.clientHeight / height, 1));
    screen.style.width = width + "px"; screen.style.height = height + "px"; screen.style.transform = `scale(${scale})`;
  }
  const resize = new ResizeObserver(fitPreview); resize.observe($(".customizer-viewport"));
  function report(error) { const message = String(error?.message || error); if (result.textContent !== message) result.textContent = message; }
  function scrollConversation() {
    const discussion = $(".customizer-discussion");
    const latest = $(".customizer-chat").lastElementChild;
    if (latest) discussion.scrollTop += latest.getBoundingClientRect().top - discussion.getBoundingClientRect().top - 12;
  }
  function renderTurn(turn) {
    const node = document.createElement("section"); node.className = "customizer-turn";
    node.innerHTML = `<article class="customizer-message customizer-user"><strong>你</strong><p></p></article>
      <article class="customizer-message customizer-assistant"><strong>界面助手</strong><p class="customizer-reply"></p>
        <small class="customizer-turn-status"></small><details class="customizer-turn-progress"><summary>修改摘要</summary><ul></ul></details></article>`;
    node.querySelector(".customizer-user p").textContent = turn.instruction;
    node.querySelector(".customizer-reply").textContent = turn.summary || turn.status;
    node.querySelector(".customizer-turn-status").textContent = turn.summary ? turn.status : "";
    for (const step of turn.steps || []) {
      const item = document.createElement("li"); item.dataset.step = step.name; item.textContent = step.message;
      node.querySelector("ul").append(item);
    }
    turn.node = node;
    return node;
  }
  function addTurn(instruction) {
    emptyConversation.hidden = true;
    const turn = {instruction, summary:"", status:"", steps:[], pending:true}; turns.push(turn);
    $(".customizer-chat").append(renderTurn(turn));
    if (turns.length === 1 && !conversation.customTitle) conversation.title = Array.from(instruction.replace(/\s+/g, " ")).slice(0, 40).join("");
    checkpoint(); renderSessions();
    scrollConversation();
    return turn;
  }
  function describeTurn(turn, message, step) {
    if (!turn) return;
    if (turn.status === message) return;
    turn.status = message;
    turn.node.querySelector(".customizer-reply").textContent = turn.summary || message;
    turn.node.querySelector(".customizer-turn-status").textContent = turn.summary ? message : "";
    if (step && !turn.node.querySelector(`[data-step="${step}"]`)) {
      const item = document.createElement("li"); item.dataset.step = step; item.textContent = message;
      turn.node.querySelector("ul").append(item);
      turn.steps.push({name:step, message});
    }
  }
  function conversationHistory() {
    return turns.flatMap(turn => [
      {role:"user", content:turn.instruction},
      {role:"assistant", content:[turn.summary, turn.status].filter(Boolean).join("\n")}
    ]);
  }
  function stage(name, message) {
    page.dataset.stage = name;
    if (message) { report(message); describeTurn(activeTurn, message, name); }
  }
  function timing() { $(".plugin-timing").textContent = `已用 ${Math.floor((performance.now() - started) / 1000)} 秒 · 已生成 ${characters.toLocaleString()} 字符`; }
  function setBusy(value) {
    busy = value; editor.querySelector('[type="submit"]').disabled = value;
    for (const field of editor.querySelectorAll("input,select,textarea")) field.disabled = value;
    $(".plugin-after").disabled = value; $(".plugin-cancel").hidden = !value || !run;
    $(".customizer-new-chat").disabled = value;
    $(".customizer-toggle-sessions").disabled = value;
    $(".customizer-sessions").inert = value;
    $(".customizer-library").inert = value;
    $(".customizer-refresh").disabled = value;
    if (value) $(".plugin-apply").disabled = true;
    if (!value) clearInterval(elapsedTimer);
  }
  function action(label, handler) {
    const button = document.createElement("button"); button.type = "button"; button.textContent = label;
    button.addEventListener("click", () => Promise.resolve().then(handler).catch(report), {signal:controller.signal});
    return button;
  }
  function update({directory, plugins}) {
    $(".plugin-directory").textContent = `保存位置：${directory}`;
    const signature = JSON.stringify(plugins);
    if (listSignature === signature) return;
    listSignature = signature;
    const selected = reference.value;
    reference.replaceChildren(new Option("新建自定义项", "")); $(".plugin-list").replaceChildren();
    for (const plugin of plugins) {
      reference.add(new Option(plugin.name, plugin.id));
      const row = document.createElement("div"); row.className = "plugin-row"; row.dataset.pluginId = plugin.id;
      const label = document.createElement("span"); label.textContent = `${plugin.name} · ${slots[plugin.slot] || "无效配置"}`;
      if (plugin.builtin && plugin.runtime) {
        const phases = {installing:"安装中", upgrading:"升级中", removing:"删除中", starting:"启动中", running:"运行中", failed:"操作失败"};
        const status = document.createElement("small"); status.className = "plugin-runtime-status";
        status.textContent = !plugin.enabled ? "已停用" : phases[plugin.runtime.phase]
          || (plugin.runtime.installed ? "已安装" : "未安装，需手动点击安装");
        label.append(document.createElement("br"), status);
      }
      row.append(label, action(plugin.enabled ? "停用" : "启用", async () => {
        await request(`/api/plugins/${encodeURIComponent(plugin.id)}/enabled`, {method:"POST", body:JSON.stringify({enabled:!plugin.enabled})}); await refresh(); await renderPreview();
      }), action("定制", async () => {
        if (busy) return;
        const session = conversation;
        $(".customizer-options").open = true;
        editor.elements.id.value = plugin.builtin ? `${plugin.id}-custom` : plugin.id;
        reference.value = plugin.id; editor.elements.target.value = plugin.slot || ""; editor.elements.instruction.focus();
        if (!plugin.builtin && plugin.revision) {
          const source = await request(`/api/plugins/${plugin.id}/files`);
          if (conversation !== session) return;
          before = source.files; proposalTurn = null; proposal = {id:plugin.id, files:{...source.files}, expectedRevision:source.plugin.revision, manifest:source.plugin, summary:"继续修改当前自定义项"};
          showProposal(); await validatePreview();
        }
      }));
      if (plugin.builtin && plugin.runtime?.manageable && openRuntime) {
        row.append(action("管理", () => openRuntime(plugin.id)));
      }
      if (!plugin.builtin && plugin.revision) row.append(action("恢复上一版", async () => {
        await request(`/api/plugins/${plugin.id}/rollback`, {method:"POST", body:JSON.stringify({expectedRevision:plugin.revision})});
        await refresh(); await renderPreview(); report("已恢复上一版本");
      }));
      if (!plugin.builtin) {
        const remove = action("删除", () => {
          if (busy || row.querySelector(".plugin-delete-confirm")) return;
          const confirm = document.createElement("div"); confirm.className = "plugin-delete-confirm";
          const message = document.createElement("p");
          message.textContent = `删除“${plugin.name}”及其上一版本？界面会立即卸载此项，编辑对话和草稿会保留。`;
          const cancel = action("取消", () => confirm.remove());
          const submit = action("确认删除", async () => {
            submit.disabled = cancel.disabled = true;
            try {
              await request(`/api/plugins/${encodeURIComponent(plugin.id)}`, {method:"DELETE", body:JSON.stringify({expectedRevision:plugin.revision})});
              await refresh(); await renderPreview(); report(`已删除“${plugin.name}”。`);
            } finally { submit.disabled = cancel.disabled = false; }
          });
          submit.className = "plugin-confirm-delete";
          confirm.append(message, submit, cancel); row.append(confirm);
        });
        remove.className = "plugin-delete"; row.append(remove);
      }
      if (plugin.error) { const error = document.createElement("p"); error.className = "plugin-error"; error.textContent = plugin.error; row.append(error); }
      $(".plugin-list").append(row);
    }
    if ([...reference.options].some(o => o.value === selected)) reference.value = selected;
  }
  async function renderPreview() {
    if (page.hidden) return;
    const version = previewVersion;
    if (!preview) {
      const {mountPreview} = await import("./preview.js");
      if (version !== previewVersion || page.hidden || controller.signal.aborted) return;
      preview = mountPreview({container:$(".customizer-screen"), source:root, basePath, request, context, getRecords});
    }
    await preview.build(candidate || null);
  }
  async function validatePreview() {
    if (!proposal) return;
    const version = ++previewVersion;
    $(".plugin-apply").disabled = true;
    try {
      const checked = await request("/api/plugins/preview", {method:"POST", signal:run?.signal, body:JSON.stringify(proposal)});
      if (version !== previewVersion || controller.signal.aborted) return false;
      candidate = checked.plugin; proposal.manifest = candidate;
      stage("preview", "正在更新整页预览…");
      await renderPreview();
      if (version !== previewVersion) return false;
      $(".plugin-apply").disabled = false;
      previewProposal = {...proposal, files:{...proposal.files}};
      stage("ready", "预览已更新，尚未应用修改。");
      saveSoon();
      return true;
    } catch (error) {
      if (version === previewVersion) stage("error", error.name === "AbortError" ? "已取消生成，尚未应用修改。" : error.message);
      return false;
    }
  }
  function showProposal() {
    $(".plugin-permissions").textContent = `${slots[proposal.manifest.slot]} · ${Object.keys(proposal.files).length} 个文件`;
    $(".plugin-files").replaceChildren(...Object.keys(proposal.files).sort().map(name => new Option(name, name)));
    showFile(); $(".plugin-preview").hidden = false;
  }
  editor.addEventListener("submit", async event => {
    event.preventDefault(); if (busy) return;
    const instruction = editor.elements.instruction.value.trim();
    if (!instruction) { editor.elements.instruction.focus(); return; }
    clearTimeout(editTimer); previewVersion++;
    const previous = {proposal, candidate, previewProposal, before, turn:proposalTurn, applicable:!$(".plugin-apply").disabled};
    const previousHistory = conversationHistory();
    const id = editor.elements.id.value.trim() || `custom-${conversation.id}`;
    editor.elements.id.value = id;
    activeTurn = addTurn(instruction);
    editor.elements.instruction.value = ""; $(".customizer-options").open = false;
    run = new AbortController(); started = performance.now(); characters = 0;
    setBusy(true); timing(); elapsedTimer = setInterval(timing, 1000);
    stage("reading", "正在读取当前界面…");
    persist();
    try {
      const existing = getRecords().find(p => p.id === id);
      before = existing ? (await request(`/api/plugins/${id}/files`, {signal:run.signal})).files : {};
      const response = await request("/api/plugins/propose-stream", {method:"POST", signal:run.signal, stream:true,
        body:JSON.stringify({id, reference:reference.value, target:editor.elements.target.value, instruction, history:previousHistory,
          ...(previous.proposal?.id === id ? {draft:previous.proposal.files, expectedRevision:previous.proposal.expectedRevision} : {})})});
      const reader = response.body.getReader(), decoder = new TextDecoder(); let pending = "", nextProposal;
      try {
        while (true) {
          const {value, done} = await reader.read();
          pending += decoder.decode(value, {stream:!done});
          const lines = pending.split("\n"); pending = lines.pop();
          if (done && pending.trim()) { lines.push(pending); pending = ""; }
          for (const line of lines) {
            if (!line.trim()) continue;
            const data = JSON.parse(line);
            if (data.stage === "error") throw new Error(data.error);
            if (data.characters !== undefined) characters = data.characters;
            stage(data.stage === "ready" ? "generated" : data.stage,
              {reading:"正在读取当前界面…", generating:"模型正在生成修改…", validating:"正在校验修改内容…", ready:"生成完成，准备预览…"}[data.stage]); timing();
            if (data.proposal) nextProposal = data.proposal;
          }
          if (done) break;
        }
      } finally { reader.releaseLock(); }
      if (!nextProposal) throw new Error("生成连接已结束，但修改内容不完整；请重试。");
      proposal = nextProposal; proposalTurn = activeTurn;
      activeTurn.summary = proposal.summary || "已按要求更新界面预览。";
      showProposal();
      if (!await validatePreview()) throw new Error(result.textContent);
      scrollConversation();
    } catch (error) {
      activeTurn.summary = "";
      stage("error", run.signal.aborted ? "已取消生成，尚未应用修改。" : error.message);
      const rebuild = candidate !== previous.candidate;
      proposal = previous.proposal; candidate = previous.candidate; before = previous.before; proposalTurn = previous.turn;
      previewProposal = previous.previewProposal;
      if (proposal) showProposal(); else $(".plugin-preview").hidden = true;
      $(".plugin-apply").disabled = !proposal || !previous.applicable;
      if (rebuild) await renderPreview().catch(report);
    }
    finally {
      activeTurn.pending = false;
      run?.abort(); run = null; activeTurn = null; setBusy(false);
      await persist(); renderSessions(); if (!page.hidden) editor.elements.instruction.focus();
    }
  }, {signal:controller.signal});
  function showFile() {
    const name = $(".plugin-files").value;
    $(".plugin-before").textContent = before[name] ?? "（新增文件）";
    $(".plugin-after").value = proposal?.files[name] ?? "";
  }
  $(".plugin-files").addEventListener("change", () => { showFile(); saveSoon(); }, {signal:controller.signal});
  $(".customizer-code").addEventListener("toggle", saveSoon, {signal:controller.signal});
  $(".plugin-after").addEventListener("input", () => {
    if (!proposal || busy) return;
    proposal.files[$(".plugin-files").value] = $(".plugin-after").value;
    saveSoon();
    previewVersion++; $(".plugin-apply").disabled = true;
    clearTimeout(editTimer); editTimer = setTimeout(validatePreview, 400);
  }, {signal:controller.signal});
  $(".plugin-cancel").addEventListener("click", () => run?.abort(), {signal:controller.signal});
  $(".customizer-new-chat").addEventListener("click", () => {
    if (busy) return;
    const next = newConversation(); conversations.unshift(next);
    $(".customizer-session-search").value = "";
    $(".customizer-sessions").hidden = true; $(".customizer-toggle-sessions").setAttribute("aria-expanded", "false");
    selectConversation(next).catch(report);
  }, {signal:controller.signal});
  $(".customizer-toggle-sessions").addEventListener("click", () => {
    const panel = $(".customizer-sessions"); panel.hidden = !panel.hidden;
    $(".customizer-toggle-sessions").setAttribute("aria-expanded", String(!panel.hidden));
    sessionEditor = null; renderSessions(); if (!panel.hidden) $(".customizer-session-search").focus();
  }, {signal:controller.signal});
  $(".customizer-session-search").addEventListener("input", renderSessions, {signal:controller.signal});
  $(".customizer-session-list").addEventListener("click", async event => {
    const button = event.target.closest("[data-session-action]");
    if (!button || busy) return;
    const row = button.closest("[data-session]");
    const item = conversations.find(session => session.id === row.dataset.session);
    if (!item) return;
    const action = button.dataset.sessionAction;
    if (action === "switch") {
      $(".customizer-sessions").hidden = true;
      $(".customizer-toggle-sessions").setAttribute("aria-expanded", "false");
      if (item !== conversation) await selectConversation(item);
    } else if (action === "rename" || action === "delete") {
      sessionEditor = {id:item.id, mode:action}; renderSessions();
      $(".customizer-session-edit input")?.focus();
    } else if (action === "save-name") {
      const name = row.querySelector("input").value.trim();
      if (!name) { row.querySelector("input").focus(); return; }
      item.title = name; item.customTitle = true; sessionEditor = null;
      await persist(); renderSessions();
    } else if (action === "confirm-delete") {
      conversations = conversations.filter(session => session !== item); sessionEditor = null;
      if (item === conversation) {
        if (!conversations.length) conversations.push(newConversation());
        await selectConversation(conversations[0]);
      } else { await persist(); renderSessions(); }
    } else if (action === "cancel") { sessionEditor = null; renderSessions(); }
  }, {signal:controller.signal});
  $(".customizer-session-list").addEventListener("keydown", event => {
    if (!event.target.matches("input")) return;
    const edit = event.target.closest(".customizer-session-edit");
    if (event.key === "Enter") { event.preventDefault(); edit.querySelector("[data-session-action=save-name]").click(); }
    if (event.key === "Escape") { event.preventDefault(); edit.querySelector("[data-session-action=cancel]").click(); }
  }, {signal:controller.signal});
  for (const name of ["input", "change", "toggle"]) editor.addEventListener(name, saveSoon, {signal:controller.signal});
  $(".plugin-reset-theme").addEventListener("click", async () => {
    try {
      await request("/api/plugins/reset-theme", {method:"POST", body:"{}"});
      candidate = null; await refresh(); await renderPreview(); report("已恢复默认皮肤，页面与组件保持运行。");
    } catch (error) { report(error); }
  }, {signal:controller.signal});
  $(".plugin-apply").addEventListener("click", async () => {
    if (!proposal || busy) return;
    setBusy(true); $(".plugin-apply").disabled = true; stage("applying", "正在应用修改…");
    describeTurn(proposalTurn, "正在应用修改…", "applying");
    try {
      await request(`/api/plugins/${proposal.id}/apply`, {method:"POST", body:JSON.stringify(proposal)});
      describeTurn(proposalTurn, "已应用到当前界面。", "applied"); proposalTurn = null;
      proposal = null; candidate = null; previewProposal = null; $(".plugin-preview").hidden = true;
      await refresh(); await renderPreview(); stage("done", "已应用，界面已更新。");
    } catch (error) { report(error); describeTurn(proposalTurn, "应用失败：" + error.message, "apply-error"); $(".plugin-apply").disabled = false; }
    finally { setBusy(false); await persist(); renderSessions(); }
  }, {signal:controller.signal});
  $(".customizer-refresh").addEventListener("click", () => (proposal ? validatePreview() : renderPreview()).catch(report), {signal:controller.signal});
  $(".customizer-size").addEventListener("change", fitPreview, {signal:controller.signal});
  function hide() {
    page.hidden = true; run?.abort(); previewVersion++; clearTimeout(editTimer);
    persist();
    preview?.destroy(); preview = null; if (application) application.inert = false; returnFocus?.focus();
  }
  function open(navigate = true) {
    if (!page.hidden) return;
    returnFocus = root.activeElement; page.hidden = false; if (application) application.inert = true;
    if (navigate && location.hash !== "#customize") { history.pushState(history.state, "", "#customize"); pushed = true; }
    ready.then(async () => {
      if (page.hidden || controller.signal.aborted) return;
      await refresh();
      if (!initialized) { setBusy(false); await selectConversation(conversation); }
      else if (!busy) await restorePreview();
    }).catch(report); editor.elements.instruction.focus();
  }
  function close() {
    hide(); if (pushed && location.hash === "#customize") history.back();
    else if (location.hash === "#customize") history.replaceState(history.state, "", location.pathname + location.search);
    pushed = false;
  }
  $("[aria-label='关闭']").addEventListener("click", close, {signal:controller.signal});
  function syncRoute() {
    if (location.hash === "#customize") open(false);
    else if (!page.hidden) hide();
  }
  for (const name of ["popstate", "hashchange"]) window.addEventListener(name, syncRoute, {signal:controller.signal});
  window.addEventListener("pagehide", () => { run?.abort(); persist(); }, {signal:controller.signal});
  setBusy(true);
  if (location.hash === "#customize") open(false);
  return {update, open, destroy:() => { hide(); controller.abort(); resize.disconnect(); clearInterval(elapsedTimer); page.remove(); }};
}
