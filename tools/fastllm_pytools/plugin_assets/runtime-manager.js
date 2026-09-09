// Native runtime management shares the plugin catalog and guarded plugin API.
export function mountRuntimeManager({root, request, refresh, getRecords, context}) {
  const dialog = document.createElement("dialog");
  dialog.className = "plugin-runtime-manager";
  dialog.setAttribute("aria-labelledby", "plugin-runtime-title");
  dialog.innerHTML = `<header><h2 id="plugin-runtime-title"></h2><button type="button" data-close>×</button></header>
    <label><span data-agent-label></span><select data-agent></select></label>
    <p data-status role="status"></p><p data-version></p><p data-note></p>
    <progress data-progress hidden></progress><p data-stage></p><pre data-error role="alert" hidden></pre>
    <footer><button type="button" data-operation="install"></button><button type="button" data-operation="upgrade"></button>
    <button type="button" data-operation="remove"></button><button type="button" data-operation="cancel" hidden></button>
    <button type="button" data-enabled></button></footer>`;
  (root.body || root).append(dialog);
  const $ = selector => dialog.querySelector(selector);
  const controller = new AbortController();
  let selected = "", pending = false, error = "", catalogSignature = "";
  const tr = (zh, en) => context().locale === "en-US" ? en : zh;
  function render() {
    if (!dialog.open) return;
    const records = getRecords().filter(p => p.builtin && p.runtime?.manageable);
    const signature = JSON.stringify(records.map(p => [p.id, p.name]));
    if (signature !== catalogSignature) {
      catalogSignature = signature;
      $("[data-agent]").replaceChildren(...records.map(p => new Option(p.name, p.id)));
    }
    const plugin = records.find(p => p.id === selected) || records[0];
    selected = plugin?.id || ""; $("[data-agent]").value = selected;
    const info = plugin?.runtime || {};
    const busy = pending || ["installing", "upgrading", "removing", "starting"].includes(info.phase);
    $("h2").textContent = tr("agent 管理", "Agent management");
    $("[data-close]").setAttribute("aria-label", tr("关闭", "Close"));
    $("[data-agent-label]").textContent = tr("选择 agent", "Select agent");
    $("[data-agent]").disabled = pending;
    const phases = {installing:tr("安装中", "Installing"), upgrading:tr("升级中", "Upgrading"),
      removing:tr("删除中", "Removing"), starting:tr("启动中", "Starting"), running:tr("运行中", "Running"), failed:tr("操作失败", "Operation failed")};
    $("[data-status]").textContent = (phases[info.phase] || (info.installed ? tr("已安装", "Installed") : tr("未安装", "Not installed")))
      + (plugin && !plugin.enabled ? tr(" · 已停用", " · Disabled") : "");
    const source = info.source === "path" ? tr("使用系统安装", "Using system installation")
      : info.source === "managed" ? tr("独立运行环境", "Private runtime") : tr("尚无可用运行环境", "No runtime available");
    $("[data-version]").textContent = `${source}${info.source === "managed" && info.version ? ` · ${info.version}` : ""}`
      + (info.targetVersion ? tr(`；当前支持版本：${info.targetVersion}`, `; supported version: ${info.targetVersion}`) : "");
    $("[data-note]").textContent = tr("无需启动模型即可安装。升级至当前支持版本，已是该版本时会重新安装。升级或删除会停止该 agent；删除仅移除独立运行环境，保留会话和工作目录。系统安装不受影响。",
      "Install without starting a model. Upgrade installs the supported version, or reinstalls it if already current. Upgrading or removing stops this agent. Removal keeps sessions, workspaces and system installations.");
    const labels = {install:info.source === "path" ? tr("安装独立版本", "Install private runtime") : tr("安装", "Install"),
      upgrade:tr("升级 / 重装", "Upgrade / reinstall"), remove:tr("删除", "Remove"), cancel:tr("取消操作", "Cancel operation")};
    for (const button of dialog.querySelectorAll("[data-operation]")) {
      const operation = button.dataset.operation;
      button.textContent = labels[operation];
      button.hidden = operation === "cancel" && !busy;
      button.disabled = !plugin || (operation === "cancel" ? pending || info.phase === "removing"
        : busy || (operation === "install" ? info.source === "managed" : !info.managed));
    }
    $("[data-enabled]").textContent = plugin?.enabled ? tr("停用", "Disable") : tr("启用", "Enable");
    $("[data-enabled]").disabled = !plugin || busy;
    const progress = $("[data-progress]");
    progress.hidden = !["installing", "upgrading"].includes(info.phase);
    if (info.total > 0) { progress.max = info.total; progress.value = info.done || 0; }
    else progress.removeAttribute("value");
    const stages = {download:tr("下载运行环境", "Downloading runtime"), extract:tr("解压运行环境", "Extracting runtime"),
      dependencies:tr("安装依赖", "Installing dependencies"), verify:tr("验证安装", "Verifying installation")};
    $("[data-stage]").textContent = stages[info.stage] || "";
    $("[data-error]").textContent = error || info.error || "";
    $("[data-error]").hidden = !$("[data-error]").textContent;
  }
  async function act(operation) {
    if (pending || !selected) return;
    const plugin = getRecords().find(p => p.id === selected);
    if (!plugin) return;
    if (operation === "remove" && !window.confirm(tr(`删除 ${plugin.name} 的独立运行环境？会话和工作目录会保留。`,
      `Remove ${plugin.name}'s private runtime? Sessions and workspaces will be kept.`))) return;
    pending = true; error = ""; render();
    try {
      const base = `/api/plugins/${encodeURIComponent(plugin.id)}`;
      await request(operation === "enabled" ? `${base}/enabled` : `${base}/runtime/${operation}`,
        {method:"POST", body:JSON.stringify(operation === "enabled" ? {enabled:!plugin.enabled} : {})});
      await refresh();
    } catch (problem) { error = problem.message; }
    finally { pending = false; render(); }
  }
  $("[data-close]").addEventListener("click", () => dialog.close(), {signal:controller.signal});
  $("[data-agent]").addEventListener("change", event => { selected = event.target.value; error = ""; render(); }, {signal:controller.signal});
  for (const button of dialog.querySelectorAll("[data-operation]")) {
    button.addEventListener("click", () => act(button.dataset.operation), {signal:controller.signal});
  }
  $("[data-enabled]").addEventListener("click", () => act("enabled"), {signal:controller.signal});
  return {update:render, open(id) {
    selected = id; error = "";
    if (!dialog.open) dialog.showModal();
    render(); refresh().catch(problem => { error = problem.message; render(); });
  }, destroy() { controller.abort(); dialog.close(); dialog.remove(); }};
}
