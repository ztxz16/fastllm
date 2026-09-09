// Shared lifecycle controls for application-owned agent pages.
export function mountNativeAgent({id, name, request, getRuntime, t, onState, onFrame, destroyContent}) {
  const $ = suffix => document.getElementById(`${id}-${suffix}`);
  const api = `/api/agents/${id}`;
  let info = {phase:"stopped"}, enabled = false, active = false, pending = false;
  let polling = false, version = 0, frame;
  const available = () => enabled && getRuntime()?.command === "server"
    && getRuntime().phase === "running" && getRuntime().ready;
  function render() {
    const busy = pending || ["starting", "installing", "upgrading", "removing"].includes(info.phase);
    const installing = ["installing", "upgrading"].includes(info.phase);
    $("status").textContent = info.phase === "removing" ? t("Removing {name}…", {name})
      : info.phase === "upgrading" ? t("Upgrading {name}…", {name})
      : installing ? t("Installing {name}…", {name}) : !available() ? t("Start an API Server before opening {name}.", {name})
      : busy ? t("Starting {name}…", {name})
      : info.phase === "running" ? t("Connected to {model}", {model:getRuntime().modelName})
      : info.phase === "failed" ? t("{name} could not start.", {name})
      : info.installed === false ? t("{name} is not installed. Click Install and open to continue.", {name})
      : t("{name} is stopped.", {name});
    $("retry").disabled = !available() || busy || info.installed == null;
    $("retry").classList.toggle("hidden", info.phase === "running");
    $("retry").textContent = info.phase === "failed" ? t("Retry")
      : info.installed === false ? t("Install and open {name}", {name}) : t("Open {name}", {name});
    $("stop").disabled = pending;
    $("stop").classList.toggle("hidden", !["installing", "upgrading", "starting", "running"].includes(info.phase));
    $("stop").textContent = installing ? t("Cancel installation") : t("Stop {name}", {name});
    $("error").textContent = info.error || "";
    $("error").classList.toggle("hidden", !info.error);
    $("install-note").classList.toggle("hidden", info.installed !== false);
    $("progress").classList.toggle("hidden", !installing);
    const stages = {download:"Downloading the runtime…", extract:"Preparing the runtime…",
      dependencies:"Installing {name} and dependencies…", verify:"Verifying the installation…"};
    $("progress-stage").textContent = t(stages[info.stage] || "Installing {name}…", {name});
    if (installing && info.total > 0) {
      $("progress-bar").max = info.total; $("progress-bar").value = info.done || 0;
      $("progress-detail").textContent = `${((info.done || 0) / 1048576).toFixed(1)} / ${(info.total / 1048576).toFixed(1)} MiB`;
    } else {
      $("progress-bar").removeAttribute("value");
      $("progress-detail").textContent = installing && info.stage === "dependencies" && info.done
        ? t("{count} package requests completed", {count:info.done}) : "";
    }
    const running = info.phase === "running" && available();
    if (!onState) {
      if (running && info.url) {
        if (!frame || frame.dataset.url !== info.url) {
          frame?.remove(); frame = document.createElement("iframe"); frame.title = name;
          frame.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms allow-downloads allow-popups allow-modals");
          frame.referrerPolicy = "no-referrer"; frame.src = info.url; frame.dataset.url = info.url;
          onFrame?.(frame);
          $("content").replaceChildren(frame);
        }
      } else { frame?.remove(); frame = null; }
    }
    $("content").classList.toggle("hidden", !running);
    $("placeholder").classList.toggle("hidden", running);
    onState?.(running ? info : {...info, phase:"stopped"}, active);
  }
  async function open(install = false) {
    if (!available() || pending || ["starting", "installing", "upgrading", "removing", "running"].includes(info.phase)) return;
    const attempt = ++version;
    pending = true; info.error = ""; render();
    try {
      const result = await request(`${api}/${install ? "install" : "open"}`, {method:"POST"});
      if (attempt === version) info = result;
    } catch (error) { if (attempt === version) info.error = error.message; }
    finally { if (attempt === version) { pending = false; render(); } }
  }
  async function poll() {
    if (!enabled || polling || pending || (!active && !["starting", "installing", "upgrading", "removing", "running"].includes(info.phase))) return;
    const attempt = version; polling = true;
    try {
      const result = await request(api);
      if (attempt === version) { info = result; render(); }
    } catch (error) { if (attempt === version && active) $("status").textContent = error.message; }
    finally { polling = false; }
  }
  $("retry").addEventListener("click", () => open(info.installed === false));
  $("stop").addEventListener("click", async () => {
    const attempt = ++version;
    pending = true; render();
    try {
      const result = await request(`${api}/stop`, {method:"POST"});
      if (attempt === version) info = result;
    } catch (error) { if (attempt === version) info.error = error.message; }
    finally { if (attempt === version) { pending = false; render(); } }
  });
  const timer = setInterval(poll, 1000);
  function destroy() { enabled = false; version++; clearInterval(timer); frame?.remove(); destroyContent?.(); }
  window.addEventListener("pagehide", destroy, {once:true});
  return {
    setEnabled(value) {
      if (enabled === value) return;
      enabled = value;
      if (!enabled) { version++; pending = false; info = {phase:"stopped"}; }
      render();
    },
    navigate(view) {
      active = view === id;
      if (active) poll().then(() => { if (active && info.phase === "stopped" && info.installed) open(); });
      render();
    },
    update() {
      if (info.sessionId && (!available() || info.sessionId !== getRuntime().sessionId)) {
        version++; pending = false; info = {phase:"stopped"};
      }
      render();
    }, destroy
  };
}
