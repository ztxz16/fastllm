// The Harness page stays mounted across navigation, independently of Studio.
export function mountHarness({request, getRuntime, t}) {
  const $ = id => document.getElementById(id);
  let info = {phase:"stopped"}, pending = false, polling = false, version = 0, frame;
  let active = false, enabled = false;
  function available() {
    const runtime = getRuntime();
    return enabled && runtime?.command === "server" && runtime.phase === "running" && runtime.ready;
  }
  function render() {
    const installing = ["installing", "upgrading"].includes(info.phase);
    const starting = pending || ["starting", "removing"].includes(info.phase) || installing;
    $("harness-status").textContent = info.phase === "removing" ? t("Removing {name}…", {name:"DeepSeek Harness"})
      : info.phase === "upgrading" ? t("Upgrading {name}…", {name:"DeepSeek Harness"})
      : installing ? t("Installing DeepSeek Harness…") : !available() ? t("Start an API Server before opening DeepSeek Harness.")
      : starting ? t("Starting DeepSeek Harness…")
      : info.phase === "running" ? t("Connected to {model}", {model:getRuntime().modelName})
      : info.phase === "failed" ? t("DeepSeek Harness could not start.")
      : info.installed === false ? t("Harness is not installed. Click Install and open Harness to continue.") : t("DeepSeek Harness is stopped.");
    $("harness-retry").disabled = !available() || starting || info.installed == null;
    $("harness-retry").classList.toggle("hidden", info.phase === "running");
    $("harness-retry").textContent = info.phase === "failed" ? t("Retry")
      : info.installed === false ? t("Install and open Harness") : t("Open Harness");
    $("harness-stop").classList.toggle("hidden", !["running", "starting", "installing", "upgrading"].includes(info.phase));
    $("harness-stop").textContent = installing ? t("Cancel installation") : t("Stop Harness");
    $("harness-stop").disabled = pending;
    $("harness-error").textContent = info.error || "";
    $("harness-error").classList.toggle("hidden", !info.error);
    $("harness-install-note").classList.toggle("hidden", info.installed !== false);
    $("harness-progress").classList.toggle("hidden", !installing);
    const stages = {download:"Downloading the runtime…", extract:"Preparing the runtime…",
      dependencies:"Installing Harness and dependencies…", verify:"Verifying the installation…"};
    $("harness-progress-stage").textContent = t(stages[info.stage] || "Installing DeepSeek Harness…");
    const progress = $("harness-progress-bar");
    if (installing && info.total > 0) {
      progress.max = info.total; progress.value = info.done || 0;
      $("harness-progress-detail").textContent = `${((info.done || 0) / 1048576).toFixed(1)} / ${(info.total / 1048576).toFixed(1)} MiB`;
    } else {
      progress.removeAttribute("value");
      $("harness-progress-detail").textContent = installing && info.stage === "dependencies" && info.done
        ? t("{count} package requests completed", {count:info.done}) : "";
    }
    if (info.phase === "running" && info.url && available()) {
      if (!frame || frame.dataset.url !== info.url) {
        frame?.remove();
        frame = document.createElement("iframe"); frame.title = "DeepSeek Harness";
        frame.setAttribute("sandbox", "allow-scripts allow-same-origin allow-forms allow-downloads allow-popups allow-modals");
        frame.referrerPolicy = "no-referrer";
        frame.src = info.url; frame.dataset.url = info.url;
        $("harness-content").replaceChildren(frame);
      }
    } else { frame?.remove(); frame = null; }
    $("harness-content").classList.toggle("hidden", !frame);
    $("harness-placeholder").classList.toggle("hidden", !!frame);
  }
  async function open(install = false) {
    if (!available() || pending || ["starting", "installing", "upgrading", "removing", "running"].includes(info.phase)) return;
    const attempt = ++version;
    pending = true; info.error = ""; render();
    try {
      const result = await request(install ? "/api/harness/install" : "/api/harness/open", {method:"POST"});
      if (attempt === version) info = result;
    } catch (error) { if (attempt === version) info = {phase:"failed", error:error.message}; }
    finally { if (attempt === version) { pending = false; render(); } }
  }
  async function poll() {
    if (!enabled || polling || pending || (!active && !["running", "starting", "installing", "upgrading"].includes(info.phase))) return;
    polling = true; const attempt = version;
    try {
      const result = await request("/api/harness");
      if (attempt === version) { info = result; render(); }
    } catch (error) { if (attempt === version && active) $("harness-status").textContent = error.message; }
    finally { polling = false; }
  }
  $("harness-retry").addEventListener("click", () => open(info.installed === false));
  $("harness-stop").addEventListener("click", async () => {
    const attempt = ++version;
    pending = true; render();
    try {
      const result = await request("/api/harness/stop", {method:"POST"});
      if (attempt === version) info = result;
    } catch (error) { if (attempt === version) info.error = error.message; }
    finally { if (attempt === version) { pending = false; render(); } }
  });
  const timer = setInterval(poll, 1000);
  window.addEventListener("pagehide", () => clearInterval(timer), {once:true});
  return {
    setEnabled(value) {
      if (enabled === value) return;
      enabled = value;
      if (!enabled) {
        version++; pending = false; info = {phase:"stopped"};
      }
      render();
    },
    destroy() { enabled = false; version++; clearInterval(timer); frame?.remove(); frame = null; },
    navigate(view) {
      active = view === "harness";
      if (active) poll().then(() => {
        if (active && info.phase === "stopped" && info.installed === true) open();
      });
      render();
    },
    update() {
      if (info.sessionId && (!available() || info.sessionId !== getRuntime().sessionId)) {
        version++; pending = false; info = {phase:"stopped"};
      }
      render();
    }
  };
}
