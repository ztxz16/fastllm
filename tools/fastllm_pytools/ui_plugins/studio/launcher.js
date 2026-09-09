import {
  state, elements, request, t, localizeServerText, showToast, friendlyError
} from "../../assets/app.js";

let webuiModulePromise;
let webuiModuleRetries = 0;
let webuiModuleLoaded = false;
const WEBUI_LOAD_TIMEOUT_MS = 30000;
function webuiIsReady() {
  return state.runtime?.command === "server" && state.runtime?.phase === "running"
    && state.runtime?.ready && Boolean(state.runtime?.sessionId);
}

function renderAgentRuntime() {
  const runtime = state.agentRuntime || {phase: "checking"};
  const phase = runtime.phase;
  const busy = state.agentRuntimeRequest || ["checking", "unchecked", "installing"].includes(phase);
  elements.agentRuntimeCard.classList.toggle("hidden", phase === "ready");
  elements.installAgentRuntime.disabled = busy || runtime.supported === false;
  elements.installAgentRuntime.textContent = t(phase === "installing" ? "Installing…"
    : phase === "failed" ? "Retry installation" : "Install Agent dependencies");
  elements.agentRuntimeTitle.textContent = t("Pi Agent");
  let message = "Install Agent dependencies with pip (about 43 MB). Uses your configured package mirror.";
  if (phase === "checking" || phase === "unchecked") message = "Checking Pi runtime…";
  if (phase === "unsupported") message = "Pi installation requires Linux x86-64, Python 3.9+ and glibc 2.17+.";
  if (phase === "failed") message = "Installation failed. See the error below and retry.";
  if (phase === "installing") message = runtime.component === "verify"
    ? "Verifying Pi runtime…" : "Installing Agent dependencies with pip…";
  elements.agentRuntimeMessage.textContent = t(message);
  elements.agentRuntimeProgress.classList.toggle("hidden", phase !== "installing");
  elements.agentRuntimeProgress.removeAttribute("value");
  elements.agentRuntimeError.classList.toggle("hidden", phase !== "failed");
  elements.agentRuntimeError.textContent = phase === "failed" ? runtime.error || "" : "";
  state.webuiComponent?.setRuntimeInstallState(runtime);
}

async function refreshAgentRuntime() {
  if (state.agentRuntimePolling || state.agentRuntimeRequest) return;
  state.agentRuntimePolling = true;
  try {
    const previous = state.agentRuntime?.phase;
    const result = await request("/api/agent-runtime");
    if (state.agentRuntimeRequest) return;
    state.agentRuntime = result;
    renderAgentRuntime();
    if (previous && previous !== "ready" && state.agentRuntime.phase === "ready") {
      state.agentConfigRefreshNeeded = true;
      if (previous === "installing") showToast(t("Pi runtime is ready. You can create an Agent now."));
    }
    if (state.agentConfigRefreshNeeded) {
      await state.webuiComponent?.refreshConfig();
      state.agentConfigRefreshNeeded = false;
    }
  } catch (_error) {
    // A temporary disconnect must not start another installation or erase progress.
  } finally { state.agentRuntimePolling = false; }
}

async function installAgentRuntime() {
  if (state.agentRuntimeRequest || state.agentRuntime?.phase === "installing" || state.agentRuntime?.supported === false) return;
  state.agentRuntimeRequest = true;
  state.agentRuntime = {...state.agentRuntime, phase: "installing", error: ""};
  renderAgentRuntime();
  try {
    state.agentRuntime = await request("/api/agent-runtime/install", {method: "POST"});
    if (state.agentRuntime.phase === "ready") state.agentConfigRefreshNeeded = true;
  } catch (error) {
    state.agentRuntime = {...state.agentRuntime, phase: "failed", error: friendlyError(error)};
  } finally { state.agentRuntimeRequest = false; }
  renderAgentRuntime();
}

function renderWebUIAvailability() {
  const ready = webuiIsReady();
  elements.openWebui.disabled = !ready;
  if ((!ready || state.webuiSessionId !== state.runtime?.sessionId)
      && (state.webuiSessionId || state.webuiLoading)) {
    state.webuiRequestId += 1;
    clearWebUILoad();
    state.webuiSessionId = "";
    state.webuiLoading = false;
    state.webuiError = "";
    destroyWebUI();
  }
  const loaded = ready && state.webuiSessionId === state.runtime.sessionId
    && !state.webuiLoading && !state.webuiError;
  elements.webuiContent.classList.toggle("hidden", !loaded);
  elements.webuiPlaceholder.classList.toggle("hidden", loaded);
  elements.webuiRetry.classList.toggle("hidden", !ready || !state.webuiError);
  elements.webuiStatus.textContent = state.webuiError ? localizeServerText(state.webuiError)
    : ready ? t("Opening WebUI…") : t("Start an API Server before opening WebUI.");
  if (state.currentView === "webui" && ready && !state.webuiLoading && !state.webuiSessionId && !state.webuiError) {
    openEmbeddedWebUI();
  }
}

function destroyWebUI() {
  state.webuiComponent?.destroy();
  state.webuiComponent = null;
  elements.webuiContent.replaceChildren();
}

function clearWebUILoad() {
  clearTimeout(state.webuiLoadTimer);
  state.webuiLoadTimer = null;
  state.webuiAbortController?.abort();
  state.webuiAbortController = null;
  // Imports cannot be aborted. Retry a stalled import with a fresh URL while
  // keeping the successfully loaded module shared across model sessions.
  if (webuiModulePromise && !webuiModuleLoaded) {
    webuiModulePromise = null;
    webuiModuleRetries += 1;
  }
}

function failWebUILoad(requestId, message = "Unable to load WebUI. Try reopening it.") {
  if (requestId !== state.webuiRequestId || !state.webuiLoading) return;
  state.webuiRequestId += 1;
  clearWebUILoad();
  state.webuiLoading = false;
  state.webuiError = message;
  destroyWebUI();
  renderWebUIAvailability();
}

async function openEmbeddedWebUI() {
  if (!webuiIsReady() || state.webuiLoading) return;
  const sessionId = state.runtime.sessionId;
  const requestId = ++state.webuiRequestId;
  state.webuiSessionId = sessionId;
  state.webuiLoading = true;
  state.webuiError = "";
  clearWebUILoad();
  const controller = new AbortController();
  state.webuiAbortController = controller;
  state.webuiLoadTimer = setTimeout(() => {
    failWebUILoad(requestId, "WebUI loading timed out. Try reopening it.");
  }, WEBUI_LOAD_TIMEOUT_MS);
  renderWebUIAvailability();
  try {
    const result = await request("/api/webui/open", {
      method: "POST", body: JSON.stringify({ sessionId }), signal: controller.signal
    });
    if (requestId !== state.webuiRequestId || sessionId !== state.runtime?.sessionId) return;
    if (!webuiModulePromise) {
      const suffix = webuiModuleRetries ? `?retry=${webuiModuleRetries}` : "";
      const pending = import(`/assets/webui/app.js${suffix}`).then(module => {
        if (webuiModulePromise === pending) webuiModuleLoaded = true;
        return module;
      }, error => {
        if (webuiModulePromise === pending) {
          webuiModulePromise = null;
          webuiModuleRetries += 1;
        }
        throw error;
      });
      webuiModulePromise = pending;
    }
    const {mountWebUI} = await webuiModulePromise;
    if (requestId !== state.webuiRequestId) return;
    const host = document.createElement("div");
    elements.webuiContent.replaceChildren(host);
    const component = await mountWebUI(host, {
      basePath: result.url, embedded: true, locale: state.locale,
      theme: window.ftllmLauncherTheme.getResolved(),
      iconUrl: "/assets/launcher-icon.png", signal: controller.signal,
      onInstallRuntime: installAgentRuntime
    });
    if (requestId !== state.webuiRequestId || sessionId !== state.runtime?.sessionId) {
      component.destroy();
      return;
    }
    state.webuiComponent = component;
    clearWebUILoad();
    state.webuiLoading = false;
    component.setLocale(state.locale);
    component.setTheme(window.ftllmLauncherTheme.getResolved());
    renderAgentRuntime();
    renderWebUIAvailability();
  } catch (error) {
    failWebUILoad(requestId, error instanceof TypeError ? "Unable to load WebUI. Try reopening it."
      : error?.message || "Unable to load WebUI. Try reopening it.");
  }
}

export {
  renderAgentRuntime, refreshAgentRuntime, installAgentRuntime, renderWebUIAvailability,
  openEmbeddedWebUI
};
