const locationQuery = new URLSearchParams(window.location.search);
const queryToken = locationQuery.get("token") || "";
if (queryToken) {
  window.sessionStorage.setItem("ftllm-launcher-token", queryToken);
  window.history.replaceState({}, "", window.location.pathname);
}
const controlToken = queryToken || window.sessionStorage.getItem("ftllm-launcher-token") || "";
const ACTIVE_RUNTIME_PHASES = new Set(["starting", "running", "stopping"]);
const ACTIVE_DOWNLOAD_PHASES = new Set(["starting", "downloading", "cancelling"]);

const state = {
  profiles: [],
  defaultProfile: null,
  currentIndex: null,
  editingConfig: null,
  runtime: null,
  download: null,
  downloadDefaults: null,
  downloadCatalog: [],
  launcherAddresses: [],
  downloadPreview: null,
  downloadTargetAutomatic: true,
  preview: null,
  dirty: false,
  logs: [],
  lastLogId: 0,
  currentView: "launch",
  hardwareLoaded: false,
  pollingRuntime: false,
  pollingDownload: false,
  pollingLogs: false,
  previewTimer: null,
  previewRequestId: 0,
  downloadPreviewTimer: null,
  downloadPreviewRequestId: 0,
  pathTimer: null,
  pathRequestId: 0
};

const elements = {};

document.addEventListener("DOMContentLoaded", initialize);

async function request(path, options = {}) {
  const headers = new Headers(options.headers || {});
  headers.set("X-FTLLM-Launcher-Token", controlToken);
  if (options.body && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  const response = await fetch(path, { ...options, headers });
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch (_error) {
      payload = { error: text };
    }
  }
  if (!response.ok) {
    throw new Error(payload.error || `请求失败（HTTP ${response.status}）`);
  }
  return payload;
}

function cacheElements() {
  const ids = [
    "app-version", "shutdown-launcher", "status-dot", "status-title", "status-message",
    "open-endpoint", "stop-runtime", "start-runtime-top", "new-profile-hero",
    "start-runtime-hero", "profile-count", "runtime-summary", "endpoint-summary",
    "runtime-progress-card", "runtime-progress-icon", "runtime-progress-title",
    "runtime-progress-label", "runtime-progress-value", "runtime-progress", "runtime-pid",
    "runtime-model", "copy-endpoint", "profile-list", "new-profile", "duplicate-profile",
    "delete-profile", "config-path", "launch-form", "save-state", "ori-field",
    "cuda-device-field", "tp-device-field", "cudapp-device-field", "moe-device-field",
    "moe-layers-field", "server-model-name-field", "server-host-field",
    "webui-max-token-field", "webui-think-field", "server-context-field",
    "server-sampling-title", "server-sampling-fields", "server-api-key-field",
    "server-hide-input-field",
    "launch-command-kicker", "launch-hero-eyebrow", "launch-hero-description",
    "endpoint-summary-label", "command-preview", "validation-messages", "save-profile",
    "start-runtime", "clear-logs", "log-count", "log-output", "log-activity",
    "refresh-hardware", "hardware-status", "hardware-grid", "path-suggestions",
    "toast-region", "download-activity", "download-form", "download-preset",
    "download-badge", "download-command", "download-validation", "download-start",
    "download-cancel", "download-use-model", "download-use-last", "download-status-icon",
    "download-status-title", "download-status-message", "download-progress-value",
    "download-progress", "download-bytes", "download-files", "download-destination",
    "launcher-address-list"
  ];
  for (const id of ids) {
    elements[toCamelCase(id)] = document.getElementById(id);
  }
}

function toCamelCase(value) {
  return value.replace(/-([a-z])/g, (_match, letter) => letter.toUpperCase());
}

async function initialize() {
  cacheElements();
  bindEvents();
  if (!controlToken) {
    showToast("缺少控制令牌，请从 ftllm launch 输出的地址重新打开。", "error", 10000);
  }
  try {
    const bootstrap = await request("/api/bootstrap");
    state.profiles = Array.isArray(bootstrap.profiles) ? bootstrap.profiles : [];
    state.defaultProfile = bootstrap.defaultProfile;
    state.runtime = bootstrap.runtime;
    state.download = bootstrap.download;
    state.downloadDefaults = bootstrap.downloadDefaults || {};
    state.downloadCatalog = Array.isArray(bootstrap.downloadCatalog) ? bootstrap.downloadCatalog : [];
    state.launcherAddresses = Array.isArray(bootstrap.launcherAddresses)
      ? bootstrap.launcherAddresses
      : [];
    elements.appVersion.textContent = `ftllm ${bootstrap.version}`;
    elements.configPath.textContent = bootstrap.configPath || "—";
    const remembered = Number(window.localStorage.getItem("ftllm-launcher-profile"));
    state.currentIndex = Number.isInteger(remembered) && remembered >= 0 && remembered < state.profiles.length
      ? remembered
      : (state.profiles.length ? 0 : null);
    state.editingConfig = cloneConfig(
      state.currentIndex === null ? state.defaultProfile : state.profiles[state.currentIndex]
    );
    if (state.currentIndex === null) {
      state.editingConfig.port = defaultServicePort(state.editingConfig.command);
    }
    const initialLogs = bootstrap.logs || { entries: [], lastId: 0 };
    state.logs = Array.isArray(initialLogs.entries) ? initialLogs.entries : [];
    state.lastLogId = Number(initialLogs.lastId || 0);
    fillForm(state.editingConfig);
    renderDownloadCatalog();
    renderLauncherAddresses();
    fillDownloadForm(state.downloadDefaults);
    renderProfiles();
    renderRuntime();
    renderDownload();
    renderLogs();
    schedulePreview(0);
    scheduleDownloadPreview(0);
    window.setInterval(refreshRuntime, 700);
    window.setInterval(refreshDownload, 700);
    window.setInterval(refreshLogs, 700);
  } catch (error) {
    showToast(`启动器初始化失败：${friendlyError(error)}`, "error", 10000);
    elements.statusTitle.textContent = "启动器连接失败";
    elements.statusMessage.textContent = friendlyError(error);
    elements.statusDot.className = "status-dot failed";
  }
}

function renderLauncherAddresses() {
  elements.launcherAddressList.replaceChildren();
  if (!state.launcherAddresses.length) {
    const empty = document.createElement("span");
    empty.className = "launcher-address-empty";
    empty.textContent = "未检测到可用的 Launcher 访问地址。";
    elements.launcherAddressList.append(empty);
    return;
  }

  for (const address of state.launcherAddresses) {
    if (!address?.url) continue;
    const scope = ["local", "lan", "public"].includes(address.scope)
      ? address.scope
      : "custom";
    const button = document.createElement("button");
    button.type = "button";
    button.className = `launcher-address ${scope}`;
    button.title = "使用当前控制令牌打开该地址";

    const label = document.createElement("span");
    label.textContent = address.label || "访问地址";
    const url = document.createElement("code");
    url.textContent = address.url;
    button.append(label, url);
    button.addEventListener("click", () => {
      const destination = new URL(address.url, window.location.href);
      if (controlToken) destination.searchParams.set("token", controlToken);
      window.open(destination, "_blank", "noopener,noreferrer");
    });
    elements.launcherAddressList.append(button);
  }
}

function bindEvents() {
  document.addEventListener("click", handleDelegatedClick);
  elements.launchForm.addEventListener("input", handleFormChange);
  elements.launchForm.addEventListener("change", handleFormChange);
  elements.downloadForm.addEventListener("input", handleDownloadChange);
  elements.downloadForm.addEventListener("change", handleDownloadChange);
  elements.downloadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await startDownload();
  });
  elements.downloadPreset.addEventListener("change", selectDownloadPreset);
  elements.launchForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    await startRuntime();
  });
  elements.newProfile.addEventListener("click", newProfile);
  elements.newProfileHero.addEventListener("click", newProfile);
  elements.duplicateProfile.addEventListener("click", duplicateProfile);
  elements.deleteProfile.addEventListener("click", deleteProfile);
  elements.saveProfile.addEventListener("click", () => saveCurrentProfile(true));
  elements.startRuntimeTop.addEventListener("click", startRuntime);
  elements.startRuntimeHero.addEventListener("click", startRuntime);
  elements.stopRuntime.addEventListener("click", stopRuntime);
  elements.openEndpoint.addEventListener("click", openEndpoint);
  elements.copyEndpoint.addEventListener("click", copyEndpoint);
  elements.clearLogs.addEventListener("click", clearLogs);
  elements.refreshHardware.addEventListener("click", loadHardware);
  elements.shutdownLauncher.addEventListener("click", shutdownLauncher);
  elements.downloadCancel.addEventListener("click", cancelDownload);
  elements.downloadUseModel.addEventListener("click", useDownloadedModel);
  elements.downloadUseLast.addEventListener("click", useDownloadedModel);
}

function handleDelegatedClick(event) {
  const nav = event.target.closest("[data-view-button]");
  if (nav) {
    switchView(nav.dataset.viewButton);
    return;
  }
  const profileButton = event.target.closest("[data-profile-index]");
  if (profileButton) {
    selectProfile(Number(profileButton.dataset.profileIndex));
  }
}

function switchView(view) {
  state.currentView = view;
  for (const button of document.querySelectorAll("[data-view-button]")) {
    button.classList.toggle("active", button.dataset.viewButton === view);
  }
  for (const panel of document.querySelectorAll(".view")) {
    panel.classList.toggle("active", panel.id === `view-${view}`);
  }
  if (view === "logs") {
    document.querySelector('[data-view-button="logs"]').classList.remove("has-activity");
    scrollLogsToBottom();
  }
  if (view === "download") {
    document.querySelector('[data-view-button="download"]').classList.remove("has-activity");
  }
  if (view === "hardware" && !state.hardwareLoaded) loadHardware();
}

function cloneConfig(config) {
  return JSON.parse(JSON.stringify(config || {}));
}

function fillForm(config) {
  for (const input of elements.launchForm.querySelectorAll("[data-field]")) {
    const value = config?.[input.dataset.field];
    if (input.type === "checkbox") {
      input.checked = Boolean(value);
    } else {
      input.value = value === null || value === undefined ? "" : String(value);
    }
  }
  state.dirty = false;
  renderSaveState();
  updateConditionalFields();
}

function collectForm() {
  const config = { ...(state.editingConfig || {}), command: "server" };
  for (const input of elements.launchForm.querySelectorAll("[data-field]")) {
    config[input.dataset.field] = input.type === "checkbox" ? input.checked : input.value;
  }
  return config;
}

function handleFormChange(event) {
  if (!event.target.matches("[data-field]")) return;
  if (event.target.dataset.field === "command") {
    const oldCommand = state.editingConfig?.command || "server";
    const portInput = elements.launchForm.querySelector('[data-field="port"]');
    const oldDefaultPort = defaultServicePort(oldCommand);
    if (!portInput.value || portInput.value === oldDefaultPort) {
      portInput.value = defaultServicePort(event.target.value);
    }
  }
  state.editingConfig = collectForm();
  state.dirty = true;
  renderSaveState();
  updateConditionalFields();
  schedulePreview();
  if (event.target.matches("[data-path-input]")) {
    schedulePathSuggestions(event.target);
  }
}

function defaultServicePort(command) {
  const preferred = command === "webui" ? 1616 : 8080;
  const launcherPort = Number(window.location.port || (window.location.protocol === "https:" ? 443 : 80));
  return String(preferred === launcherPort ? preferred + 1 : preferred);
}

function updateConditionalFields() {
  const config = collectForm();
  const isWebui = config.command === "webui";
  elements.cudaDeviceField.classList.toggle("hidden", config.device !== "cuda");
  elements.tpDeviceField.classList.toggle("hidden", config.device !== "tp");
  elements.cudappDeviceField.classList.toggle("hidden", config.device !== "cudapp");
  elements.oriField.classList.toggle("hidden", !String(config.model || "").toLowerCase().endsWith(".gguf"));
  elements.moeDeviceField.classList.toggle("hidden", !config.enable_moe_hybrid);
  elements.moeLayersField.classList.toggle("hidden", !config.enable_moe_hybrid);
  elements.serverModelNameField.classList.toggle("hidden", isWebui);
  elements.serverHostField.classList.toggle("hidden", isWebui);
  elements.webuiMaxTokenField.classList.toggle("hidden", !isWebui);
  elements.webuiThinkField.classList.toggle("hidden", !isWebui);
  elements.serverContextField.classList.toggle("hidden", isWebui);
  elements.serverSamplingTitle.classList.toggle("hidden", isWebui);
  elements.serverSamplingFields.classList.toggle("hidden", isWebui);
  elements.serverApiKeyField.classList.toggle("hidden", isWebui);
  elements.serverHideInputField.classList.toggle("hidden", isWebui);
  elements.launchCommandKicker.textContent = isWebui ? "FTLLM WEBUI" : "FTLLM SERVER";
  elements.launchHeroEyebrow.textContent = isWebui ? "LOCAL CHAT WEBUI" : "LOCAL MODEL API";
  elements.launchHeroDescription.replaceChildren();
  elements.launchHeroDescription.append(
    "选择模型与设备，Launcher 会生成并托管 ",
    Object.assign(document.createElement("code"), { textContent: isWebui ? "ftllm webui" : "ftllm server" }),
    isWebui ? "，就绪后可直接打开聊天页面。" : "，启动进度和日志都留在这个页面。"
  );
  elements.endpointSummaryLabel.textContent = isWebui ? "WebUI 地址" : "API 地址";
}

function renderSaveState() {
  elements.saveState.className = "save-state";
  if (state.dirty) {
    elements.saveState.classList.add("dirty");
    elements.saveState.textContent = "未保存";
  } else if (state.currentIndex === null) {
    elements.saveState.textContent = "新配置";
  } else {
    elements.saveState.classList.add("saved");
    elements.saveState.textContent = "已保存";
  }
}

function renderProfiles() {
  elements.profileList.replaceChildren();
  elements.profileCount.textContent = String(state.profiles.length);
  if (!state.profiles.length) {
    const empty = document.createElement("div");
    empty.className = "empty-profile";
    empty.textContent = "还没有保存配置";
    elements.profileList.append(empty);
    return;
  }
  state.profiles.forEach((profile, index) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "profile-item";
    button.dataset.profileIndex = String(index);
    button.classList.toggle("active", index === state.currentIndex);

    const avatar = document.createElement("span");
    avatar.className = "profile-avatar";
    avatar.textContent = firstVisibleCharacter(profile.name || profile.model_name || "F");
    const copy = document.createElement("span");
    copy.className = "profile-copy";
    const title = document.createElement("strong");
    title.textContent = profile.name || profile.model_name || "未命名配置";
    const path = document.createElement("small");
    path.textContent = profile.model || "尚未设置模型";
    copy.append(title, path);
    button.append(avatar, copy);
    if (profileIsRunning(profile)) {
      const running = document.createElement("span");
      running.className = "profile-running";
      running.title = "正在运行";
      button.append(running);
    }
    elements.profileList.append(button);
  });
}

function firstVisibleCharacter(value) {
  return Array.from(String(value).trim())[0]?.toUpperCase() || "F";
}

function profileIsRunning(profile) {
  const runtime = state.runtime;
  if (!runtime?.pid) return false;
  return (
    (runtime.profileName && runtime.profileName === profile.name)
    || runtime.model === profile.model
  );
}

function selectProfile(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.profiles.length) return;
  if (state.dirty && !window.confirm("当前修改尚未保存，仍要切换配置吗？")) return;
  state.currentIndex = index;
  window.localStorage.setItem("ftllm-launcher-profile", String(index));
  state.editingConfig = cloneConfig(state.profiles[index]);
  fillForm(state.editingConfig);
  renderProfiles();
  schedulePreview(0);
}

function newProfile() {
  if (state.dirty && !window.confirm("当前修改尚未保存，仍要新建配置吗？")) return;
  state.currentIndex = null;
  state.editingConfig = cloneConfig(state.defaultProfile);
  state.editingConfig.port = defaultServicePort(state.editingConfig.command);
  const used = new Set(state.profiles.map((item) => item.name));
  let sequence = 1;
  while (used.has(`配置${sequence}`)) sequence += 1;
  state.editingConfig.name = `配置${sequence}`;
  fillForm(state.editingConfig);
  state.dirty = true;
  renderSaveState();
  renderProfiles();
  schedulePreview(0);
  document.querySelector('[data-field="name"]').focus();
}

function duplicateProfile() {
  const config = collectForm();
  state.currentIndex = null;
  state.editingConfig = cloneConfig(config);
  state.editingConfig.name = `${config.name || "配置"} 副本`;
  fillForm(state.editingConfig);
  state.dirty = true;
  renderSaveState();
  renderProfiles();
  schedulePreview(0);
}

async function deleteProfile() {
  if (state.currentIndex === null) {
    newProfile();
    return;
  }
  const profile = state.profiles[state.currentIndex];
  if (!window.confirm(`删除配置“${profile.name || "未命名配置"}”？不会删除模型文件。`)) return;
  try {
    const result = await request(`/api/profiles/${state.currentIndex}`, { method: "DELETE" });
    state.profiles = result.profiles;
    state.currentIndex = state.profiles.length ? Math.min(state.currentIndex, state.profiles.length - 1) : null;
    state.editingConfig = cloneConfig(
      state.currentIndex === null ? state.defaultProfile : state.profiles[state.currentIndex]
    );
    fillForm(state.editingConfig);
    renderProfiles();
    schedulePreview(0);
    showToast("配置已删除。", "success");
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}

async function saveCurrentProfile(showSuccess = false) {
  const config = collectForm();
  const result = await request("/api/profiles", {
    method: "POST",
    body: JSON.stringify({ index: state.currentIndex, config })
  });
  state.profiles = result.profiles;
  state.currentIndex = result.index;
  state.editingConfig = cloneConfig(result.profile);
  state.dirty = false;
  window.localStorage.setItem("ftllm-launcher-profile", String(result.index));
  renderProfiles();
  renderSaveState();
  if (showSuccess) showToast("启动配置已保存。", "success");
  return result.profile;
}

function schedulePreview(delay = 160) {
  window.clearTimeout(state.previewTimer);
  const requestId = ++state.previewRequestId;
  state.previewTimer = window.setTimeout(() => updatePreview(requestId), delay);
}

async function updatePreview(requestId) {
  try {
    const preview = await request("/api/preview", {
      method: "POST",
      body: JSON.stringify(collectForm())
    });
    if (requestId !== state.previewRequestId) return;
    state.preview = preview;
    elements.commandPreview.textContent = preview.command || "请先完成有效配置。";
    renderLaunchValidation(preview.errors || []);
    if (!state.runtime?.pid) elements.endpointSummary.textContent = preview.endpoint || "—";
    updateActionAvailability();
  } catch (error) {
    if (requestId !== state.previewRequestId) return;
    state.preview = { errors: [friendlyError(error)] };
    elements.commandPreview.textContent = "无法生成命令。";
    renderLaunchValidation(state.preview.errors);
    updateActionAvailability();
  }
}

function renderValidation(container, errors, successMessage) {
  container.replaceChildren();
  const messages = errors.length ? errors : [successMessage];
  for (const message of messages) {
    const item = document.createElement("div");
    item.className = errors.length ? "validation-message" : "validation-message ok";
    item.textContent = message;
    container.append(item);
  }
}

function renderLaunchValidation(errors) {
  renderValidation(elements.validationMessages, errors, "配置检查通过，可以启动。");
}

function renderDownloadValidation(errors) {
  renderValidation(elements.downloadValidation, errors, "下载配置检查通过，可以开始下载。");
}

async function startRuntime() {
  try {
    const preview = await request("/api/preview", {
      method: "POST",
      body: JSON.stringify(collectForm())
    });
    state.preview = preview;
    renderLaunchValidation(preview.errors || []);
    if (preview.errors?.length) {
      showToast("请先修正配置错误。", "error");
      return;
    }
    const config = await saveCurrentProfile(false);
    state.runtime = await request("/api/runtime/start", {
      method: "POST",
      body: JSON.stringify(config)
    });
    renderRuntime();
    switchView("launch");
    showToast("模型服务已开始启动。", "success");
  } catch (error) {
    showToast(friendlyError(error), "error", 7000);
    await refreshRuntime();
  }
}

async function stopRuntime() {
  try {
    state.runtime = await request("/api/runtime/stop", { method: "POST" });
    renderRuntime();
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}

async function refreshRuntime() {
  if (state.pollingRuntime) return;
  state.pollingRuntime = true;
  try {
    state.runtime = await request("/api/runtime");
    renderRuntime();
  } catch (_error) {
    // The launcher may be shutting down; avoid a repeating toast.
  } finally {
    state.pollingRuntime = false;
  }
}

function renderRuntime() {
  const runtime = state.runtime || { phase: "stopped", message: "尚未启动" };
  const phase = runtime.phase || "stopped";
  const isWebui = runtime.command === "webui";
  const titles = {
    stopped: "服务未启动",
    starting: isWebui ? "正在启动聊天 WebUI" : "正在启动模型",
    running: isWebui ? "聊天 WebUI 运行中" : "本地 API 运行中",
    stopping: "正在停止服务",
    failed: isWebui ? "WebUI 启动异常" : "模型服务异常"
  };
  const summaries = {
    stopped: "未启动",
    starting: "启动中",
    running: "运行中",
    stopping: "停止中",
    failed: "启动失败"
  };
  elements.statusDot.className = `status-dot ${phase}`;
  elements.statusTitle.textContent = titles[phase] || phase;
  elements.statusMessage.textContent = runtime.message || "—";
  elements.runtimeSummary.textContent = summaries[phase] || phase;
  if (runtime.endpoint) elements.endpointSummary.textContent = runtime.endpoint;
  const active = runtimeIsActive(runtime);
  const showProgress = active || phase === "failed";
  elements.runtimeProgressCard.classList.toggle("hidden", !showProgress);
  elements.runtimeProgressTitle.textContent = titles[phase] || phase;
  elements.runtimeProgressLabel.textContent = runtime.progressLabel || runtime.message || "—";
  const percent = Math.max(0, Math.min(100, Number(runtime.progress || 0)));
  elements.runtimeProgressValue.textContent = runtime.progressIndeterminate ? "处理中" : `${Math.round(percent)}%`;
  elements.runtimeProgress.classList.toggle("indeterminate", Boolean(runtime.progressIndeterminate));
  if (runtime.progressIndeterminate) {
    elements.runtimeProgress.removeAttribute("value");
  } else {
    elements.runtimeProgress.value = percent;
  }
  elements.runtimePid.textContent = runtime.pid || "—";
  elements.runtimeModel.textContent = runtime.modelName || basename(runtime.model) || "—";
  elements.runtimeProgressIcon.textContent = phase === "running" ? "✓" : (phase === "failed" ? "!" : "▷");
  elements.stopRuntime.classList.toggle("hidden", !active);
  elements.stopRuntime.disabled = phase === "stopping";
  elements.openEndpoint.disabled = !runtime.ready;
  elements.openEndpoint.textContent = isWebui ? "打开 WebUI" : "打开 API 文档";
  renderProfiles();
  updateActionAvailability();
}

function updateActionAvailability() {
  const active = runtimeIsActive(state.runtime);
  const invalid = Boolean(state.preview?.errors?.length);
  for (const button of [elements.startRuntime, elements.startRuntimeTop, elements.startRuntimeHero]) {
    button.disabled = active || invalid;
  }
}

function runtimeIsActive(runtime) {
  return Boolean(runtime?.pid) || ACTIVE_RUNTIME_PHASES.has(runtime?.phase);
}

function basename(value) {
  const parts = String(value || "").replace(/\\/g, "/").split("/").filter(Boolean);
  return parts[parts.length - 1] || "";
}

function openEndpoint() {
  if (!state.runtime?.endpoint || !state.runtime.ready) return;
  const endpoint = state.runtime.endpoint.replace(/\/$/, "");
  const url = state.runtime.command === "webui" ? endpoint : `${endpoint}/docs`;
  window.open(url, "_blank", "noopener,noreferrer");
}

async function copyEndpoint() {
  const endpoint = state.runtime?.endpoint || state.preview?.endpoint;
  if (!endpoint) return;
  try {
    await navigator.clipboard.writeText(endpoint);
    showToast("API 地址已复制。", "success");
  } catch (_error) {
    showToast("浏览器未允许复制，请手动复制顶部 API 地址。", "error");
  }
}

function renderDownloadCatalog() {
  elements.downloadPreset.replaceChildren();
  for (const group of state.downloadCatalog) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = group.label || group.id || "模型";
    for (const model of group.models || []) {
      const option = document.createElement("option");
      option.value = model.id;
      option.textContent = model.label || model.id;
      optgroup.append(option);
    }
    if (optgroup.children.length) elements.downloadPreset.append(optgroup);
  }
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = "自定义模型 ID";
  elements.downloadPreset.append(custom);
}

function fillDownloadForm(config) {
  for (const input of elements.downloadForm.querySelectorAll("[data-download-field]")) {
    const value = config?.[input.dataset.downloadField];
    input.value = value === null || value === undefined ? "" : String(value);
  }
  const modelId = String(config?.modelId || "");
  elements.downloadPreset.value = downloadCatalogHas(modelId) ? modelId : "custom";
  state.downloadTargetAutomatic = true;
}

function collectDownloadForm() {
  const config = {};
  for (const input of elements.downloadForm.querySelectorAll("[data-download-field]")) {
    config[input.dataset.downloadField] = input.value;
  }
  return config;
}

function downloadCatalogHas(modelId) {
  return state.downloadCatalog.some((group) =>
    (group.models || []).some((model) => model.id === modelId)
  );
}

function downloadTargetFor(modelId) {
  const template = String(state.downloadDefaults?.targetDir || "");
  const templateModel = basename(state.downloadDefaults?.modelId || "");
  const slash = Math.max(template.lastIndexOf("/"), template.lastIndexOf("\\"));
  const root = slash >= 0 ? template.slice(0, slash) : template;
  const separator = template.includes("\\") && !template.includes("/") ? "\\" : "/";
  const name = basename(modelId) || templateModel || "model";
  return root ? `${root}${separator}${name}` : name;
}

function selectDownloadPreset() {
  if (elements.downloadPreset.value === "custom") {
    elements.downloadForm.querySelector('[data-download-field="modelId"]').focus();
    return;
  }
  const modelInput = elements.downloadForm.querySelector('[data-download-field="modelId"]');
  const targetInput = elements.downloadForm.querySelector('[data-download-field="targetDir"]');
  modelInput.value = elements.downloadPreset.value;
  if (state.downloadTargetAutomatic) targetInput.value = downloadTargetFor(modelInput.value);
  scheduleDownloadPreview(0);
}

function handleDownloadChange(event) {
  if (!event.target.matches("[data-download-field]")) return;
  const field = event.target.dataset.downloadField;
  if (field === "modelId") {
    elements.downloadPreset.value = downloadCatalogHas(event.target.value) ? event.target.value : "custom";
    if (state.downloadTargetAutomatic) {
      const targetInput = elements.downloadForm.querySelector('[data-download-field="targetDir"]');
      targetInput.value = downloadTargetFor(event.target.value);
    }
  } else if (field === "targetDir") {
    state.downloadTargetAutomatic = false;
  }
  scheduleDownloadPreview();
  if (event.target.matches("[data-path-input]")) schedulePathSuggestions(event.target);
}

function scheduleDownloadPreview(delay = 160) {
  window.clearTimeout(state.downloadPreviewTimer);
  const requestId = ++state.downloadPreviewRequestId;
  state.downloadPreviewTimer = window.setTimeout(
    () => updateDownloadPreview(requestId),
    delay
  );
}

async function updateDownloadPreview(requestId) {
  try {
    const preview = await request("/api/download/preview", {
      method: "POST",
      body: JSON.stringify(collectDownloadForm())
    });
    if (requestId !== state.downloadPreviewRequestId) return;
    state.downloadPreview = preview;
    elements.downloadCommand.textContent = state.downloadPreview.command || "请先完成有效配置。";
    renderDownloadValidation(state.downloadPreview.errors || []);
  } catch (error) {
    if (requestId !== state.downloadPreviewRequestId) return;
    state.downloadPreview = { errors: [friendlyError(error)] };
    elements.downloadCommand.textContent = "无法生成下载命令。";
    renderDownloadValidation(state.downloadPreview.errors);
  }
  renderDownload();
}

async function startDownload() {
  try {
    const preview = await request("/api/download/preview", {
      method: "POST",
      body: JSON.stringify(collectDownloadForm())
    });
    state.downloadPreview = preview;
    elements.downloadCommand.textContent = preview.command || "请先完成有效配置。";
    renderDownloadValidation(preview.errors || []);
    if (preview.errors?.length) {
      showToast("请先修正下载配置错误。", "error");
      return;
    }
    state.download = await request("/api/download/start", {
      method: "POST",
      body: JSON.stringify(collectDownloadForm())
    });
    renderDownload();
    switchView("download");
    showToast("模型下载已开始。", "success");
  } catch (error) {
    showToast(friendlyError(error), "error", 7000);
    await refreshDownload();
  }
}

async function cancelDownload() {
  try {
    state.download = await request("/api/download/cancel", { method: "POST" });
    renderDownload();
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}

async function refreshDownload() {
  if (state.pollingDownload) return;
  state.pollingDownload = true;
  const previousPhase = state.download?.phase;
  try {
    state.download = await request("/api/download");
    if (state.download.phase !== previousPhase) {
      if (state.download.phase === "completed") {
        showToast("模型下载完成，可以直接用于启动。", "success", 6000);
        if (state.currentView !== "download") {
          document.querySelector('[data-view-button="download"]').classList.add("has-activity");
        }
      } else if (state.download.phase === "failed") {
        showToast(state.download.message || "模型下载失败。", "error", 7000);
      }
    }
    renderDownload();
  } catch (_error) {
    // Ignore transient polling errors while the launcher exits.
  } finally {
    state.pollingDownload = false;
  }
}

function renderDownload() {
  const download = state.download || { phase: "idle", progress: 0, message: "尚未开始下载" };
  const phase = download.phase || "idle";
  const active = ACTIVE_DOWNLOAD_PHASES.has(phase);
  const titles = {
    idle: "尚未开始下载",
    starting: "正在启动下载",
    downloading: "正在下载模型",
    cancelling: "正在取消下载",
    cancelled: "下载已取消",
    completed: "模型下载完成",
    failed: "模型下载失败"
  };
  const badges = {
    idle: "未开始", starting: "连接中", downloading: "下载中", cancelling: "取消中",
    cancelled: "已取消", completed: "已完成", failed: "失败"
  };
  elements.downloadBadge.className = `save-state ${phase === "completed" ? "saved" : phase}`;
  elements.downloadBadge.textContent = badges[phase] || phase;
  elements.downloadStatusTitle.textContent = titles[phase] || phase;
  elements.downloadStatusMessage.textContent = download.message || "—";
  elements.downloadStatusIcon.textContent = phase === "completed" ? "✓" : (phase === "failed" ? "!" : "↓");
  const percent = Math.max(0, Math.min(100, Number(download.progress || 0)));
  elements.downloadProgressValue.textContent = download.progressIndeterminate
    ? "处理中"
    : `${Math.round(percent * 10) / 10}%`;
  if (download.progressIndeterminate) {
    elements.downloadProgress.removeAttribute("value");
    elements.downloadProgress.classList.add("indeterminate");
  } else {
    elements.downloadProgress.value = percent;
    elements.downloadProgress.classList.remove("indeterminate");
  }
  elements.downloadBytes.textContent = download.totalBytes
    ? `${formatBytes(download.downloadedBytes)} / ${formatBytes(download.totalBytes)}`
    : (download.downloadedBytes ? formatBytes(download.downloadedBytes) : "—");
  elements.downloadFiles.textContent = download.totalFiles
    ? `${download.completedFiles || 0} / ${download.totalFiles}`
    : "—";
  elements.downloadDestination.textContent = download.destination || collectDownloadForm().targetDir || "—";
  elements.downloadStart.disabled = active || Boolean(state.downloadPreview?.errors?.length);
  elements.downloadCancel.classList.toggle("hidden", !active);
  elements.downloadCancel.disabled = phase === "cancelling";
  const completed = phase === "completed" && Boolean(download.destination);
  elements.downloadUseModel.classList.toggle("hidden", !completed);
  elements.downloadUseLast.classList.toggle("hidden", !completed);
  const nav = document.querySelector('[data-view-button="download"]');
  if (active) nav.classList.add("has-activity");
  else if (state.currentView === "download") nav.classList.remove("has-activity");
}

function useDownloadedModel() {
  const destination = state.download?.destination;
  if (!destination) return;
  const modelInput = elements.launchForm.querySelector('[data-field="model"]');
  modelInput.value = destination;
  state.editingConfig = collectForm();
  state.dirty = true;
  renderSaveState();
  updateConditionalFields();
  schedulePreview(0);
  switchView("launch");
  modelInput.focus();
  showToast("已将下载目录填入当前启动配置。", "success");
}

async function refreshLogs() {
  if (state.pollingLogs) return;
  state.pollingLogs = true;
  try {
    const result = await request(`/api/logs?since=${state.lastLogId}`);
    const entries = Array.isArray(result.entries) ? result.entries : [];
    if (entries.length) {
      const wasNearBottom = isLogNearBottom();
      state.logs.push(...entries);
      if (state.logs.length > 1500) state.logs.splice(0, state.logs.length - 1500);
      state.lastLogId = Number(result.lastId || state.lastLogId);
      renderLogs(wasNearBottom);
      if (state.currentView !== "logs") {
        document.querySelector('[data-view-button="logs"]').classList.add("has-activity");
      }
    }
  } catch (_error) {
    // Ignore transient polling errors while the launcher exits.
  } finally {
    state.pollingLogs = false;
  }
}

function renderLogs(forceBottom = true) {
  elements.logOutput.replaceChildren();
  elements.logCount.textContent = `${state.logs.length} 条`;
  if (!state.logs.length) {
    const empty = document.createElement("div");
    empty.className = "log-empty";
    empty.textContent = "启动模型后，日志会显示在这里。";
    elements.logOutput.append(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const entry of state.logs) {
    const line = document.createElement("div");
    line.className = `log-line ${entry.level || "info"}`;
    const timestamp = document.createElement("span");
    timestamp.className = "log-time";
    timestamp.textContent = formatTime(entry.timestamp);
    const source = document.createElement("span");
    source.className = "log-source";
    source.textContent = entry.source || "ftllm";
    const message = document.createElement("span");
    message.className = "log-message";
    message.textContent = entry.message || "";
    line.append(timestamp, source, message);
    fragment.append(line);
  }
  elements.logOutput.append(fragment);
  if (forceBottom) scrollLogsToBottom();
}

function formatTime(timestamp) {
  const date = new Date(Number(timestamp) * 1000);
  return Number.isNaN(date.getTime()) ? "--:--:--" : date.toLocaleTimeString("zh-CN", { hour12: false });
}

function isLogNearBottom() {
  return elements.logOutput.scrollHeight - elements.logOutput.scrollTop - elements.logOutput.clientHeight < 70;
}

function scrollLogsToBottom() {
  window.requestAnimationFrame(() => {
    elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
  });
}

async function clearLogs() {
  try {
    await request("/api/logs", { method: "DELETE" });
    state.logs = [];
    renderLogs();
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}

function schedulePathSuggestions(input) {
  window.clearTimeout(state.pathTimer);
  const requestId = ++state.pathRequestId;
  state.pathTimer = window.setTimeout(async () => {
    const prefix = input.value || "";
    if (!prefix) {
      elements.pathSuggestions.replaceChildren();
      return;
    }
    try {
      const pathQuery = new URLSearchParams({
        prefix,
        directories_only: String(input.hasAttribute("data-directories-only"))
      });
      const result = await request(`/api/paths?${pathQuery}`);
      if (requestId !== state.pathRequestId || input.value !== prefix) return;
      elements.pathSuggestions.replaceChildren();
      for (const value of result.paths || []) {
        const option = document.createElement("option");
        option.value = value;
        elements.pathSuggestions.append(option);
      }
    } catch (_error) {
      // Path completion is optional.
    }
  }, 180);
}

async function loadHardware() {
  elements.refreshHardware.disabled = true;
  elements.hardwareStatus.textContent = "正在读取硬件信息…";
  try {
    const modelPath = collectForm().model || "";
    const report = await request(`/api/hardware?model_path=${encodeURIComponent(modelPath)}`);
    state.hardwareLoaded = true;
    renderHardware(report);
    elements.hardwareStatus.textContent = `检测完成 · ${report.platform} · Python ${report.python}`;
  } catch (error) {
    elements.hardwareStatus.textContent = `检测失败：${friendlyError(error)}`;
  } finally {
    elements.refreshHardware.disabled = false;
  }
}

function renderHardware(report) {
  elements.hardwareGrid.replaceChildren();
  elements.hardwareGrid.append(
    hardwareCard("CPU", "C", report.cpu?.model || "未知 CPU", [
      ["逻辑线程", String(report.cpu?.logical || "—")],
      ["当前可用", String(report.cpu?.available || "—")],
      ["NUMA 节点", String(report.numa?.length || 0)]
    ]),
    hardwareCard("内存", "M", "系统内存", [
      ["总容量", formatBytes(report.memory?.total)],
      ["当前可用", formatBytes(report.memory?.available)],
      ["可用比例", formatRatio(report.memory?.available, report.memory?.total)]
    ]),
    gpuHardwareCard(report.gpus || []),
    hardwareCard("存储与构建", "D", report.disk?.path || "模型所在磁盘", [
      ["磁盘容量", formatBytes(report.disk?.total)],
      ["磁盘可用", formatBytes(report.disk?.free)],
      ["CUDA 构建", report.build?.USE_CUDA ? "已启用" : "未启用"],
      ["ROCm 构建", report.build?.USE_ROCM ? "已启用" : "未启用"],
      ["NUMA 构建", report.build?.USE_NUMAS ? "已启用" : "未启用"]
    ])
  );
}

function hardwareCard(title, icon, subtitle, rows) {
  const card = document.createElement("article");
  card.className = "hardware-card";
  const heading = document.createElement("div");
  heading.className = "hardware-card-heading";
  const badge = document.createElement("span");
  badge.className = "hardware-card-icon";
  badge.textContent = icon;
  const copy = document.createElement("div");
  const strong = document.createElement("strong");
  strong.textContent = title;
  const small = document.createElement("small");
  small.textContent = subtitle;
  copy.append(strong, small);
  heading.append(badge, copy);
  const body = document.createElement("div");
  body.className = "hardware-rows";
  for (const [label, value] of rows) {
    const row = document.createElement("div");
    row.className = "hardware-row";
    const key = document.createElement("span");
    key.textContent = label;
    const data = document.createElement("strong");
    data.textContent = value;
    row.append(key, data);
    body.append(row);
  }
  card.append(heading, body);
  return card;
}

function gpuHardwareCard(gpus) {
  const card = document.createElement("article");
  card.className = "hardware-card wide";
  const heading = document.createElement("div");
  heading.className = "hardware-card-heading";
  const icon = document.createElement("span");
  icon.className = "hardware-card-icon";
  icon.textContent = "G";
  const copy = document.createElement("div");
  const title = document.createElement("strong");
  title.textContent = "GPU";
  const subtitle = document.createElement("small");
  subtitle.textContent = gpus.length ? `${gpus.length} 张 NVIDIA GPU` : "未检测到 nvidia-smi 或 NVIDIA GPU";
  copy.append(title, subtitle);
  heading.append(icon, copy);
  const list = document.createElement("div");
  list.className = "gpu-list";
  if (!gpus.length) {
    const empty = document.createElement("div");
    empty.className = "gpu-item";
    empty.textContent = "未通过 nvidia-smi 检测到 NVIDIA GPU；其他设备请按实际构建配置选择。";
    list.append(empty);
  }
  for (const gpu of gpus) {
    const item = document.createElement("div");
    item.className = "gpu-item";
    const info = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = `GPU ${gpu.index} · ${gpu.name}`;
    const detail = document.createElement("small");
    detail.textContent = `${gpu.memoryFreeMiB} / ${gpu.memoryTotalMiB} MiB 可用 · 驱动 ${gpu.driver}`;
    info.append(name, detail);
    const health = document.createElement("span");
    health.textContent = `${gpu.utilization}% · ${gpu.temperature}℃`;
    item.append(info, health);
    list.append(item);
  }
  card.append(heading, list);
  return card;
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!bytes) return "未知";
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** index)).toFixed(index >= 3 ? 1 : 0)} ${units[index]}`;
}

function formatRatio(value, total) {
  const numerator = Number(value || 0);
  const denominator = Number(total || 0);
  return denominator ? `${Math.round(numerator * 100 / denominator)}%` : "未知";
}

async function shutdownLauncher() {
  if (!window.confirm("退出 Launcher？正在运行的模型服务和下载任务也会停止。")) return;
  try {
    await request("/api/shutdown", { method: "POST" });
    showToast("Launcher 正在退出…", "success", 5000);
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}

function showToast(message, tone = "", duration = 3500) {
  const toast = document.createElement("div");
  toast.className = `toast ${tone}`.trim();
  toast.textContent = message;
  elements.toastRegion.append(toast);
  window.setTimeout(() => toast.remove(), duration);
}

function friendlyError(error) {
  return String(error?.message || error || "未知错误");
}
