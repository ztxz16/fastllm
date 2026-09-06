const locationQuery = new URLSearchParams(window.location.search);
const queryToken = locationQuery.get("token") || "";
if (queryToken) {
  window.sessionStorage.setItem("ftllm-launcher-token", queryToken);
  window.history.replaceState({}, "", window.location.pathname);
}
const controlToken = queryToken || window.sessionStorage.getItem("ftllm-launcher-token") || "";
const ACTIVE_RUNTIME_PHASES = new Set(["starting", "running", "stopping"]);
const ACTIVE_DOWNLOAD_PHASES = new Set(["starting", "downloading", "cancelling"]);
const DEFAULT_LOCALE = "zh-CN";
const SUPPORTED_LOCALES = new Set(["zh-CN", "en-US"]);
const LOCALE_STORAGE_KEY = "ftllm-launcher-locale";
const WEBUI_LOAD_TIMEOUT_MS = 30000;
const AUTOMATIC_CONFIGURATION_DEFAULTS = Object.freeze({
  device: "auto",
  cuda_device_id: "0",
  tp: "2",
  cudapp: "2",
  dtype: "auto",
  threads: "auto",
  gpu_mem_ratio: "0.9",
  max_batch: "auto",
  chunked_prefill_size: "auto",
  kv_cache_dtype: "auto",
  kv_cache_limit: "auto",
  tokens: "auto",
  enable_moe_hybrid: false,
  moe_device: "numa",
  moe_device_layers: "-1",
  moe_device_custom: "",
  moe_dtype: "auto",
  moe_atype: "auto",
  ngram_device: "auto",
  speculative_algorithm: "auto",
  speculative_draft_model_path: "",
  mtp: "auto",
  draft_tokens: "auto"
});
const AUTOMATIC_CONFIGURATION_FIELDS = new Set(Object.keys(AUTOMATIC_CONFIGURATION_DEFAULTS));
// The HTML template is the Chinese fallback; JavaScript and backend messages use
// English message IDs. Locale resources provide the corresponding translations.
const localeCache = new Map();
const capturedStaticText = [];
const capturedStaticAttributes = [];
let profileRenderSignature = "";
let webuiModulePromise;
let webuiModuleRetries = 0;
let webuiModuleLoaded = false;

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
  profileQuery: "",
  profileFilter: "all",
  webuiSessionId: "",
  webuiLoading: false,
  webuiError: "",
  webuiRequestId: 0,
  webuiLoadTimer: null,
  webuiAbortController: null,
  webuiComponent: null,
  locale: DEFAULT_LOCALE,
  staticMessages: {},
  messages: {},
  messagePatterns: [],
  hardwareLoaded: false,
  hardwareReport: null,
  hardwareStatus: "idle",
  hardwareError: "",
  pollingRuntime: false,
  pollingDownload: false,
  pollingLogs: false,
  previewTimer: null,
  previewRequestId: 0,
  automaticConfigTimer: null,
  automaticConfigRequestId: 0,
  automaticConfigPending: false,
  automaticConfigAppliedModel: "",
  automaticConfigStatus: { phase: "idle" },
  downloadPreviewTimer: null,
  downloadPreviewRequestId: 0,
  pathTimer: null,
  pathRequestId: 0,
  folderPickerRequestId: 0,
  folderPickerLoading: false,
  folderPickerResult: null,
  folderPickerError: "",
  confirmationResolve: null,
  confirmationRestoreFocus: null
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
    throw new Error(localizeServerText(payload.error) || t("Request failed (HTTP {status}).", { status: response.status }));
  }
  return payload;
}

function cacheElements() {
  const ids = [
    "app-version", "shutdown-launcher", "status-dot", "status-title", "status-message",
    "open-endpoint", "stop-runtime", "profile-count", "profile-list", "new-profile",
    "current-view-title", "profile-search", "profile-results",
    "open-webui", "webui-placeholder", "webui-status",
    "webui-content", "webui-retry",
    "config-path", "profile-editor-modal", "profile-editor-title", "launch-form",
    "close-profile-editor", "save-state", "ori-field", "auto-configure-profile",
    "clear-profile-config", "automatic-config-status",
    "cuda-device-field", "tp-device-field", "cudapp-device-field", "moe-device-field",
    "moe-device-custom-field", "moe-layers-field", "server-model-name-field", "server-host-field",
    "webui-max-token-field", "webui-think-field",
    "server-sampling-title", "server-sampling-fields", "server-api-key-field",
    "server-hide-input-field", "launch-command-kicker", "command-preview",
    "validation-messages", "save-profile",
    "start-runtime", "clear-logs", "log-count", "log-output", "log-activity",
    "refresh-hardware", "hardware-status", "hardware-grid", "path-suggestions",
    "choose-model-folder", "folder-picker-modal", "folder-picker-title",
    "folder-picker-close", "folder-picker-current", "folder-picker-up",
    "folder-picker-list", "folder-picker-status", "folder-picker-cancel",
    "folder-picker-select",
    "confirmation-modal", "confirmation-icon", "confirmation-kicker",
    "confirmation-title", "confirmation-message", "confirmation-cancel",
    "confirmation-confirm",
    "toast-region", "download-activity", "download-form", "download-preset",
    "download-badge", "download-command", "download-validation", "download-start",
    "download-cancel", "download-use-model", "download-use-last", "download-status-icon",
    "download-status-title", "download-status-message", "download-progress-value",
    "download-progress", "download-bytes", "download-files", "download-destination",
    "launcher-address-list", "language-select"
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
  captureStaticMessages();
  await initializeLocale();
  bindEvents();
  if (!controlToken) {
    showToast(t("Missing control token. Reopen the URL printed by ftllm launch."), "error", 10000);
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
    state.currentIndex = null;
    state.editingConfig = cloneConfig(state.defaultProfile);
    state.editingConfig.port = defaultServicePort(state.editingConfig.command);
    const initialLogs = bootstrap.logs || { entries: [], lastId: 0 };
    state.logs = Array.isArray(initialLogs.entries) ? initialLogs.entries : [];
    state.lastLogId = Number(initialLogs.lastId || 0);
    fillForm(state.editingConfig);
    renderDownloadCatalog();
    renderLauncherAddresses();
    fillDownloadForm(state.downloadDefaults);
    renderRuntime();
    renderDownload();
    renderLogs();
    schedulePreview(0);
    scheduleDownloadPreview(0);
    window.setInterval(refreshRuntime, 700);
    window.setInterval(refreshDownload, 700);
    window.setInterval(refreshLogs, 700);
  } catch (error) {
    showToast(t("Launcher initialization failed: {error}", { error: friendlyError(error) }), "error", 10000);
    elements.statusTitle.textContent = t("Launcher connection failed");
    elements.statusMessage.textContent = friendlyError(error);
    elements.statusDot.className = "status-dot failed";
  }
}

function normalizeLocale(locale) {
  const value = String(locale || "").toLowerCase();
  return value.startsWith("zh") ? "zh-CN" : "en-US";
}

function preferredLocale() {
  const saved = window.localStorage.getItem(LOCALE_STORAGE_KEY);
  if (SUPPORTED_LOCALES.has(saved)) return saved;
  const browserLocales = Array.isArray(navigator.languages) && navigator.languages.length
    ? navigator.languages
    : [navigator.language];
  for (const locale of browserLocales) {
    const language = String(locale || "").toLowerCase();
    if (language.startsWith("zh")) return "zh-CN";
    if (language.startsWith("en")) return "en-US";
  }
  return "en-US";
}

function captureStaticMessages() {
  const walker = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
  while (walker.nextNode()) {
    const node = walker.currentNode;
    if (!node.nodeValue?.trim()) continue;
    const parentTag = node.parentElement?.tagName;
    if (parentTag === "SCRIPT" || parentTag === "STYLE") continue;
    const match = node.nodeValue.match(/^(\s*)([\s\S]*?)(\s*)$/);
    capturedStaticText.push({
      node,
      prefix: match?.[1] || "",
      source: match?.[2] || node.nodeValue,
      suffix: match?.[3] || ""
    });
  }
  for (const element of document.body.querySelectorAll("*")) {
    for (const name of ["aria-label", "placeholder", "title"]) {
      if (element.hasAttribute(name)) {
        capturedStaticAttributes.push({ element, name, source: element.getAttribute(name) });
      }
    }
  }
}

async function fetchLocale(locale) {
  if (localeCache.has(locale)) return localeCache.get(locale);
  const response = await fetch(`/assets/locales/${encodeURIComponent(locale)}.json`);
  if (!response.ok) throw new Error(`Unable to load locale ${locale} (HTTP ${response.status}).`);
  const resource = await response.json();
  if (!resource || resource.locale !== locale || typeof resource.messages !== "object") {
    throw new Error(`Invalid locale resource: ${locale}.`);
  }
  localeCache.set(locale, resource);
  return resource;
}

async function initializeLocale() {
  const preferred = preferredLocale();
  try {
    await activateLocale(preferred);
  } catch (error) {
    console.warn(`Failed to load ${preferred} locale:`, error);
    if (preferred !== DEFAULT_LOCALE) {
      try {
        await activateLocale(DEFAULT_LOCALE);
        return;
      } catch (fallbackError) {
        console.warn(`Failed to load ${DEFAULT_LOCALE} locale:`, fallbackError);
      }
    }
    applyStaticTranslations();
  }
}

async function activateLocale(locale) {
  const normalized = normalizeLocale(locale);
  const resource = await fetchLocale(normalized);
  state.locale = normalized;
  state.staticMessages = resource.static || {};
  state.messages = resource.messages || {};
  state.messagePatterns = (resource.patterns || []).flatMap((pattern) => {
    try {
      return [{ expression: new RegExp(pattern.source), target: pattern.target }];
    } catch (_error) {
      return [];
    }
  });
  document.documentElement.lang = normalized;
  elements.languageSelect.value = normalized;
  applyStaticTranslations();
}

function applyStaticTranslations() {
  for (const item of capturedStaticText) {
    if (!item.node.isConnected) continue;
    const translated = state.staticMessages[item.source] ?? item.source;
    item.node.nodeValue = `${item.prefix}${translated}${item.suffix}`;
  }
  for (const item of capturedStaticAttributes) {
    if (!item.element.isConnected) continue;
    item.element.setAttribute(item.name, state.staticMessages[item.source] ?? item.source);
  }
}

function interpolate(message, values = {}) {
  return String(message).replace(/\{([a-zA-Z0-9_]+)\}/g, (match, name) => (
    Object.prototype.hasOwnProperty.call(values, name) ? String(values[name]) : match
  ));
}

function t(message, values = {}) {
  return interpolate(state.messages[message] ?? message, values);
}

function localizeDisplayCommand(command) {
  return String(command || "").replace(
    "[Token injected through environment]",
    `[${t("Token injected through environment")}]`
  );
}

function localizeServerText(value) {
  const source = String(value ?? "");
  if (source.includes("\n")) {
    return source.split("\n").map((line) => localizeServerText(line)).join("\n");
  }
  if (Object.prototype.hasOwnProperty.call(state.messages, source)) return state.messages[source];
  for (const pattern of state.messagePatterns) {
    if (pattern.expression.test(source)) return source.replace(pattern.expression, pattern.target);
  }
  return source;
}

async function changeLocale(locale) {
  const previous = state.locale;
  elements.languageSelect.disabled = true;
  try {
    await activateLocale(locale);
    window.localStorage.setItem(LOCALE_STORAGE_KEY, state.locale);
    renderLocalizedContent();
  } catch (error) {
    elements.languageSelect.value = previous;
    showToast(t("Unable to switch language: {error}", { error: String(error?.message || error) }), "error");
  } finally {
    elements.languageSelect.disabled = false;
  }
}

function renderLocalizedContent() {
  state.webuiComponent?.setLocale(state.locale);
  renderViewTitle();
  updateConditionalFields();
  renderAutomaticConfigurationStatus();
  renderProfileEditorTitle();
  renderSaveState();
  renderRuntime();
  renderLauncherAddresses();
  renderDownloadCatalog();
  renderDownload();
  renderLogs(false);
  if (state.preview) renderLaunchValidation(state.preview.errors || []);
  if (state.downloadPreview) renderDownloadValidation(state.downloadPreview.errors || []);
  renderHardwareStatus();
  if (state.hardwareReport) renderHardware(state.hardwareReport);
  if (!elements.folderPickerModal.classList.contains("hidden")) renderFolderPicker();
}

function renderLauncherAddresses() {
  elements.launcherAddressList.replaceChildren();
  if (!state.launcherAddresses.length) {
    const empty = document.createElement("span");
    empty.className = "launcher-address-empty";
    empty.textContent = t("No Launcher access address was detected.");
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
    button.title = t("Open this address with the current control token");

    const label = document.createElement("span");
    const labels = {
      local: t("Local address"),
      lan: t("LAN address"),
      public: t("Public address"),
      custom: t("Access address")
    };
    label.textContent = labels[scope];
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
  elements.webuiRetry.addEventListener("click", () => {
    state.webuiError = "";
    openEmbeddedWebUI();
  });
  elements.profileSearch.addEventListener("input", (event) => {
    state.profileQuery = event.target.value;
    renderProfiles();
  });
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
  elements.autoConfigureProfile.addEventListener("click", () => configureProfileAutomatically());
  elements.clearProfileConfig.addEventListener("click", clearProfileInferenceConfiguration);
  elements.closeProfileEditor.addEventListener("click", () => closeProfileEditor());
  elements.profileEditorModal.addEventListener("click", (event) => {
    if (event.target === elements.profileEditorModal) closeProfileEditor();
  });
  elements.chooseModelFolder.addEventListener("click", openFolderPicker);
  elements.folderPickerClose.addEventListener("click", () => closeFolderPicker());
  elements.folderPickerCancel.addEventListener("click", () => closeFolderPicker());
  elements.folderPickerSelect.addEventListener("click", selectCurrentFolder);
  elements.folderPickerUp.addEventListener("click", () => {
    if (state.folderPickerResult?.parent) loadFolderPicker(state.folderPickerResult.parent);
  });
  elements.folderPickerModal.addEventListener("click", (event) => {
    if (event.target === elements.folderPickerModal) closeFolderPicker();
  });
  elements.confirmationCancel.addEventListener("click", () => settleConfirmation(false));
  elements.confirmationConfirm.addEventListener("click", () => settleConfirmation(true));
  elements.confirmationModal.addEventListener("click", (event) => {
    if (event.target === elements.confirmationModal) settleConfirmation(false);
  });
  document.addEventListener("keydown", (event) => {
    if (!elements.confirmationModal.classList.contains("hidden")) {
      if (event.key === "Escape") {
        event.preventDefault();
        settleConfirmation(false);
      } else if (event.key === "Tab") {
        trapConfirmationFocus(event);
      }
      return;
    }
    if (event.key === "Tab") {
      const dialog = !elements.folderPickerModal.classList.contains("hidden")
        ? elements.folderPickerModal
        : !elements.profileEditorModal.classList.contains("hidden") ? elements.profileEditorModal : null;
      if (dialog) trapDialogFocus(event, dialog);
      return;
    }
    if (event.key !== "Escape") return;
    if (!elements.folderPickerModal.classList.contains("hidden")) {
      event.preventDefault();
      closeFolderPicker();
      return;
    }
    if (elements.profileEditorModal.classList.contains("hidden")) return;
    event.preventDefault();
    closeProfileEditor();
  });
  elements.saveProfile.addEventListener("click", saveProfileAndClose);
  elements.stopRuntime.addEventListener("click", stopRuntime);
  elements.openEndpoint.addEventListener("click", openEndpoint);
  elements.clearLogs.addEventListener("click", clearLogs);
  elements.refreshHardware.addEventListener("click", loadHardware);
  elements.shutdownLauncher.addEventListener("click", shutdownLauncher);
  elements.downloadCancel.addEventListener("click", cancelDownload);
  elements.downloadUseModel.addEventListener("click", useDownloadedModel);
  elements.downloadUseLast.addEventListener("click", useDownloadedModel);
  elements.languageSelect.addEventListener("change", (event) => changeLocale(event.target.value));
}

function handleDelegatedClick(event) {
  if (event.target.closest("[data-new-profile]")) {
    newProfile();
    return;
  }
  if (event.target.closest("[data-clear-profile-filters]")) {
    resetProfileFilters();
    renderProfiles();
    elements.profileSearch.focus();
    return;
  }
  const filter = event.target.closest("[data-profile-filter]");
  if (filter) {
    state.profileFilter = filter.dataset.profileFilter;
    renderProfileFilters();
    renderProfiles();
    return;
  }
  const sectionButton = event.target.closest("[data-editor-section]");
  if (sectionButton) {
    const section = document.getElementById(`editor-${sectionButton.dataset.editorSection}`);
    if (section instanceof HTMLDetailsElement) section.open = true;
    section?.scrollIntoView({ block: "start" });
    // Keep the jump links available while making the section keyboard-accessible.
    const heading = section?.querySelector(".section-title, summary");
    if (heading) {
      heading.setAttribute("tabindex", "-1");
      heading.focus({ preventScroll: true });
    }
    return;
  }
  const profileAction = event.target.closest("[data-profile-action]");
  if (profileAction) {
    const index = Number(profileAction.dataset.profileIndex);
    const action = profileAction.dataset.profileAction;
    if (action === "start") startSavedProfile(index);
    if (action === "edit") editProfile(index);
    if (action === "delete") deleteProfile(index);
    return;
  }
  const nav = event.target.closest("[data-view-button], [data-open-view]");
  if (nav) {
    switchView(nav.dataset.viewButton || nav.dataset.openView);
    return;
  }
}

function switchView(view) {
  state.currentView = view;
  for (const button of document.querySelectorAll("[data-view-button]")) {
    const selected = button.dataset.viewButton === view;
    button.classList.toggle("active", selected);
    if (selected) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  for (const panel of document.querySelectorAll(".view")) {
    panel.classList.toggle("active", panel.id === `view-${view}`);
  }
  document.querySelector(".app-shell").classList.toggle("webui-active", view === "webui");
  elements.openWebui.classList.toggle("hidden", view === "webui");
  if (view === "webui") renderWebUIAvailability();
  if (view === "logs") {
    document.querySelector('[data-view-button="logs"]').classList.remove("has-activity");
    scrollLogsToBottom();
  }
  if (view === "download") {
    document.querySelector('[data-view-button="download"]').classList.remove("has-activity");
  }
  if (view === "hardware" && !state.hardwareLoaded) loadHardware();
  renderViewTitle();
}

function renderViewTitle() {
  const titles = {
    launch: t("Launch service"),
    webui: t("Studio"),
    download: t("Download model"),
    logs: t("Runtime logs"),
    hardware: t("Hardware")
  };
  elements.currentViewTitle.textContent = titles[state.currentView] || titles.launch;
}

function renderProfileFilters() {
  for (const button of document.querySelectorAll("[data-profile-filter]")) {
    const selected = button.dataset.profileFilter === state.profileFilter;
    button.classList.toggle("active", selected);
    button.setAttribute("aria-pressed", String(selected));
  }
}

function resetProfileFilters() {
  state.profileQuery = "";
  state.profileFilter = "all";
  elements.profileSearch.value = "";
  renderProfileFilters();
}

function createIcon(name) {
  const icon = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  icon.classList.add("icon");
  icon.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#icon-${name}`);
  icon.append(use);
  return icon;
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
  const changedField = event.target.dataset.field;
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
  if (changedField === "model" && state.automaticConfigPending) {
    scheduleAutomaticConfiguration();
  } else if (
    state.automaticConfigPending
    && AUTOMATIC_CONFIGURATION_FIELDS.has(changedField)
  ) {
    cancelPendingAutomaticConfiguration();
    state.automaticConfigStatus = { phase: "cancelled" };
    renderAutomaticConfigurationStatus();
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
  elements.moeDeviceCustomField.classList.toggle(
    "hidden",
    !config.enable_moe_hybrid || config.moe_device !== "custom"
  );
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
}

function automaticConfigurationFingerprint(config = collectForm()) {
  const values = { model: String(config.model || "").trim() };
  for (const field of AUTOMATIC_CONFIGURATION_FIELDS) values[field] = config[field];
  return JSON.stringify(values);
}

function cancelPendingAutomaticConfiguration() {
  window.clearTimeout(state.automaticConfigTimer);
  state.automaticConfigTimer = null;
  state.automaticConfigPending = false;
  state.automaticConfigAppliedModel = "";
  state.automaticConfigRequestId += 1;
}

function prepareAutomaticConfigurationForNewProfile() {
  cancelPendingAutomaticConfiguration();
  state.automaticConfigPending = true;
  state.automaticConfigStatus = { phase: "waiting" };
  renderAutomaticConfigurationStatus();
}

function scheduleAutomaticConfiguration(delay = 650) {
  window.clearTimeout(state.automaticConfigTimer);
  state.automaticConfigRequestId += 1;
  const model = String(collectForm().model || "").trim();
  if (!model) {
    state.automaticConfigStatus = { phase: "waiting" };
    renderAutomaticConfigurationStatus();
    return;
  }
  if (
    model === state.automaticConfigAppliedModel
    && state.automaticConfigStatus?.phase === "applied"
  ) return;
  state.automaticConfigStatus = { phase: "waiting" };
  renderAutomaticConfigurationStatus();
  state.automaticConfigTimer = window.setTimeout(() => {
    state.automaticConfigTimer = null;
    configureProfileAutomatically({ automatic: true });
  }, delay);
}

async function configureProfileAutomatically({ automatic = false } = {}) {
  const current = collectForm();
  const model = String(current.model || "").trim();
  if (!model) {
    state.automaticConfigStatus = { phase: "missing-model" };
    renderAutomaticConfigurationStatus();
    if (!automatic) showToast(t("Choose a local model before using automatic configuration."), "error");
    return;
  }
  window.clearTimeout(state.automaticConfigTimer);
  state.automaticConfigTimer = null;
  const requestId = ++state.automaticConfigRequestId;
  const fingerprint = automaticConfigurationFingerprint(current);
  const keepAutomaticForNewProfile = state.automaticConfigPending && state.currentIndex === null;
  state.automaticConfigStatus = { phase: "loading" };
  renderAutomaticConfigurationStatus();
  try {
    const recommendation = await request("/api/recommend", {
      method: "POST",
      body: JSON.stringify({ model, name: current.name || current.model_name || "" })
    });
    if (
      requestId !== state.automaticConfigRequestId
      || elements.profileEditorModal.classList.contains("hidden")
    ) return;
    if (fingerprint !== automaticConfigurationFingerprint()) {
      state.automaticConfigPending = false;
      state.automaticConfigStatus = { phase: "stale" };
      renderAutomaticConfigurationStatus();
      return;
    }
    if (!recommendation?.config || typeof recommendation.config !== "object") {
      throw new Error(t("The automatic configuration response is invalid."));
    }
    const recommendedFields = {};
    for (const [field, value] of Object.entries(recommendation.config)) {
      if (AUTOMATIC_CONFIGURATION_FIELDS.has(field)) recommendedFields[field] = value;
    }
    state.editingConfig = { ...collectForm(), ...recommendedFields };
    fillForm(state.editingConfig);
    state.dirty = true;
    state.automaticConfigPending = keepAutomaticForNewProfile;
    state.automaticConfigAppliedModel = model;
    state.automaticConfigStatus = { phase: "applied", recommendation };
    renderSaveState();
    renderAutomaticConfigurationStatus();
    updateConditionalFields();
    schedulePreview(0);
    showToast(
      automatic
        ? t("The new model was configured automatically. Review the recommendation before saving.")
        : t("Recommended inference settings were applied. Review them before saving."),
      "success",
      4600
    );
  } catch (error) {
    if (requestId !== state.automaticConfigRequestId) return;
    state.automaticConfigStatus = {
      phase: "error",
      error: friendlyError(error)
    };
    renderAutomaticConfigurationStatus();
  }
}

function clearProfileInferenceConfiguration() {
  cancelPendingAutomaticConfiguration();
  state.editingConfig = {
    ...collectForm(),
    ...AUTOMATIC_CONFIGURATION_DEFAULTS
  };
  fillForm(state.editingConfig);
  state.dirty = true;
  state.automaticConfigStatus = { phase: "cleared" };
  renderSaveState();
  renderAutomaticConfigurationStatus();
  updateConditionalFields();
  schedulePreview(0);
  showToast(t("Optional inference settings were cleared."), "success");
}

function automaticRecommendationDescription(recommendation) {
  const detected = recommendation?.detected || {};
  const config = recommendation?.config || {};
  const hardware = recommendation?.hardware || {};
  const adjustments = recommendation?.adjustments || {};
  const strategyTitles = {
    automatic: t("Automatic device selection"),
    cuda: t("Single-GPU configuration"),
    tensor_parallel: t("Tensor-parallel configuration"),
    hybrid_numa: t("GPU + NUMA hybrid inference"),
    hybrid_cpu: t("GPU + CPU hybrid inference"),
    hybrid_disk: t("GPU + disk hybrid inference"),
    numa: t("NUMA CPU configuration"),
    cpu: t("CPU configuration"),
    cpu_disk: t("CPU + disk hybrid inference")
  };
  let device = t("automatic device selection");
  if (config.device === "cuda") {
    device = config.enable_moe_hybrid
      ? t("CUDA GPU {device} + {moeDevice} MoE", {
          device: config.cuda_device_id || "0",
          moeDevice: String(config.moe_device || "CPU").toUpperCase()
        })
      : t("CUDA GPU {device}", { device: config.cuda_device_id || "0" });
  } else if (config.device === "tp") {
    device = t("GPU tensor parallelism ({devices})", { devices: config.tp || "—" });
  } else if (config.device === "numa") {
    device = t("NUMA CPU");
  } else if (config.device === "cpu") {
    device = config.enable_moe_hybrid
      ? t("CPU + disk MoE")
      : "CPU";
  }
  const dtype = config.dtype === "auto" ? t("model-native precision") : config.dtype;
  const kind = detected.isMoe ? t("MoE model") : t("dense model");
  const size = Number(detected.parameterBillions || 0);
  const summary = size > 0
    ? t("Detected a {size}B {kind}; recommended {device} with {dtype}.", {
        size: String(Math.round(size * 100) / 100), kind, device, dtype
      })
    : t("Recommended {device} with {dtype} from the available model and hardware information.", {
        device, dtype
      });
  const details = [];
  if (detected.architecture) {
    details.push(t("Detected architecture: {architecture}.", { architecture: detected.architecture }));
  }
  if (Number(detected.weightGiB || 0) > 0) {
    details.push(t("Detected model weights: {size} GiB.", { size: detected.weightGiB }));
  }
  if (config.device === "tp") {
    details.push(t("The model is sharded across {count} GPUs for throughput and capacity.", {
      count: String((hardware.selectedGpuIds || []).length || String(config.tp || "").split(",").filter(Boolean).length)
    }));
  }
  if (config.enable_moe_hybrid) {
    const placement = config.moe_device_layers === "-1"
      ? t("all MoE layers")
      : t("the trailing {count} MoE layers", { count: config.moe_device_layers });
    details.push(t("MoE experts use {device}; placement: {placement}.", {
      device: String(config.moe_device || "").toUpperCase(),
      placement
    }));
  }
  if (adjustments.precisionAdjusted) {
    details.push(t("The weight type was adjusted to fit the available memory."));
  }
  if (adjustments.ngramOnDisk) {
    details.push(t("The large N-gram table was placed on disk to preserve host memory."));
  }
  if (adjustments.metadataLimited) {
    details.push(t("Some model metadata was unavailable; verify the recommendation before launch."));
  }
  return {
    title: strategyTitles[recommendation?.strategy] || t("Automatic configuration applied"),
    summary,
    details
  };
}

function renderAutomaticConfigurationStatus() {
  const status = state.automaticConfigStatus || { phase: "idle" };
  const loading = status.phase === "loading";
  elements.autoConfigureProfile.disabled = loading;
  elements.clearProfileConfig.disabled = loading;
  elements.autoConfigureProfile.textContent = loading
    ? t("Analyzing...")
    : t("Automatic configuration");
  elements.automaticConfigStatus.className = "automatic-config-status";
  elements.automaticConfigStatus.replaceChildren();
  if (status.phase === "idle") {
    elements.automaticConfigStatus.classList.add("hidden");
    updateActionAvailability();
    return;
  }
  let title = "";
  let message = "";
  let details = [];
  if (status.phase === "waiting") {
    title = t("Automatic configuration is ready");
    message = t("Choose a local model path. It will be configured automatically.");
  } else if (status.phase === "missing-model") {
    title = t("Model path required");
    message = t("Choose a local model before using automatic configuration.");
    elements.automaticConfigStatus.classList.add("error");
  } else if (status.phase === "loading") {
    title = t("Analyzing model and hardware...");
    message = t("Reading model metadata, weight size, GPU memory, system memory, and NUMA topology.");
    elements.automaticConfigStatus.classList.add("loading");
  } else if (status.phase === "applied") {
    const description = automaticRecommendationDescription(status.recommendation);
    title = description.title;
    message = description.summary;
    details = description.details;
    elements.automaticConfigStatus.classList.add("success");
  } else if (status.phase === "cleared") {
    title = t("Inference configuration cleared");
    message = t("Optional inference settings were reset; model and service fields were preserved.");
  } else if (status.phase === "cancelled") {
    title = t("Automatic configuration paused");
    message = t("Your manual inference settings were kept. Use Automatic configuration to run it again.");
  } else if (status.phase === "stale") {
    title = t("Recommendation was not applied");
    message = t("The configuration changed while analysis was running. Run automatic configuration again if needed.");
  } else if (status.phase === "error") {
    title = t("Automatic configuration failed");
    message = status.error || t("Unknown error");
    elements.automaticConfigStatus.classList.add("error");
  }
  const heading = document.createElement("strong");
  heading.textContent = title;
  const copy = document.createElement("span");
  copy.textContent = message;
  elements.automaticConfigStatus.append(heading, copy);
  if (details.length) {
    const list = document.createElement("ul");
    for (const detail of details) {
      const item = document.createElement("li");
      item.textContent = detail;
      list.append(item);
    }
    elements.automaticConfigStatus.append(list);
  }
  updateActionAvailability();
}

function renderSaveState() {
  elements.saveState.className = "save-state";
  if (state.dirty) {
    elements.saveState.classList.add("dirty");
    elements.saveState.textContent = t("Unsaved");
  } else if (state.currentIndex === null) {
    elements.saveState.textContent = t("New profile");
  } else {
    elements.saveState.classList.add("saved");
    elements.saveState.textContent = t("Saved");
  }
}

function renderProfiles() {
  const active = runtimeIsActive(state.runtime);
  const runningIndex = findRunningProfileIndex();
  const signature = JSON.stringify([
    state.locale,
    state.profileQuery,
    state.profileFilter,
    state.profiles.map((profile) => [
      profile.name,
      profile.model_name,
      profile.command,
      profile.model,
      profile.device,
      profile.cuda_device_id,
      profile.tp,
      profile.cudapp,
      profile.dtype,
      profile.port
    ]),
    active,
    runningIndex,
    state.runtime?.phase || "stopped"
  ]);
  if (signature === profileRenderSignature) {
    updateProfileRuntime();
    return;
  }
  profileRenderSignature = signature;
  elements.profileList.replaceChildren();
  elements.profileCount.textContent = String(state.profiles.length);
  const query = state.profileQuery.trim().toLocaleLowerCase();
  const profiles = state.profiles.map((profile, index) => ({ profile, index })).filter(({ profile }) => {
    const mode = profile.command === "webui" ? "webui" : "server";
    return (state.profileFilter === "all" || mode === state.profileFilter)
      && [profile.name, profile.model_name, profile.model].some(
        (value) => String(value || "").toLocaleLowerCase().includes(query)
      );
  });
  elements.profileResults.textContent = t("{count} launch items shown", { count: profiles.length });
  if (!profiles.length) {
    const filtered = state.profiles.length > 0;
    const empty = document.createElement("div");
    empty.className = "empty-profile";
    const icon = document.createElement("span");
    icon.className = "empty-profile-icon";
    icon.append(createIcon(filtered ? "search" : "grid"));
    const title = document.createElement("strong");
    title.textContent = filtered ? t("No matching launch items") : t("Your first model starts here");
    const detail = document.createElement("small");
    detail.textContent = filtered
      ? t("Try another name or model path, or clear the filters.")
      : t("Choose a local model and save a profile to launch an API or chat service.");
    const action = document.createElement("button");
    action.type = "button";
    action.className = "primary-button";
    if (filtered) action.dataset.clearProfileFilters = "";
    else action.dataset.newProfile = "";
    action.textContent = filtered ? t("Clear filters") : t("Add launch item");
    empty.append(icon, title, detail, action);
    elements.profileList.append(empty);
    return;
  }
  profiles.forEach(({ profile, index }) => {
    const item = document.createElement("article");
    item.className = "profile-item";
    const running = index === runningIndex;
    item.classList.toggle("running", running);

    const avatar = document.createElement("span");
    avatar.className = "profile-avatar";
    if (profile.command === "webui") avatar.classList.add("webui");
    avatar.textContent = firstVisibleCharacter(profile.name || profile.model_name || "F");
    const copy = document.createElement("div");
    copy.className = "profile-copy";
    const titleRow = document.createElement("div");
    titleRow.className = "profile-title-row";
    const title = document.createElement("strong");
    title.textContent = profile.name || profile.model_name || t("Unnamed profile");
    const mode = document.createElement("span");
    mode.className = "profile-mode";
    mode.textContent = profile.command === "webui" ? t("Chat WebUI") : t("API Server");
    titleRow.append(title);
    const metadata = document.createElement("div");
    metadata.className = "profile-metadata";
    for (const value of [profileDeviceLabel(profile), profile.dtype || "auto", t("Port {port}", { port: profile.port || "—" })]) {
      const detail = document.createElement("span");
      detail.textContent = value;
      metadata.append(detail);
    }
    const path = document.createElement("small");
    path.className = "profile-path";
    path.textContent = profile.model || t("Model not set");
    path.title = profile.model || "";
    copy.append(mode, titleRow, path, metadata);

    const actions = document.createElement("div");
    actions.className = "project-actions";
    const start = profileActionButton("start", index, t("Start"), "start");
    start.disabled = active;
    if (running) {
      const phaseLabels = {
        starting: t("Starting"),
        running: t("Running"),
        stopping: t("Stopping")
      };
      const status = document.createElement("span");
      status.className = `profile-live-status ${state.runtime?.phase || "running"}`;
      status.textContent = phaseLabels[state.runtime?.phase] || t("Running");
      titleRow.append(status);
      start.textContent = phaseLabels[state.runtime?.phase] || t("Running");
    }
    if (active && !running) start.title = t("Stop the running service before starting another item.");
    actions.append(
      start,
      profileActionButton("edit", index, t("Edit"), "edit"),
      profileActionButton("delete", index, t("Delete"), "delete danger")
    );
    item.append(avatar, copy, actions);

    if (running) item.append(renderProfileRuntime());
    elements.profileList.append(item);
  });
}

function profileActionButton(action, index, label, variant = "") {
  const button = document.createElement("button");
  button.type = "button";
  button.className = `project-action ${variant}`.trim();
  button.dataset.profileAction = action;
  button.dataset.profileIndex = String(index);
  button.textContent = label;
  return button;
}

function profileDeviceLabel(profile) {
  const devices = {
    auto: t("Auto device"),
    cuda: "CUDA",
    tp: "TP",
    cudapp: "CUDA:PP",
    cpu: "CPU",
    numa: "NUMA"
  };
  const device = devices[profile.device] || String(profile.device || t("Auto device"));
  const ids = { cuda: profile.cuda_device_id || "0", tp: profile.tp, cudapp: profile.cudapp };
  return ids[profile.device] ? `${device} · ${ids[profile.device]}` : device;
}

function renderProfileRuntime() {
  const runtime = state.runtime || {};
  const panel = document.createElement("div");
  panel.className = "project-runtime";
  const status = document.createElement("div");
  status.className = "project-runtime-status";
  const label = document.createElement("span");
  label.className = "profile-runtime-label";
  label.textContent = localizeServerText(runtime.progressLabel || runtime.message) || t("Starting");
  const value = document.createElement("strong");
  value.className = "profile-runtime-value";
  const percent = Math.max(0, Math.min(100, Number(runtime.progress || 0)));
  value.textContent = runtime.progressIndeterminate ? t("Processing") : `${Math.round(percent)}%`;
  status.append(label, value);
  panel.append(status);
  if (runtime.phase === "starting") {
    const progress = document.createElement("progress");
    progress.className = "progress-track profile-runtime-progress";
    progress.max = 100;
    if (runtime.progressIndeterminate) {
      progress.classList.add("indeterminate");
    } else {
      progress.value = percent;
    }
    panel.append(progress);
  }
  if (runtime.endpoint) {
    const endpoint = document.createElement("code");
    endpoint.className = "profile-runtime-endpoint";
    endpoint.textContent = runtime.endpoint;
    panel.append(endpoint);
  }
  return panel;
}

function updateProfileRuntime() {
  const panel = elements.profileList.querySelector(".project-runtime");
  if (!panel) return;
  const runtime = state.runtime || {};
  const percent = Math.max(0, Math.min(100, Number(runtime.progress || 0)));
  const label = panel.querySelector(".profile-runtime-label");
  const value = panel.querySelector(".profile-runtime-value");
  const progress = panel.querySelector(".profile-runtime-progress");
  if (label) label.textContent = localizeServerText(runtime.progressLabel || runtime.message) || t("Starting");
  if (value) value.textContent = runtime.progressIndeterminate ? t("Processing") : `${Math.round(percent)}%`;
  if (progress) {
    progress.classList.toggle("indeterminate", Boolean(runtime.progressIndeterminate));
    if (runtime.progressIndeterminate) progress.removeAttribute("value");
    else progress.value = percent;
  }
  let endpoint = panel.querySelector(".profile-runtime-endpoint");
  if (runtime.endpoint) {
    if (!endpoint) {
      endpoint = document.createElement("code");
      endpoint.className = "profile-runtime-endpoint";
      panel.append(endpoint);
    }
    endpoint.textContent = runtime.endpoint;
  } else {
    endpoint?.remove();
  }
}

function firstVisibleCharacter(value) {
  return Array.from(String(value).trim())[0]?.toUpperCase() || "F";
}

function findRunningProfileIndex() {
  const runtime = state.runtime;
  if (!runtimeIsActive(runtime)) return -1;
  if (runtime.profileName) {
    const namedIndex = state.profiles.findIndex((profile) => profile.name === runtime.profileName);
    if (namedIndex >= 0) return namedIndex;
  }
  return runtime.model
    ? state.profiles.findIndex((profile) => profile.model === runtime.model)
    : -1;
}

function editProfile(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.profiles.length) return;
  cancelPendingAutomaticConfiguration();
  state.automaticConfigStatus = { phase: "idle" };
  state.currentIndex = index;
  state.editingConfig = cloneConfig(state.profiles[index]);
  fillForm(state.editingConfig);
  showProfileEditor();
  schedulePreview(0);
  elements.profileEditorTitle.focus({ preventScroll: true });
}

function newProfile() {
  cancelPendingAutomaticConfiguration();
  state.currentIndex = null;
  state.editingConfig = cloneConfig(state.defaultProfile);
  state.editingConfig.port = defaultServicePort(state.editingConfig.command);
  const used = new Set(state.profiles.map((item) => item.name));
  let sequence = 1;
  while (used.has(t("Profile {number}", { number: sequence }))) sequence += 1;
  state.editingConfig.name = t("Profile {number}", { number: sequence });
  fillForm(state.editingConfig);
  state.dirty = true;
  renderSaveState();
  showProfileEditor();
  prepareAutomaticConfigurationForNewProfile();
  schedulePreview(0);
  elements.launchForm.querySelector('[data-field="model"]').focus();
}

function showProfileEditor() {
  renderProfileEditorTitle();
  renderAutomaticConfigurationStatus();
  elements.profileEditorModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  elements.launchForm.scrollTop = 0;
}

function renderProfileEditorTitle() {
  elements.profileEditorTitle.textContent = state.currentIndex === null
    ? t("Add launch item")
    : t("Edit launch item");
}

async function closeProfileEditor(force = false) {
  if (!force && state.dirty) {
    const discard = await showConfirmation({
      tone: "warning",
      icon: "↩",
      kicker: t("Unsaved changes"),
      title: t("Discard your changes?"),
      message: t("Discard unsaved changes and return to the launch item list?"),
      cancelLabel: t("Keep editing"),
      confirmLabel: t("Discard changes")
    });
    if (!discard) return false;
  }
  cancelPendingAutomaticConfiguration();
  state.automaticConfigStatus = { phase: "idle" };
  state.currentIndex = null;
  state.dirty = false;
  if (!elements.folderPickerModal.classList.contains("hidden")) {
    closeFolderPicker(false);
  }
  elements.profileEditorModal.classList.add("hidden");
  document.body.classList.remove("modal-open");
  renderProfiles();
  elements.newProfile.focus({ preventScroll: true });
  return true;
}

async function deleteProfile(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.profiles.length) return;
  const profile = state.profiles[index];
  const name = profile.name || t("Unnamed profile");
  const confirmed = await showConfirmation({
    tone: "danger",
    icon: "×",
    kicker: t("Delete launch item"),
    title: t("Delete “{name}”?", { name }),
    message: t("This removes only the saved launch configuration. Model files remain on disk."),
    cancelLabel: t("Cancel"),
    confirmLabel: t("Delete")
  });
  if (!confirmed) return;
  try {
    const result = await request(`/api/profiles/${index}`, { method: "DELETE" });
    state.profiles = result.profiles;
    renderProfiles();
    showToast(t("Profile deleted."), "success");
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
  resetProfileFilters();
  renderSaveState();
  if (showSuccess) showToast(t("Launch profile saved."), "success");
  return result.profile;
}

async function saveProfileAndClose() {
  try {
    await saveCurrentProfile(true);
    closeProfileEditor(true);
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
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
    elements.commandPreview.textContent = preview.command || t("Complete a valid configuration first.");
    renderLaunchValidation(preview.errors || []);
    updateActionAvailability();
  } catch (error) {
    if (requestId !== state.previewRequestId) return;
    state.preview = { errors: [friendlyError(error)] };
    elements.commandPreview.textContent = t("Unable to generate command.");
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
    item.textContent = localizeServerText(message);
    container.append(item);
  }
}

function renderLaunchValidation(errors) {
  renderValidation(elements.validationMessages, errors, t("Configuration is valid and ready to launch."));
}

function renderDownloadValidation(errors) {
  renderValidation(elements.downloadValidation, errors, t("Download configuration is valid and ready."));
}

async function startSavedProfile(index) {
  if (!Number.isInteger(index) || index < 0 || index >= state.profiles.length) return;
  const config = cloneConfig(state.profiles[index]);
  try {
    const preview = await request("/api/preview", {
      method: "POST",
      body: JSON.stringify(config)
    });
    if (preview.errors?.length) {
      cancelPendingAutomaticConfiguration();
      state.automaticConfigStatus = { phase: "idle" };
      state.currentIndex = index;
      state.editingConfig = config;
      fillForm(config);
      state.preview = preview;
      elements.commandPreview.textContent = preview.command || t("Complete a valid configuration first.");
      renderLaunchValidation(preview.errors);
      showProfileEditor();
      showToast(t("Fix the configuration errors first."), "error");
      return;
    }
    state.runtime = await request("/api/runtime/start", {
      method: "POST",
      body: JSON.stringify(config)
    });
    renderRuntime();
    showToast(t("Model service is starting."), "success");
  } catch (error) {
    showToast(friendlyError(error), "error", 7000);
    await refreshRuntime();
  }
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
      showToast(t("Fix the configuration errors first."), "error");
      return;
    }
    const config = await saveCurrentProfile(false);
    state.runtime = await request("/api/runtime/start", {
      method: "POST",
      body: JSON.stringify(config)
    });
    closeProfileEditor(true);
    renderRuntime();
    switchView("launch");
    showToast(t("Model service is starting."), "success");
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
  const runtime = state.runtime || { phase: "stopped", message: "Model has not been started" };
  const phase = runtime.phase || "stopped";
  const isWebui = runtime.command === "webui";
  const titles = {
    stopped: t("Service not started"),
    starting: isWebui ? t("Starting chat WebUI") : t("Starting model"),
    running: isWebui ? t("Chat WebUI is running") : t("Local API is running"),
    stopping: t("Stopping service"),
    failed: isWebui ? t("WebUI startup failed") : t("Model service failed")
  };
  elements.statusDot.className = `status-dot ${phase}`;
  elements.statusTitle.textContent = titles[phase] || phase;
  elements.statusMessage.textContent = localizeServerText(runtime.message) || "—";
  const active = runtimeIsActive(runtime);
  elements.stopRuntime.classList.toggle("hidden", !active);
  elements.stopRuntime.disabled = phase === "stopping";
  elements.openEndpoint.disabled = !runtime.ready;
  elements.openEndpoint.textContent = isWebui ? t("Open WebUI") : t("Open API documentation");
  renderWebUIAvailability();
  renderProfiles();
  updateActionAvailability();
}

function updateActionAvailability() {
  const active = runtimeIsActive(state.runtime);
  const invalid = Boolean(state.preview?.errors?.length);
  const automaticBusy = state.automaticConfigStatus?.phase === "loading" || (
    state.automaticConfigPending
    && state.automaticConfigStatus?.phase === "waiting"
    && Boolean(String(collectForm().model || "").trim())
  );
  elements.startRuntime.disabled = active || invalid || automaticBusy;
  elements.saveProfile.disabled = automaticBusy;
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

function renderDownloadCatalog() {
  const selected = elements.downloadPreset.value;
  elements.downloadPreset.replaceChildren();
  for (const group of state.downloadCatalog) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = localizeServerText(group.label || group.id || t("Models"));
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
  custom.textContent = t("Custom model ID");
  elements.downloadPreset.append(custom);
  elements.downloadPreset.value = [...elements.downloadPreset.options].some((option) => option.value === selected)
    ? selected
    : "custom";
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
    elements.downloadCommand.textContent = localizeDisplayCommand(state.downloadPreview.command)
      || t("Complete a valid configuration first.");
    renderDownloadValidation(state.downloadPreview.errors || []);
  } catch (error) {
    if (requestId !== state.downloadPreviewRequestId) return;
    state.downloadPreview = { errors: [friendlyError(error)] };
    elements.downloadCommand.textContent = t("Unable to generate download command.");
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
    elements.downloadCommand.textContent = localizeDisplayCommand(preview.command)
      || t("Complete a valid configuration first.");
    renderDownloadValidation(preview.errors || []);
    if (preview.errors?.length) {
      showToast(t("Fix the download configuration errors first."), "error");
      return;
    }
    state.download = await request("/api/download/start", {
      method: "POST",
      body: JSON.stringify(collectDownloadForm())
    });
    renderDownload();
    switchView("download");
    showToast(t("Model download started."), "success");
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
        showToast(t("Model download completed and is ready to launch."), "success", 6000);
        if (state.currentView !== "download") {
          document.querySelector('[data-view-button="download"]').classList.add("has-activity");
        }
      } else if (state.download.phase === "failed") {
        showToast(localizeServerText(state.download.message) || t("Model download failed."), "error", 7000);
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
  const download = state.download || { phase: "idle", progress: 0, message: "Download has not started" };
  const phase = download.phase || "idle";
  const active = ACTIVE_DOWNLOAD_PHASES.has(phase);
  const titles = {
    idle: t("Download has not started"),
    starting: t("Starting download"),
    downloading: t("Downloading model"),
    cancelling: t("Cancelling download"),
    cancelled: t("Download cancelled"),
    completed: t("Model download completed"),
    failed: t("Model download failed")
  };
  const badges = {
    idle: t("Not started"), starting: t("Connecting"), downloading: t("Downloading"), cancelling: t("Cancelling"),
    cancelled: t("Cancelled"), completed: t("Completed"), failed: t("Failed")
  };
  elements.downloadBadge.className = `save-state ${phase === "completed" ? "saved" : phase}`;
  elements.downloadBadge.textContent = badges[phase] || phase;
  elements.downloadStatusTitle.textContent = titles[phase] || phase;
  elements.downloadStatusMessage.textContent = localizeServerText(download.message) || "—";
  elements.downloadStatusIcon.textContent = phase === "completed" ? "✓" : (phase === "failed" ? "!" : "↓");
  const percent = Math.max(0, Math.min(100, Number(download.progress || 0)));
  elements.downloadProgressValue.textContent = download.progressIndeterminate
    ? t("Processing")
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
  cancelPendingAutomaticConfiguration();
  state.currentIndex = null;
  state.editingConfig = cloneConfig(state.defaultProfile);
  state.editingConfig.port = defaultServicePort(state.editingConfig.command);
  state.editingConfig.name = basename(destination) || t("New profile");
  state.editingConfig.model = destination;
  fillForm(state.editingConfig);
  const modelInput = elements.launchForm.querySelector('[data-field="model"]');
  state.dirty = true;
  renderSaveState();
  updateConditionalFields();
  schedulePreview(0);
  switchView("launch");
  showProfileEditor();
  prepareAutomaticConfigurationForNewProfile();
  scheduleAutomaticConfiguration(0);
  modelInput.focus();
  showToast(t("A new launch item was created and is being configured automatically."), "success");
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
  elements.logCount.textContent = t("{count} entries", { count: state.logs.length });
  if (!state.logs.length) {
    const empty = document.createElement("div");
    empty.className = "log-empty";
    empty.textContent = t("Logs will appear here after the model starts.");
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
    message.textContent = localizeDisplayCommand(localizeServerText(entry.message));
    line.append(timestamp, source, message);
    fragment.append(line);
  }
  elements.logOutput.append(fragment);
  if (forceBottom) scrollLogsToBottom();
}

function formatTime(timestamp) {
  const date = new Date(Number(timestamp) * 1000);
  return Number.isNaN(date.getTime()) ? "--:--:--" : date.toLocaleTimeString(state.locale, { hour12: false });
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

function openFolderPicker() {
  const modelInput = elements.launchForm.querySelector('[data-field="model"]');
  elements.folderPickerModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  elements.folderPickerTitle.focus({ preventScroll: true });
  loadFolderPicker(modelInput.value || "");
}

function closeFolderPicker(restoreFocus = true) {
  state.folderPickerRequestId += 1;
  state.folderPickerLoading = false;
  elements.folderPickerModal.classList.add("hidden");
  if (elements.profileEditorModal.classList.contains("hidden")) {
    document.body.classList.remove("modal-open");
  }
  if (restoreFocus && !elements.profileEditorModal.classList.contains("hidden")) {
    elements.chooseModelFolder.focus({ preventScroll: true });
  }
}

async function loadFolderPicker(path) {
  const requestId = ++state.folderPickerRequestId;
  state.folderPickerLoading = true;
  state.folderPickerError = "";
  renderFolderPicker();
  try {
    const query = new URLSearchParams({ path: String(path || "") });
    const result = await request(`/api/folders?${query}`);
    if (
      requestId !== state.folderPickerRequestId
      || elements.folderPickerModal.classList.contains("hidden")
    ) return;
    if (!result || typeof result.path !== "string" || !Array.isArray(result.folders)) {
      throw new Error(t("The folder browser response is invalid."));
    }
    state.folderPickerResult = result;
  } catch (error) {
    if (requestId !== state.folderPickerRequestId) return;
    state.folderPickerError = friendlyError(error);
  } finally {
    if (requestId !== state.folderPickerRequestId) return;
    state.folderPickerLoading = false;
    renderFolderPicker();
  }
}

function renderFolderPicker() {
  elements.folderPickerList.replaceChildren();
  elements.folderPickerStatus.className = "folder-picker-status";
  elements.folderPickerStatus.textContent = "";
  elements.folderPickerSelect.disabled = (
    state.folderPickerLoading
    || Boolean(state.folderPickerError)
    || !state.folderPickerResult?.path
  );
  elements.folderPickerUp.disabled = (
    state.folderPickerLoading
    || Boolean(state.folderPickerError)
    || !state.folderPickerResult?.parent
  );

  if (state.folderPickerLoading) {
    elements.folderPickerCurrent.textContent = state.folderPickerResult?.path || t("Loading folders...");
    const loading = document.createElement("div");
    loading.className = "folder-picker-placeholder loading";
    loading.textContent = t("Loading folders...");
    elements.folderPickerList.append(loading);
    return;
  }

  if (state.folderPickerError) {
    elements.folderPickerCurrent.textContent = state.folderPickerResult?.path || "—";
    elements.folderPickerStatus.classList.add("error");
    elements.folderPickerStatus.textContent = t("Unable to browse folders: {error}", {
      error: state.folderPickerError
    });
    const empty = document.createElement("div");
    empty.className = "folder-picker-placeholder error";
    empty.textContent = t("This folder could not be opened.");
    elements.folderPickerList.append(empty);
    return;
  }

  const result = state.folderPickerResult;
  elements.folderPickerCurrent.textContent = result?.path || "—";
  elements.folderPickerCurrent.title = result?.path || "";
  const folders = result?.folders || [];
  if (!folders.length) {
    const empty = document.createElement("div");
    empty.className = "folder-picker-placeholder";
    empty.textContent = t("This folder has no subfolders.");
    elements.folderPickerList.append(empty);
  } else {
    const fragment = document.createDocumentFragment();
    for (const folder of folders) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "folder-picker-entry";
      button.setAttribute("role", "option");
      button.title = folder.path;
      const icon = document.createElement("span");
      icon.className = "folder-picker-entry-icon";
      icon.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.className = "folder-picker-entry-name";
      name.textContent = folder.name;
      const arrow = document.createElement("span");
      arrow.className = "folder-picker-entry-arrow";
      arrow.setAttribute("aria-hidden", "true");
      arrow.textContent = "›";
      button.append(icon, name, arrow);
      button.addEventListener("click", () => loadFolderPicker(folder.path));
      fragment.append(button);
    }
    elements.folderPickerList.append(fragment);
  }
  if (result?.truncated) {
    elements.folderPickerStatus.textContent = t("Only the first {count} folders are shown.", {
      count: String(folders.length)
    });
  }
}

function selectCurrentFolder() {
  const path = state.folderPickerResult?.path;
  if (!path) return;
  const modelInput = elements.launchForm.querySelector('[data-field="model"]');
  modelInput.value = path;
  modelInput.dispatchEvent(new Event("input", { bubbles: true }));
  closeFolderPicker(false);
  modelInput.focus({ preventScroll: true });
}

async function loadHardware() {
  elements.refreshHardware.disabled = true;
  state.hardwareStatus = "loading";
  state.hardwareError = "";
  renderHardwareStatus();
  try {
    const modelPath = collectForm().model || "";
    const report = await request(`/api/hardware?model_path=${encodeURIComponent(modelPath)}`);
    state.hardwareLoaded = true;
    state.hardwareReport = report;
    state.hardwareStatus = "loaded";
    renderHardware(report);
  } catch (error) {
    state.hardwareStatus = "failed";
    state.hardwareError = friendlyError(error);
  } finally {
    renderHardwareStatus();
    elements.refreshHardware.disabled = false;
  }
}

function renderHardwareStatus() {
  if (state.hardwareStatus === "loading") {
    elements.hardwareStatus.textContent = t("Reading hardware information...");
  } else if (state.hardwareStatus === "loaded" && state.hardwareReport) {
    elements.hardwareStatus.textContent = t("Detection completed · {platform} · Python {python}", {
      platform: state.hardwareReport.platform,
      python: state.hardwareReport.python
    });
  } else if (state.hardwareStatus === "failed") {
    elements.hardwareStatus.textContent = t("Detection failed: {error}", { error: state.hardwareError });
  }
}

function renderHardware(report) {
  elements.hardwareGrid.replaceChildren();
  elements.hardwareGrid.append(
    hardwareCard("CPU", "C", report.cpu?.model || t("Unknown CPU"), [
      [t("Logical threads"), String(report.cpu?.logical || "—")],
      [t("Currently available"), String(report.cpu?.available || "—")],
      [t("NUMA nodes"), String(report.numa?.length || 0)]
    ]),
    hardwareCard(t("Memory"), "M", t("System memory"), [
      [t("Total capacity"), formatBytes(report.memory?.total)],
      [t("Currently available"), formatBytes(report.memory?.available)],
      [t("Available ratio"), formatRatio(report.memory?.available, report.memory?.total)]
    ]),
    gpuHardwareCard(report.gpus || []),
    hardwareCard(t("Storage and build"), "D", report.disk?.path || t("Model disk"), [
      [t("Disk capacity"), formatBytes(report.disk?.total)],
      [t("Disk available"), formatBytes(report.disk?.free)],
      [t("CUDA build"), report.build?.USE_CUDA ? t("Enabled") : t("Disabled")],
      [t("ROCm build"), report.build?.USE_ROCM ? t("Enabled") : t("Disabled")],
      [t("NUMA build"), report.build?.USE_NUMAS ? t("Enabled") : t("Disabled")]
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
  subtitle.textContent = gpus.length
    ? t("{count} NVIDIA GPUs", { count: gpus.length })
    : t("nvidia-smi or an NVIDIA GPU was not detected");
  copy.append(title, subtitle);
  heading.append(icon, copy);
  const list = document.createElement("div");
  list.className = "gpu-list";
  if (!gpus.length) {
    const empty = document.createElement("div");
    empty.className = "gpu-item";
    empty.textContent = t("No NVIDIA GPU was detected through nvidia-smi. Select other devices according to the build configuration.");
    list.append(empty);
  }
  for (const gpu of gpus) {
    const item = document.createElement("div");
    item.className = "gpu-item";
    const info = document.createElement("div");
    const name = document.createElement("strong");
    name.textContent = `GPU ${gpu.index} · ${gpu.name}`;
    const detail = document.createElement("small");
    detail.textContent = t("{free} / {total} MiB available · driver {driver}", {
      free: gpu.memoryFreeMiB,
      total: gpu.memoryTotalMiB,
      driver: gpu.driver
    });
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
  if (!bytes) return t("Unknown");
  const units = ["B", "KiB", "MiB", "GiB", "TiB"];
  const index = Math.min(units.length - 1, Math.floor(Math.log(bytes) / Math.log(1024)));
  return `${(bytes / (1024 ** index)).toFixed(index >= 3 ? 1 : 0)} ${units[index]}`;
}

function formatRatio(value, total) {
  const numerator = Number(value || 0);
  const denominator = Number(total || 0);
  return denominator ? `${Math.round(numerator * 100 / denominator)}%` : t("Unknown");
}

async function shutdownLauncher() {
  const confirmed = await showConfirmation({
    tone: "danger",
    icon: "■",
    kicker: t("Shut down Launcher"),
    title: t("Exit Launcher?"),
    message: t("Exit Launcher? Running model services and downloads will also stop."),
    cancelLabel: t("Keep Launcher running"),
    confirmLabel: t("Exit Launcher")
  });
  if (!confirmed) return;
  try {
    await request("/api/shutdown", { method: "POST" });
    showToast(t("Launcher is shutting down..."), "success", 5000);
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}

function showConfirmation({
  tone = "warning",
  icon = "!",
  kicker,
  title,
  message,
  cancelLabel,
  confirmLabel
}) {
  if (state.confirmationResolve) return Promise.resolve(false);
  state.confirmationRestoreFocus = document.activeElement instanceof HTMLElement
    ? document.activeElement
    : null;
  elements.confirmationModal.dataset.tone = tone;
  elements.confirmationIcon.textContent = icon;
  elements.confirmationKicker.textContent = kicker;
  elements.confirmationTitle.textContent = title;
  elements.confirmationMessage.textContent = message;
  elements.confirmationCancel.textContent = cancelLabel;
  elements.confirmationConfirm.textContent = confirmLabel;
  elements.confirmationConfirm.className = `confirmation-confirm-button ${tone}`;
  elements.confirmationModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  window.requestAnimationFrame(() => {
    elements.confirmationConfirm.focus({ preventScroll: true });
  });
  return new Promise((resolve) => {
    state.confirmationResolve = resolve;
  });
}

function settleConfirmation(confirmed) {
  const resolve = state.confirmationResolve;
  if (!resolve) return;
  const restoreFocus = state.confirmationRestoreFocus;
  state.confirmationResolve = null;
  state.confirmationRestoreFocus = null;
  elements.confirmationModal.classList.add("hidden");
  if (
    elements.profileEditorModal.classList.contains("hidden")
    && elements.folderPickerModal.classList.contains("hidden")
  ) {
    document.body.classList.remove("modal-open");
  }
  if (restoreFocus?.isConnected) restoreFocus.focus({ preventScroll: true });
  resolve(Boolean(confirmed));
}

function trapConfirmationFocus(event) {
  const controls = [elements.confirmationCancel, elements.confirmationConfirm];
  const first = controls[0];
  const last = controls[controls.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function trapDialogFocus(event, dialog) {
  const controls = [...dialog.querySelectorAll(
    'button, input, select, textarea, summary, a[href], [tabindex="0"]'
  )].filter((element) => !element.disabled && element.getClientRects().length > 0);
  if (!controls.length) return;
  const first = controls[0];
  const last = controls[controls.length - 1];
  if (event.shiftKey && (document.activeElement === first || !controls.includes(document.activeElement))) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && (document.activeElement === last || !dialog.contains(document.activeElement))) {
    event.preventDefault();
    first.focus();
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
  return localizeServerText(error?.message || error || t("Unknown error"));
}

function webuiIsReady() {
  return state.runtime?.command === "server" && state.runtime?.phase === "running"
    && state.runtime?.ready && Boolean(state.runtime?.sessionId);
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
      iconUrl: "/assets/launcher-icon.png", signal: controller.signal
    });
    if (requestId !== state.webuiRequestId || sessionId !== state.runtime?.sessionId) {
      component.destroy();
      return;
    }
    state.webuiComponent = component;
    clearWebUILoad();
    state.webuiLoading = false;
    component.setLocale(state.locale);
    renderWebUIAvailability();
  } catch (error) {
    failWebUILoad(requestId, error instanceof TypeError ? "Unable to load WebUI. Try reopening it."
      : error?.message || "Unable to load WebUI. Try reopening it.");
  }
}
