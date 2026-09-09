import {
  resetProfileSearch, cloneConfig, fillForm, collectForm, isSimpleNewProfile,
  configurationModeDescription, openAutomaticConfigurationDialog, closeAutomaticConfigurationDialog,
  handleConfigurationModeChange, handleFormChange, defaultServicePort, updateConditionalFields,
  configureProfileAutomatically, clearProfileInferenceConfiguration,
  renderAutomaticConfigurationStatus, renderSaveState, renderProfiles, editProfile, newProfile,
  renderProfileEditorTitle, closeProfileEditor, deleteProfile, saveProfileAndClose, schedulePreview,
  renderLaunchValidation, focusValidationError, startSavedProfile, startRuntime, openFolderPicker,
  closeFolderPicker, loadFolderPicker, renderFolderPicker, selectCurrentFolder
} from "../ui_plugins/models/app.js";
import {
  renderDownloadValidation, renderDownloadCatalog, fillDownloadForm, selectDownloadPreset,
  handleDownloadChange, scheduleDownloadPreview, startDownload, cancelDownload, refreshDownload,
  renderDownload, useDownloadedModel
} from "../ui_plugins/downloads/app.js";
import {
  refreshLogs, renderLogs, scrollLogsToBottom, clearLogs
} from "../ui_plugins/logs/app.js";
import {
  loadHardware, renderHardwareStatus, renderHardware
} from "../ui_plugins/hardware/app.js";
import {
  renderAgentRuntime, refreshAgentRuntime, installAgentRuntime, renderWebUIAvailability,
  openEmbeddedWebUI
} from "../ui_plugins/studio/launcher.js";

import {
  mountPluginHost
} from "../plugin-core/host.js";
import {mountHarness} from "../ui_plugins/harness/app.js";
import {mountOpenCode} from "../ui_plugins/opencode/app.js";
import {mountCodex} from "../ui_plugins/codex/app.js";

let pluginHost, harness, nativeAgents = {};
const locationQuery = new URLSearchParams(window.location.search);
const queryToken = locationQuery.get("token") || "";
if (queryToken) {
  window.sessionStorage.setItem("ftllm-launcher-token", queryToken);
  window.history.replaceState({}, "", window.location.pathname);
}
const controlToken = queryToken || window.sessionStorage.getItem("ftllm-launcher-token") || "";
const ACTIVE_RUNTIME_PHASES = new Set(["starting", "running", "stopping"]);
const DEFAULT_LOCALE = "zh-CN";
const SUPPORTED_LOCALES = new Set(["zh-CN", "en-US"]);
const LOCALE_STORAGE_KEY = "ftllm-launcher-locale";
const INFERENCE_SPEED_TIMEOUT_MS = 3000;
// The HTML template is the Chinese fallback; JavaScript and backend messages use
// English message IDs. Locale resources provide the corresponding translations.
const localeCache = new Map();
const capturedStaticText = [];
const capturedStaticAttributes = [];

const state = {
  profiles: [],
  defaultProfile: null,
  currentIndex: null,
  editingConfig: null,
  speculativeCountField: "draft_tokens",
  automaticConfigDialogPreviousStatus: null,
  runtime: null,
  agentRuntime: null,
  agentRuntimePolling: false,
  agentRuntimeRequest: false,
  agentConfigRefreshNeeded: false,
  inferenceSpeedSession: null,
  inferenceSpeedSamples: {},
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
  folderPickerSelectedFile: "",
  folderPickerError: "",
  folderPickerField: "model",
  folderPickerTrigger: null,
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
  if (options.stream && response.ok) return response;
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
    "inference-status-bar", "prefill-speed", "decode-speed",
    "context-window-metric", "context-window",
    "open-endpoint", "stop-runtime", "profile-count", "profile-list", "new-profile",
    "current-view-title", "profile-search", "profile-results",
    "open-webui", "webui-placeholder", "webui-status",
    "webui-content", "webui-retry", "agent-runtime-card", "agent-runtime-title",
    "agent-runtime-message", "agent-runtime-progress", "agent-runtime-error", "install-agent-runtime",
    "config-path", "profile-editor-modal", "profile-editor-title", "launch-form",
    "close-profile-editor", "save-state", "ori-field", "auto-configure-profile",
    "clear-profile-config", "automatic-config-status",
    "profile-parameters", "profile-editor-description", "configuration-mode-options",
    "configuration-mode-description", "automatic-config-actions",
    "configuration-mode-settings", "automatic-config-dialog", "automatic-mode-options",
    "automatic-mode-description", "automatic-enable-speculative-decoding",
    "automatic-config-dialog-close", "automatic-config-dialog-cancel",
    "automatic-config-dialog-apply", "automatic-config-dialog-error",
    "cuda-device-field", "tp-device-field", "cudapp-device-field", "moe-device-field",
    "moe-device-custom-field", "moe-layers-field", "server-model-name-field", "server-host-field",
    "webui-max-token-field", "webui-think-field",
    "server-context-field", "server-sampling-title", "server-sampling-fields", "server-api-key-field",
    "server-hide-input-field", "launch-command-kicker", "command-preview",
    "validation-messages", "validation-summary", "launch-action-hint", "save-profile",
    "speculative-mtp-field", "speculative-draft-tokens-field", "speculative-path-field", "speculative-mode-hint",
    "start-runtime", "clear-logs", "log-count", "log-output",
    "refresh-hardware", "hardware-status", "hardware-grid", "path-suggestions",
    "choose-model-folder", "choose-draft-model-folder", "folder-picker-modal", "folder-picker-title",
    "folder-picker-close", "folder-picker-current", "folder-picker-up",
    "folder-picker-location", "folder-picker-drive-field", "folder-picker-drive",
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
    "launcher-address-list", "language-select", "theme-select"
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
  initializeTheme();
  captureStaticMessages();
  await initializeLocale();
  bindEvents();
  harness = mountHarness({request, getRuntime:() => state.runtime, t});
  const agentOptions = {request, getRuntime:() => state.runtime, t};
  nativeAgents = {opencode:mountOpenCode(agentOptions), codex:mountCodex({...agentOptions,
    chooseDirectory:(input, trigger) => openFolderPicker(null, trigger, {input, directoriesOnly:true})})};
  pluginHost = await mountPluginHost({request, navigation:document.querySelector(".navigation"),
    container:document.querySelector(".page-scroll"), navigate:switchView,
    nativePages:{harness, ...nativeAgents},
    context:() => ({locale:state.locale, theme:window.ftllmLauncherTheme.getResolved()}),
    studioCall:(capability, args) => {
      if (!state.webuiComponent) throw new Error("请先打开工作室");
      return state.webuiComponent.pluginCall(capability, args);
    }});
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
    // Expire speed samples even if a runtime request stalls or the connection drops.
    window.setInterval(renderInferenceSpeed, 250);
    window.setInterval(refreshDownload, 700);
    window.setInterval(refreshLogs, 700);
    refreshAgentRuntime();
    window.setInterval(refreshAgentRuntime, 1000);
  } catch (error) {
    showToast(t("Launcher initialization failed: {error}", { error: friendlyError(error) }), "error", 10000);
    elements.statusTitle.textContent = t("Launcher connection failed");
    elements.statusMessage.textContent = friendlyError(error);
    elements.statusDot.className = "status-dot failed";
  }
}

function initializeTheme() {
  const theme = window.ftllmLauncherTheme;
  elements.themeSelect.value = theme.getPreference();
  elements.themeSelect.addEventListener("change", () => theme.setPreference(elements.themeSelect.value));
  window.addEventListener("ftllm-theme-change", event => {
    elements.themeSelect.value = event.detail.preference;
    state.webuiComponent?.setTheme(event.detail.theme);
  });
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
  renderAgentRuntime();
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
  elements.installAgentRuntime.addEventListener("click", installAgentRuntime);
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
  elements.configurationModeOptions.addEventListener("change", handleConfigurationModeChange);
  elements.autoConfigureProfile.addEventListener("click", openAutomaticConfigurationDialog);
  elements.automaticConfigDialogClose.addEventListener("click", () => closeAutomaticConfigurationDialog());
  elements.automaticConfigDialogCancel.addEventListener("click", () => closeAutomaticConfigurationDialog());
  elements.automaticConfigDialog.addEventListener("cancel", (event) => {
    event.preventDefault();
    closeAutomaticConfigurationDialog();
  });
  elements.automaticModeOptions.addEventListener("change", () => {
    elements.automaticModeDescription.textContent = configurationModeDescription(
      elements.automaticModeOptions.querySelector("input:checked")?.value || "long_context"
    );
  });
  elements.automaticConfigDialogApply.addEventListener("click", async () => {
    const selection = {
      config_mode: elements.automaticModeOptions.querySelector("input:checked")?.value || "long_context",
      enable_speculative_decoding: elements.automaticEnableSpeculativeDecoding.checked
    };
    if (await configureProfileAutomatically({ selection })) {
      closeAutomaticConfigurationDialog(false);
    }
  });
  elements.clearProfileConfig.addEventListener("click", clearProfileInferenceConfiguration);
  elements.closeProfileEditor.addEventListener("click", () => closeProfileEditor());
  elements.profileEditorModal.addEventListener("click", (event) => {
    if (event.target === elements.profileEditorModal) closeProfileEditor();
  });
  elements.chooseModelFolder.addEventListener("click", () => openFolderPicker("model", elements.chooseModelFolder));
  elements.chooseDraftModelFolder.addEventListener("click", () => openFolderPicker("speculative_draft_model_path", elements.chooseDraftModelFolder));
  elements.validationSummary.addEventListener("click", () => focusValidationError());
  elements.folderPickerClose.addEventListener("click", () => closeFolderPicker());
  elements.folderPickerCancel.addEventListener("click", () => closeFolderPicker());
  elements.folderPickerSelect.addEventListener("click", selectCurrentFolder);
  elements.folderPickerLocation.addEventListener("submit", (event) => {
    event.preventDefault();
    loadFolderPicker(elements.folderPickerCurrent.value.trim());
  });
  elements.folderPickerDrive.addEventListener("change", () => {
    if (elements.folderPickerDrive.value) loadFolderPicker(elements.folderPickerDrive.value);
  });
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
    // The native modal handles focus trapping and Escape before the editor.
    if (elements.automaticConfigDialog.open) return;
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
  if (event.target.closest("[data-clear-profile-search]")) {
    resetProfileSearch();
    renderProfiles();
    elements.profileSearch.focus();
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
  document.querySelector(".app-shell").classList.toggle("webui-active", ["webui", "harness", "opencode", "codex"].includes(view));
  harness?.navigate(view);
  for (const agent of Object.values(nativeAgents)) agent.navigate(view);
  elements.openWebui.classList.toggle("hidden", view === "webui");
  if (view === "webui") renderWebUIAvailability();
  if (view === "logs") scrollLogsToBottom();
  if (view === "download") {
    document.querySelector('[data-view-button="download"]').classList.remove("has-activity");
  }
  if (view === "hardware" && !state.hardwareLoaded) loadHardware();
  renderViewTitle();
}

function renderViewTitle() {
  document.querySelector(".management-label").textContent = ["webui", "harness", "opencode", "codex"].includes(state.currentView)
    ? "agent" : t("Model management");
  const titles = {
    launch: t("Launch service"),
    webui: t("Studio"),
    harness: "DeepSeek Harness",
    opencode: "OpenCode",
    codex: "Codex",
    download: t("Download model"),
    logs: t("Runtime logs"),
    hardware: t("Hardware")
  };
  elements.currentViewTitle.textContent = pluginHost?.title(state.currentView) || titles[state.currentView] || titles.launch;
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
  renderInferenceSpeed();
  renderContextWindow();
  renderWebUIAvailability();
  harness?.update();
  for (const agent of Object.values(nativeAgents)) agent.update();
  renderProfiles();
  updateActionAvailability();
}

function renderInferenceSpeed() {
  const runtime = state.runtime || {};
  const visible = runtime.command === "server" && runtime.phase === "running" && runtime.ready;
  elements.inferenceStatusBar.classList.toggle("hidden", !visible);
  if (!visible || state.inferenceSpeedSession !== runtime.sessionId) {
    state.inferenceSpeedSamples = {};
    state.inferenceSpeedSession = runtime.sessionId;
  }
  const now = performance.now();
  for (const kind of ["prefill", "decode"]) {
    const speed = runtime.speed?.[`${kind}TokensPerSecond`];
    const updatedAt = runtime.speed?.[`${kind}UpdatedAt`];
    let sample = state.inferenceSpeedSamples[kind];
    const valid = visible && typeof updatedAt === "number" && Number.isFinite(updatedAt);
    if (valid && sample?.updatedAt !== updatedAt) {
      // Use the server clock for sample age and a local monotonic deadline.
      // Repeated polls of the same sample must not extend its lifetime.
      const age = typeof runtime.reportedAt === "number" && Number.isFinite(runtime.reportedAt)
        ? Math.max(0, (runtime.reportedAt - updatedAt) * 1000) : 0;
      sample = { updatedAt, expiresAt: now + Math.max(0, INFERENCE_SPEED_TIMEOUT_MS - age) };
      state.inferenceSpeedSamples[kind] = sample;
    }
    const active = valid && sample && now < sample.expiresAt
      && typeof speed === "number" && Number.isFinite(speed) && speed > 0;
    const value = elements[`${kind}Speed`];
    value.textContent = active
      ? speed.toLocaleString(state.locale, { maximumFractionDigits: 1 })
      : "0";
    value.classList.toggle("active", active);
  }
}

function renderContextWindow() {
  const tokens = state.runtime?.contextWindowTokens;
  const known = Number.isSafeInteger(tokens) && tokens > 0;
  elements.contextWindow.textContent = known
    ? (tokens >= 1024
      ? `${(Math.floor(tokens / 1024 * 100) / 100).toLocaleString(state.locale, { maximumFractionDigits: 2 })}K`
      : tokens.toLocaleString(state.locale))
    : "—";
  elements.contextWindow.classList.toggle("active", known);
  elements.contextWindowMetric.title = known
    ? t("Available context per session (input + output): {tokens} tokens. 1K = 1024 tokens.", {
      tokens: tokens.toLocaleString(state.locale)
    })
    : t("Context capacity has not been reported yet.");
}

function updateActionAvailability() {
  const active = runtimeIsActive(state.runtime);
  const invalid = Boolean(state.preview?.errors?.length);
  const automaticBusy = state.automaticConfigStatus?.phase === "loading" || (
    state.automaticConfigPending
    && state.automaticConfigStatus?.phase === "waiting"
    && Boolean(String(collectForm().model || "").trim())
  );
  const missingModel = isSimpleNewProfile() && !String(collectForm().model || "").trim();
  elements.startRuntime.disabled = active || invalid || automaticBusy || missingModel;
  elements.saveProfile.disabled = automaticBusy || missingModel;
  const hint = active
    ? t("A service is running. Saved changes apply on the next model start.")
    : invalid ? t("Fix the highlighted fields before starting. You can still save an unfinished profile.")
    : automaticBusy ? t("Wait for automatic configuration to finish.")
    : missingModel ? t("Choose a model before saving or starting.") : "";
  elements.launchActionHint.textContent = hint;
  elements.startRuntime.title = hint;
  elements.startRuntime.setAttribute("aria-describedby", "launch-action-hint");
  renderSaveState();
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


export {
  state, elements, request, t, localizeDisplayCommand, localizeServerText, switchView, createIcon,
  renderValidation, refreshRuntime, renderRuntime, updateActionAvailability, runtimeIsActive,
  basename, showConfirmation, showToast, friendlyError
};
