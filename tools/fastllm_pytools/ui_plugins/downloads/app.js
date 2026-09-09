import {
  state, elements, request, t, localizeDisplayCommand, localizeServerText, switchView,
  renderValidation, basename, showToast, friendlyError
} from "../../assets/app.js";
import {
  cloneConfig, fillForm, defaultServicePort, updateConditionalFields,
  cancelPendingAutomaticConfiguration, prepareAutomaticConfigurationForNewProfile,
  scheduleAutomaticConfiguration, renderSaveState, showProfileEditor, schedulePreview,
  schedulePathSuggestions
} from "../models/app.js";
import {
  formatBytes
} from "../hardware/app.js";

const ACTIVE_DOWNLOAD_PHASES = new Set(["starting", "downloading", "cancelling"]);
function renderDownloadValidation(errors) {
  renderValidation(elements.downloadValidation, errors, t("Download configuration is valid and ready."));
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


export {
  renderDownloadValidation, renderDownloadCatalog, fillDownloadForm, selectDownloadPreset,
  handleDownloadChange, scheduleDownloadPreview, startDownload, cancelDownload, refreshDownload,
  renderDownload, useDownloadedModel
};
