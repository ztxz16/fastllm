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
  elements.downloadCatalog.replaceChildren();
  for (const group of state.downloadCatalog) {
    const optgroup = document.createElement("optgroup");
    optgroup.label = localizeServerText(group.label || group.id || t("Models"));
    const section = document.createElement("section");
    section.className = `download-catalog-group accent-${group.accent || "slate"}`;
    const heading = document.createElement("div");
    heading.className = "download-catalog-heading";
    const title = document.createElement("div");
    const eyebrow = document.createElement("span");
    eyebrow.className = "download-catalog-eyebrow";
    eyebrow.textContent = "MODEL CLUSTER";
    const headingText = document.createElement("h2");
    headingText.textContent = localizeServerText(group.label || group.id || t("Models"));
    title.append(eyebrow, headingText);
    const description = document.createElement("p");
    description.textContent = localizeServerText(group.description || "");
    heading.append(title, description);
    const models = document.createElement("div");
    models.className = "download-catalog-models";
    for (const model of group.models || []) {
      const preset = model.preset || model.id;
      const option = document.createElement("option");
      option.value = preset;
      option.textContent = model.label || model.id;
      optgroup.append(option);
      const button = document.createElement("button");
      button.type = "button";
      button.className = "download-model-option";
      button.dataset.preset = preset;
      button.setAttribute("aria-pressed", preset === selected ? "true" : "false");
      const modelLabel = document.createElement("strong");
      modelLabel.textContent = model.label || model.id;
      const modelMeta = document.createElement("small");
      modelMeta.textContent = localizeServerText(model.meta || model.id);
      button.append(modelLabel, modelMeta);
      button.addEventListener("click", () => {
        elements.downloadPreset.value = preset;
        selectDownloadPreset();
      });
      models.append(button);
    }
    if (optgroup.children.length) elements.downloadPreset.append(optgroup);
    if (models.children.length) {
      section.append(heading, models);
      elements.downloadCatalog.append(section);
    }
  }
  const custom = document.createElement("option");
  custom.value = "custom";
  custom.textContent = t("Custom model ID");
  elements.downloadPreset.append(custom);
  elements.downloadPreset.value = [...elements.downloadPreset.options].some((option) => option.value === selected)
    ? selected
    : "custom";
  const active = elements.downloadPreset.value;
  for (const button of elements.downloadCatalog.querySelectorAll("[data-preset]")) {
    button.setAttribute("aria-pressed", button.dataset.preset === active ? "true" : "false");
  }
}

function findDownloadModel(preset) {
  for (const group of state.downloadCatalog) {
    const model = (group.models || []).find((item) => (item.preset || item.id) === preset);
    if (model) return model;
  }
  return null;
}

function findDownloadPreset(modelId, include = "") {
  let fallback = null;
  for (const group of state.downloadCatalog) {
    for (const model of group.models || []) {
      if (model.id !== modelId) continue;
      const preset = model.preset || model.id;
      if (String(model.include || "") === String(include || "")) return preset;
      if (!fallback && !model.include) fallback = preset;
    }
  }
  return fallback;
}

function fillDownloadForm(config) {
  for (const input of elements.downloadForm.querySelectorAll("[data-download-field]")) {
    const value = config?.[input.dataset.downloadField];
    input.value = value === null || value === undefined ? "" : String(value);
  }
  const modelId = String(config?.modelId || "");
  elements.downloadPreset.value = findDownloadPreset(modelId, config?.include || "") || "custom";
  state.downloadTargetAutomatic = true;
  renderDownloadCatalog();
}

function collectDownloadForm() {
  const config = {};
  for (const input of elements.downloadForm.querySelectorAll("[data-download-field]")) {
    config[input.dataset.downloadField] = input.value;
  }
  return config;
}

function downloadTargetFor(modelId, include = "") {
  const template = String(state.downloadDefaults?.targetDir || "");
  const templateModel = basename(state.downloadDefaults?.modelId || "");
  const slash = Math.max(template.lastIndexOf("/"), template.lastIndexOf("\\"));
  const root = slash >= 0 ? template.slice(0, slash) : template;
  const separator = template.includes("\\") && !template.includes("/") ? "\\" : "/";
  const variant = String(include || "").split(/[,;\n]/)[0].trim()
    .replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  const name = `${basename(modelId) || templateModel || "model"}${variant ? `-${variant}` : ""}`;
  return root ? `${root}${separator}${name}` : name;
}

function selectDownloadPreset() {
  if (elements.downloadPreset.value === "custom") {
    renderDownloadCatalog();
    elements.downloadForm.querySelector('[data-download-field="modelId"]').focus();
    return;
  }
  const model = findDownloadModel(elements.downloadPreset.value);
  if (!model) return;
  const modelInput = elements.downloadForm.querySelector('[data-download-field="modelId"]');
  const targetInput = elements.downloadForm.querySelector('[data-download-field="targetDir"]');
  const includeInput = elements.downloadForm.querySelector('[data-download-field="include"]');
  modelInput.value = model.id;
  if (includeInput) includeInput.value = model.include || "";
  if (state.downloadTargetAutomatic) targetInput.value = downloadTargetFor(modelInput.value, model.include);
  renderDownloadCatalog();
  scheduleDownloadPreview(0);
}

function handleDownloadChange(event) {
  if (!event.target.matches("[data-download-field]")) return;
  const field = event.target.dataset.downloadField;
  if (field === "modelId") {
    const includeInput = elements.downloadForm.querySelector('[data-download-field="include"]');
    elements.downloadPreset.value = findDownloadPreset(event.target.value, includeInput?.value || "") || "custom";
    if (state.downloadTargetAutomatic) {
      const targetInput = elements.downloadForm.querySelector('[data-download-field="targetDir"]');
      targetInput.value = downloadTargetFor(event.target.value, includeInput?.value || "");
    }
  } else if (field === "targetDir") {
    state.downloadTargetAutomatic = false;
  } else if (field === "include") {
    const modelInput = elements.downloadForm.querySelector('[data-download-field="modelId"]');
    elements.downloadPreset.value = findDownloadPreset(modelInput?.value || "", event.target.value) || "custom";
  }
  renderDownloadCatalog();
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
  const destination = state.download?.modelPath || state.download?.destination;
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
  showToast(t(state.editingConfig.config_mode === "auto"
    ? "Choose a model and service settings. Devices, memory and inference settings are planned when the service starts."
    : "A new launch item was created and is being configured automatically."), "success");
}


export {
  renderDownloadValidation, renderDownloadCatalog, fillDownloadForm, selectDownloadPreset,
  handleDownloadChange, scheduleDownloadPreview, startDownload, cancelDownload, refreshDownload,
  renderDownload, useDownloadedModel
};
