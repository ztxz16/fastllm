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
  elements.downloadValidation.hidden = !errors.length;
}

const categoryLabels = {
  language: "Language models", image: "Image generation", video: "Video generation",
  world: "World models", dialogue: "Audiovisual dialogue"
};

let selectedPreset = "";
let customDraft = null;
let startingDownload = false;

function selectedModel() {
  for (const group of state.downloadCatalog) {
    const model = group.models.find(item => item.preset === selectedPreset);
    if (model) return {group, model};
  }
  return {};
}

function renderCategories(activeCategory) {
  const categories = [...new Set(state.downloadCatalog.map(group => group.category)), "custom"];
  for (const category of categories) {
    let input = elements.downloadCategory.querySelector(`input[value="${category}"]`);
    if (!input) {
      input = document.createElement("input");
      input.type = "radio";
      input.name = "download-category";
      input.value = category;
      const label = document.createElement("label");
      label.append(input, document.createElement("span"));
      elements.downloadCategory.append(label);
    }
    input.checked = category === activeCategory;
    input.nextElementSibling.textContent = category === "custom" ? t("Custom repository") : t(categoryLabels[category]);
  }
}

function renderDownloadCatalog() {
  const {group, model} = selectedModel();
  const catalog = state.downloadCatalog;
  renderCategories(group?.category || "custom");
  elements.downloadCatalogCount.textContent = t("{models} models · {versions} releases", {
    models: catalog.length, versions: catalog.reduce((count, item) => count + item.models.length, 0)
  });
  const groups = catalog.filter(item => item.category === group?.category);
  elements.downloadFamily.replaceChildren(...groups.map(item => new Option(localizeServerText(item.label), item.id)));
  elements.downloadFamily.closest("label").hidden = !group;
  elements.downloadVersions.hidden = !group;
  elements.downloadSourcePanel.hidden = !group;
  elements.downloadCacheHint.hidden = !model?.sources;
  for (const row of elements.downloadForm.querySelectorAll("[data-custom-download]")) row.hidden = Boolean(group);
  elements.downloadTarget.title = elements.downloadTarget.value;
  elements.downloadTarget.scrollLeft = elements.downloadTarget.scrollWidth;
  renderSelection(group, model);
  if (!group) return;
  elements.downloadFamily.value = group.id;
  renderVersions(group, model);
  elements.downloadNote.textContent = model.sources || model.downloadGB ? localizeServerText(model.meta || "") : "";
  const sources = model.sources || [{repo: model.id, files: model.include ? model.include.split(",") : []}];
  const hubs = [...new Set(sources.map(src => src.hub || "ModelScope"))];
  elements.downloadSourceCount.textContent = hubs.join(" + ");
  elements.downloadSources.replaceChildren();
  for (const src of sources) {
    const row = document.createElement("li");
    const details = document.createElement("details");
    const summary = document.createElement("summary");
    const role = document.createElement("strong");
    role.textContent = localizeServerText(src.label || "模型权重");
    const repo = document.createElement("span");
    repo.textContent = (hubs.length > 1 ? (src.hub || "ModelScope") + " · " : "") + src.repo;
    summary.append(role, repo);
    const scope = document.createElement("code");
    scope.textContent = src.files.length ? src.files.map(name => {
      const path = [src.into, src.rename?.[name] || name].filter(Boolean).join("/");
      return path === name ? name : name + " → " + path;
    }).join("\n") : t("Entire repository");
    details.append(summary, scope);
    row.append(details);
    elements.downloadSources.append(row);
  }
}

function renderSelection(group, model) {
  elements.downloadSelection.textContent = group
    ? [localizeServerText(group.label), localizeServerText(model.label)].join(" · ")
    : downloadField("modelId").value.trim() || t("Custom repository");
  renderMobileSummary(model);
  elements.downloadEstimate.replaceChildren();
  if (model?.downloadGB) {
    const label = document.createElement("small");
    label.textContent = t("Estimated package size");
    const size = document.createElement("strong");
    size.textContent = String(model.downloadGB);
    const unit = document.createElement("span");
    unit.textContent = "GB";
    elements.downloadEstimate.append(label, size, unit);
  } else {
    const hint = document.createElement("small");
    hint.textContent = t("Size available when the download connects.");
    elements.downloadEstimate.append(hint);
  }
}

function renderMobileSummary(model) {
  const active = ACTIVE_DOWNLOAD_PHASES.has(state.download?.phase);
  elements.downloadReviewLabel.textContent = t(active ? "View download progress" : "Review download");
  elements.downloadReview.setAttribute("aria-controls", active ? "download-status-title" : "download-checkout-title");
  if (active) {
    elements.downloadReviewValue.textContent = elements.downloadProgressValue.textContent;
  } else {
    elements.downloadReviewValue.textContent = model?.downloadGB
      ? t("About {size} GB", {size: model.downloadGB}) : elements.downloadSelection.textContent;
  }
}

function renderVersions(group, model) {
  const list = elements.downloadOptions;
  if (list.dataset.family !== group.id) {
    list.replaceChildren();
    list.dataset.family = group.id;
    for (const variant of group.models) {
      const row = document.createElement("label");
      row.className = "download-model-option";
      const input = document.createElement("input");
      input.type = "radio";
      input.name = "download-version";
      input.value = variant.preset;
      input.dataset.preset = variant.preset;
      row.append(input, document.createElement("strong"), document.createElement("small"));
      list.append(row);
    }
  }
  for (const [index, variant] of group.models.entries()) {
    const row = list.children[index];
    row.querySelector("input").checked = variant.preset === model.preset;
    row.querySelector("strong").textContent = localizeServerText(variant.label);
    row.querySelector("small").textContent = variant.downloadGB
      ? t("About {size} GB", {size: variant.downloadGB})
      : localizeServerText(variant.meta || "");
  }
}

function downloadField(name) {
  return elements.downloadForm.querySelector(`[data-download-field="${name}"]`);
}

function fillDownloadForm(config) {
  for (const input of elements.downloadForm.querySelectorAll("[data-download-field]")) {
    input.value = config?.[input.dataset.downloadField] ?? "";
  }
  selectedPreset = state.downloadCatalog.flatMap(group => group.models).find(model =>
    !model.sources && model.id === config?.modelId && (model.include || "") === (config?.include || ""))?.preset || "";
  state.downloadTargetAutomatic = true;
  renderDownloadCatalog();
}

function collectDownloadForm() {
  const config = {};
  for (const input of elements.downloadForm.querySelectorAll("[data-download-field]")) {
    config[input.dataset.downloadField] = input.value;
  }
  const {model} = selectedModel();
  if (model?.sources) config.bundle = model.preset;
  return config;
}

function downloadTargetFor(modelId, include = "", bundle = "") {
  const template = String(state.downloadDefaults?.targetDir || "");
  const slash = Math.max(template.lastIndexOf("/"), template.lastIndexOf("\\"));
  const variant = include.split(/[,;\n]/)[0].trim().replace(/[^A-Za-z0-9._-]+/g, "-").replace(/^-+|-+$/g, "");
  const name = bundle || [basename(modelId) || "model", variant].filter(Boolean).join("-");
  return template.slice(0, slash + 1) + name;
}

function selectDownloadPreset(preset) {
  if (!selectedPreset) customDraft = collectDownloadForm();
  selectedPreset = preset;
  const {model} = selectedModel();
  downloadField("modelId").value = model?.id ?? customDraft?.modelId ?? "";
  downloadField("include").value = model ? model.include || "" : customDraft?.include || "";
  downloadField("revision").value = model ? "master" : customDraft?.revision || "master";
  if (state.downloadTargetAutomatic) {
    downloadField("targetDir").value = downloadTargetFor(downloadField("modelId").value, downloadField("include").value,
      model?.sources ? model.preset : "");
  }
  renderDownloadCatalog();
  scheduleDownloadPreview(0);
}

function handleCatalogChange(event) {
  if (event.target.name === "download-category") {
    const group = state.downloadCatalog.find(item => item.category === event.target.value);
    selectDownloadPreset(group?.models[0].preset || "");
    if (!group) downloadField("modelId").focus();
  } else if (event.target.name === "download-version") {
    selectDownloadPreset(event.target.value);
  } else if (event.target === elements.downloadFamily) {
    const group = state.downloadCatalog.find(item => item.id === event.target.value);
    if (group) selectDownloadPreset(group.models[0].preset);
  }
}

function handleDownloadChange(event) {
  if (!event.target.matches("[data-download-field]")) return;
  const field = event.target.dataset.downloadField;
  if (field === "modelId" && !selectedPreset) renderSelection();
  if (field === "targetDir") {
    state.downloadTargetAutomatic = !event.target.value;
    event.target.title = event.target.value;
  } else if ((field === "modelId" || field === "include") && state.downloadTargetAutomatic) {
    downloadField("targetDir").value = downloadTargetFor(downloadField("modelId").value, downloadField("include").value);
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
  if (startingDownload || ACTIVE_DOWNLOAD_PHASES.has(state.download?.phase)) return;
  startingDownload = true;
  const config = collectDownloadForm();
  renderDownload();
  try {
    const preview = await request("/api/download/preview", {
      method: "POST",
      body: JSON.stringify(config)
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
      body: JSON.stringify(config)
    });
    renderDownload();
    switchView("download");
    showToast(t("Model download started."), "success");
  } catch (error) {
    showToast(friendlyError(error), "error", 7000);
    await refreshDownload();
  } finally {
    startingDownload = false;
    renderDownload();
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
  elements.downloadTransfer.hidden = phase === "idle";
  elements.downloadTransfer.dataset.phase = phase;
  const title = titles[phase] || phase;
  if (elements.downloadStatusTitle.textContent !== title) elements.downloadStatusTitle.textContent = title;
  const stages = {fetch: "Fetching component", verify: "Checking component files", convert: "Converting LingBot assets on CPU"};
  elements.downloadStatusMessage.hidden = phase === "idle" || phase === "completed";
  elements.downloadStatusMessage.textContent = phase === "downloading" && stages[download.stage]
    ? [t(stages[download.stage]), download.component].filter(Boolean).join(" · ")
    : localizeServerText(download.message) || "—";
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
  elements.downloadDestination.textContent = download.destination || "";
  elements.downloadStart.disabled = startingDownload || active || Boolean(state.downloadPreview?.errors?.length);
  elements.downloadCancel.classList.toggle("hidden", !active);
  elements.downloadCancel.disabled = phase === "cancelling";
  const completed = phase === "completed" && Boolean(download.destination);
  elements.downloadUseModel.classList.toggle("hidden", !completed);
  renderMobileSummary(selectedModel().model);
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
  renderDownloadValidation, renderDownloadCatalog, fillDownloadForm, handleCatalogChange,
  handleDownloadChange, scheduleDownloadPreview, startDownload, cancelDownload, refreshDownload,
  renderDownload, useDownloadedModel
};
