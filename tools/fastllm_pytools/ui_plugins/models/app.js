import {
  state, elements, request, t, localizeServerText, switchView, createIcon, refreshRuntime,
  renderRuntime, updateActionAvailability, runtimeIsActive, basename, showConfirmation, showToast,
  friendlyError
} from "../../assets/app.js";

let profileRenderSignature = "";
let folderPickerTarget = null, folderPickerDirectoriesOnly = false;
const AUTOMATIC_CONFIGURATION_DEFAULTS = Object.freeze({
  device: "auto",
  cuda_device_id: "0",
  tp: "2",
  cudapp: "2",
  threads: "auto",
  gpu_mem_ratio: "0.9",
  low_gpu_mem: false,
  max_batch: "auto",
  max_context_length: "auto",
  kv_cache_dtype: "auto",
  kv_cache_limit: "auto",
  tokens: "auto",
  enable_moe_hybrid: false,
  moe_device: "numa",
  moe_device_layers: "-1",
  moe_device_custom: "",
  moe_atype: "auto",
  ngram_device: "auto",
  speculative_algorithm: "auto",
  speculative_draft_model_path: "",
  mtp: "auto",
  draft_tokens: "auto"
});
const AUTOMATIC_CONFIGURATION_FIELDS = new Set(Object.keys(AUTOMATIC_CONFIGURATION_DEFAULTS));
function resetProfileSearch() {
  state.profileQuery = "";
  elements.profileSearch.value = "";
}

function cloneConfig(config) {
  return JSON.parse(JSON.stringify(config || {}));
}

function fillForm(config) {
  state.speculativeCountField = initialSpeculativeTokenField(config || {});
  for (const input of elements.launchForm.querySelectorAll("[data-field]")) {
    const value = config?.[input.dataset.field];
    if (input.type === "checkbox") {
      input.checked = Boolean(value);
    } else {
      input.value = value === null || value === undefined ? "" : String(value);
    }
  }
  for (const input of elements.configurationModeOptions.querySelectorAll("[data-config-mode]")) {
    input.checked = input.value === (config?.config_mode || "custom");
  }
  state.dirty = false;
  renderSaveState();
  updateConditionalFields();
  renderProfilePresentation();
}

function collectForm() {
  const config = { ...(state.editingConfig || {}), command: "server" };
  for (const input of elements.launchForm.querySelectorAll("[data-field]")) {
    config[input.dataset.field] = input.type === "checkbox" ? input.checked : input.value;
  }
  config.config_mode = elements.configurationModeOptions.querySelector("[data-config-mode]:checked")?.value || "custom";
  return config;
}

function isSimpleNewProfile() {
  return state.currentIndex === null && collectForm().config_mode !== "custom";
}

function renderProfilePresentation() {
  const simple = isSimpleNewProfile();
  const mode = collectForm().config_mode;
  elements.launchForm.classList.toggle("simple-profile", simple);
  elements.profileParameters.classList.toggle("hidden", simple);
  elements.automaticConfigActions.classList.toggle("hidden", simple);
  elements.configurationModeSettings.classList.toggle("hidden", state.currentIndex !== null);
  elements.profileEditorDescription.textContent = state.currentIndex === null
    ? t("Choose a model and configuration mode, then save or start your service.")
    : t("Edit all settings, or click Automatic configuration to choose a mode.");
  elements.configurationModeDescription.textContent = configurationModeDescription(mode);
}

function configurationModeDescription(mode) {
  const descriptions = {
    long_context: t("One conversation at a time; context capacity follows the model and available memory."),
    high_concurrency: t("Automatic batching; context capacity follows the model and available memory."),
    custom: t("Set parameters yourself, or use automatic configuration as a starting point.")
  };
  return descriptions[mode] || descriptions.custom;
}

function openAutomaticConfigurationDialog() {
  if (state.currentIndex === null) {
    configureProfileAutomatically();
    return;
  }
  const config = collectForm();
  const mode = config.config_mode === "high_concurrency" ? "high_concurrency" : "long_context";
  elements.automaticModeOptions.replaceChildren();
  for (const card of elements.configurationModeOptions.children) {
    if (card.querySelector("input").value === "custom") continue;
    const clone = card.cloneNode(true);
    const input = clone.querySelector("input");
    input.name = "automatic-configuration-mode";
    delete input.dataset.configMode;
    input.checked = input.value === mode;
    elements.automaticModeOptions.append(clone);
  }
  elements.automaticEnableSpeculativeDecoding.checked = config.enable_speculative_decoding;
  elements.automaticModeDescription.textContent = configurationModeDescription(mode);
  state.automaticConfigDialogPreviousStatus = state.automaticConfigStatus;
  elements.automaticConfigDialog.showModal();
  renderAutomaticConfigurationStatus();
  elements.automaticModeOptions.querySelector("input:checked")?.focus();
}

function closeAutomaticConfigurationDialog(cancelled = true) {
  if (!elements.automaticConfigDialog.open) return;
  if (cancelled) {
    cancelPendingAutomaticConfiguration();
    state.automaticConfigStatus = state.automaticConfigDialogPreviousStatus || { phase: "idle" };
  }
  elements.automaticConfigDialog.close();
  state.automaticConfigDialogPreviousStatus = null;
  renderAutomaticConfigurationStatus();
  elements.autoConfigureProfile.focus({ preventScroll: true });
}

function handleConfigurationModeChange() {
  cancelPendingAutomaticConfiguration();
  state.editingConfig = collectForm();
  state.dirty = true;
  state.automaticConfigStatus = { phase: "idle" };
  renderProfilePresentation();
  if (isSimpleNewProfile()) {
    prepareAutomaticConfigurationForNewProfile();
    scheduleAutomaticConfiguration(0);
  } else {
    elements.launchForm.querySelector("#editor-advanced").open = true;
  }
  renderSaveState();
  renderAutomaticConfigurationStatus();
  schedulePreview(0);
}

function handleFormChange(event) {
  if (!event.target.matches("[data-field]")) return;
  const changedField = event.target.dataset.field;
  if (changedField === "speculative_algorithm" && event.target.value === "off") {
    for (const [field, value] of Object.entries({ mtp: "0", draft_tokens: "auto", speculative_draft_model_path: "" })) {
      elements.launchForm.querySelector(`[data-field="${field}"]`).value = value;
    }
    elements.launchForm.querySelector('[data-field="enable_speculative_decoding"]').checked = false;
  } else if (changedField === "speculative_algorithm" && event.target.value !== state.editingConfig?.speculative_algorithm) {
    const mtp = elements.launchForm.querySelector('[data-field="mtp"]');
    const draft = elements.launchForm.querySelector('[data-field="draft_tokens"]');
    if (event.target.value === "mtp") {
      const count = Number(draft.value) > 0 ? draft.value : mtp.value;
      mtp.value = Number(count) >= 1 && Number(count) <= 8 ? count
        : collectForm().speculative_draft_model_path ? "auto" : "3";
      draft.value = "auto";
    } else if (["dflash", "dspark"].includes(event.target.value)) {
      if (Number(mtp.value) > 0) draft.value = mtp.value;
      mtp.value = "auto";
    }
  } else if (changedField === "mtp") {
    elements.launchForm.querySelector('[data-field="draft_tokens"]').value = "auto";
  } else if (changedField === "draft_tokens") {
    elements.launchForm.querySelector('[data-field="mtp"]').value = "auto";
  }
  if (changedField === "speculative_algorithm") {
    state.speculativeCountField = initialSpeculativeTokenField(collectForm());
  }
  if (changedField === "speculative_draft_model_path" && !event.target.value.trim()
      && collectForm().speculative_algorithm === "mtp" && state.speculativeCountField === "draft_tokens") {
    elements.launchForm.querySelector('[data-field="mtp"]').value = collectForm().draft_tokens;
    elements.launchForm.querySelector('[data-field="draft_tokens"]').value = "auto";
    state.speculativeCountField = "mtp";
  }
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
  if (changedField === "enable_speculative_decoding") {
    prepareAutomaticConfigurationForNewProfile();
    if (state.automaticConfigPending) scheduleAutomaticConfiguration(0);
  } else if (changedField === "model" && state.automaticConfigPending) {
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
  const speculativeOff = config.speculative_algorithm === "off";
  const mtpCount = speculativeTokenField() === "mtp";
  elements.speculativeMtpField.classList.toggle("hidden", speculativeOff || !mtpCount);
  elements.speculativeDraftTokensField.classList.toggle("hidden", speculativeOff || mtpCount);
  elements.speculativePathField.classList.toggle("hidden", speculativeOff);
  for (const field of ["mtp", "draft_tokens", "speculative_draft_model_path"]) {
    elements.launchForm.querySelector(`[data-field="${field}"]`).disabled = speculativeOff
      || (field === "mtp" && !mtpCount) || (field === "draft_tokens" && mtpCount);
  }
  elements.chooseDraftModelFolder.disabled = speculativeOff;
  const speculativeHints = {
    off: t("Speculative decoding is off. No draft model will be loaded."),
    mtp: t("Built-in MTP needs 1–8 draft tokens. For external MTP, choose a checkpoint file or folder; auto uses its default."),
    dflash: t("Choose a DFlash2 model folder. Auto uses the checkpoint's draft token count; MTP is disabled."),
    dspark: t("Choose a DSpark model folder, or set the token count for built-in DSpark. MTP is disabled."),
    auto: t("Detect the algorithm from the draft model or existing MTP settings. Without either, speculative decoding stays off.")
  };
  elements.speculativeModeHint.textContent = speculativeHints[config.speculative_algorithm] || speculativeHints.auto;
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

function initialSpeculativeTokenField(config) {
  if (config.speculative_algorithm === "mtp") {
    const externalCount = config.speculative_draft_model_path && !(Number(config.mtp) > 0) && Number(config.draft_tokens) > 0;
    return externalCount ? "draft_tokens" : "mtp";
  }
  return (!config.speculative_algorithm || config.speculative_algorithm === "auto") && Number(config.mtp) > 0
    ? "mtp" : "draft_tokens";
}

function speculativeTokenField() {
  return state.speculativeCountField;
}

function automaticConfigurationFingerprint(config = collectForm()) {
  const values = {
    model: String(config.model || "").trim(),
    config_mode: config.config_mode,
    enable_speculative_decoding: config.enable_speculative_decoding
  };
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
  state.automaticConfigPending = isSimpleNewProfile();
  state.automaticConfigStatus = { phase: state.automaticConfigPending ? "waiting" : "idle" };
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

async function configureProfileAutomatically({ automatic = false, selection = null } = {}) {
  const current = collectForm();
  const requested = { ...current, ...selection };
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
      body: JSON.stringify({
        model, name: current.name || current.model_name || "", config_mode: requested.config_mode,
        enable_speculative_decoding: requested.enable_speculative_decoding
      })
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
    state.editingConfig = {
      ...collectForm(), ...selection, ...recommendedFields,
      // Drop inherited overrides so automatic configuration uses the runtime default.
      chunked_prefill_size: "auto"
    };
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
    return true;
  } catch (error) {
    if (requestId !== state.automaticConfigRequestId) return;
    state.automaticConfigStatus = {
      phase: "error",
      error: friendlyError(error)
    };
    renderAutomaticConfigurationStatus();
    return false;
  }
}

function clearProfileInferenceConfiguration() {
  cancelPendingAutomaticConfiguration();
  state.editingConfig = {
    ...collectForm(),
    ...AUTOMATIC_CONFIGURATION_DEFAULTS,
    chunked_prefill_size: "auto"
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
  const kind = detected.isMoe ? t("MoE model") : t("dense model");
  const size = Number(detected.parameterBillions || 0);
  const summary = size > 0
    ? t("Detected a {size}B {kind}; recommended {device}. Weight type is unchanged.", {
        size: String(Math.round(size * 100) / 100), kind, device
      })
    : t("Recommended {device} from the available model and hardware information. Weight type is unchanged.", {
        device
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
  elements.automaticConfigDialogApply.disabled = loading;
  elements.automaticConfigDialogApply.textContent = loading ? t("Analyzing...") : t("Apply configuration");
  elements.automaticEnableSpeculativeDecoding.disabled = loading;
  for (const input of elements.automaticModeOptions.querySelectorAll("input")) input.disabled = loading;
  const dialogError = status.phase === "error" ? status.error
    : status.phase === "missing-model" ? t("Choose a local model before using automatic configuration.") : "";
  elements.automaticConfigDialogError.textContent = dialogError || "";
  elements.automaticConfigDialogError.classList.toggle("hidden", !dialogError);
  elements.autoConfigureProfile.disabled = loading;
  elements.clearProfileConfig.disabled = loading;
  elements.autoConfigureProfile.textContent = loading
    ? t("Analyzing...")
    : t("Automatic configuration");
  elements.automaticConfigStatus.className = "automatic-config-status";
  elements.automaticConfigStatus.replaceChildren();
  if (status.phase === "idle" || (isSimpleNewProfile() && status.phase === "waiting" && !collectForm().model.trim())) {
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
    if (isSimpleNewProfile()) {
      title = t("Configuration is ready");
      message = t("Your model is configured for the selected mode. Save it or start the service.");
      details = [];
    }
    if (status.recommendation?.speculative?.requested) {
      const messages = {
        enabled: t("Built-in MTP detected. Speculative decoding is enabled (3 draft tokens)."),
        no_mtp: t("This model does not declare built-in MTP. Speculative decoding remains off."),
        missing_weights: t("Built-in MTP weights are missing or incomplete. Speculative decoding remains off."),
        unsupported_architecture: t("Automatic MTP is not supported for this model architecture. Speculative decoding remains off."),
        cuda_required: t("MTP requires a supported CUDA configuration. Speculative decoding remains off."),
        unverified: t("Could not verify built-in MTP from the local model files. Speculative decoding remains off.")
      };
      details.push(messages[status.recommendation.speculative.reason] || messages.unverified);
    }
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
    const pending = configurationNeedsRestart();
    elements.saveState.classList.add(pending ? "pending" : "saved");
    elements.saveState.textContent = pending ? t("Saved · restart required") : t("Saved");
  }
}

function configurationNeedsRestart() {
  return state.currentIndex !== null && state.currentIndex === findRunningProfileIndex()
    && state.preview?.runtimeSessionId === state.runtime?.sessionId
    && state.preview?.matchesRunningConfig === false;
}

function renderProfiles() {
  const active = runtimeIsActive(state.runtime);
  const runningIndex = findRunningProfileIndex();
  const signature = JSON.stringify([
    state.locale,
    state.profileQuery,
    state.profiles.map((profile) => [
      profile.name,
      profile.model_name,
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
  const profiles = state.profiles.map((profile, index) => ({ profile, index })).filter(({ profile }) =>
    [profile.name, profile.model_name, profile.model].some(
      (value) => String(value || "").toLocaleLowerCase().includes(query)
    )
  );
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
      ? t("Try another name or model path, or clear the search.")
      : t("Choose a local model and save a profile to launch an API Server.");
    const action = document.createElement("button");
    action.type = "button";
    action.className = "primary-button";
    if (filtered) action.dataset.clearProfileSearch = "";
    else action.dataset.newProfile = "";
    action.textContent = filtered ? t("Clear search") : t("Add launch item");
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
    avatar.textContent = firstVisibleCharacter(profile.name || profile.model_name || "F");
    const copy = document.createElement("div");
    copy.className = "profile-copy";
    const titleRow = document.createElement("div");
    titleRow.className = "profile-title-row";
    const title = document.createElement("strong");
    title.textContent = profile.name || profile.model_name || t("Unnamed profile");
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
    copy.append(titleRow, path, metadata);

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
  elements.launchForm.querySelector("#editor-advanced").open = !isSimpleNewProfile();
}

function renderProfileEditorTitle() {
  elements.profileEditorTitle.textContent = state.currentIndex === null
    ? t("Add launch item")
    : t("Edit launch item");
  renderProfilePresentation();
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
  closeAutomaticConfigurationDialog();
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
  await ensureProfileConfiguration();
  const config = collectForm();
  if (isSimpleNewProfile()) {
    const base = basename(config.model) || config.name;
    const used = new Set(state.profiles.map((profile) => profile.name));
    config.name = base;
    for (let suffix = 2; used.has(config.name); suffix += 1) config.name = `${base} (${suffix})`;
  }
  const result = await request("/api/profiles", {
    method: "POST",
    body: JSON.stringify({ index: state.currentIndex, config })
  });
  state.profiles = result.profiles;
  state.currentIndex = result.index;
  state.editingConfig = cloneConfig(result.profile);
  state.dirty = false;
  resetProfileSearch();
  renderSaveState();
  if (showSuccess) showToast(runtimeIsActive(state.runtime)
    ? t("Profile saved. The running service is unchanged; new settings apply on the next model start.")
    : t("Launch profile saved."), "success");
  return result.profile;
}

async function ensureProfileConfiguration() {
  if (!isSimpleNewProfile()) return;
  if (state.automaticConfigStatus?.phase === "applied"
      && state.automaticConfigAppliedModel === String(collectForm().model || "").trim()) return;
  if (!await configureProfileAutomatically({ automatic: true })) {
    throw new Error(t("Automatic configuration must finish before saving. Retry or choose custom configuration."));
  }
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
    renderSaveState();
    updateActionAvailability();
  } catch (error) {
    if (requestId !== state.previewRequestId) return;
    state.preview = { errors: [friendlyError(error)] };
    elements.commandPreview.textContent = t("Unable to generate command.");
    renderLaunchValidation(state.preview.errors);
    updateActionAvailability();
  }
}

function renderLaunchValidation(errors) {
  const showErrors = errors.length && (!isSimpleNewProfile() || Boolean(collectForm().model.trim()));
  for (const input of elements.launchForm.querySelectorAll('[aria-invalid="true"]')) {
    input.removeAttribute("aria-invalid");
    const descriptions = (input.getAttribute("aria-describedby") || "").split(/\s+/).filter(id => id && !id.startsWith("launch-error-"));
    if (descriptions.length) input.setAttribute("aria-describedby", descriptions.join(" "));
    else input.removeAttribute("aria-describedby");
  }
  elements.launchForm.querySelectorAll("[data-field-error]").forEach(node => node.remove());
  elements.validationMessages.classList.toggle("hidden", !showErrors);
  elements.validationMessages.replaceChildren();
  elements.validationSummary.classList.toggle("hidden", !showErrors);
  elements.validationSummary.textContent = showErrors ? t("Errors: {count} · locate", { count: errors.length }) : "";
  if (!showErrors) return;
  const details = state.preview?.fieldErrors || [];
  const byInput = new Map();
  for (const message of errors) {
    const detail = details.find(item => item.message === message);
    const inputs = [...new Set((detail?.fields || []).map(validationInput).filter(Boolean))];
    for (const input of inputs) {
      if (!byInput.has(input)) byInput.set(input, []);
      if (!byInput.get(input).includes(message)) byInput.get(input).push(message);
    }
    const item = document.createElement("button");
    item.type = "button";
    item.className = "validation-message";
    item.textContent = localizeServerText(message);
    item.addEventListener("click", () => focusValidationError(inputs[0]));
    elements.validationMessages.append(item);
  }
  for (const [input, messages] of byInput) {
    const id = `launch-error-${input.dataset.field}`;
    const note = document.createElement("small");
    note.id = id;
    note.dataset.fieldError = input.dataset.field;
    note.className = "field-error";
    note.textContent = messages.map(localizeServerText).join(" ");
    (input.closest(".field, .switch-field") || input.parentElement).append(note);
    input.setAttribute("aria-invalid", "true");
    input.setAttribute("aria-describedby", [input.getAttribute("aria-describedby"), id].filter(Boolean).join(" "));
  }
}

function validationInput(field) {
  if (["mtp", "draft_tokens"].includes(field)) field = speculativeTokenField();
  const fallback = { device_custom: "device", moe_dtype_custom: "moe_dtype" };
  const inputs = [...elements.launchForm.querySelectorAll("[data-field]")];
  return inputs.find(input => input.dataset.field === field)
    || inputs.find(input => input.dataset.field === fallback[field]);
}

function focusValidationError(input) {
  input ||= [...elements.launchForm.querySelectorAll('[aria-invalid="true"]')].find(node => !node.disabled);
  if (!input) {
    elements.validationMessages.scrollIntoView({ block: "center" });
    elements.validationMessages.querySelector("button")?.focus({ preventScroll: true });
    return;
  }
  const advanced = input.closest("details");
  if (advanced) advanced.open = true;
  input.scrollIntoView({ block: "center" });
  input.focus({ preventScroll: true });
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
    await ensureProfileConfiguration();
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

function openFolderPicker(field, trigger, {input: target, directoriesOnly = false} = {}) {
  state.folderPickerField = field;
  state.folderPickerTrigger = trigger;
  const input = target || elements.launchForm.querySelector(`[data-field="${field}"]`);
  folderPickerTarget = input;
  folderPickerDirectoriesOnly = directoriesOnly;
  state.folderPickerResult = null;
  elements.folderPickerModal.classList.remove("hidden");
  document.body.classList.add("modal-open");
  elements.folderPickerTitle.focus({ preventScroll: true });
  loadFolderPicker(input.value || (!directoriesOnly && collectForm().model) || "");
}

function closeFolderPicker(restoreFocus = true) {
  state.folderPickerRequestId += 1;
  state.folderPickerLoading = false;
  elements.folderPickerModal.classList.add("hidden");
  if (elements.profileEditorModal.classList.contains("hidden")) {
    document.body.classList.remove("modal-open");
  }
  if (restoreFocus) {
    state.folderPickerTrigger?.focus({ preventScroll: true });
  }
}

async function loadFolderPicker(path) {
  const requestId = ++state.folderPickerRequestId;
  state.folderPickerLoading = true;
  state.folderPickerError = "";
  state.folderPickerSelectedFile = "";
  elements.folderPickerCurrent.value = String(path || "");
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
    state.folderPickerSelectedFile = folderPickerDirectoriesOnly ? "" : result.selectedFile || "";
    elements.folderPickerCurrent.value = result.path;
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
  elements.folderPickerTitle.textContent = folderPickerDirectoriesOnly ? t("Choose a workspace folder")
    : state.folderPickerField === "speculative_draft_model_path"
    ? t("Choose a draft model file or folder") : t("Choose a model file or folder");
  document.getElementById("folder-picker-description").textContent = folderPickerDirectoriesOnly
    ? t("Browse the machine running Launcher. Open a folder, then select the current folder.")
    : t("Browse the machine running Launcher. Open a folder, then select a file or use the current folder.");
  elements.folderPickerModal.querySelector(".folder-picker-kicker").textContent = folderPickerDirectoriesOnly
    ? "WORKSPACE LOCATION" : "MODEL LOCATION";
  const drives = state.folderPickerResult?.drives || [];
  elements.folderPickerDriveField.classList.toggle("hidden", !drives.length);
  elements.folderPickerDrive.replaceChildren();
  const currentDrive = elements.folderPickerCurrent.value.match(/^[a-z]:[\\/]/i)?.[0].replace("/", "\\").toUpperCase();
  if (!drives.some((drive) => drive.path.toUpperCase() === currentDrive)) {
    const placeholder = document.createElement("option");
    placeholder.value = "";
    placeholder.textContent = t("Select a drive");
    elements.folderPickerDrive.append(placeholder);
  }
  for (const drive of drives) {
    const option = document.createElement("option");
    option.value = drive.path;
    option.textContent = drive.name;
    option.selected = drive.path.toUpperCase() === currentDrive;
    elements.folderPickerDrive.append(option);
  }
  elements.folderPickerList.replaceChildren();
  elements.folderPickerStatus.className = "folder-picker-status";
  elements.folderPickerStatus.textContent = "";
  elements.folderPickerSelect.textContent = state.folderPickerSelectedFile
    ? t("Select this file") : t("Select this folder");
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
    const loading = document.createElement("div");
    loading.className = "folder-picker-placeholder loading";
    loading.textContent = t("Loading folders...");
    elements.folderPickerList.append(loading);
    return;
  }

  if (state.folderPickerError) {
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
  const folders = result?.folders || [];
  const files = folderPickerDirectoriesOnly ? [] : result?.files || [];
  if (!folders.length && !files.length) {
    const empty = document.createElement("div");
    empty.className = "folder-picker-placeholder";
    empty.textContent = t("This folder is empty.");
    elements.folderPickerList.append(empty);
  } else {
    const fragment = document.createDocumentFragment();
    for (const entry of [...folders.map((folder) => ({ ...folder, isDirectory: true })), ...files]) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "folder-picker-entry";
      button.title = entry.path;
      button.dataset.entryPath = entry.path;
      const selected = entry.path === state.folderPickerSelectedFile;
      button.classList.toggle("selected", selected);
      if (!entry.isDirectory) button.setAttribute("aria-pressed", String(selected));
      const icon = document.createElement("span");
      icon.className = "folder-picker-entry-icon";
      if (!entry.isDirectory) icon.classList.add("file-icon");
      icon.setAttribute("aria-hidden", "true");
      const name = document.createElement("span");
      name.className = "folder-picker-entry-name";
      name.textContent = entry.name;
      const arrow = document.createElement("span");
      arrow.className = "folder-picker-entry-arrow";
      arrow.setAttribute("aria-hidden", "true");
      arrow.textContent = entry.isDirectory ? "›" : (selected ? "✓" : "");
      button.append(icon, name, arrow);
      button.addEventListener("click", () => {
        if (entry.isDirectory) {
          loadFolderPicker(entry.path);
        } else {
          state.folderPickerSelectedFile = selected ? "" : entry.path;
          renderFolderPicker();
          elements.folderPickerList.querySelector(`[data-entry-path="${CSS.escape(entry.path)}"]`)?.focus();
        }
      });
      fragment.append(button);
    }
    elements.folderPickerList.append(fragment);
  }
  if (result?.truncated) {
    elements.folderPickerStatus.textContent = t("Only the first {count} entries are shown.", {
      count: String(folders.length + files.length)
    });
  }
}

function selectCurrentFolder() {
  if (state.folderPickerLoading || state.folderPickerError) return;
  const path = state.folderPickerSelectedFile || state.folderPickerResult?.path;
  if (!path) return;
  const input = folderPickerTarget;
  input.value = path;
  input.dispatchEvent(new Event("input", { bubbles: true }));
  closeFolderPicker(false);
  input.focus({ preventScroll: true });
}


export {
  resetProfileSearch, cloneConfig, fillForm, collectForm, isSimpleNewProfile,
  configurationModeDescription, openAutomaticConfigurationDialog, closeAutomaticConfigurationDialog,
  handleConfigurationModeChange, handleFormChange, defaultServicePort, updateConditionalFields,
  cancelPendingAutomaticConfiguration, prepareAutomaticConfigurationForNewProfile,
  scheduleAutomaticConfiguration, configureProfileAutomatically, clearProfileInferenceConfiguration,
  renderAutomaticConfigurationStatus, renderSaveState, renderProfiles, editProfile, newProfile,
  showProfileEditor, renderProfileEditorTitle, closeProfileEditor, deleteProfile,
  saveProfileAndClose, schedulePreview, renderLaunchValidation, focusValidationError,
  startSavedProfile, startRuntime, schedulePathSuggestions, openFolderPicker, closeFolderPicker,
  loadFolderPicker, renderFolderPicker, selectCurrentFolder
};
