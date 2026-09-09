import {
  state, elements, request, t, friendlyError
} from "../../assets/app.js";
import {
  collectForm
} from "../models/app.js";

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


export {
  loadHardware, renderHardwareStatus, renderHardware, formatBytes
};
