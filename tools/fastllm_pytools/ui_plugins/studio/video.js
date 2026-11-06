import {state, request, t, showToast, friendlyError} from "../../assets/app.js";

let mode = null, initialized = false, polling = false, submitting = false;
let selectedService = "", jobSignature = "", refreshAvailability;
const results = new Map();
const $ = id => document.getElementById(id);
const services = () => state.runtime?.services || (state.runtime?.sessionId ? [state.runtime] : []);
const ready = item => item?.ready && item.phase === "running" && item.command === "server";

function options(select, items, selected) {
  const signature = JSON.stringify([state.locale, items.map(item => [item.sessionId, item.profileName, item.phase])]);
  if (select.dataset.signature !== signature) {
    select.replaceChildren(...items.map(item => {
      const option = document.createElement("option"); option.value = item.sessionId;
      option.textContent = `${item.profileName || item.modelName} · ${t(item.phase)}`;
      return option;
    }));
    select.dataset.signature = signature;
  }
  select.value = items.some(item => item.sessionId === selected) ? selected : (items[0]?.sessionId || "");
}

export function chatRuntime() {
  const items = services().filter(item => ready(item) && (item.kind || "llm") === "llm");
  return items.find(item => item.sessionId === state.studioChatId) || items[0] || null;
}

export function openVideoService(sessionId) {
  mode = "video";
  $("video-service").value = sessionId;
  refreshAvailability();
  pollJobs();
}

export function renderStudioMedia(onChange) {
  refreshAvailability = onChange;
  if (!initialized) {
    initialized = true;
    $("studio-chat-tab").onclick = () => { mode = "chat"; refreshAvailability(); };
    $("studio-video-tab").onclick = () => { mode = "video"; refreshAvailability(); pollJobs(); };
    $("studio-chat-select").onchange = event => { state.studioChatId = event.target.value; refreshAvailability(); };
    $("video-service").onchange = () => { selectedService = ""; renderStudioMedia(refreshAvailability); };
    $("video-form").onsubmit = generate;
    setInterval(() => { if (mode === "video" && state.currentView === "webui") pollJobs(); }, 2000);
    window.addEventListener("pagehide", clearResults);
  }
  const chat = chatRuntime();
  const videos = services().filter(item => item.capabilities?.includes("video.generate"));
  const previous = $("video-service").value;
  options($("video-service"), videos, previous);
  options($("studio-chat-select"), services().filter(item => ready(item) && (item.kind || "llm") === "llm"), chat?.sessionId);
  if (mode === null) mode = chat ? "chat" : videos.length ? "video" : null;
  const videoMode = mode === "video";
  $("video-studio").classList.toggle("hidden", !videoMode);
  $("studio-chat-select-label").classList.toggle("hidden", videoMode);
  $("studio-video-tab").classList.toggle("selected", videoMode);
  $("studio-chat-tab").classList.toggle("selected", !videoMode);
  const service = videos.find(item => item.sessionId === $("video-service").value);
  $("video-generate").disabled = submitting || !ready(service);
  $("video-service-status").textContent = !service ? t("Start a video model in Model management first.")
    : ready(service) ? t("Ready · tasks run in the background") : t(service.phase);
  const pika = service?.modelName === "pika-h3-flex";
  $("video-steps").disabled = pika;
  $("video-steps").placeholder = pika ? "6" : t("Model default");
  if (selectedService !== (service?.sessionId || "")) {
    selectedService = service?.sessionId || "";
    jobSignature = ""; clearResults(); $("video-jobs").replaceChildren();
    if (ready(service)) pollJobs();
  }
  return {videoMode, ready: Boolean(chat || videos.some(ready))};
}

function clearResults() {
  for (const value of results.values()) URL.revokeObjectURL(value);
  results.clear();
}

async function generate(event) {
  event.preventDefault();
  if (submitting || !selectedService) return;
  const serviceId = selectedService;
  const body = {prompt: $("video-prompt").value, size: $("video-size").value, seconds: Number($("video-seconds").value)};
  if ($("video-seed").value) body.seed = Number($("video-seed").value);
  if (!$("video-steps").disabled && $("video-steps").value) body.num_inference_steps = Number($("video-steps").value);
  let requestBody = JSON.stringify(body);
  const first = $("video-first-frame").files[0], last = $("video-last-frame").files[0];
  if ((first?.size || 0) + (last?.size || 0) > 31 * 1024 * 1024) {
    showToast(t("Reference images must total less than 31 MiB."), "error"); return;
  }
  if (first || last) {
    requestBody = new FormData();
    Object.entries(body).forEach(([key, value]) => requestBody.append(key, String(value)));
    if (first) requestBody.append("first_frame", first);
    if (last) requestBody.append("last_frame", last);
  }
  submitting = true; renderStudioMedia(refreshAvailability);
  try {
    await request(`/api/media/${serviceId}/videos`, {method: "POST", body: requestBody});
    showToast(t("Video generation queued."), "success");
    await pollJobs();
  } catch (error) { showToast(friendlyError(error), "error"); }
  finally { submitting = false; renderStudioMedia(refreshAvailability); }
}

async function pollJobs() {
  const serviceId = selectedService;
  if (polling || !serviceId || !ready(services().find(item => item.sessionId === serviceId))) return;
  polling = true;
  try {
    const response = await request(`/api/media/${serviceId}/videos`);
    if (serviceId !== selectedService) return;
    const jobs = (response.data || []).slice().sort((a, b) => b.created_at - a.created_at);
    const signature = JSON.stringify([state.locale, jobs]);
    if (signature === jobSignature) return;
    jobSignature = signature;
    // Keep existing cards/video elements so progress polls never reset playback.
    const root = $("video-jobs");
    for (const [index, job] of jobs.entries()) {
      let card = [...root.children].find(node => node.dataset.jobId === job.id);
      if (!card) {
        card = document.createElement("article"); card.className = "video-job"; card.dataset.jobId = job.id;
        card.innerHTML = '<strong></strong><p></p><progress max="100"></progress><small></small><div class="video-job-actions"></div>';
        root.insertBefore(card, root.children[index] || null);
      }
      card.querySelector("strong").textContent = `${t(job.status)} · ${job.size} · ${job.id}`;
      card.querySelector("p").textContent = job.prompt;
      card.querySelector("progress").value = Number(job.progress || 0);
      card.querySelector("small").textContent = job.error?.message ||
        (["completed", "failed", "cancelled"].includes(job.status) ? t(job.status)
          : `${job.stage || ""} ${job.current || 0}/${job.total || 0}`);
      const actions = card.querySelector(".video-job-actions");
      if (card.dataset.status !== job.status) {
        card.dataset.status = job.status; actions.replaceChildren();
        if (["queued", "in_progress"].includes(job.status)) {
          const cancel = document.createElement("button"); cancel.type = "button"; cancel.className = "secondary-button"; cancel.textContent = t("Cancel");
          cancel.onclick = async () => {
            cancel.disabled = true;
            try { await request(`/api/media/${serviceId}/videos/${job.id}/cancel`, {method: "POST"}); jobSignature = ""; }
            catch (error) { cancel.disabled = false; showToast(friendlyError(error), "error"); }
          };
          actions.append(cancel);
        } else if (job.status === "completed") {
          const preview = document.createElement("button"); preview.type = "button"; preview.className = "primary-button"; preview.textContent = t("Preview and download");
          preview.onclick = async () => {
            preview.disabled = true;
            try {
              const response = await request(`/api/media/${serviceId}/videos/${job.id}/content`, {stream: true});
              const blob = await response.blob();
              if (serviceId !== selectedService || !card.isConnected) return;
              const url = URL.createObjectURL(blob); results.set(job.id, url);
              const video = document.createElement("video"); video.src = url; video.controls = true; video.preload = "metadata";
              const download = document.createElement("a"); download.href = url; download.download = `${job.id}.mp4`; download.textContent = t("Download video");
              card.append(video); actions.replaceChildren(download);
            } catch (error) { preview.disabled = false; showToast(friendlyError(error), "error"); }
          };
          actions.append(preview);
        }
      }
      const button = actions.querySelector("button"), download = actions.querySelector("a");
      if (button) button.textContent = t(job.status === "completed" ? "Preview and download" : "Cancel");
      if (download) download.textContent = t("Download video");
    }
    for (const card of [...root.children]) if (!jobs.some(job => job.id === card.dataset.jobId)) {
      URL.revokeObjectURL(results.get(card.dataset.jobId)); results.delete(card.dataset.jobId); card.remove();
    }
  } catch (error) { $("video-service-status").textContent = friendlyError(error); }
  finally { polling = false; }
}
