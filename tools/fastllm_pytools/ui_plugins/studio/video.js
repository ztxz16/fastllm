import { state, t, request, friendlyError } from "../../assets/app.js";
import { renderImageStudio, selectImageService } from "./image.js";
import { VideoWorkspace } from "./video-workspace.js";

let mode = null,
  initialized = false,
  refreshAvailability,
  workspace;
let requestedVideoService = "";
let selectedDemo = "", openedDemo = "";
let demoRequest = 0;
const $ = (id) => document.getElementById(id);
const services = () => state.runtime?.services || (state.runtime?.sessionId ? [state.runtime] : []);
const ready = (item) => item?.ready && item.phase === "running" && item.command === "server";

function options(select, items, selected) {
  const signature = JSON.stringify([
    state.locale,
    items.map((item) => [item.sessionId, item.profileName, item.phase]),
  ]);
  if (select.dataset.signature !== signature) {
    select.replaceChildren(
      ...items.map((item) => {
        const option = document.createElement("option");
        option.value = item.sessionId;
        option.textContent = `${item.profileName || item.modelName} · ${t(item.phase)}`;
        return option;
      }),
    );
    select.dataset.signature = signature;
  }
  select.value = items.some((item) => item.sessionId === selected) ? selected : items[0]?.sessionId || "";
}

export function chatRuntime() {
  const items = services().filter((item) => ready(item) && (item.kind || "llm") === "llm");
  return items.find((item) => item.sessionId === state.studioChatId) || items[0] || null;
}

export function openVideoService(sessionId) {
  mode = "video";
  requestedVideoService = sessionId;
  refreshAvailability();
}

export function openImageService(sessionId) {
  mode = "image";
  selectImageService(sessionId);
  refreshAvailability();
}

export function openInteractiveService(sessionId) {
  selectedDemo = sessionId;
  mode = "interactive";
  refreshAvailability();
}

function updateInteractiveDemo(id) {
  if (id === openedDemo) return;
  openedDemo = id;
  const attempt = ++demoRequest;
  const error = $("interactive-error");
  $("interactive-frame")?.remove();
  error.hidden = true;
  if (!id) return;
  request(`/api/services/${id}/demo`, {method: "POST"}).then(result => {
    if (attempt !== demoRequest) return;
    const frame = document.createElement("iframe");
    frame.id = "interactive-frame";
    frame.title = t("Interactive demo");
    frame.allow = "camera; microphone; autoplay; fullscreen";
    frame.src = result.url;
    $("interactive-studio").append(frame);
  }).catch(reason => {
    if (attempt === demoRequest) {
      error.textContent = friendlyError(reason);
      error.hidden = false;
    }
  });
}

export function renderStudioMedia(onChange) {
  refreshAvailability = onChange;
  if (!initialized) {
    initialized = true;
    workspace = new VideoWorkspace($("video-studio"), {
      request: (id, path, options) => request(`/api/media/${id}${path}`, options),
      friendlyError,
    });
    $("studio-chat-tab").onclick = () => {
      mode = "chat";
      refreshAvailability();
    };
    $("studio-video-tab").onclick = () => {
      mode = "video";
      refreshAvailability();
    };
    $("studio-demo-tab").onclick = () => {
      if (!$("interactive-error").hidden) openedDemo = "";
      mode = "interactive";
      refreshAvailability();
    };
    $("studio-demo-service").onchange = event => openInteractiveService(event.target.value);
    $("studio-image-tab").onclick = () => {
      mode = "image";
      refreshAvailability();
    };
    $("studio-chat-select").onchange = (event) => {
      state.studioChatId = event.target.value;
      refreshAvailability();
    };
  }
  const chat = chatRuntime();
  const videos = services().filter((item) => item.capabilities?.includes("video.generate"));
  options(
    $("studio-chat-select"),
    services().filter((item) => ready(item) && (item.kind || "llm") === "llm"),
    chat?.sessionId,
  );
  const images = services().filter((item) => item.capabilities?.includes("image.generate") && !item.capabilities?.includes("interactive.demo"));
  const demos = services().filter(item => ready(item) && item.capabilities?.includes("interactive.demo"));
  if (mode === "interactive" && !demos.length) mode = null;
  if (mode === null) mode = chat ? "chat" : videos.length ? "video" : images.length ? "image" : demos.length ? "interactive" : null;
  const demoMode = mode === "interactive";
  options($("studio-demo-service"), demos, selectedDemo);
  selectedDemo = $("studio-demo-service").value;
  $("studio-demo-tab").classList.toggle("hidden", !demos.length);
  $("studio-demo-tab").classList.toggle("selected", demoMode);
  $("studio-demo-service").classList.toggle("hidden", !demoMode || demos.length < 2);
  $("interactive-studio").classList.toggle("hidden", !demoMode);
  if (!demos.some(item => item.sessionId === openedDemo)) updateInteractiveDemo("");
  if (demoMode && state.currentView === "webui") updateInteractiveDemo(selectedDemo);
  const videoMode = mode === "video";
  const mediaMode = videoMode || mode === "image" || demoMode;
  const imageReady = renderImageStudio(mode === "image");
  workspace.update({
    services: videos.map((item) => ({
      id: item.sessionId,
      name: item.profileName || item.modelName,
      modelName: item.modelName,
      ready: ready(item),
    })),
    active: videoMode && state.currentView === "webui",
    locale: state.locale,
    requestedService: requestedVideoService,
  });
  requestedVideoService = "";
  $("video-studio").classList.toggle("hidden", !videoMode);
  $("studio-chat-select-label").classList.toggle("hidden", mediaMode);
  $("studio-video-tab").classList.toggle("selected", videoMode);
  $("studio-chat-tab").classList.toggle("selected", !mediaMode);
  return { mediaMode, ready: Boolean(chat || videos.some(ready) || imageReady || demos.length) };
}
