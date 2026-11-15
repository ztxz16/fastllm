import { state, t, request, friendlyError } from "../../assets/app.js";
import { renderImageStudio, selectImageService } from "./image.js";
import { VideoWorkspace } from "./video-workspace.js";

let mode = null,
  initialized = false,
  refreshAvailability,
  workspace;
let requestedVideoService = "";
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
  const images = services().filter((item) => item.capabilities?.includes("image.generate"));
  if (mode === null) mode = chat ? "chat" : videos.length ? "video" : images.length ? "image" : null;
  const videoMode = mode === "video";
  const mediaMode = videoMode || mode === "image";
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
  return { mediaMode, ready: Boolean(chat || videos.some(ready) || imageReady) };
}
