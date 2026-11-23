import { state, request } from "../../assets/app.js";
import { ImageWorkspace } from "./image-workspace.js";
let workspace,
  requestedService = "";
const services = () => state.runtime?.services || (state.runtime?.sessionId ? [state.runtime] : []);
export function selectImageService(id) {
  requestedService = id;
}
export function renderImageStudio(active) {
  if (!workspace)
    workspace = new ImageWorkspace(document.getElementById("image-studio"), {
      id: "image",
      locale: state.locale,
      request: (path, options, serviceId) => request(`/api/media/${serviceId}${path}`, options),
      endpoint: (serviceId) => services().find((service) => service.sessionId === serviceId)?.endpoint,
    });
  const items = services()
    .filter((item) => item.capabilities?.includes("image.generate") && !item.capabilities?.includes("interactive.demo"))
    .map((item) => ({
      id: item.sessionId,
      name: item.profileName || item.modelName,
      ready: item.ready && item.phase === "running" && item.command === "server",
    }));
  workspace.update({
    services: items,
    active: active && state.currentView === "webui",
    locale: state.locale,
    requestedService,
  });
  requestedService = "";
  document.getElementById("image-studio").classList.toggle("hidden", !active);
  document.getElementById("studio-image-tab").classList.toggle("selected", active);
  return items.some((item) => item.ready);
}
