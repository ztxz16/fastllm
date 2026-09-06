import {mountWebUI} from "./app.js";
const container = document.getElementById("webui-root");
const host = document.createElement("div");
container.append(host);
mountWebUI(host, {basePath:container.dataset.basePath}).catch(error => {
  container.textContent = `WebUI: ${error.message}`;
});
