import {mountNativeAgent} from "../../plugin-core/native-agent.js";

export function mountOpenCode(options) {
  let frame;
  function syncAppearance() {
    if (!frame?.isConnected) return;
    const theme = document.documentElement.dataset.theme === "dark" ? "dark" : "light";
    frame.contentWindow?.postMessage({type:"ftllm:opencode-appearance", theme}, new URL(frame.src).origin);
  }
  function ready(event) {
    if (frame && event.source === frame.contentWindow && event.origin === new URL(frame.src).origin
        && event.data?.type === "ftllm:opencode-ready") syncAppearance();
  }
  window.addEventListener("message", ready);
  window.addEventListener("ftllm-theme-change", syncAppearance);
  return mountNativeAgent({...options, id:"opencode", name:"OpenCode",
    onFrame(value) { frame = value; frame.addEventListener("load", syncAppearance); },
    destroyContent() {
      window.removeEventListener("message", ready);
      window.removeEventListener("ftllm-theme-change", syncAppearance);
      frame = null;
    }
  });
}
