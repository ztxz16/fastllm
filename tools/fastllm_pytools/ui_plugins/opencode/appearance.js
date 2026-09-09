// Runs on OpenCode's separate origin. Only the registered Launcher parent may
// synchronize appearance; this bridge accepts no API calls or executable content.
(() => {
  const parents = JSON.parse(document.currentScript.dataset.parents || "[]");
  if (window.parent === window) return;
  document.documentElement.dataset.ftllmEmbedded = "";
  function apply(theme) {
    if (theme !== "light" && theme !== "dark") return;
    document.documentElement.dataset.ftllmTheme = theme;
    // OpenCode's native Theme provider listens to this storage key, so code
    // highlighting, dialogs and terminal colors change along with the page.
    const key = "opencode-color-scheme";
    let oldValue = null;
    try { oldValue = localStorage.getItem(key); localStorage.setItem(key, theme); } catch (_) {}
    window.dispatchEvent(new StorageEvent("storage", {key, oldValue, newValue:theme}));
  }
  window.addEventListener("message", event => {
    if (event.source !== window.parent || !parents.includes(event.origin)
        || event.data?.type !== "ftllm:opencode-appearance") return;
    apply(event.data.theme);
  });
  function ready() {
    for (const origin of parents) window.parent.postMessage({type:"ftllm:opencode-ready"}, origin);
  }
  ready();
  window.addEventListener("DOMContentLoaded", ready, {once:true});
})();
