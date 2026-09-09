// Run before styles are loaded so the first paint uses the saved appearance.
(() => {
  const storageKey = "ftllm-launcher-theme";
  const choices = new Set(["light", "dark"]);
  let preference = "light";
  try {
    const saved = window.localStorage.getItem(storageKey);
    if (choices.has(saved)) preference = saved;
  } catch (_) { /* Appearance still works when browser storage is unavailable. */ }

  function apply() {
    const theme = preference;
    document.documentElement.dataset.theme = theme;
    window.dispatchEvent(new CustomEvent("ftllm-theme-change", {detail: {preference, theme}}));
  }

  window.ftllmLauncherTheme = {
    getPreference: () => preference,
    getResolved: () => document.documentElement.dataset.theme,
    setPreference(value) {
      if (!choices.has(value)) return;
      preference = value;
      try { window.localStorage.setItem(storageKey, preference); } catch (_) {}
      apply();
    }
  };
  window.addEventListener("storage", event => {
    if (event.key !== storageKey && event.key !== null) return;
    preference = choices.has(event.newValue) ? event.newValue : "light";
    apply();
  });
  apply();
})();
