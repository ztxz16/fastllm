// Only validated, declarative values reach the application shell. Plugin CSS
// stays in its iframe; this adapter owns and restores every property it changes.
export function mountAppearance({root, basePath, changed, studio = Boolean(root.host), styles = true}) {
  const element = root.host || root.documentElement;
  const style = document.createElement("link");
  style.rel = "stylesheet"; style.href = basePath + "/plugin-core/appearance.css";
  if (styles) (root.head || root).append(style);
  const properties = new Map();
  const colors = {
    background:studio ? ["--canvas"] : ["--bg"], surface:["--surface"],
    surfaceMuted:studio ? ["--soft"] : ["--surface-soft"],
    text:studio ? ["--ink", "--text"] : ["--ink"], mutedText:["--muted", "--subtle", "--faint"],
    border:["--line", "--line-strong"], primary:["--primary", "--brand", "--action", "--focus"],
    primaryHover:["--primary-dark", "--brand-strong", "--action-hover"], primaryText:["--plugin-primary-text"],
    primarySoft:["--primary-soft", "--brand-soft"], sidebar:["--plugin-sidebar"], topbar:["--plugin-topbar"]
  };
  const fonts = {system:'system-ui, "PingFang SC", "Microsoft YaHei", sans-serif',
    serif:'Georgia, "Songti SC", serif', mono:'ui-monospace, "SFMono-Regular", Consolas, monospace'};
  let theme = null, signature = "", palette = {};
  function set(name, value) {
    if (!properties.has(name)) properties.set(name, [element.style.getPropertyValue(name), element.style.getPropertyPriority(name)]);
    element.style.setProperty(name, value);
  }
  function clear() {
    for (const [name, [value, priority]] of properties) {
      if (value) element.style.setProperty(name, value, priority);
      else element.style.removeProperty(name);
    }
    properties.clear();
    delete element.dataset.pluginSkin; delete element.dataset.pluginSidebar;
    delete element.dataset.pluginSpacing;
  }
  function render() {
    const next = JSON.stringify([theme, element.dataset.theme]);
    if (signature === next) return;
    signature = next; clear(); palette = {};
    if (theme) {
      element.dataset.pluginSkin = theme.id;
      palette = theme.theme[element.dataset.theme === "dark" ? "dark" : "light"] || {};
      for (const [key, value] of Object.entries(palette)) for (const name of colors[key]) set(name, value);
      const layout = theme.theme.layout || {};
      if (layout.font) set("font-family", fonts[layout.font]);
      if (layout.radius !== undefined) set("--plugin-radius", layout.radius + "px");
      if (!studio) {
        if (layout.sidebarSide) element.dataset.pluginSidebar = layout.sidebarSide;
        if (layout.spacing) element.dataset.pluginSpacing = layout.spacing;
        if (layout.sidebarWidth) set("--plugin-sidebar-width", layout.sidebarWidth + "px");
        if (layout.contentWidth) set("--plugin-content-width", layout.contentWidth + "px");
      }
    }
    changed();
  }
  const observer = new MutationObserver(render);
  observer.observe(element, {attributes:true, attributeFilter:["data-theme"]});
  return {
    update(records) {
      theme = records.find(p => !p.builtin && p.enabled && p.revision && p.slot === "theme") || null;
      render();
    },
    palette:() => palette,
    destroy() { observer.disconnect(); clear(); style.remove(); }
  };
}
