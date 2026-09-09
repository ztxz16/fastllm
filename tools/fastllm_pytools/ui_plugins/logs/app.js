import {
  state, elements, request, t, localizeDisplayCommand, localizeServerText, showToast, friendlyError
} from "../../assets/app.js";

async function refreshLogs() {
  if (state.pollingLogs) return;
  state.pollingLogs = true;
  try {
    const result = await request(`/api/logs?since=${state.lastLogId}`);
    const entries = Array.isArray(result.entries) ? result.entries : [];
    if (entries.length) {
      const wasNearBottom = isLogNearBottom();
      state.logs.push(...entries);
      if (state.logs.length > 1500) state.logs.splice(0, state.logs.length - 1500);
      state.lastLogId = Number(result.lastId || state.lastLogId);
      renderLogs(wasNearBottom);
    }
  } catch (_error) {
    // Ignore transient polling errors while the launcher exits.
  } finally {
    state.pollingLogs = false;
  }
}

function renderLogs(forceBottom = true) {
  elements.logOutput.replaceChildren();
  elements.logCount.textContent = t("{count} entries", { count: state.logs.length });
  if (!state.logs.length) {
    const empty = document.createElement("div");
    empty.className = "log-empty";
    empty.textContent = t("Logs will appear here after the model starts.");
    elements.logOutput.append(empty);
    return;
  }
  const fragment = document.createDocumentFragment();
  for (const entry of state.logs) {
    const line = document.createElement("div");
    line.className = `log-line ${entry.level || "info"}`;
    const timestamp = document.createElement("span");
    timestamp.className = "log-time";
    timestamp.textContent = formatTime(entry.timestamp);
    const source = document.createElement("span");
    source.className = "log-source";
    source.textContent = entry.source || "ftllm";
    const message = document.createElement("span");
    message.className = "log-message";
    message.textContent = localizeDisplayCommand(localizeServerText(entry.message));
    line.append(timestamp, source, message);
    fragment.append(line);
  }
  elements.logOutput.append(fragment);
  if (forceBottom) scrollLogsToBottom();
}

function formatTime(timestamp) {
  const date = new Date(Number(timestamp) * 1000);
  return Number.isNaN(date.getTime()) ? "--:--:--" : date.toLocaleTimeString(state.locale, { hour12: false });
}

function isLogNearBottom() {
  return elements.logOutput.scrollHeight - elements.logOutput.scrollTop - elements.logOutput.clientHeight < 70;
}

function scrollLogsToBottom() {
  window.requestAnimationFrame(() => {
    elements.logOutput.scrollTop = elements.logOutput.scrollHeight;
  });
}

async function clearLogs() {
  try {
    await request("/api/logs", { method: "DELETE" });
    state.logs = [];
    renderLogs();
  } catch (error) {
    showToast(friendlyError(error), "error");
  }
}


export {
  refreshLogs, renderLogs, scrollLogsToBottom, clearLogs
};
