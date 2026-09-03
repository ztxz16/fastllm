"use strict";

const query = new URLSearchParams(window.location.search);
const state = query.get("state") || "starting";
const detail = query.get("detail") || "";
const useChinese = (navigator.language || "").toLowerCase().startsWith("zh");
const messages = useChinese ? {
  starting: "正在启动本地服务…",
  failed: "启动失败",
  note: "界面仅连接本机，模型与配置不会上传。",
  retry: "重试",
  quit: "退出",
} : {
  starting: "Starting the local service…",
  failed: "Startup failed",
  note: "The interface connects locally; models and profiles are not uploaded.",
  retry: "Retry",
  quit: "Quit",
};

document.documentElement.lang = useChinese ? "zh-CN" : "en";
document.getElementById("status").textContent = messages[state] || messages.starting;
document.getElementById("detail").textContent = detail;
document.getElementById("local-note").textContent = messages.note;
document.getElementById("retry").textContent = messages.retry;
document.getElementById("quit").textContent = messages.quit;
if (state === "failed") {
  document.getElementById("spinner").hidden = true;
  document.getElementById("actions").hidden = false;
  document.body.classList.add("failed");
}
