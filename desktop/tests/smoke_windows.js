"use strict";

// Run with the bundled Electron's ELECTRON_RUN_AS_NODE=1. No npm dependencies.
const assert = require("node:assert/strict");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const { spawn, execFileSync } = require("node:child_process");
const { setTimeout: delay } = require("node:timers/promises");

async function freePort() {
  const server = net.createServer();
  await new Promise((resolve) => server.listen(0, "127.0.0.1", resolve));
  const port = server.address().port;
  await new Promise((resolve) => server.close(resolve));
  return port;
}

async function until(operation, timeout = 60_000) {
  const end = Date.now() + timeout;
  let lastError;
  while (Date.now() < end) {
    try {
      const result = await operation();
      if (result) return result;
    } catch (error) {
      if (error.fatal) throw error;
      lastError = error;
    }
    await delay(200);
  }
  throw lastError || new Error(`Timed out after ${timeout} ms`);
}

async function connect(url) {
  const socket = new WebSocket(url);
  await new Promise((resolve, reject) => {
    socket.addEventListener("open", resolve, { once: true });
    socket.addEventListener("error", reject, { once: true });
  });
  const pending = new Map();
  const exceptions = [];
  let id = 0;
  socket.addEventListener("message", ({ data }) => {
    const message = JSON.parse(data);
    if (message.method === "Runtime.exceptionThrown") exceptions.push(message.params.exceptionDetails);
    const entry = pending.get(message.id);
    if (entry) {
      clearTimeout(entry.timer);
      pending.delete(message.id);
      if (message.error) entry.reject(new Error(JSON.stringify(message.error)));
      else entry.resolve(message.result);
    }
  });
  function send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const requestId = ++id;
      const timer = setTimeout(() => {
        pending.delete(requestId);
        reject(new Error(`CDP timeout: ${method}`));
      }, 120_000);
      pending.set(requestId, { resolve, reject, timer });
      socket.send(JSON.stringify({ id: requestId, method, params }));
    });
  }
  async function evaluate(expression) {
    const result = await send("Runtime.evaluate", { expression, awaitPromise: true, returnByValue: true });
    if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
    return result.result.value;
  }
  return { send, evaluate, exceptions, close: () => socket.close() };
}

async function main() {
  assert.equal(process.platform, "win32");
  const bundle = path.resolve(process.argv[2]);
  const output = path.resolve(process.argv[3]);
  const model = process.argv[4] && path.resolve(process.argv[4]);
  const tensorParallel = Number(process.argv[5] || 1);
  assert.ok(Number.isInteger(tensorParallel) && tensorParallel >= 1 && tensorParallel <= 64);
  fs.mkdirSync(output, { recursive: true });
  const temporary = fs.mkdtempSync(path.join(output, "run-"));
  const debugPort = await freePort();
  const launcherPort = await freePort();
  const environment = Object.fromEntries(Object.entries(process.env).filter(([key]) =>
    !/^(PYTHON|CUDA|CONDA|VIRTUAL_ENV|PIP_|ELECTRON_|FTLLM_|FASTLLM_|PATH$|HF_HOME$|MODELSCOPE_CACHE$)/i.test(key)));
  environment.PATH = path.join(environment.SystemRoot || "C:\\Windows", "System32");
  environment.FTLLM_LAUNCHER_DATA_DIR = path.join(temporary, "data");
  environment.FTLLM_LAUNCHER_PORT = String(launcherPort);
  environment.FTLLM_DESKTOP_SMOKE_TEST = "1";
  const log = fs.openSync(path.join(temporary, "electron.log"), "w");
  const executable = path.join(bundle, "FastLLM-Launcher.exe");
  const child = spawn(executable, [`--remote-debugging-port=${debugPort}`], {
    cwd: temporary, env: environment, windowsHide: true, stdio: ["ignore", log, log],
  });
  const exit = new Promise((resolve, reject) => {
    child.once("error", reject);
    child.once("exit", (code) => resolve(code));
  });
  let cdp;
  let modelPort;
  try {
    const page = await until(async () => {
      if (child.exitCode !== null) throw new Error(`Electron exited ${child.exitCode}; see ${temporary}`);
      const targets = await (await fetch(`http://127.0.0.1:${debugPort}/json/list`)).json();
      return targets.find((item) => item.type === "page" && item.url.startsWith(`http://127.0.0.1:${launcherPort}/`));
    });
    cdp = await connect(page.webSocketDebuggerUrl);
    await cdp.send("Runtime.enable");
    await until(() => cdp.evaluate(`Boolean(document.querySelector('[data-view-button="webui"]'))`));
    const identity = await cdp.evaluate("({ title: document.title, ua: navigator.userAgent, node: typeof require })");
    const buildInfo = JSON.parse(fs.readFileSync(path.join(bundle, "support", "BUILD-INFO.json"), "utf8"));
    assert.ok(identity.ua.split(" ").includes(`Electron/${buildInfo.desktop.electron_version}`));
    assert.equal(identity.node, "undefined");
    async function api(endpoint, body) {
      const options = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
      return cdp.evaluate(`fetch(${JSON.stringify(endpoint)}, ${JSON.stringify(options)}).then(async r => {
        if (!r.ok) throw new Error(r.status + ': ' + await r.text()); return r.json(); })`);
    }
    assert.ok((await api("/api/bootstrap")).defaultProfile);
    assert.ok((await api("/api/hardware")).memory.total > 0);
    for (const asset of ["app.js", "styles.css", "template.html"]) {
      assert.equal(await cdp.evaluate(`fetch('/assets/webui/${asset}').then(r => r.status)`), 200);
    }
    assert.equal(await cdp.evaluate("import('/assets/webui/app.js').then(m => typeof m.mountWebUI)"), "function");
    console.log("[OK] Actual Electron BrowserWindow, sandboxed renderer, authenticated API and bundled Studio module");
    if (model) {
      modelPort = await freePort();
      await api("/api/runtime/start", {
        command: "server", model, model_name: "electron-test", dtype: "auto",
        device: tensorParallel > 1 ? "tp" : "cuda",
        tp: Array.from({ length: tensorParallel }, (_, index) => index).join(","),
        max_batch: "1", enable_thinking: "false",
        host: "127.0.0.1", port: String(modelPort), threads: "4", max_context_length: "2048",
      });
      await until(async () => {
        const state = await api("/api/runtime");
        if (["failed", "error", "exited"].includes(state.phase)) {
          throw Object.assign(new Error(JSON.stringify((await api("/api/logs")).entries.slice(-10))), { fatal: true });
        }
        return state.ready;
      }, 240_000);
      await cdp.evaluate(`document.querySelector('[data-view-button="webui"]').click()`);
      await until(() => cdp.evaluate(`!document.querySelector('#webui-content').classList.contains('hidden')
        && Boolean(document.querySelector('#webui-content').firstElementChild?.shadowRoot?.querySelector('#newChat'))`), 60_000);
      const completion = await (await fetch(`http://127.0.0.1:${modelPort}/v1/chat/completions`, {
        method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: "electron-test", messages: [{ role: "user", content: "Say hello." }],
          max_tokens: 16, temperature: 0, stream: false }), signal: AbortSignal.timeout(90_000),
      })).json();
      assert.ok(completion.choices[0].message.content.trim());
      console.log("[OK] Electron -> GPU model -> embedded Studio -> real chat completion");
      // Leave the model running: closing the actual native window MUST stop it.
    }
    const screenshot = await cdp.send("Page.captureScreenshot", { format: "png" });
    fs.writeFileSync(path.join(temporary, "electron-window.png"), Buffer.from(screenshot.data, "base64"));
    assert.deepEqual(cdp.exceptions, []);
    const powershell = path.join(environment.PATH, "WindowsPowerShell", "v1.0", "powershell.exe");
    execFileSync(powershell, ["-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File",
      path.join(__dirname, "close_window.ps1"), "-ProcessId", String(child.pid)], { windowsHide: true });
    const code = await Promise.race([exit, delay(25_000, undefined, { ref: false }).then(() => { throw new Error("Electron window close did not exit"); })]);
    assert.equal(code, 0);
    for (const port of [launcherPort, modelPort].filter(Boolean)) {
      await assert.rejects(fetch(`http://127.0.0.1:${port}/`, { signal: AbortSignal.timeout(2000) }));
    }
    const desktopLog = fs.readFileSync(path.join(temporary, "data/logs/desktop.log"), "utf8");
    assert.match(desktopLog, /ftllm launch exited \(code=0/);
    assert.doesNotMatch(desktopLog, /token=(?!\[redacted\])/);
    fs.writeFileSync(path.join(temporary, "RESULT.json"), JSON.stringify({
      passed: true, electron: identity.ua, isolatedPath: environment.PATH,
      model: model || null, modelTensorParallel: model ? tensorParallel : null,
      nativeWindowClose: true, managedPortsClosed: true,
    }, null, 2));
    console.log(`[OK] Native window close exits Electron and managed processes; report: ${temporary}`);
  } finally {
    cdp?.close();
    if (child.exitCode === null) {
      try { execFileSync(path.join(environment.PATH, "taskkill.exe"), ["/PID", String(child.pid), "/T", "/F"], { windowsHide: true }); } catch (_) {}
    }
    fs.closeSync(log);
  }
}

main().catch((error) => {
  console.error(error);
  fs.writeFileSync(path.join(path.resolve(process.argv[3]), "FAILURE.txt"), error.stack || String(error));
  process.exitCode = 1;
});
