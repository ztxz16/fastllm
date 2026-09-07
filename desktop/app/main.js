"use strict";

const { app, BrowserWindow, session } = require("electron");
const { spawn } = require("node:child_process");
const fs = require("node:fs");
const net = require("node:net");
const path = require("node:path");
const { pathToFileURL } = require("node:url");

const {
  LineBuffer,
  buildFtllmEnvironment,
  extractControlUrl,
  redactControlTokens,
} = require("./runtime");

const APP_NAME = "FastLLM Launcher";
const STARTUP_TIMEOUT_MS = 60_000;
const SHUTDOWN_TIMEOUT_MS = 8_000;
const loadingPagePath = path.join(__dirname, "loading.html");
const loadingPageUrl = pathToFileURL(loadingPagePath);
const packagedRoot = path.dirname(process.execPath);
const runtimeRoot = path.resolve(
  process.env.FTLLM_RUNTIME_DIR || packagedRoot,
);

function makeWritableDataRoot(preferredRoot) {
  const candidates = [preferredRoot, path.join(app.getPath("appData"), APP_NAME)];
  for (const candidate of candidates) {
    try {
      fs.mkdirSync(candidate, { recursive: true, mode: 0o700 });
      fs.accessSync(candidate, fs.constants.R_OK | fs.constants.W_OK);
      return candidate;
    } catch (error) {
      process.stderr.write(`Cannot use data directory ${candidate}: ${error.message}\n`);
    }
  }
  throw new Error("No writable data directory is available.");
}

const dataRoot = makeWritableDataRoot(
  path.resolve(process.env.FTLLM_LAUNCHER_DATA_DIR || path.join(packagedRoot, "data")),
);
app.setName(APP_NAME);
app.setPath("userData", path.join(dataRoot, "electron"));

let mainWindow = null;
let launcherProcess = null;
let launcherPort = null;
let launcherOrigin = null;
let launcherToken = null;
let startupTimer = null;
let stopPromise = null;
let logStream = null;
let starting = false;
let stopping = false;
let restarting = false;
let quitRequested = false;
let forceQuit = false;

function appendLog(source, line) {
  const safeLine = redactControlTokens(line).replace(/[\r\n]+$/g, "");
  if (!safeLine) {
    return;
  }
  const record = `${new Date().toISOString()} [${source}] ${safeLine}\n`;
  process.stdout.write(record);
  if (logStream) {
    logStream.write(record);
  }
}

function loadingPage(state, detail = "") {
  if (!mainWindow || mainWindow.isDestroyed()) {
    return Promise.resolve();
  }
  return mainWindow.loadFile(loadingPagePath, {
    query: { state, detail: redactControlTokens(String(detail)).slice(0, 4000) },
  });
}

function showFailure(detail) {
  return loadingPage("failed", detail).catch((error) => {
    appendLog("desktop", `Failed to show the startup error: ${error.message}`);
  });
}

function isHttpUrl(value) {
  try {
    return ["http:", "https:"].includes(new URL(value).protocol);
  } catch (_error) {
    return false;
  }
}

function isLoadingPageUrl(value) {
  try {
    const parsed = new URL(value);
    parsed.search = "";
    parsed.hash = "";
    return parsed.href === loadingPageUrl.href;
  } catch (_error) {
    return false;
  }
}

function openChildWindow(url) {
  if (!isHttpUrl(url)) {
    return;
  }
  const child = new BrowserWindow({
    width: 1180,
    height: 820,
    minWidth: 760,
    minHeight: 560,
    show: false,
    autoHideMenuBar: true,
    backgroundColor: "#f5f7fb",
    icon: path.join(__dirname, "icon.png"),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  child.once("ready-to-show", () => child.show());
  child.webContents.setWindowOpenHandler(() => ({ action: "deny" }));
  child.webContents.on("will-navigate", (event, target) => {
    if (!isHttpUrl(target)) {
      event.preventDefault();
    }
  });
  child.loadURL(url).catch((error) => {
    appendLog("desktop", `Failed to open ${url}: ${error.message}`);
  });
}

function attachWindowGuards(window) {
  window.webContents.setWindowOpenHandler(({ url }) => {
    openChildWindow(url);
    return { action: "deny" };
  });
  window.webContents.on("will-navigate", (event, url) => {
    if (url === "ftllm-action://restart") {
      event.preventDefault();
      restartLauncher().catch((error) => {
        appendLog("desktop", `Failed to restart ftllm: ${error.message}`);
        showFailure(error.message);
      });
      return;
    }
    if (url === "ftllm-action://quit") {
      event.preventDefault();
      app.quit();
      return;
    }
    let allowed = isLoadingPageUrl(url);
    try {
      allowed = allowed || (launcherOrigin && new URL(url).origin === launcherOrigin);
    } catch (_error) {
      allowed = false;
    }
    if (!allowed) {
      event.preventDefault();
      openChildWindow(url);
    }
  });
  window.webContents.on("before-input-event", (event, input) => {
    const reload = input.key === "F5"
      || ((input.control || input.meta) && input.key.toLowerCase() === "r");
    if (reload) {
      event.preventDefault();
    }
  });
}

function createWindow() {
  mainWindow = new BrowserWindow({
    width: 1320,
    height: 900,
    minWidth: 980,
    minHeight: 680,
    show: false,
    autoHideMenuBar: true,
    title: APP_NAME,
    backgroundColor: "#f5f7fb",
    icon: path.join(__dirname, "icon.png"),
    webPreferences: {
      contextIsolation: true,
      nodeIntegration: false,
      sandbox: true,
    },
  });
  attachWindowGuards(mainWindow);
  mainWindow.once("ready-to-show", () => mainWindow && mainWindow.show());
  mainWindow.on("closed", () => {
    mainWindow = null;
  });
  return loadingPage("starting");
}

function reserveLoopbackPort() {
  const configured = Number(process.env.FTLLM_LAUNCHER_PORT || 0);
  if (Number.isInteger(configured) && configured >= 1 && configured <= 65535) {
    return Promise.resolve(configured);
  }
  return new Promise((resolve, reject) => {
    const server = net.createServer();
    server.unref();
    server.once("error", reject);
    server.listen({ host: "127.0.0.1", port: 0, exclusive: true }, () => {
      const address = server.address();
      server.close((error) => {
        if (error) {
          reject(error);
        } else {
          resolve(address.port);
        }
      });
    });
  });
}

function validateRuntime() {
  const python = path.join(runtimeRoot, "runtime", "bin", "python3");
  const cli = path.join(runtimeRoot, "runtime", "lib");
  if (!fs.existsSync(python) || !fs.existsSync(cli)) {
    throw new Error(`Bundled FastLLM runtime is incomplete: ${runtimeRoot}`);
  }
  return python;
}

function configureControlHeader() {
  session.defaultSession.webRequest.onBeforeSendHeaders((details, callback) => {
    if (launcherOrigin && launcherToken) {
      try {
        if (new URL(details.url).origin === launcherOrigin) {
          details.requestHeaders["X-FTLLM-Launcher-Token"] = launcherToken;
        }
      } catch (_error) {
        // Leave malformed or non-URL requests untouched.
      }
    }
    callback({ requestHeaders: details.requestHeaders });
  });
  session.defaultSession.setPermissionRequestHandler(
    (_webContents, _permission, callback) => callback(false),
  );
}

async function showLauncher(controlUrl, child) {
  if (launcherProcess !== child) {
    return;
  }
  const parsed = new URL(controlUrl);
  launcherOrigin = parsed.origin;
  launcherToken = parsed.searchParams.get("token");
  clearTimeout(startupTimer);
  startupTimer = null;
  if (!mainWindow || mainWindow.isDestroyed()) {
    return;
  }
  await mainWindow.loadURL(controlUrl);
  mainWindow.setTitle(APP_NAME);
}

function consumeOutput(source, decoder, child, childPort, chunk) {
  for (const line of decoder.push(chunk)) {
    appendLog(source, line);
    const controlUrl = extractControlUrl(line, childPort);
    if (controlUrl && launcherProcess === child && !launcherOrigin) {
      showLauncher(controlUrl, child).catch((error) => {
        if (launcherProcess !== child) {
          return;
        }
        appendLog("desktop", `Failed to load launcher UI: ${error.message}`);
        showFailure(error.message);
      });
    }
  }
}

async function startLauncher() {
  if (launcherProcess || starting || stopping || quitRequested) {
    return;
  }
  starting = true;

  let python;
  try {
    launcherOrigin = null;
    launcherToken = null;
    await loadingPage("starting");
    python = validateRuntime();
    launcherPort = await reserveLoopbackPort();
  } catch (error) {
    appendLog("desktop", error.message);
    await showFailure(error.message);
    return;
  } finally {
    starting = false;
  }
  if (quitRequested) {
    return;
  }

  const configPath = path.join(dataRoot, "config", "fastllm", "tui_commands.json");
  fs.mkdirSync(path.dirname(configPath), { recursive: true, mode: 0o700 });
  const environment = buildFtllmEnvironment(runtimeRoot, dataRoot);
  const arguments_ = [
    "-m", "ftllm.cli", "launch",
    "--host", "127.0.0.1",
    "--port", String(launcherPort),
    "--no-browser",
    "--config", configPath,
  ];
  appendLog("desktop", `Starting bundled ftllm launch on 127.0.0.1:${launcherPort}`);
  launcherProcess = spawn(python, arguments_, {
    cwd: dataRoot,
    env: environment,
    detached: true,
    stdio: ["ignore", "pipe", "pipe"],
  });
  const child = launcherProcess;
  const childPort = launcherPort;
  const stdout = new LineBuffer();
  const stderr = new LineBuffer();
  child.stdout.on("data", (chunk) => {
    consumeOutput("ftllm", stdout, child, childPort, chunk);
  });
  child.stderr.on("data", (chunk) => {
    consumeOutput("ftllm", stderr, child, childPort, chunk);
  });
  child.once("error", (error) => {
    appendLog("desktop", `Failed to start ftllm: ${error.message}`);
    if (launcherProcess === child) {
      launcherProcess = null;
      clearTimeout(startupTimer);
      startupTimer = null;
      showFailure(error.message);
    }
  });
  child.once("exit", (code, signal) => {
    for (const line of [...stdout.flush(), ...stderr.flush()]) {
      appendLog("ftllm", line);
    }
    appendLog("desktop", `ftllm launch exited (code=${code}, signal=${signal || "none"})`);
    if (launcherProcess !== child) {
      return;
    }
    launcherProcess = null;
    clearTimeout(startupTimer);
    startupTimer = null;
    if (quitRequested || restarting || stopping) {
      return;
    }
    if (code === 0 && launcherOrigin) {
      app.quit();
      return;
    }
    const logPath = path.join(dataRoot, "logs", "desktop.log");
    showFailure(`ftllm launch exited with code ${code ?? "unknown"}. See ${logPath}`);
  });

  const timer = setTimeout(() => {
    if (startupTimer === timer) {
      startupTimer = null;
    }
    if (!launcherOrigin && launcherProcess === child) {
      const detail = `Timed out waiting for ftllm launch on port ${childPort}.`;
      appendLog("desktop", detail);
      showFailure(detail);
    }
  }, STARTUP_TIMEOUT_MS);
  startupTimer = timer;
}

function signalLauncher(child, signal) {
  const exited = !child || child.exitCode !== null || child.signalCode !== null;
  if (exited || child.pid === undefined) {
    return;
  }
  try {
    process.kill(-child.pid, signal);
  } catch (_groupError) {
    try {
      child.kill(signal);
    } catch (_processError) {
      // The process already exited.
    }
  }
}

function stopLauncher() {
  clearTimeout(startupTimer);
  startupTimer = null;
  if (stopPromise) {
    return stopPromise;
  }
  if (!launcherProcess) {
    return Promise.resolve();
  }
  stopping = true;
  const child = launcherProcess;
  appendLog("desktop", "Stopping ftllm launch and its managed processes");
  stopPromise = new Promise((resolve) => {
    let settled = false;
    let forceTimer = null;
    const finish = () => {
      if (settled) {
        return;
      }
      settled = true;
      clearTimeout(forceTimer);
      child.off("close", finish);
      stopPromise = null;
      stopping = false;
      resolve();
    };
    child.once("close", finish);
    signalLauncher(child, "SIGTERM");
    forceTimer = setTimeout(() => {
      if (child.exitCode === null && child.signalCode === null) {
        appendLog(
          "desktop",
          "ftllm did not stop in time; terminating its process group",
        );
        signalLauncher(child, "SIGKILL");
      }
      finish();
    }, SHUTDOWN_TIMEOUT_MS).unref();
  });
  return stopPromise;
}

async function restartLauncher() {
  if (restarting) {
    return;
  }
  restarting = true;
  try {
    await loadingPage("starting");
    await stopLauncher();
  } finally {
    restarting = false;
  }
  if (!quitRequested) {
    await startLauncher();
  }
}

const hasSingleInstanceLock = app.requestSingleInstanceLock();
if (!hasSingleInstanceLock) {
  app.quit();
} else {
  app.on("second-instance", () => {
    if (!mainWindow) {
      return;
    }
    if (mainWindow.isMinimized()) {
      mainWindow.restore();
    }
    mainWindow.show();
    mainWindow.focus();
  });

  app.whenReady().then(async () => {
    const logDirectory = path.join(dataRoot, "logs");
    fs.mkdirSync(logDirectory, { recursive: true, mode: 0o700 });
    logStream = fs.createWriteStream(path.join(logDirectory, "desktop.log"), {
      flags: "a",
      mode: 0o600,
    });
    logStream.on("error", (error) => {
      process.stderr.write(`Cannot write desktop log: ${error.message}\n`);
      logStream = null;
    });
    configureControlHeader();
    await createWindow();
    await startLauncher();
  }).catch((error) => {
    appendLog("desktop", `Desktop startup failed: ${error.stack || error.message}`);
    showFailure(error.message);
  });

  app.on("activate", () => {
    if (!mainWindow) {
      createWindow().then(startLauncher).catch((error) => {
        appendLog("desktop", `Failed to recreate the desktop window: ${error.message}`);
        showFailure(error.message);
      });
    }
  });

  app.on("window-all-closed", () => app.quit());
  app.on("before-quit", (event) => {
    if (forceQuit) {
      return;
    }
    if (quitRequested) {
      if (launcherProcess) {
        event.preventDefault();
      }
      return;
    }
    quitRequested = true;
    if (!launcherProcess) {
      if (logStream) {
        logStream.end();
      }
      return;
    }
    event.preventDefault();
    stopLauncher().finally(() => {
      forceQuit = true;
      if (logStream) {
        logStream.end();
      }
      app.quit();
    });
  });
}
