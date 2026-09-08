"use strict";

const fs = require("node:fs");
const path = require("node:path");
const { StringDecoder } = require("node:string_decoder");

const CONTROL_URL_PATTERN = /https?:\/\/[^\s<>"']*[?&]token=[A-Za-z0-9_-]+/g;
const CONTROL_TOKEN_PATTERN = /([?&]token=)[^&#\s<>"']+/g;

function normalizedHostname(url) {
  return url.hostname.replace(/^\[/, "").replace(/\]$/, "").toLowerCase();
}

function isLoopbackUrl(value, expectedPort = undefined) {
  let parsed;
  try {
    parsed = new URL(value);
  } catch (_error) {
    return false;
  }
  const hostname = normalizedHostname(parsed);
  if (parsed.protocol !== "http:" || !["127.0.0.1", "localhost", "::1"].includes(hostname)) {
    return false;
  }
  if (expectedPort !== undefined && Number(parsed.port) !== Number(expectedPort)) {
    return false;
  }
  return Boolean(parsed.searchParams.get("token"));
}

function extractControlUrl(text, expectedPort = undefined) {
  const candidates = String(text).match(CONTROL_URL_PATTERN) || [];
  return candidates.find((candidate) => isLoopbackUrl(candidate, expectedPort)) || null;
}

function redactControlTokens(text) {
  return String(text).replace(CONTROL_TOKEN_PATTERN, "$1[redacted]");
}

class LineBuffer {
  constructor() {
    this.pending = "";
    this.decoder = new StringDecoder("utf8");
  }

  push(chunk) {
    const combined = this.pending + (Buffer.isBuffer(chunk) ? this.decoder.write(chunk) : String(chunk));
    const lines = combined.split(/\r?\n/);
    this.pending = lines.pop() || "";
    return lines;
  }

  flush() {
    const line = this.pending + this.decoder.end();
    this.pending = "";
    return line ? [line] : [];
  }
}

function pythonExecutable(runtimeRoot, platform = process.platform) {
  return path.join(runtimeRoot, "runtime", ...(platform === "win32" ? ["python.exe"] : ["bin", "python3"]));
}

function pythonSitePackages(runtimeRoot, platform = process.platform) {
  if (platform === "win32") {
    return path.join(runtimeRoot, "runtime", "Lib", "site-packages");
  }
  const libraryRoot = path.join(runtimeRoot, "runtime", "lib");
  if (!fs.existsSync(libraryRoot)) {
    return null;
  }
  const entry = fs.readdirSync(libraryRoot, { withFileTypes: true })
    .filter((candidate) => candidate.isDirectory() && /^python\d+\.\d+$/.test(candidate.name))
    .sort((left, right) => left.name.localeCompare(right.name))[0];
  return entry ? path.join(libraryRoot, entry.name, "site-packages") : null;
}

function bundledLibraryDirectories(runtimeRoot, platform = process.platform) {
  const windows = platform === "win32";
  const directories = windows
    ? [path.join(runtimeRoot, "runtime"), path.join(runtimeRoot, "runtime", "DLLs"),
      path.join(pythonSitePackages(runtimeRoot, platform), "ftllm")]
    : [path.join(runtimeRoot, "runtime", "lib")];
  const sitePackages = pythonSitePackages(runtimeRoot, platform);
  const nvidiaRoot = sitePackages && path.join(sitePackages, "nvidia");
  if (nvidiaRoot && fs.existsSync(nvidiaRoot)) {
    const components = fs.readdirSync(nvidiaRoot, { withFileTypes: true })
      .sort((left, right) => left.name.localeCompare(right.name));
    for (const component of components) {
      if (!component.isDirectory()) {
        continue;
      }
      const libraryDirectory = path.join(nvidiaRoot, component.name, windows ? "bin" : "lib");
      if (fs.existsSync(libraryDirectory)) {
        directories.push(libraryDirectory);
      }
    }
  }
  return directories;
}

function findCertificateBundle(runtimeRoot, platform) {
  const sitePackages = pythonSitePackages(runtimeRoot, platform);
  if (!sitePackages) {
    return null;
  }
  const certificate = path.join(sitePackages, "certifi", "cacert.pem");
  return fs.existsSync(certificate) ? certificate : null;
}

function buildFtllmEnvironment(runtimeRoot, dataRoot, baseEnvironment = process.env, platform = process.platform) {
  const environment = { ...baseEnvironment };
  const windows = platform === "win32";
  const pythonBin = path.dirname(pythonExecutable(runtimeRoot, platform));
  const libraryDirectories = bundledLibraryDirectories(runtimeRoot, platform);
  if (!windows && environment.LD_LIBRARY_PATH) {
    libraryDirectories.push(environment.LD_LIBRARY_PATH);
  }
  // Windows environment keys are case insensitive. Node otherwise selects only
  // one of Path/PATH when spawning, which can accidentally retain the host path.
  const pathKey = Object.keys(environment).find((key) => key.toUpperCase() === "PATH");
  const inheritedPath = pathKey ? environment[pathKey] : (windows ? "" : "/usr/bin:/bin");
  for (const key of Object.keys(environment)) {
    if (["PYTHONHOME", "PYTHONPATH", ...(windows ? ["PATH"] : [])].includes(key.toUpperCase())) {
      delete environment[key];
    }
  }

  environment.FTLLM_HOME = runtimeRoot;
  environment.PATH = windows
    ? [runtimeRoot, path.join(runtimeRoot, "tools"), pythonBin, ...libraryDirectories, inheritedPath].join(";")
    : `${runtimeRoot}:${pythonBin}:${inheritedPath}`;
  environment.PYTHONDONTWRITEBYTECODE = "1";
  environment.PYTHONNOUSERSITE = "1";
  environment.PYTHONUNBUFFERED = "1";
  environment.PYTHONUTF8 = "1";
  environment.XDG_CONFIG_HOME = path.join(dataRoot, "config");
  environment.XDG_CACHE_HOME = path.join(dataRoot, "cache");
  environment.HF_HOME ||= path.join(dataRoot, "cache", "huggingface");
  environment.MODELSCOPE_CACHE ||= path.join(dataRoot, "cache", "modelscope");
  if (windows) {
    delete environment.LD_LIBRARY_PATH;
  } else {
    environment.LD_LIBRARY_PATH = libraryDirectories.join(":");
  }

  const certificate = findCertificateBundle(runtimeRoot, platform);
  if (certificate && !environment.SSL_CERT_FILE) {
    environment.SSL_CERT_FILE = certificate;
  }
  return environment;
}

module.exports = {
  LineBuffer,
  buildFtllmEnvironment,
  bundledLibraryDirectories,
  extractControlUrl,
  isLoopbackUrl,
  pythonExecutable,
  pythonSitePackages,
  redactControlTokens,
};
