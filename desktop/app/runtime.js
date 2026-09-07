"use strict";

const fs = require("node:fs");
const path = require("node:path");

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
  }

  push(chunk) {
    const combined = this.pending + String(chunk);
    const lines = combined.split(/\r?\n/);
    this.pending = lines.pop() || "";
    return lines;
  }

  flush() {
    const line = this.pending;
    this.pending = "";
    return line ? [line] : [];
  }
}

function pythonSitePackages(runtimeRoot) {
  const libraryRoot = path.join(runtimeRoot, "runtime", "lib");
  if (!fs.existsSync(libraryRoot)) {
    return null;
  }
  const entry = fs.readdirSync(libraryRoot, { withFileTypes: true })
    .filter((candidate) => candidate.isDirectory() && /^python\d+\.\d+$/.test(candidate.name))
    .sort((left, right) => left.name.localeCompare(right.name))[0];
  return entry ? path.join(libraryRoot, entry.name, "site-packages") : null;
}

function bundledLibraryDirectories(runtimeRoot) {
  const directories = [path.join(runtimeRoot, "runtime", "lib")];
  const sitePackages = pythonSitePackages(runtimeRoot);
  const nvidiaRoot = sitePackages && path.join(sitePackages, "nvidia");
  if (nvidiaRoot && fs.existsSync(nvidiaRoot)) {
    const components = fs.readdirSync(nvidiaRoot, { withFileTypes: true })
      .sort((left, right) => left.name.localeCompare(right.name));
    for (const component of components) {
      if (!component.isDirectory()) {
        continue;
      }
      const libraryDirectory = path.join(nvidiaRoot, component.name, "lib");
      if (fs.existsSync(libraryDirectory)) {
        directories.push(libraryDirectory);
      }
    }
  }
  return directories;
}

function findCertificateBundle(runtimeRoot) {
  const sitePackages = pythonSitePackages(runtimeRoot);
  if (!sitePackages) {
    return null;
  }
  const certificate = path.join(sitePackages, "certifi", "cacert.pem");
  return fs.existsSync(certificate) ? certificate : null;
}

function buildFtllmEnvironment(runtimeRoot, dataRoot, baseEnvironment = process.env) {
  const environment = { ...baseEnvironment };
  const pythonBin = path.join(runtimeRoot, "runtime", "bin");
  const libraryDirectories = bundledLibraryDirectories(runtimeRoot);
  if (environment.LD_LIBRARY_PATH) {
    libraryDirectories.push(environment.LD_LIBRARY_PATH);
  }

  environment.FTLLM_HOME = runtimeRoot;
  environment.PATH = `${runtimeRoot}:${pythonBin}:${environment.PATH || "/usr/bin:/bin"}`;
  environment.PYTHONDONTWRITEBYTECODE = "1";
  environment.PYTHONNOUSERSITE = "1";
  environment.PYTHONUNBUFFERED = "1";
  environment.PYTHONUTF8 = "1";
  environment.XDG_CONFIG_HOME = path.join(dataRoot, "config");
  environment.XDG_CACHE_HOME = path.join(dataRoot, "cache");
  environment.LD_LIBRARY_PATH = libraryDirectories.join(":");
  delete environment.PYTHONHOME;
  delete environment.PYTHONPATH;

  const certificate = findCertificateBundle(runtimeRoot);
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
  redactControlTokens,
};
