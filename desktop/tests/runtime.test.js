"use strict";

const assert = require("node:assert/strict");
const fs = require("node:fs");
const os = require("node:os");
const path = require("node:path");
const test = require("node:test");

const {
  LineBuffer,
  buildFtllmEnvironment,
  extractControlUrl,
  isLoopbackUrl,
  redactControlTokens,
} = require("../app/runtime");

test("extractControlUrl accepts only the expected loopback listener", () => {
  const line = "  Local address: http://127.0.0.1:8123/?token=secret_123";
  assert.equal(extractControlUrl(line, 8123), "http://127.0.0.1:8123/?token=secret_123");
  assert.equal(
    extractControlUrl("http://localhost:8123/?language=zh&token=secret_123", 8123),
    "http://localhost:8123/?language=zh&token=secret_123",
  );
  assert.equal(extractControlUrl(line, 8124), null);
  assert.equal(extractControlUrl("http://example.com:8123/?token=secret", 8123), null);
  assert.equal(isLoopbackUrl("http://[::1]:8123/?token=secret", 8123), true);
  assert.equal(isLoopbackUrl("https://127.0.0.1:8123/?token=secret", 8123), false);
});

test("redactControlTokens does not persist launcher credentials", () => {
  assert.equal(
    redactControlTokens("Open http://127.0.0.1:8000/?token=very-secret now"),
    "Open http://127.0.0.1:8000/?token=[redacted] now",
  );
  assert.equal(
    redactControlTokens("Open http://localhost:8000/?language=zh&token=secret-value#top"),
    "Open http://localhost:8000/?language=zh&token=[redacted]#top",
  );
});

test("LineBuffer preserves split output chunks", () => {
  const decoder = new LineBuffer();
  assert.deepEqual(decoder.push("first\nsec"), ["first"]);
  assert.deepEqual(decoder.push("ond\nthird\n"), ["second", "third"]);
  assert.deepEqual(decoder.flush(), []);
});

test("buildFtllmEnvironment isolates Python and portable data", () => {
  const temporary = fs.mkdtempSync(path.join(os.tmpdir(), "ftllm-desktop-test-"));
  try {
    const runtimeRoot = path.join(temporary, "ftllm");
    const sitePackages = path.join(runtimeRoot, "runtime", "lib", "python3.11", "site-packages");
    fs.mkdirSync(path.join(sitePackages, "nvidia", "cublas", "lib"), { recursive: true });
    fs.mkdirSync(path.join(sitePackages, "certifi"), { recursive: true });
    fs.writeFileSync(path.join(sitePackages, "certifi", "cacert.pem"), "test");
    const dataRoot = path.join(temporary, "data");
    const environment = buildFtllmEnvironment(runtimeRoot, dataRoot, {
      PATH: "/usr/bin",
      PYTHONHOME: "/bad",
      PYTHONPATH: "/also-bad",
      LD_LIBRARY_PATH: "/host/lib",
    });
    assert.equal(environment.PYTHONHOME, undefined);
    assert.equal(environment.PYTHONPATH, undefined);
    assert.equal(environment.PYTHONDONTWRITEBYTECODE, "1");
    assert.equal(environment.XDG_CONFIG_HOME, path.join(dataRoot, "config"));
    assert.match(environment.PATH, /runtime\/bin:\/usr\/bin$/);
    assert.match(environment.LD_LIBRARY_PATH, /nvidia\/cublas\/lib/);
    assert.equal(environment.SSL_CERT_FILE, path.join(sitePackages, "certifi", "cacert.pem"));
  } finally {
    fs.rmSync(temporary, { recursive: true, force: true });
  }
});
