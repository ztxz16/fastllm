// Focused form-state regressions; no browser, server or dependencies required.
import assert from "node:assert/strict";
import { ImageWorkspace } from "../../tools/fastllm_pytools/ui_plugins/studio/image-workspace.js";

const ws = Object.create(ImageWorkspace.prototype);
const fields = Object.fromEntries(
  ["generate", "files", "reuse", "status", "quality", "ratio"].map((k) => [k, {}]),
);
ws.$ = (name) => fields[name];
ws.locale = "en-US";
for (const quality of ["2k", "4k"]) {
  fields.quality.value = quality;
  for (const [ratio, expected] of Object.entries({
    square: 1,
    landscape: 4 / 3,
    portrait: 3 / 4,
    wide: 16 / 9,
    tall: 9 / 16,
  })) {
    fields.ratio.value = ratio;
    const [w, h] = ws.size().split("x").map(Number);
    assert.ok(Math.abs(w / h - expected) < 1e-12, `${quality} ${ratio} has incorrect geometry`);
    assert.ok(w * h <= 20 * 1024 * 1024);
  }
}
const modes = [{}, {}, {}];
Object.assign(ws, {
  files: [],
  mode: "multi",
  ready: true,
  submitting: false,
  preparingFiles: false,
  root: { querySelectorAll: () => modes },
});
ws.renderReferences = () => {};
ws.renderCurl = () => {};
ws.error = (message) => {
  ws.lastError = message;
};
let finish;
let closed = false;
globalThis.createImageBitmap = () =>
  new Promise((resolve) => {
    finish = resolve;
  });
const first = new File(["first"], "first.png", { type: "image/png" });
const pending = ws.addFiles([first]);
assert.equal(ws.preparingFiles, true);
assert.equal(fields.generate.disabled, true);
assert.equal(fields.files.disabled, true);
assert.ok(modes.every((b) => b.disabled));
ws.setMode("edit");
assert.equal(ws.mode, "multi");
await ws.addFiles([new File(["second"], "second.png", { type: "image/png" })]);
await ws.generate(); // Must return before attempting to submit a half-prepared input.
await ws.reuse();
finish({
  width: 64,
  height: 96,
  close() {
    closed = true;
  },
});
await pending;
assert.equal(closed, true);
assert.deepEqual(
  ws.files.map((item) => item.file.name),
  ["first.png"],
);
assert.equal(ws.preparingFiles, false);
assert.equal(fields.generate.disabled, false);
assert.equal(fields.files.disabled, false);
URL.revokeObjectURL(ws.files[0].url);
globalThis.createImageBitmap = async () => {
  throw Error("invalid image");
};
await ws.addFiles([first]);
assert.equal(ws.lastError, "invalid image");
assert.equal(ws.preparingFiles, false);
assert.equal(fields.generate.disabled, false);
console.log("PASS aspect ratios, upload serialization, mode/submit/reuse guards and failure recovery");
