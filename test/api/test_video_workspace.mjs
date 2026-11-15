// Exercise actual workspace request/state methods without a browser, GPU or model.
import assert from "node:assert/strict";
import { VideoWorkspace } from "../../tools/fastllm_pytools/ui_plugins/studio/video-workspace.js";

function fixture(request) {
  const nodes = new Map();
  const workspace = Object.assign(Object.create(VideoWorkspace.prototype), {
    service: "first",
    selected: "job",
    services: [{ id: "first", ready: true }],
    jobs: [{ id: "job", status: "completed", size: "640x384", seconds: 5 }],
    results: new Map(),
    frames: new Map(),
    references: [],
    loading: new Set(),
    cancelling: new Set(),
    revision: 0,
    locale: "zh-CN",
    active: false,
    disposed: false,
    options: { request, friendlyError: (error) => error.message },
    availability() {},
    renderJobs() {},
    renderReferences() {},
    $(id) {
      if (!nodes.has(id))
        nodes.set(id, {
          value: "",
          dataset: {},
          replaceChildren(...children) {
            this.children = children;
          },
          pause() {
            this.pauses = (this.pauses || 0) + 1;
          },
          load() {
            this.loads = (this.loads || 0) + 1;
          },
          removeAttribute(name) {
            delete this[name];
          },
          focus() {},
        });
      return nodes.get(id);
    },
  });
  workspace.$("prompt").value = "A quiet lake";
  workspace.$("size").value = "640x384";
  workspace.$("seconds").value = "5";
  return workspace;
}
const deferred = () => {
  let resolve;
  const promise = new Promise((r) => {
    resolve = r;
  });
  return { promise, resolve };
};

// Re-rendering a completed result for locale/progress updates must preserve playback.
const playback = fixture();
playback.results.set("first/job", "blob:video");
playback.renderSelection();
const player = playback.$("player");
player.currentTime = 2.25;
playback.locale = "en-US";
playback.renderSelection();
playback.renderSelection();
assert.equal(player.pauses, 1);
assert.equal(player.currentTime, 2.25);
assert.equal(player.src, "blob:video");
assert.equal(playback.$("download").hidden, false);

// Switching to Pika shows the actual fixed value and restores the H3 custom value on return.
const steps = fixture();
steps.$("steps").value = "50";
steps.services[0].modelName = "pika-h3-flex";
VideoWorkspace.prototype.availability.call(steps);
assert.equal(steps.$("steps").value, "6");
assert.equal(steps.$("steps").disabled, true);
steps.services[0].modelName = "minimax-h3";
VideoWorkspace.prototype.availability.call(steps);
assert.equal(steps.$("steps").value, "50");
assert.equal(steps.$("steps").disabled, false);

// A response from a previous service cannot populate the newly selected service.
const content = deferred();
const switched = fixture(async () => content.promise);
const preview = switched.preview();
switched.service = "second";
content.resolve({ blob: async () => new Blob(["video"]) });
await preview;
assert.equal(switched.results.size, 0);
assert.equal(switched.loading.size, 0);

// A pending list response cannot undo a newer cancellation or accepted job.
const stale = deferred();
const polling = fixture(async () => stale.promise);
const poll = polling.poll();
polling.jobs[0].status = "cancelled";
++polling.revision;
stale.resolve({ data: [{ id: "job", status: "queued" }] });
await poll;
assert.equal(polling.jobs[0].status, "cancelled");

// Frame mode sends original bytes and optional settings; text mode excludes retained frames.
const requests = [];
const submission = fixture(async (service, path, options) => {
  requests.push({ service, path, options });
  return { id: "new", status: "queued" };
});
submission.poll = async () => {};
submission.mode = "frames";
submission.$("seed").value = "0";
submission.$("steps").value = "50";
submission.$("steps").disabled = true; // Pika must omit a stale custom H3 step count.
submission.frames.set("first", { file: new File(["first-frame-bytes"], "first.png", { type: "image/png" }) });
submission.frames.set("last", { file: new File(["last-frame-bytes"], "last.webp", { type: "image/webp" }) });
await submission.generate({ preventDefault() {} });
const multipart = requests[0].options.body;
assert.ok(multipart instanceof FormData);
assert.equal(await multipart.get("first_frame").text(), "first-frame-bytes");
assert.equal(await multipart.get("last_frame").text(), "last-frame-bytes");
assert.equal(multipart.get("seed"), "0");
assert.equal(multipart.has("num_inference_steps"), false);
submission.mode = "text";
submission.$("steps").disabled = false;
await submission.generate({ preventDefault() {} });
assert.deepEqual(JSON.parse(requests[1].options.body), {
  prompt: "A quiet lake",
  size: "640x384",
  seconds: 5,
  seed: 0,
  num_inference_steps: 50,
});

// Failed requests always re-enable submission and allow retry.
const failed = fixture(async () => {
  throw Error("network unavailable");
});
await failed.generate({ preventDefault() {} });
assert.equal(failed.submitting, false);
assert.equal(failed.$("error").textContent, "network unavailable");

// Mixed references keep original bytes, media types and user order; retained keyframes are excluded.
submission.mode = "reference";
submission.references = [
  { kind: "image", file: new File(["scene"], "scene.png", { type: "image/png" }) },
  { kind: "video", duration: 3, file: new File(["motion"], "motion.mp4", { type: "video/mp4" }) },
  { kind: "image", file: new File(["person"], "person.jpg", { type: "image/jpeg" }) },
  { kind: "audio", duration: 4, file: new File(["sound"], "sound.wav", { type: "audio/wav" }) },
];
assert.deepEqual(submission.referenceLabels(), ["Picture 1", "Video 1", "Picture 2", "Audio 1"]);
submission.moveReference(2, -1);
assert.deepEqual(submission.referenceLabels(), ["Picture 1", "Picture 2", "Video 1", "Audio 1"]);
await submission.generate({ preventDefault() {} });
const mixed = requests.at(-1).options.body;
assert.deepEqual(await Promise.all(mixed.getAll("reference").map((file) => file.text())), [
  "scene",
  "person",
  "motion",
  "sound",
]);
assert.deepEqual(
  mixed.getAll("reference").map((file) => file.type),
  ["image/png", "image/jpeg", "video/mp4", "audio/wav"],
);
assert.equal(mixed.has("first_frame"), false);
assert.equal(mixed.has("last_frame"), false);
submission.removeReference(2);
assert.deepEqual(submission.referenceLabels(), ["Picture 1", "Picture 2", "Audio 1"]);

// Validate real backend boundaries before sending, without imposing image requirements on video-only refs.
const sample = (kind, duration, size = 1) => ({ kind, duration, file: { size } });
assert.equal(submission.referenceError([sample("video", 15)], true), "");
assert.equal(submission.referenceError([sample("image"), sample("audio", 15)], true), "");
for (const invalid of [
  [],
  [sample("audio", 3)],
  Array(10).fill(sample("image")),
  Array(4).fill(sample("video", 2)),
  [...Array(9).fill(sample("image")), ...Array(3).fill(sample("video", 2)), sample("audio", 2)],
  [sample("video", 1.9)],
  [sample("video", 15.1)],
  [sample("video", 8), sample("video", 8)],
  [sample("image"), sample("audio", 8), sample("audio", 8)],
  [sample("image", null, 128 * 1024 * 1024 + 1)],
  [sample("image", null, 128 * 1024 * 1024), sample("image", null, 128 * 1024 * 1024)],
])
  assert.notEqual(submission.referenceError(invalid, true), "");
const before = requests.length;
submission.references = [];
await submission.generate({ preventDefault() {} });
assert.equal(requests.length, before);
submission.references = [sample("image")];
submission.referenceAvailable = false;
await submission.generate({ preventDefault() {} });
assert.equal(requests.length, before);

// An old service health response cannot change the current model's reference availability.
const pendingHealth = deferred();
const health = fixture(async () => pendingHealth.promise);
const healthTask = health.loadCapabilities();
health.service = "second";
pendingHealth.resolve({ reference_model_available: false });
await healthTask;
assert.notEqual(health.referenceAvailable, false);

console.log(
  "PASS mixed reference bytes/order/numbering, validation limits, mode isolation and capability isolation",
);
console.log(
  "PASS playback retention, service isolation, stale polls, ordered frame bytes, Pika defaults and failure recovery",
);
