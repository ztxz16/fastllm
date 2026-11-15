const presets = [
  [
    "自然风光",
    "Nature",
    "A quiet alpine lake at sunrise. Mist drifts over the water as the camera slowly glides forward. Pine trees sway gently, warm light reaches the mountain peaks. Natural colors, cinematic photography, soft wind and birdsong.",
  ],
  [
    "产品特写",
    "Product",
    "An amber glass perfume bottle on a travertine pedestal. The camera slowly orbits the bottle as warm sunlight moves across the glass. Olive leaves sway in the foreground, refined editorial photography, soft ambient sound.",
  ],
  [
    "电影人像",
    "Portrait",
    "A woman in a linen coat walks through a sunlit old town. The camera tracks gently beside her, a light breeze moves her hair. Shallow depth of field, warm film colors, natural footsteps and distant street ambience.",
  ],
];
const terminal = (status) => ["completed", "failed", "cancelled"].includes(status);

export class VideoWorkspace {
  constructor(root, options) {
    this.root = root;
    this.options = options;
    this.services = [];
    this.service = "";
    this.jobs = [];
    this.selected = "";
    this.mode = "text";
    this.results = new Map();
    this.frames = new Map();
    this.references = [];
    this.addingReferences = false;
    this.referenceAvailable = null;
    this.healthService = "";
    this.health = null;
    this.loading = new Set();
    this.cancelling = new Set();
    this.polling = false;
    this.submitting = false;
    this.revision = 0;
    this.disposed = false;
    for (const [id, file] of [
      ["h3-workspace-style", "image-workspace.css"],
      ["h3-video-style", "video-workspace.css"],
    ]) {
      if (document.getElementById(id)) continue;
      const style = document.createElement("link");
      style.id = id;
      style.rel = "stylesheet";
      style.href = new URL(file, import.meta.url).href;
      document.head.append(style);
    }
    root.classList.add("h3-workspace", "h3-video-workspace");
    root.innerHTML = /* HTML */ ` <div class="h3-heading">
        <div>
          <div class="h3-eyebrow">MINIMAX H3 / VIDEO STUDIO</div>
          <h2 data-copy="title"></h2>
          <p data-copy="subtitle"></p>
        </div>
        <span class="h3-badge">LOCAL CREATIVE STUDIO</span>
      </div>
      <div class="h3-layout">
        <form id="video-form" class="h3-controls">
          <div class="h3-service">
            <label for="video-service" data-copy="model"></label
            ><select id="video-service"></select>
            <div id="video-runtime" class="h3-runtime" role="status" hidden></div>
          </div>
          <div class="h3-modes" role="group" data-label="mode">
            <button type="button" data-mode="text" data-copy="textMode"></button
            ><button type="button" data-mode="frames" data-copy="frameMode"></button
            ><button type="button" data-mode="reference" data-copy="referenceMode"></button>
          </div>
          <div id="video-frames" class="h3-field" hidden>
            <div class="h3-frame-grid">
              ${["first", "last"]
                .map(
                  (slot) => `<div class="h3-frame-slot">
                <div class="h3-field-head"><label for="video-${slot}-frame" data-copy="${slot}"></label><button type="button" class="h3-text-button" data-remove="${slot}" data-copy="remove" hidden></button></div>
                <label class="h3-drop h3-frame-drop"><img data-frame="${slot}" alt="" hidden/><span class="h3-frame-placeholder"><b>＋</b><small data-copy="upload"></small></span><input id="video-${slot}-frame" type="file" accept="image/*" data-label="${slot}"/></label>
                <p class="h3-frame-name" data-name="${slot}"></p>
              </div>`,
                )
                .join("")}
            </div>
            <p class="h3-hint" data-copy="frameHint"></p>
          </div>
          <div id="video-references" class="h3-field" hidden>
            <div class="h3-field-head">
              <label for="video-reference-files" data-copy="references"></label><span>01 — REFERENCES</span>
            </div>
            <label class="h3-drop h3-reference-drop"
              ><b>＋</b><span data-copy="addReferences"></span><small data-copy="referenceTypes"></small
              ><input
                id="video-reference-files"
                type="file"
                multiple
                accept="image/*,video/*,audio/*"
                data-label="addReferences"
            /></label>
            <p id="video-reference-summary" class="h3-hint" role="status"></p>
            <div id="video-reference-list" class="h3-reference-list"></div>
            <p class="h3-hint" data-copy="referenceHint"></p>
            <details class="h3-reference-help">
              <summary data-copy="referenceLimits"></summary>
              <p class="h3-hint" data-copy="referenceLimitHint"></p>
            </details>
          </div>
          <p id="video-reference-unavailable" class="h3-hint" hidden data-copy="referenceUnavailable"></p>
          <div class="h3-field">
            <div class="h3-field-head">
              <label for="video-prompt" data-copy="prompt"></label><span>01 — STORY</span>
            </div>
            <textarea
              id="video-prompt"
              required
              maxlength="16000"
              data-placeholder="promptPlaceholder"
            ></textarea>
            <div class="h3-presets" id="video-presets"></div>
            <p class="h3-hint" data-copy="promptHint"></p>
          </div>
          <div class="h3-row">
            <div class="h3-field">
              <label for="video-size" data-copy="size"></label
              ><select id="video-size">
                <option>640x384</option>
                <option>384x640</option>
                <option>832x480</option>
                <option>1344x768</option>
                <option>768x1344</option>
              </select>
            </div>
            <div class="h3-field">
              <label for="video-seconds" data-copy="duration"></label
              ><select id="video-seconds">
                <option value="5">5 s</option>
                <option value="10">10 s</option>
                <option value="14">14 s</option>
              </select>
            </div>
          </div>
          <details class="h3-video-advanced">
            <summary data-copy="advanced"></summary>
            <div class="h3-row">
              <div class="h3-field">
                <label for="video-seed" data-copy="seed"></label
                ><input id="video-seed" type="number" min="0" max="2147483647" data-placeholder="random" />
              </div>
              <div class="h3-field">
                <label for="video-steps" data-copy="steps"></label
                ><input id="video-steps" type="number" min="2" max="100" />
              </div>
            </div>
            <p class="h3-hint" id="video-step-hint"></p>
          </details>
          <button id="video-generate" class="h3-submit" type="submit"></button>
          <p id="video-service-status" class="h3-status" role="status"></p>
          <div id="video-error" class="h3-error" role="alert"></div>
        </form>
        <div class="h3-stage">
          <div class="h3-canvas h3-video-canvas">
            <span class="h3-canvas-label">YOUR NEXT SCENE</span>
            <div id="video-empty" class="h3-empty">
              <div class="h3-film-art" aria-hidden="true"><span>▶</span></div>
              <h3 data-copy="emptyTitle"></h3>
              <p data-copy="emptyHint"></p>
            </div>
            <video
              id="video-player"
              controls
              playsinline
              preload="metadata"
              hidden
              data-label="preview"
            ></video>
          </div>
          <div id="video-progress-panel" class="h3-progress" hidden>
            <div class="h3-progress-head">
              <span id="video-progress-label" role="status"></span
              ><button id="video-cancel" type="button"></button>
            </div>
            <progress id="video-progress" max="100" data-label="progress"></progress>
            <p id="video-job-error" class="h3-error" role="alert"></p>
          </div>
          <div class="h3-result-footer">
            <span id="video-result-info" class="h3-result-info"></span>
            <div class="h3-result-actions">
              <button id="video-preview" type="button" hidden></button
              ><a id="video-download" hidden data-copy="download"></a>
            </div>
          </div>
          <div class="h3-history-heading">
            <span data-copy="history"></span
            ><button id="video-refresh" class="h3-text-button" type="button" data-copy="refresh"></button>
          </div>
          <p id="video-history-empty" class="h3-hint" data-copy="historyEmpty"></p>
          <div id="video-jobs" class="h3-video-history" role="group" data-label="history"></div>
          <p class="h3-hint h3-agent-note" data-copy="agentHint"></p>
        </div>
      </div>`;
    this.$("form").onsubmit = (event) => this.generate(event);
    this.$("service").onchange = (event) => this.changeService(event.target.value);
    this.$("refresh").onclick = () => this.poll();
    this.$("preview").onclick = () => this.preview();
    this.$("cancel").onclick = () => this.cancel();
    this.$("reference-files").onchange = () => this.addReferences([...this.$("reference-files").files]);
    for (const button of root.querySelectorAll("[data-mode]"))
      button.onclick = () => this.setMode(button.dataset.mode);
    for (const slot of ["first", "last"]) {
      this.$(`${slot}-frame`).onchange = () => this.setFrame(slot);
      root.querySelector(`[data-remove="${slot}"]`).onclick = () => this.clearFrame(slot);
    }
    this.$("player").onerror = () => {
      if (this.$("player").hasAttribute("src"))
        this.error(
          this.tr(
            "视频无法播放，请下载后使用本地播放器打开。",
            "This video cannot play here. Download it to open in a local player.",
          ),
        );
    };
    this.timer = setInterval(() => {
      if (this.active) this.poll();
    }, 2000);
    this.pagehide = (event) => {
      if (!event.persisted) this.dispose();
    };
    window.addEventListener("pagehide", this.pagehide);
  }
  $(name) {
    return this.root.querySelector(`#video-${name}`);
  }
  tr(zh, en) {
    return this.locale === "en-US" ? en : zh;
  }
  status(value) {
    const labels = {
      queued: ["排队中", "Queued"],
      in_progress: ["生成中", "Generating"],
      completed: ["已完成", "Completed"],
      failed: ["失败", "Failed"],
      cancelled: ["已取消", "Cancelled"],
    };
    return labels[value] ? this.tr(...labels[value]) : value;
  }
  localize() {
    const copy = {
      title: ["让画面，开始讲故事。", "Give your vision a little motion."],
      subtitle: [
        "用文字构思镜头，让图片、动作与声音成为创作灵感。",
        "Build a scene from words, images, motion and sound.",
      ],
      model: ["视频模型", "Video model"],
      mode: ["创作模式", "Creation mode"],
      textMode: ["文生视频", "Text to video"],
      frameMode: ["首尾帧动画", "Animate frames"],
      referenceMode: ["参考素材", "References"],
      references: ["参考素材", "Reference materials"],
      addReferences: ["添加参考素材", "Add references"],
      referenceTypes: ["图片 · 视频 · 音频，可多选", "Images, video and audio · select multiple"],
      referenceHint: [
        "点击素材编号插入提示词。排序后编号会更新，请检查描述中的对应关系。",
        "Click a label to insert it into the prompt. After reordering, check that your prompt matches the new labels.",
      ],
      referenceLimits: ["素材数量与上传限制", "Reference limits"],
      referenceLimitHint: [
        "最多 9 张图、3 段视频、3 段音频，合计 12 份。音视频每段 2–15 秒，视频累计与独立音频累计分别不超过 15 秒。音频需搭配图片或视频；此模式不使用首尾帧。单文件 ≤128 MiB，素材合计 ≤255 MiB（为请求预留空间）。",
        "Up to 9 images, 3 videos and 3 audio clips, 12 files total. Each clip: 2–15 s; video and standalone audio totals: 15 s each. Audio requires an image or video. Keyframes are excluded. Up to 128 MiB per file, 255 MiB of files per request (with room for form data).",
      ],
      referenceUnavailable: [
        "此服务未安装参考模型权重，参考素材模式暂不可用。",
        "Reference model weights are missing from this service.",
      ],
      first: ["首帧图片（可选）", "First frame (optional)"],
      last: ["尾帧图片（可选）", "Last frame (optional)"],
      remove: ["移除", "Remove"],
      upload: ["点击上传", "Upload image"],
      frameHint: [
        "可使用首帧、尾帧或两者。单图 ≤128 MiB，合计 ≤255 MiB。",
        "Use a first frame, a last frame, or both. Up to 128 MiB each, 255 MiB combined.",
      ],
      prompt: ["镜头描述", "Describe your scene"],
      promptPlaceholder: [
        "描述主体、动作、镜头运动与声音…",
        "Describe the subject, movement, camera and sound…",
      ],
      promptHint: [
        "写清画面如何变化；镜头运动与声音也可以一起描述。",
        "Describe how the scene unfolds, including camera movement and sound.",
      ],
      size: ["分辨率", "Resolution"],
      duration: ["时长", "Duration"],
      advanced: ["高级设置", "Advanced settings"],
      seed: ["随机种子", "Seed"],
      steps: ["采样步数", "Sampling steps"],
      random: ["随机", "Random"],
      emptyTitle: ["下一段故事，从这里开场", "Every story starts with a scene"],
      emptyHint: [
        "写下镜头描述，或带上参考素材。生成后可在这里预览和下载视频。",
        "Start with a description or reference materials. Preview and download your video here.",
      ],
      preview: ["生成的视频", "Generated video"],
      progress: ["视频生成进度", "Video generation progress"],
      download: ["下载视频", "Download video"],
      history: ["最近的创作", "Recent creations"],
      refresh: ["刷新", "Refresh"],
      historyEmpty: ["这里将保存本次服务的创作记录。", "Your creations from this service will appear here."],
      agentHint: [
        "也可以在对话工作区告诉 Pi Agent 你的创意，让它调用视频模型并保存结果。",
        "You can also ask Pi Agent in the chat workspace to create a video and save the result.",
      ],
    };
    for (const [key, attr] of [
      ["copy", null],
      ["label", "aria-label"],
      ["placeholder", "placeholder"],
    ]) {
      for (const node of this.root.querySelectorAll(`[data-${key}]`)) {
        const text = this.tr(...copy[node.dataset[key]]);
        if (attr) node.setAttribute(attr, text);
        else node.textContent = text;
      }
    }
    this.$("presets").replaceChildren(
      ...presets.map(([zh, en, prompt]) => {
        const button = document.createElement("button");
        button.type = "button";
        button.textContent = this.tr(zh, en);
        button.onclick = () => {
          this.$("prompt").value = prompt;
        };
        return button;
      }),
    );
    this.setMode(this.mode);
    this.renderReferences();
    this.renderJobs();
  }
  update({ services, active, locale, requestedService }) {
    this.services = services;
    this.active = active;
    if (locale !== this.locale) {
      this.locale = locale;
      this.localize();
    }
    const signature = JSON.stringify([locale, services.map((item) => [item.id, item.name, item.ready])]);
    if (signature !== this.serviceSignature) {
      this.serviceSignature = signature;
      this.$("service").replaceChildren(
        ...(services.length
          ? services
          : [{ id: "", name: this.tr("请先启动视频模型", "Start a video model first") }]
        ).map((item) => {
          const option = document.createElement("option");
          option.value = item.id;
          option.textContent = item.name;
          return option;
        }),
      );
    }
    const wanted = requestedService || this.service;
    const id = services.some((item) => item.id === wanted) ? wanted : services[0]?.id || "";
    if (id !== this.service) this.changeService(id);
    this.$("service").value = id;
    this.availability();
    if (services.find((item) => item.id === id)?.ready && this.healthService !== id) this.loadCapabilities();
    if (active) this.poll();
  }
  availability() {
    this.showRuntime();
    const service = this.services.find((item) => item.id === this.service);
    this.$("generate").disabled =
      this.submitting ||
      this.addingReferences ||
      !service?.ready ||
      (this.mode === "reference" && this.referenceAvailable === false);
    this.$("service").disabled = this.submitting || !this.services.length;
    this.$("generate").textContent = this.submitting
      ? this.tr("正在提交…", "Submitting…")
      : this.tr("生成视频  ↗", "Generate video  ↗");
    this.$("service-status").textContent = service?.ready
      ? this.tr("服务已就绪 · 任务在后台运行", "Ready · tasks run in the background")
      : this.tr("请先在模型管理中启动视频模型", "Start a video model in Model management first.");
    const pika = service?.modelName === "pika-h3-flex";
    if (pika && !this.$("steps").disabled) {
      this.customSteps = this.$("steps").value;
      this.$("steps").value = "6";
    } else if (!pika && this.$("steps").disabled) {
      this.$("steps").value = this.customSteps || "";
    }
    this.$("steps").disabled = pika;
    this.$("steps").placeholder = this.$("steps").disabled ? "6" : this.tr("模型默认", "Model default");
    this.$("step-hint").textContent = this.$("steps").disabled
      ? this.tr("Pika 固定使用 6 步。", "Pika uses a fixed 6-step schedule.")
      : this.tr("留空使用 MiniMax H3 模型默认步数。", "Leave blank to use the MiniMax H3 model default.");
  }
  changeService(id) {
    ++this.revision;
    this.service = id;
    this.healthService = "";
    this.health = null;
    this.referenceAvailable = null;
    this.showReferenceAvailability();
    this.jobs = [];
    this.selected = "";
    this.clearResults();
    this.renderJobs();
    this.availability();
    this.error("");
    if (this.active) this.poll();
  }
  setMode(mode) {
    this.mode = mode;
    this.$("frames").hidden = mode !== "frames";
    this.$("references").hidden = mode !== "reference";
    for (const button of this.root.querySelectorAll("[data-mode]"))
      button.setAttribute("aria-pressed", String(button.dataset.mode === mode));
    this.availability();
  }
  async loadCapabilities() {
    const id = this.service;
    this.healthService = id;
    try {
      const health = await this.options.request(id, "/health");
      if (this.disposed || this.service !== id) return;
      this.health = health;
      this.referenceAvailable = health.reference_model_available !== false;
      this.showReferenceAvailability();
      this.availability();
    } catch {
      // Older services may omit the health route; generation still validates their inputs.
    }
  }
  showReferenceAvailability() {
    this.root.querySelector('[data-mode="reference"]').disabled = this.referenceAvailable === false;
    this.$("reference-unavailable").hidden = this.referenceAvailable !== false;
  }
  showRuntime() {
    const root = this.$("runtime"), health = this.health;
    root.replaceChildren();
    const precision = this.mode === "reference" ? health?.reference_dit_precision : health?.dit_precision;
    root.hidden = !precision;
    if (!precision) return;
    const mode = health.dit_parallel_mode || "single";
    const count = mode === "sp" ? health.dit_sequence_parallel_size : health.dit_tensor_parallel_size;
    const values = [precision.toUpperCase(), mode === "single" ? this.tr("单卡", "Single GPU") : `${mode.toUpperCase()} × ${count || 1}`];
    for (const value of values) {
      const badge = document.createElement("span");
      badge.textContent = value;
      root.append(badge);
    }
  }
  error(message) {
    this.$("error").textContent = message;
  }
  clearFrame(slot) {
    const frame = this.frames.get(slot);
    if (frame) URL.revokeObjectURL(frame.url);
    this.frames.delete(slot);
    this.$(`${slot}-frame`).value = "";
    const image = this.root.querySelector(`[data-frame="${slot}"]`);
    image.hidden = true;
    image.removeAttribute("src");
    this.root.querySelector(`[data-remove="${slot}"]`).hidden = true;
    this.root.querySelector(`[data-name="${slot}"]`).textContent = "";
  }
  setFrame(slot) {
    const file = this.$(`${slot}-frame`).files[0];
    if (!file) return;
    const other = this.frames.get(slot === "first" ? "last" : "first");
    if (
      !file.type.startsWith("image/") ||
      !file.size ||
      file.size > 128 * 1024 * 1024 ||
      file.size + (other?.file.size || 0) > 255 * 1024 * 1024
    ) {
      this.$(`${slot}-frame`).value = "";
      this.error(
        this.tr(
          "请使用非空图片：单图 ≤128 MiB，合计 ≤255 MiB。",
          "Use non-empty images: up to 128 MiB each, 255 MiB combined.",
        ),
      );
      return;
    }
    const old = this.frames.get(slot);
    if (old) URL.revokeObjectURL(old.url);
    const url = URL.createObjectURL(file);
    this.frames.set(slot, { file, url });
    const image = this.root.querySelector(`[data-frame="${slot}"]`);
    image.src = url;
    image.hidden = false;
    this.root.querySelector(`[data-remove="${slot}"]`).hidden = false;
    this.root.querySelector(`[data-name="${slot}"]`).textContent = file.name;
    this.error("");
  }
  referenceError(items, requireVisual = false) {
    const counts = { image: 0, video: 0, audio: 0 },
      durations = { video: 0, audio: 0 };
    for (const item of items) {
      if (!["image", "video", "audio"].includes(item.kind) || !item.file.size)
        return this.tr("请上传非空的图片、视频或音频。", "Upload non-empty images, video or audio.");
      counts[item.kind]++;
      if (item.file.size > 128 * 1024 * 1024)
        return this.tr("单个素材不能超过 128 MiB。", "Each reference must be at most 128 MiB.");
      if (item.kind !== "image" && item.duration != null) {
        if (!Number.isFinite(item.duration) || item.duration < 2 || item.duration > 15)
          return this.tr(
            "每段视频、音频的时长须为 2–15 秒。",
            "Each video or audio clip must be 2–15 seconds.",
          );
        durations[item.kind] += item.duration;
      }
    }
    if (counts.image > 9 || counts.video > 3 || counts.audio > 3 || items.length > 12)
      return this.tr(
        "最多 9 张图、3 段视频、3 段音频，合计不超过 12 份。",
        "Use up to 9 images, 3 videos, 3 audio clips, and 12 files total.",
      );
    if (durations.video > 15 || durations.audio > 15)
      return this.tr(
        "视频总时长、独立音频总时长分别不能超过 15 秒。",
        "Video and standalone audio must each total at most 15 seconds.",
      );
    if (items.reduce((sum, item) => sum + item.file.size, 0) > 255 * 1024 * 1024)
      return this.tr("素材总大小不能超过 255 MiB。", "References must total at most 255 MiB.");
    if (requireVisual && !counts.image && !counts.video)
      return this.tr(
        "请至少添加一张图片或一段视频，音频不能单独作为参考。",
        "Add at least one image or video; audio cannot be the only reference.",
      );
    return "";
  }
  async inspectReference(item) {
    if (item.kind === "image") return item;
    item.duration = await new Promise((resolve) => {
      const media = document.createElement(item.kind);
      let timer;
      const finish = (duration) => {
        clearTimeout(timer);
        media.onloadedmetadata = media.onerror = null;
        media.removeAttribute("src");
        media.load();
        resolve(duration);
      };
      // FFmpeg accepts more formats than browsers. Let the server validate unknown durations.
      media.onloadedmetadata = () => finish(media.duration);
      media.onerror = () => finish(null);
      timer = setTimeout(() => finish(null), 8000);
      media.preload = "metadata";
      media.src = item.url;
    });
    return item;
  }
  async addReferences(files) {
    if (this.addingReferences || !files.length) return;
    this.$("reference-files").value = "";
    const added = files.map((file) => ({ file, kind: file.type.split("/")[0] }));
    const error = this.referenceError([...this.references, ...added]);
    if (error) {
      this.error(error);
      return;
    }
    this.addingReferences = true;
    this.$("reference-files").disabled = true;
    this.renderReferences();
    this.availability();
    this.error("");
    try {
      for (const item of added) item.url = URL.createObjectURL(item.file);
      await Promise.all(added.map((item) => this.inspectReference(item)));
      if (this.disposed) return;
      const error = this.referenceError([...this.references, ...added]);
      if (error) {
        this.error(error);
        return;
      }
      this.references.push(...added);
    } finally {
      for (const item of added)
        if (!this.references.includes(item) && item.url) URL.revokeObjectURL(item.url);
      this.addingReferences = false;
      this.$("reference-files").disabled = false;
      this.renderReferences();
      this.availability();
    }
  }
  moveReference(index, delta) {
    const target = index + delta;
    if (target < 0 || target >= this.references.length) return;
    [this.references[index], this.references[target]] = [this.references[target], this.references[index]];
    this.renderReferences();
  }
  removeReference(index) {
    const [item] = this.references.splice(index, 1);
    if (item) URL.revokeObjectURL(item.url);
    this.renderReferences();
    this.error("");
  }
  referenceLabels() {
    const counts = { image: 0, video: 0, audio: 0 },
      names = { image: "Picture", video: "Video", audio: "Audio" };
    return this.references.map((item) => `${names[item.kind]} ${++counts[item.kind]}`);
  }
  renderReferences() {
    const labels = this.referenceLabels();
    this.$("reference-summary").textContent = this.addingReferences
      ? this.tr("正在读取素材…", "Reading references…")
      : this.tr(`已添加 ${this.references.length} / 12 份素材`, `${this.references.length} / 12 references`);
    this.$("reference-list").replaceChildren(
      ...this.references.map((item, index) => {
        const card = document.createElement("div");
        card.className = "h3-reference-card";
        const header = document.createElement("div");
        header.className = "h3-reference-head";
        const label = document.createElement("button");
        label.type = "button";
        label.className = "h3-reference-label";
        label.textContent = labels[index];
        label.title = this.tr("插入提示词", "Insert into prompt");
        label.onclick = () => {
          const prompt = this.$("prompt");
          const space =
            prompt.selectionStart && !/\s/.test(prompt.value[prompt.selectionStart - 1]) ? " " : "";
          prompt.setRangeText(`${space}${labels[index]} `, prompt.selectionStart, prompt.selectionEnd, "end");
          prompt.focus();
        };
        header.append(label);
        const actions = document.createElement("div");
        actions.className = "h3-reference-actions";
        for (const [symbol, title, delta] of [
          ["↑", this.tr("上移", "Move up"), -1],
          ["↓", this.tr("下移", "Move down"), 1],
          ["×", this.tr("移除", "Remove"), 0],
        ]) {
          const button = document.createElement("button");
          button.type = "button";
          button.textContent = symbol;
          button.title = title;
          button.setAttribute("aria-label", `${title} ${labels[index]}`);
          button.disabled = (delta < 0 && index === 0) || (delta > 0 && index === this.references.length - 1);
          button.onclick = () => (delta ? this.moveReference(index, delta) : this.removeReference(index));
          actions.append(button);
        }
        header.append(actions);
        const media = document.createElement(item.kind === "image" ? "img" : item.kind);
        media.src = item.url;
        media.className = "h3-reference-preview";
        if (item.kind === "image") media.alt = item.file.name;
        else {
          media.controls = true;
          media.preload = "metadata";
          media.setAttribute("playsinline", "");
        }
        const name = document.createElement("p");
        name.className = "h3-reference-name";
        name.textContent = item.file.name;
        name.title = item.file.name;
        const meta = document.createElement("small");
        meta.textContent =
          `${(item.file.size / 1024 / 1024).toFixed(1)} MiB` +
          (item.kind === "image"
            ? ""
            : item.duration == null
              ? this.tr(" · 时长由服务端校验", " · Duration checked by server")
              : ` · ${item.duration.toFixed(2)} s`);
        card.append(header, media, name, meta);
        return card;
      }),
    );
  }
  async generate(event) {
    event.preventDefault();
    if (
      this.submitting ||
      this.addingReferences ||
      !this.services.find((item) => item.id === this.service)?.ready
    )
      return;
    if (this.mode === "reference") {
      const error =
        this.referenceAvailable === false
          ? this.tr("此服务缺少参考模型权重。", "Reference model weights are missing.")
          : this.referenceError(this.references, true);
      if (error) {
        this.error(error);
        return;
      }
    }
    if (!this.$("prompt").value.trim()) {
      this.$("prompt").focus();
      return;
    }
    const id = this.service;
    const body = {
      prompt: this.$("prompt").value,
      size: this.$("size").value,
      seconds: Number(this.$("seconds").value),
    };
    if (this.$("seed").value) body.seed = Number(this.$("seed").value);
    if (!this.$("steps").disabled && this.$("steps").value)
      body.num_inference_steps = Number(this.$("steps").value);
    let requestBody = JSON.stringify(body);
    if ((this.mode === "frames" && this.frames.size) || this.mode === "reference") {
      requestBody = new FormData();
      Object.entries(body).forEach(([key, value]) => requestBody.append(key, String(value)));
      if (this.mode === "reference") {
        for (const { file } of this.references) requestBody.append("reference", file);
      } else {
        for (const [slot, { file }] of this.frames) requestBody.append(`${slot}_frame`, file);
      }
    }
    this.submitting = true;
    this.availability();
    this.error("");
    try {
      const job = await this.options.request(id, "/videos", { method: "POST", body: requestBody });
      if (this.disposed || id !== this.service) return;
      ++this.revision;
      this.selected = job.id;
      // The current poll may have started before submission; show the accepted job immediately.
      this.jobs = [job, ...this.jobs.filter((item) => item.id !== job.id)];
      this.renderJobs();
      await this.poll();
    } catch (error) {
      if (id === this.service) this.error(this.options.friendlyError(error));
    } finally {
      this.submitting = false;
      this.availability();
    }
  }
  async poll() {
    const id = this.service;
    if (this.disposed || this.polling || !this.services.find((item) => item.id === id)?.ready) return;
    const revision = this.revision;
    this.polling = true;
    try {
      const response = await this.options.request(id, "/videos");
      if (this.disposed || id !== this.service || revision !== this.revision) return;
      this.jobs = (response.data || []).slice().sort((a, b) => b.created_at - a.created_at);
      if (!this.jobs.some((job) => job.id === this.selected)) this.selected = this.jobs[0]?.id || "";
      this.renderJobs();
    } catch (error) {
      if (id === this.service) this.error(this.options.friendlyError(error));
    } finally {
      this.polling = false;
      if (id !== this.service && this.active) this.poll();
    }
  }
  renderJobs() {
    const root = this.$("jobs");
    for (const [index, job] of this.jobs.entries()) {
      let card = [...root.children].find((node) => node.dataset.jobId === job.id);
      if (!card) {
        card = document.createElement("button");
        card.type = "button";
        card.className = "video-job";
        card.dataset.jobId = job.id;
        card.innerHTML = "<small></small><p></p><span></span>";
        card.onclick = () => {
          this.selected = job.id;
          this.error("");
          this.renderJobs();
        };
      }
      if (root.children[index] !== card) root.insertBefore(card, root.children[index] || null);
      card.classList.toggle("selected", job.id === this.selected);
      card.setAttribute("aria-pressed", String(job.id === this.selected));
      card.dataset.status = job.status;
      card.querySelector("small").textContent = this.status(job.status);
      card.querySelector("p").textContent = job.prompt;
      card.querySelector("span").textContent = [job.size, job.seconds ? `${job.seconds} s` : ""]
        .filter(Boolean)
        .join(" · ");
    }
    for (const card of [...root.children])
      if (!this.jobs.some((job) => job.id === card.dataset.jobId)) card.remove();
    for (const [key, url] of this.results)
      if (!this.jobs.some((job) => `${this.service}/${job.id}` === key)) {
        URL.revokeObjectURL(url);
        this.results.delete(key);
      }
    this.$("history-empty").hidden = this.jobs.length > 0;
    this.renderSelection();
  }
  renderSelection() {
    const job = this.jobs.find((item) => item.id === this.selected);
    const key = `${this.service}/${this.selected}`;
    const url = job && this.results.get(key);
    const player = this.$("player");
    // Progress, locale and history updates must never replace the active player or reset its clock.
    if (player.dataset.key !== (url ? key : "")) {
      player.pause();
      player.dataset.key = url ? key : "";
      if (url) player.src = url;
      else {
        player.removeAttribute("src");
        player.load();
      }
    }
    player.hidden = !url;
    this.$("empty").hidden = Boolean(url);
    this.$("result-info").textContent = job
      ? [
          this.status(job.status),
          job.size,
          job.seconds ? `${job.seconds} s` : "",
          job.seed != null ? `Seed ${job.seed}` : "",
        ]
          .filter(Boolean)
          .join(" · ")
      : this.tr("预览区域 · 支持全屏播放", "Preview · fullscreen playback available");
    this.$("progress-panel").hidden = !job || job.status === "completed";
    this.$("progress-label").textContent = job
      ? `${this.status(job.status)}${!terminal(job.status) ? ` · ${Math.round(Number(job.progress || 0))}%` : ""}`
      : "";
    this.$("progress").value = Number(job?.progress || 0);
    this.$("progress").hidden = !job || terminal(job.status);
    this.$("job-error").textContent = job?.error?.message || "";
    this.$("cancel").hidden = !job || terminal(job.status);
    this.$("cancel").disabled = this.cancelling.has(key);
    this.$("cancel").textContent = this.cancelling.has(key)
      ? this.tr("正在取消…", "Cancelling…")
      : this.tr("取消任务", "Cancel task");
    this.$("preview").hidden = !job || job.status !== "completed" || Boolean(url);
    this.$("preview").disabled = this.loading.has(key);
    this.$("preview").textContent = this.loading.has(key)
      ? this.tr("正在加载视频…", "Loading video…")
      : this.tr("预览并下载", "Preview and download");
    this.$("download").hidden = !url;
    if (url) {
      this.$("download").href = url;
      this.$("download").download = `${job.id}.mp4`;
    } else this.$("download").removeAttribute("href");
  }
  async preview() {
    const id = this.service,
      jobId = this.selected,
      key = `${id}/${jobId}`;
    if (this.loading.has(key)) return;
    this.loading.add(key);
    this.renderSelection();
    this.error("");
    try {
      const response = await this.options.request(id, `/videos/${jobId}/content`, { stream: true });
      const blob = await response.blob();
      if (this.disposed || id !== this.service || !this.jobs.some((job) => job.id === jobId)) return;
      this.results.set(key, URL.createObjectURL(blob));
    } catch (error) {
      if (id === this.service) this.error(this.options.friendlyError(error));
    } finally {
      this.loading.delete(key);
      this.renderSelection();
    }
  }
  async cancel() {
    const id = this.service,
      jobId = this.selected,
      key = `${id}/${jobId}`;
    if (this.cancelling.has(key)) return;
    this.cancelling.add(key);
    this.renderSelection();
    try {
      const job = await this.options.request(id, `/videos/${jobId}/cancel`, { method: "POST" });
      if (this.disposed || id !== this.service) return;
      ++this.revision;
      this.jobs = this.jobs.map((item) => (item.id === jobId ? job : item));
      this.renderJobs();
    } catch (error) {
      if (id === this.service) this.error(this.options.friendlyError(error));
    } finally {
      this.cancelling.delete(key);
      this.renderSelection();
    }
  }
  clearResults() {
    this.$("player").pause();
    this.$("player").removeAttribute("src");
    this.$("player").load();
    this.$("player").dataset.key = "";
    for (const url of this.results.values()) URL.revokeObjectURL(url);
    this.results.clear();
  }
  dispose() {
    this.disposed = true;
    clearInterval(this.timer);
    this.clearResults();
    for (const slot of ["first", "last"]) this.clearFrame(slot);
    for (const item of this.references) URL.revokeObjectURL(item.url);
    this.references = [];
    window.removeEventListener("pagehide", this.pagehide);
  }
}
