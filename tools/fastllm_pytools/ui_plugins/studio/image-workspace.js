// Shared by ftllm launch and the standalone MiniMax H3 Image demo.
const defaultPrompt =
  "A red panda wearing a tiny blue scarf, sitting on a moss-covered rock beside a clear mountain stream. Golden autumn forest, soft morning sunlight, detailed fur, natural colors, cinematic wildlife photography. No text or watermark.";
const presets = {
  text: [
    ["自然摄影", "Photography", defaultPrompt],
    [
      "产品大片",
      "Product",
      "An elegant amber glass perfume bottle on a pale travertine pedestal, surrounded by olive branches. Soft afternoon sunlight, warm shadows, refined editorial product photography, no text.",
    ],
    [
      "概念插画",
      "Illustration",
      "A cozy little bookshop floating on a cloud above a quiet seaside town, golden evening light, intricate hand-painted watercolor illustration, warm cream and sage palette.",
    ],
  ],
  edit: [
    [
      "换背景",
      "Background",
      "Keep the subject and composition from <Picture 1>. Replace the background with a snowy winter forest under soft morning light. Preserve the subject’s identity and fine details.",
    ],
    [
      "改风格",
      "Restyle",
      "Reimagine <Picture 1> as a delicate watercolor illustration on textured ivory paper. Preserve the subject, pose and composition, with soft sage and warm ochre colors.",
    ],
    [
      "保留主体",
      "Keep subject",
      "Preserve the subject from <Picture 1>, including its identity and distinctive details. Create a new cinematic scene with soft golden-hour lighting and a calm natural background.",
    ],
  ],
  multi: [
    [
      "主体与场景",
      "Subject & scene",
      "Place the main subject from <Picture 1> naturally into the scene from <Picture 2>. Preserve the subject’s identity and proportions, and match lighting, perspective and shadows to the new environment.",
    ],
    [
      "配色与风格",
      "Color & style",
      "Keep the subject and composition from <Picture 1>, and apply the color palette, lighting and artistic style of <Picture 2>. Create one coherent, detailed image.",
    ],
  ],
};
export class ImageWorkspace {
  constructor(root, options) {
    this.root = root;
    this.options = options;
    this.locale = options.locale || "zh-CN";
    this.mode = "text";
    this.files = [];
    this.jobs = [];
    this.urls = new Map();
    this.service = "";
    this.ready = false;
    this.active = true;
    this.submitting = false;
    this.preparingFiles = false;
    this.selected = "";
    this.polling = false;
    if (!document.getElementById("h3-workspace-style")) {
      const style = document.createElement("link");
      style.id = "h3-workspace-style";
      style.rel = "stylesheet";
      style.href = new URL("./image-workspace.css", import.meta.url).href;
      document.head.append(style);
    }
    root.classList.add("h3-workspace");
    root.innerHTML = /* HTML */ `<div class="h3-heading">
        <div>
          <div class="h3-eyebrow">MINIMAX H3 / IMAGE STUDIO</div>
          <h2 data-copy="title"></h2>
          <p data-copy="subtitle"></p>
        </div>
        <span class="h3-badge">LOCAL CREATIVE STUDIO</span>
      </div>
      <div class="h3-layout">
        <form class="h3-controls">
          <div class="h3-service">
            <label data-copy="model"></label
            ><select data-el="service" aria-label="图像模型"></select>
          </div>
          <div class="h3-modes" role="group" aria-label="创作模式">
            <button type="button" data-mode="text"></button><button type="button" data-mode="edit"></button
            ><button type="button" data-mode="multi"></button>
          </div>
          <div data-el="reference-field" class="h3-field" hidden>
            <div class="h3-field-head">
              <label data-copy="references"></label><span data-el="count"></span>
            </div>
            <label class="h3-drop" data-el="drop"
              ><b>＋</b><span data-copy="upload"></span><small>PNG · JPG · WEBP</small
              ><input
                data-el="files"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                multiple
                aria-label="上传参考图"
            /></label>
            <div class="h3-references" data-el="references"></div>
            <p class="h3-hint" data-copy="referenceHint"></p>
          </div>
          <div class="h3-field">
            <div class="h3-field-head">
              <label for="${options.id}-prompt" data-copy="prompt"></label><span>01 — PROMPT</span>
            </div>
            <textarea id="${options.id}-prompt" data-el="prompt" required maxlength="16000"></textarea>
            <div class="h3-presets" data-el="presets"></div>
            <p class="h3-hint" data-copy="promptHint"></p>
          </div>
          <div class="h3-row">
            <div class="h3-field">
              <label for="${options.id}-ratio" data-copy="ratio"></label
              ><select id="${options.id}-ratio" data-el="ratio">
                <option value="square">1:1</option>
                <option value="landscape">4:3</option>
                <option value="portrait">3:4</option>
                <option value="wide">16:9</option>
                <option value="tall">9:16</option>
              </select>
            </div>
            <div class="h3-field">
              <label for="${options.id}-quality" data-copy="quality"></label
              ><select id="${options.id}-quality" data-el="quality">
                <option value="2k">2K</option>
                <option value="4k">4K</option>
              </select>
            </div>
          </div>
          <div class="h3-field">
            <label for="${options.id}-seed" data-copy="seed"></label
            ><input id="${options.id}-seed" data-el="seed" type="number" min="0" max="2147483647" />
          </div>
          <button class="h3-submit" data-el="generate" type="submit"></button>
          <p class="h3-status" data-el="status" role="status"></p>
          <div class="h3-error" data-el="error" role="alert"></div>
          <details class="h3-api">
            <summary data-copy="api"></summary>
            <pre data-el="curl"></pre>
            <button type="button" data-el="copy" data-copy="copy"></button>
          </details>
        </form>
        <section class="h3-stage">
          <div class="h3-canvas" data-el="canvas">
            <span class="h3-canvas-label">YOUR NEXT CREATION</span>
            <div class="h3-empty" data-el="empty">
              <div class="h3-empty-art"></div>
              <h3 data-copy="emptyTitle"></h3>
              <p data-copy="emptyHint"></p>
            </div>
            <img data-el="result" alt="生成的图片" hidden />
          </div>
          <div class="h3-progress" data-el="progress-box" hidden>
            <div class="h3-progress-head">
              <span data-el="progress-text"></span><button type="button" data-el="cancel"></button>
            </div>
            <progress max="100" data-el="progress"></progress>
          </div>
          <div class="h3-result-footer">
            <span class="h3-result-info" data-el="result-info"></span>
            <div class="h3-result-actions" data-el="result-actions" hidden>
              <button type="button" data-el="reuse"></button><a data-el="download" download></a>
            </div>
          </div>
          <div class="h3-history-heading">
            <span data-copy="recent"></span
            ><button type="button" class="h3-text-button" data-el="refresh" data-copy="refresh"></button>
          </div>
          <div class="h3-history" data-el="history"></div>
        </section>
      </div>
      <dialog class="h3-zoom" data-el="zoom">
        <button type="button" data-el="close-zoom" aria-label="关闭大图">×</button
        ><img data-el="zoom-image" alt="生成图片的大图" />
      </dialog>`;
    this.$ = (name) => root.querySelector(`[data-el="${name}"]`);
    this.$("prompt").value = defaultPrompt;
    root.querySelector("form").onsubmit = (e) => {
      e.preventDefault();
      this.generate();
    };
    for (const button of root.querySelectorAll("[data-mode]"))
      button.onclick = () => this.setMode(button.dataset.mode);
    this.$("service").onchange = () => this.changeService(this.$("service").value);
    this.$("files").onchange = async () => {
      await this.addFiles([...this.$("files").files]);
      this.$("files").value = "";
    };
    const drop = this.$("drop");
    drop.ondragover = (e) => {
      e.preventDefault();
      drop.classList.add("dragover");
    };
    drop.ondragleave = () => drop.classList.remove("dragover");
    drop.ondrop = (e) => {
      e.preventDefault();
      drop.classList.remove("dragover");
      this.addFiles([...e.dataTransfer.files]);
    };
    for (const field of ["prompt", "ratio", "quality", "seed"])
      this.$(field).oninput = () => this.renderCurl();
    this.$("cancel").onclick = () => this.cancel();
    this.$("refresh").onclick = () => this.poll();
    this.$("reuse").onclick = () => this.reuse();
    this.$("canvas").onclick = () => {
      if (this.$("result").hidden) return;
      this.$("zoom-image").src = this.$("result").src;
      this.$("zoom").showModal();
    };
    this.$("close-zoom").onclick = () => this.$("zoom").close();
    this.$("copy").onclick = async () => {
      try {
        await navigator.clipboard.writeText(this.$("curl").textContent);
        this.$("copy").textContent = this.tr("已复制", "Copied");
      } catch (e) {
        this.error(this.tr("请选中命令复制。", "Select the command to copy it."));
      }
    };
    this.timer = setInterval(() => {
      if (this.active) this.poll();
    }, 2000);
    window.addEventListener("pagehide", () => this.dispose(), { once: true });
    this.renderText();
    this.setMode("text");
  }
  tr(zh, en) {
    return this.locale === "en-US" ? en : zh;
  }
  error(message) {
    this.$("error").textContent = message || "";
  }
  renderText() {
    const copy = {
      title: ["让想象，有迹可循。", "Give your imagination a place."],
      subtitle: [
        "从文字出发，或带上参考图，创作下一幅画面。",
        "Start with words, bring a reference, make something new.",
      ],
      model: ["图像模型", "Image model"],
      references: ["参考图", "References"],
      upload: ["点击上传，或将图片拖到这里", "Choose images or drop them here"],
      referenceHint: [
        "按上传顺序使用 <Picture 1>、<Picture 2> 指定图片；可用箭头调整顺序。最多 16 张，总上传不超过 32 MB。",
        "Use <Picture 1>, <Picture 2>, … in upload order. Reorder with the arrows. Up to 16 images, 32 MB total.",
      ],
      prompt: ["画面描述", "Your prompt"],
      promptHint: [
        "详细描述完整画面，英文提示词通常效果更好。",
        "Describe the complete scene. Detailed English prompts usually work best.",
      ],
      ratio: ["画面比例", "Aspect ratio"],
      quality: ["输出尺寸", "Resolution"],
      seed: ["随机种子（可选）", "Seed (optional)"],
      api: ["查看 API 调用示例", "View API example"],
      copy: ["复制命令", "Copy command"],
      emptyTitle: ["你的下一幅作品，从这里开始", "Your next creation starts here"],
      emptyHint: [
        "写一段描述，或上传参考图。\n生成完成后，预览、下载，也可以继续编辑。",
        "Write a prompt or add a reference. Preview, download, then keep creating.",
      ],
      recent: ["最近的创作", "Recent creations"],
      refresh: ["刷新", "Refresh"],
    };
    for (const [key, value] of Object.entries(copy))
      for (const el of this.root.querySelectorAll(`[data-copy="${key}"]`)) el.textContent = this.tr(...value);
    this.$("seed").placeholder = this.tr("留空随机", "Random");
    this.$("reuse").textContent = this.tr("作为参考图", "Use as reference");
    this.$("download").textContent = this.tr("下载 PNG ↓", "Download PNG ↓");
    this.$("cancel").textContent = this.tr("取消任务", "Cancel");
    for (const [mode, zh, en] of [
      ["text", "文生图", "Text to image"],
      ["edit", "单图编辑", "Edit image"],
      ["multi", "多图组合", "Combine images"],
    ])
      this.root.querySelector(`[data-mode="${mode}"]`).textContent = this.tr(zh, en);
    this.$("service").setAttribute("aria-label", this.tr("图像模型", "Image model"));
    this.$("files").setAttribute("aria-label", this.tr("上传参考图", "Upload reference images"));
    this.renderMode();
    this.renderJobs();
    this.renderCurl();
    this.availability();
  }
  update({ services, active = true, locale = this.locale, requestedService }) {
    this.active = active;
    this.services = services;
    if (locale !== this.locale) {
      this.locale = locale;
      this.renderText();
    }
    const select = this.$("service"),
      signature = JSON.stringify(services.map((s) => [s.id, s.name, s.ready]));
    if (signature !== this.serviceSignature) {
      select.replaceChildren(
        ...services.map((s) => {
          const option = document.createElement("option");
          option.value = s.id;
          option.textContent = s.name;
          return option;
        }),
      );
      this.serviceSignature = signature;
    }
    const id = services.some((s) => s.id === requestedService)
      ? requestedService
      : services.some((s) => s.id === this.service)
        ? this.service
        : services[0]?.id || "";
    select.value = id;
    if (id !== this.service) this.changeService(id);
    this.ready = Boolean(services.find((s) => s.id === id)?.ready);
    this.availability();
  }
  changeService(id) {
    this.service = id;
    this.ready = Boolean(this.services?.find((s) => s.id === id)?.ready);
    this.jobs = [];
    this.selected = "";
    this.clearResults();
    this.renderJobs();
    this.availability();
    if (this.ready) this.poll();
  }
  availability() {
    this.$("generate").disabled = !this.ready || this.submitting || this.preparingFiles;
    this.$("files").disabled = this.preparingFiles;
    this.$("reuse").disabled = this.preparingFiles;
    for (const button of this.root.querySelectorAll("[data-mode]")) button.disabled = this.preparingFiles;
    if (this.preparingFiles) {
      for (const button of this.root.querySelectorAll(".h3-ref-actions button")) button.disabled = true;
    }
    this.$("generate").textContent = this.preparingFiles
      ? this.tr("正在准备参考图…", "Preparing references…")
      : this.submitting
        ? this.tr("正在提交…", "Submitting…")
        : this.mode === "text"
          ? this.tr("开始生成  ↗", "Generate image  ↗")
          : this.tr("开始创作  ↗", "Create image  ↗");
    this.$("status").textContent = this.ready
      ? this.tr("服务已就绪 · 6 步生成 · 任务在后台运行", "Ready · 6 steps · tasks run in the background")
      : this.tr("请先在启动台启动图像模型", "Start an image model in the launcher first");
  }
  setMode(mode) {
    if (this.preparingFiles) return;
    this.mode = mode;
    this.error("");
    this.renderMode();
    this.renderCurl();
    this.availability();
  }
  renderMode() {
    for (const button of this.root.querySelectorAll("[data-mode]"))
      button.setAttribute("aria-pressed", String(button.dataset.mode === this.mode));
    this.$("reference-field").hidden = this.mode === "text";
    this.$("files").multiple = this.mode === "multi";
    this.$("presets").replaceChildren(
      ...presets[this.mode].map(([zh, en, prompt]) => {
        const b = document.createElement("button");
        b.type = "button";
        b.textContent = this.tr(zh, en);
        b.onclick = () => {
          this.$("prompt").value = prompt;
          this.renderCurl();
        };
        return b;
      }),
    );
    this.renderReferences();
  }
  async addFiles(files) {
    if (this.preparingFiles || !files.length) return;
    if (this.mode === "text") this.setMode(files.length > 1 ? "multi" : "edit");
    this.preparingFiles = true;
    this.availability();
    try {
      this.error("");
      const limit = this.mode === "edit" ? 1 : 16;
      if (this.files.length + files.length > limit)
        throw Error(
          this.tr(
            this.mode === "edit" ? "单图编辑只能使用 1 张图片；请切换到多图组合。" : "最多上传 16 张参考图。",
            this.mode === "edit"
              ? "Switch to Combine images to use more than one reference."
              : "Use up to 16 reference images.",
          ),
        );
      const pending = [];
      let bytes = this.files.reduce((n, f) => n + f.file.size, 0),
        pixels = this.files.reduce((n, f) => n + f.width * f.height, 0);
      try {
        for (let file of files) {
          if (!["image/png", "image/jpeg", "image/webp"].includes(file.type) || file.size > 96 * 1024 * 1024)
            throw Error(
              this.tr(
                "请上传 PNG、JPG 或 WebP 图片，单个文件不超过 96 MB。",
                "Use PNG, JPEG or WebP files up to 96 MB.",
              ),
            );
          const bitmap = await createImageBitmap(file, { imageOrientation: "from-image" });
          const { width, height } = bitmap;
          try {
            pixels += width * height;
            if (
              Math.min(width, height) < 32 ||
              Math.max(width, height) > 8192 ||
              width * height > 20 * 1024 * 1024 ||
              Math.max(width, height) > 4 * Math.min(width, height) ||
              pixels > 64 * 1024 * 1024
            )
              throw Error(
                this.tr(
                  "图片需为 32–8192 像素，单张最多 20 MP，宽高比在 1:4–4:1，总像素不超过 64 MP。",
                  "Images must be 32–8192 px, up to 20 MP each / 64 MP combined, with aspect ratios within 1:4–4:1.",
                ),
              );
            // Native PNG results use stored DEFLATE blocks; recompress them losslessly before upload.
            if (file.type === "image/png" && file.size > 8 * 1024 * 1024) {
              const canvas = document.createElement("canvas");
              canvas.width = width;
              canvas.height = height;
              canvas.getContext("2d").drawImage(bitmap, 0, 0);
              const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/png"));
              if (!blob) throw Error("PNG encoding failed");
              file = new File([blob], file.name, { type: "image/png" });
            }
          } finally {
            bitmap.close();
          }
          bytes += file.size;
          if (bytes > 31 * 1024 * 1024)
            throw Error(
              this.tr("图片总大小超过 31 MB，请压缩后再上传。", "Keep the combined image size below 31 MB."),
            );
          pending.push({ file, width, height, url: URL.createObjectURL(file) });
        }
        this.files.push(...pending);
      } catch (e) {
        for (const p of pending) URL.revokeObjectURL(p.url);
        throw e;
      }
      this.renderCurl();
    } catch (e) {
      this.error(e.message);
    } finally {
      this.preparingFiles = false;
      this.renderReferences();
      this.availability();
    }
  }
  renderReferences() {
    this.$("count").textContent = `${this.files.length} / ${this.mode === "edit" ? 1 : 16}`;
    this.$("references").replaceChildren(
      ...this.files.map((item, i) => {
        const card = document.createElement("div");
        card.className = "h3-reference";
        const img = document.createElement("img");
        img.src = item.url;
        img.alt = `Picture ${i + 1}`;
        const label = document.createElement("span");
        label.textContent = `Picture ${i + 1}`;
        const actions = document.createElement("div");
        actions.className = "h3-ref-actions";
        for (const [text, name, fn, disabled] of [
          [
            "←",
            this.tr("前移参考图", "Move reference earlier"),
            () => {
              [this.files[i - 1], this.files[i]] = [this.files[i], this.files[i - 1]];
            },
            i === 0,
          ],
          [
            "→",
            this.tr("后移参考图", "Move reference later"),
            () => {
              [this.files[i + 1], this.files[i]] = [this.files[i], this.files[i + 1]];
            },
            i === this.files.length - 1,
          ],
          [
            "×",
            this.tr("移除参考图", "Remove reference"),
            () => {
              URL.revokeObjectURL(item.url);
              this.files.splice(i, 1);
            },
            false,
          ],
        ]) {
          const b = document.createElement("button");
          b.type = "button";
          b.textContent = text;
          b.setAttribute("aria-label", `${name} ${i + 1}`);
          b.disabled = disabled;
          b.onclick = () => {
            fn();
            this.renderReferences();
            this.renderCurl();
          };
          actions.append(b);
        }
        card.append(img, label, actions);
        return card;
      }),
    );
  }
  size() {
    const sizes = {
      "2k": {
        square: "2048x2048",
        landscape: "2368x1776",
        portrait: "1776x2368",
        wide: "2736x1539",
        tall: "1539x2736",
      },
      "4k": {
        square: "4096x4096",
        landscape: "4736x3552",
        portrait: "3552x4736",
        wide: "5472x3078",
        tall: "3078x5472",
      },
    };
    return sizes[this.$("quality").value][this.$("ratio").value];
  }
  body() {
    const body = {
      model: "minimax-h3-image",
      prompt: this.$("prompt").value,
      size: this.size(),
      num_inference_steps: 6,
    };
    if (this.$("seed").value !== "") body.seed = Number(this.$("seed").value);
    return body;
  }
  renderCurl() {
    if (!this.$("curl")) return;
    const endpoint = this.options.endpoint?.(this.service) || "http://127.0.0.1:8009",
      body = { ...this.body(), response_format: "b64_json" };
    const auth = ' -H "Authorization: Bearer $FASTLLM_API_KEY"';
    this.$("curl").textContent =
      this.mode === "text"
        ? `curl ${endpoint}/v1/images/generations${auth} \\\n  -H 'Content-Type: application/json' --data-binary @- <<'JSON'\n${JSON.stringify(body, null, 2)}\nJSON`
        : `curl ${endpoint}/v1/images/edits${auth} \\\n  -F prompt=@prompt.txt \\\n  -F size=${body.size} -F response_format=b64_json${body.seed === undefined ? "" : ` -F seed=${body.seed}`} \\\n${Array.from({ length: Math.max(this.mode === "multi" ? 2 : 1, this.files.length) }, (_, i) => `  -F 'image[]=@reference-${i + 1}.png'`).join(" \\\n")}\n\n# ${this.tr("将上面的画面描述保存为 prompt.txt", "Save your prompt as prompt.txt")}`;
  }
  async generate() {
    if (!this.ready || this.submitting || this.preparingFiles) return;
    this.error("");
    if (this.mode === "edit" && this.files.length !== 1) {
      this.error(this.tr("请上传一张参考图。", "Add one reference image."));
      return;
    }
    if (this.mode === "multi" && this.files.length < 2) {
      this.error(this.tr("多图组合需要至少两张参考图。", "Add at least two reference images."));
      return;
    }
    const service = this.service,
      body = this.body();
    let payload = JSON.stringify(body);
    if (this.mode !== "text") {
      payload = new FormData();
      for (const [key, value] of Object.entries(body)) payload.append(key, String(value));
      for (const item of this.files) payload.append("image[]", item.file, item.file.name);
    }
    this.submitting = true;
    this.availability();
    try {
      const job = await this.options.request("/images/jobs", { method: "POST", body: payload }, service);
      if (this.service === service) {
        this.selected = job.id;
        this.jobs.unshift(job);
        this.clearPreview();
        this.renderJobs();
        await this.poll();
      }
    } catch (e) {
      this.error(e.message);
    } finally {
      this.submitting = false;
      this.availability();
    }
  }
  async poll() {
    const service = this.service;
    if (this.polling || !service || !this.ready) return;
    this.polling = true;
    try {
      const result = await this.options.request("/images/jobs", {}, service);
      if (service !== this.service) return;
      this.jobs = (result.data || []).sort((a, b) => b.created_at - a.created_at);
      if (!this.jobs.some((j) => j.id === this.selected)) this.selected = this.jobs[0]?.id || "";
      this.renderJobs();
      await this.loadPreview();
    } catch (e) {
      this.error(e.message);
    } finally {
      this.polling = false;
    }
  }
  jobStatus(job) {
    return this.tr(
      ...({
        queued: ["排队中", "Queued"],
        in_progress: ["生成中", "Generating"],
        cancelling: ["正在取消", "Cancelling"],
        cancelled: ["已取消", "Cancelled"],
        failed: ["生成失败", "Failed"],
        completed: ["已完成", "Completed"],
      }[job.status] || [job.status, job.status]),
    );
  }
  renderJobs() {
    const job = this.jobs.find((j) => j.id === this.selected),
      running = job && !["completed", "cancelled", "failed"].includes(job.status);
    this.$("progress-box").hidden = !running;
    if (running) {
      const stage = this.tr(
        ...({
          text: ["理解提示词与参考图", "Understanding prompt and references"],
          reference: ["编码参考图", "Encoding references"],
          prepare: ["准备生成", "Preparing generation"],
          denoise: ["生成画面", "Generating image"],
          decode: ["解码图片", "Decoding image"],
        }[job.stage] || ["等待生成", "Waiting to generate"]),
      );
      this.$("progress-text").textContent =
        job.status === "cancelling"
          ? this.tr("正在取消，等待当前阶段结束…", "Cancelling after the current stage…")
          : `${stage} ${job.total ? `${job.current}/${job.total}` : ""}`;
      this.$("progress").value = job.progress || 0;
      this.$("cancel").disabled = job.status === "cancelling";
    }
    if (job?.status === "failed") this.error(job.error?.message || this.jobStatus(job));
    this.$("result-info").textContent = job
      ? `${this.jobStatus(job)} · ${job.size} · Seed ${job.seed}${job.reference_count ? this.tr(` · ${job.reference_count} 张参考图`, ` · ${job.reference_count} references`) : ""}`
      : this.tr("预览区域 · 点击图片可放大", "Preview · click the image to expand");
    const signature = JSON.stringify([
      this.locale,
      this.selected,
      this.jobs.map((j) => [j.id, j.status, j.progress]),
    ]);
    if (signature !== this.historySignature) {
      this.historySignature = signature;
      this.$("history").replaceChildren(
        ...this.jobs.slice(0, 9).map((j) => {
          const b = document.createElement("button");
          b.type = "button";
          b.className = j.id === this.selected ? "selected" : "";
          const title = document.createElement("strong");
          title.textContent = j.prompt;
          const status = document.createElement("small");
          status.textContent = this.jobStatus(j);
          const size = document.createElement("small");
          size.textContent = j.size;
          b.append(status, title, size);
          b.onclick = () => {
            this.selected = j.id;
            this.error("");
            this.clearPreview();
            this.renderJobs();
            this.loadPreview();
          };
          return b;
        }),
      );
    }
  }
  async loadPreview() {
    const job = this.jobs.find((j) => j.id === this.selected),
      service = this.service;
    if (!job || job.status !== "completed") return;
    const key = `${service}/${job.id}`;
    let entry = this.urls.get(key);
    if (!entry) {
      if (this.loading === key) return;
      this.loading = key;
      try {
        const response = await this.options.request(
          `/images/jobs/${job.id}/content`,
          { stream: true },
          service,
        );
        const blob = await response.blob();
        if (service !== this.service) return;
        entry = { blob, url: URL.createObjectURL(blob) };
        this.urls.set(key, entry);
        while (this.urls.size > 9) {
          const [first, value] = this.urls.entries().next().value;
          URL.revokeObjectURL(value.url);
          this.urls.delete(first);
        }
      } catch (e) {
        this.error(e.message);
        return;
      } finally {
        if (this.loading === key) this.loading = "";
      }
    }
    if (service !== this.service || job.id !== this.selected) return;
    this.$("result").src = entry.url;
    this.$("result").hidden = false;
    this.$("empty").hidden = true;
    this.$("canvas").setAttribute("role", "button");
    this.$("canvas").setAttribute("aria-label", this.tr("放大生成图片", "Expand generated image"));
    this.$("result-actions").hidden = false;
    this.$("download").href = entry.url;
    this.$("download").download = job.id + ".png";
  }
  async cancel() {
    const job = this.jobs.find((j) => j.id === this.selected);
    if (!job) return;
    this.$("cancel").disabled = true;
    try {
      await this.options.request(`/images/jobs/${job.id}/cancel`, { method: "POST" }, this.service);
      await this.poll();
    } catch (e) {
      this.error(e.message);
      this.$("cancel").disabled = false;
    }
  }
  async reuse() {
    if (this.preparingFiles) return;
    const entry = this.urls.get(`${this.service}/${this.selected}`);
    if (!entry) return;
    for (const item of this.files) URL.revokeObjectURL(item.url);
    this.files = [];
    this.setMode("edit");
    this.$("prompt").value = presets.edit[0][2];
    await this.addFiles([new File([entry.blob], "reference.png", { type: "image/png" })]);
    this.renderCurl();
    this.$("prompt").focus();
  }
  clearPreview() {
    this.$("result").hidden = true;
    this.$("result").removeAttribute("src");
    this.$("empty").hidden = false;
    this.$("result-actions").hidden = true;
    this.$("canvas").removeAttribute("role");
    this.$("canvas").removeAttribute("aria-label");
  }
  clearResults() {
    for (const entry of this.urls.values()) URL.revokeObjectURL(entry.url);
    this.urls.clear();
    this.clearPreview();
  }
  dispose() {
    clearInterval(this.timer);
    this.clearResults();
    for (const item of this.files) URL.revokeObjectURL(item.url);
  }
}
