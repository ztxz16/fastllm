// One Annex B decoder and one audio clock per reply. The packet header contains
// lengths for all access units, so HTTP read boundaries need not match frames.
class DuoAvcPlayer {
  static async supported() {
    if (!window.VideoDecoder || !window.AudioContext) return false;
    return (await VideoDecoder.isConfigSupported({codec: "avc1.42c01f"})).supported;
  }

  constructor(canvas, audio, onPacket = () => {}, onDraw = () => {}) {
    this.canvas = canvas;
    this.audio = audio;
    this.onPacket = onPacket;
    this.onDraw = onDraw;
    this.abort = new AbortController();
    this.frames = [];
    this.sources = new Set();
    this.playEnd = 0;
    this.block = 0;
    this.decoder = new VideoDecoder({
      output: frame => {
        const batch = this.batch;
        if (!batch) { frame.close(); return; }
        batch.frames.push(frame);
        if (batch.frames.length === batch.count) batch.resolve();
      },
      error: error => this.batch?.reject(error),
    });
    this.decoder.configure({codec: "avc1.42c01f", optimizeForLatency: true});
  }

  close() {
    this.abort.abort();
    this.batch?.reject(new DOMException("Playback stopped", "AbortError"));
    this.batch?.frames.splice(0).forEach(frame => frame.close());
    if (this.decoder.state !== "closed") this.decoder.close();
    cancelAnimationFrame(this.raf);
    for (const {frame} of this.frames.splice(0)) frame.close();
    for (const src of this.sources) src.stop();
    this.sources.clear();
  }

  draw() {
    while (this.frames.length && this.frames[0].time <= this.audio.currentTime) {
      const {frame, block, index} = this.frames.shift();
      if (this.canvas.width !== frame.displayWidth || this.canvas.height !== frame.displayHeight) {
        this.canvas.width = frame.displayWidth;
        this.canvas.height = frame.displayHeight;
      }
      this.canvas.getContext("2d").drawImage(frame, 0, 0);
      frame.close();
      this.onDraw(block, index);
    }
    this.raf = this.frames.length ? requestAnimationFrame(() => this.draw()) : 0;
  }

  async packet(bytes) {
    const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.length);
    const block = view.getUint32(0, true), count = view.getUint32(4, true), samples = view.getUint32(8, true);
    if (block !== this.block++ || count !== 17 || samples < 22666 || samples > 22667)
      throw Error("Invalid Duo block header");
    let offset = 12 + 4 * count;
    const pcm = offset;
    offset += samples * 4;
    const units = [];
    for (let i = 0; i < count; ++i) {
      const word = view.getUint32(12 + i * 4, true), length = word & 0x7fffffff;
      if (!length || offset + length > bytes.length) throw Error("Invalid H.264 access unit");
      units.push(new EncodedVideoChunk({type: word >>> 31 ? "key" : "delta",
        timestamp: Math.round((block * count + i) * 1e6 / 24), data: bytes.subarray(offset, offset + length)}));
      offset += length;
    }
    if (offset !== bytes.length) throw Error("Unexpected Duo media bytes");
    this.onPacket(block, count, samples, bytes);
    const audio = this.audio.createBuffer(2, samples, 32000);
    for (let ch = 0; ch < 2; ++ch) {
      const dst = audio.getChannelData(ch);
      for (let i = 0; i < samples; ++i) dst[i] = view.getInt16(pcm + i * 4 + ch * 2, true) / 32768;
    }
    const frames = [];
    await new Promise((resolve, reject) => {
      this.batch = {frames, count, resolve, reject};
      for (const unit of units) this.decoder.decode(unit);
    });
    this.batch = null;
    // Match the official player's 45 ms playout cushion.
    const start = Math.max(this.audio.currentTime + 0.045, this.playEnd);
    this.playEnd = start + audio.duration;
    const src = this.audio.createBufferSource();
    src.buffer = audio;
    src.connect(this.audio.destination);
    this.sources.add(src);
    src.onended = () => { this.sources.delete(src); src.disconnect(); };
    src.start(start);
    frames.forEach((frame, index) => this.frames.push({frame, block, index, time: start + index / 24}));
    if (!this.raf) this.draw();
  }

  async play(url) {
    const response = await fetch(url, {signal: this.abort.signal});
    if (!response.ok) throw Error("Cannot open Duo media stream");
    const reader = response.body.getReader();
    let data = new Uint8Array(0);
    try {
      for (;;) {
        const {done, value} = await reader.read();
        if (done) break;
        const next = new Uint8Array(data.length + value.length);
        next.set(data); next.set(value, data.length); data = next;
        let offset = 0;
        while (data.length - offset >= 4) {
          const size = new DataView(data.buffer, data.byteOffset + offset).getUint32(0, true);
          if (!size || size > 32 * 1024 * 1024) throw Error("Invalid Duo packet length");
          if (data.length - offset - 4 < size) break;
          await this.packet(data.subarray(offset + 4, offset + 4 + size));
          offset += 4 + size;
        }
        data = data.subarray(offset);
      }
      if (data.length) throw Error("Incomplete Duo media packet");
    } finally {
      reader.releaseLock();
    }
  }
}
