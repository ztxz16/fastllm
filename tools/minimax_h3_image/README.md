# MiniMax-H3-Image native generation and editing

This backend loads the original [MiniMaxAI/MiniMax-H3-Image](https://huggingface.co/MiniMaxAI/MiniMax-H3-Image) weights directly. It runs the image model's six distilled PDD steps in C++/CUDA and writes a PNG. No Torch, Diffusers, Python inference runtime or converted transformer checkpoint is required. The Python launcher manages processes and HTTP proxying only. PNG decoding is native C++; JPEG/WebP decoding uses FFmpeg/ffprobe executables, without invoking Python.

The independent model implementation is `include/models/minimax_h3_image/model.h` and `src/models/minimax_h3_image/model.cpp`. It reuses the H3 CUDA runtime and Qwen3-VL text encoder, with a separate image transformer, 2D ViT VAE and sampling pipeline. The shared H3 VAE encoder accepts the image checkpoint’s key names while preserving the video checkpoint path.

## Automatic FP8 selection

Use the existing `--model` argument, or the launcher's model path. A complete native
model directory keeps BF16. Selecting an FP8 `.safetensors` file automatically enables
native E4M3 projections; no precision or extra checkpoint flag is needed. The loader
reads tensor dtypes and shapes, not the filename. A quantized directory with multiple
checkpoints requires selecting the exact file.

The tested checkpoint is `unsloth/MiniMax-H3-Image-FP8`'s
`MiniMax-H3-Image-FP8.safetensors` (per-output-channel scales). Native H3 E4M3 tensors
with scalar scales are also recognized; Comfy quantization metadata identifies its already-packed QKV layout. INT8 and FP4 checkpoints return a clear error;
they are not silently dequantized or treated as FP8.

These files contain the DiT only. Keep the original complete `MiniMax-H3-Image` base
beside the quantized repository, or provide a `base_model` directory/symlink inside it.
The loader searches locally up to two parent directories; it rejects missing or
ambiguous bases and does not download anything. Text encoder, VAE, AdaLN, refiners and
PDD heads use the original base. Only the 50 main blocks' QKV/output and FC1/FC2
projections use FP8 weights and dynamic per-token FP8 activations, with FP32 GEMM
accumulation and BF16 output. Quantization changes pixels and exceeds the BF16
operator-alignment tolerance; it is an explicitly selected model, not a BF16 fallback.

```bash
CUDA_VISIBLE_DEVICES=4,5 build/src/models/minimax_h3/minimax_h3_image_server \
  --model /workspace2/hfmodels/unsloth--MiniMax-H3-Image-FP8/MiniMax-H3-Image-FP8.safetensors \
  --devices 0,1 --port 8009
```

FP8 needs CUDA 11.8+ and SM89+. The fused SM120 CUTLASS backend uses the normal
CUDA library environment. Only the cuBLAS FP8 fallback on multiple Blackwell GPUs
requires cuBLAS 13.2.1+ to avoid CUB-9487; the loader checks that path's library version. Multi-GPU FP8 uses SP
(`auto` selects it); explicit TP is unsupported. Existing GPU/host budgets apply,
including layer streaming. FP8 file reads use bounded staging and no persistent CPU
weight cache. `/health` reports `dit_precision`; the studio displays it. Text-to-image,
single-reference edits and multi-reference requests use the same API in either precision.
SM120 builds with CUDA 12.8+ and the bundled CUTLASS use an FP32-accumulating
FP8 GEMM whose epilogue applies both activation and weight scales directly to
BF16 output. Shape-based tile selection and vectorized activation quantization
are automatic; no additional model or precision flags are needed. Other GPU
architectures retain the cuBLASLt path. The fused epilogue removes an intermediate
BF16 rounding boundary, so its outputs need not be pixel-identical to the previous
FP8 implementation; operator validation uses relative L2 error.
Inference remains C++/CUDA/CUTLASS/cuBLAS/NCCL with no Torch dependency.

## Build and run

```bash
cmake -S . -B build -DUSE_CUDA=ON -DCUDA_ARCH=120 \
  -DBUILD_HY_NOMAS=ON -DBUILD_QWEN_IMAGE=ON -DBUILD_MINIMAX_H3=ON \
  -DPY_API=OFF -DCMAKE_BUILD_TYPE=Release
cmake --build build --target minimax_h3_image_generate minimax_h3_image_server -j8

CUDA_VISIBLE_DEVICES=3 build/src/models/minimax_h3/minimax_h3_image_generate \
  --model /workspace2/hfmodels/MiniMax-H3-Image \
  --prompt 'A red panda beside a mountain stream, autumn forest, natural wildlife photograph.' \
  --width 2048 --height 2048 --seed 42 --output red-panda.png
```

Select available GPUs before running. `--device` addresses the visible CUDA devices, so it defaults to `0` after setting `CUDA_VISIBLE_DEVICES`. The tested device is an RTX PRO 5000 Blackwell with 72 GB memory, including constrained-memory runs described below.

Both CLI and HTTP server accept `--devices 0,1`. Text encoding uses tensor parallelism.
With `--parallel-mode auto` (default), resident DiT weights use sequence parallelism: each
GPU handles a contiguous token partition, with head-group K/V exchange overlapped with
attention. Each rank writes its local K/V directly into the exchange buffer; only peer
segments are copied. Large aligned BF16 projections tune two cuBLASLt candidates once per
shape, keeping FP32 accumulation; this changes cold-start timing and may change rounding.
On two SM120 devices, padded local sequences of 512–8192 rows use seven head groups
across two persistent attention streams. CUDA events order each group after its K/V
transfer and join both streams before the output projection. The scratch arena is reused
within DiT and released before VAE batching; the attention math is unchanged. Other
architectures, rank counts and sequence sizes retain serial group execution. Native tensor
pools explicitly grant access to participating peer-capable devices for direct peer copies.
BF16 low-memory mode uses tensor-parallel weight shards and layer streaming instead.
`--parallel-mode tp|sp` selects either mode explicitly; BF16 SP requires at least 60 GiB available
per device. The final PDD projection runs on local rows before gathering the 96-wide velocity, avoiding a full hidden-state exchange. The reference VAE encoder and sampler use the first device. Resident BF16 VAE
decoding splits independent tiles across devices, preserving tile order and blend boundaries.
The build must find NCCL (`FASTLLM_NCCL_INCLUDE_DIR` and `FASTLLM_NCCL_LIBRARY` can point
to a nonstandard installation). Inference remains native C++/CUDA/NCCL without Torch.

```bash
CUDA_VISIBLE_DEVICES=4,5 build/src/models/minimax_h3/minimax_h3_image_generate \
  --model /path/to/MiniMax-H3-Image --devices 0,1 \
  --prompt "A red panda beside a mountain stream, autumn forest" \
  --width 2048 --height 2048 --seed 42 --output red-panda.png
```

In TP mode, `--tp-reduction activation` is the default: it rounds DiT row-projection partials to BF16 before
reduction to reduce communication. `--tp-reduction fp32` retains FP32 partials until after their
sum. Both retain FP32 accumulation in GEMMs; text-encoder reductions always use FP32.
Changing rank count or reduction precision can change generated pixels. Checkpoint tests compare
each block with identical input tensors using relative L2, and report accumulated drift separately.
The GPU limit applies to **each device**; all ranks share one host-cache budget. The launcher maps
selected GPUs to logical indices automatically. Device counts 1/2/4/8 satisfy the model dimensions;
the multi-device validation currently covers two GPUs.

The model directory must contain:

- `transformer/config.json`, `model.safetensors.index.json` and its native shards.
- `vae/config.json` and `diffusion_pytorch_model.safetensors`.
- `pdd_config.json` and `pdd_fused_heads.safetensors`.
- `tokenizer/` and the original MiniMax-H3 `text_encoder/`. A symlink to an existing text encoder is supported.

Defaults are BF16 transformer/VAE compute, FP32 timestep/input/output projections, six steps and a 2048×2048 canvas. `--vae-precision fp32` provides a slower diagnostic alternative. The text encoder reads hidden state 50, takes the prompt verbatim and streams its GPU weights while retaining a budgeted CPU weight cache. Fixed AdaLN timestep tables are reused across requests; their large source matrices are released after preparation. With sufficient free VRAM, DiT and VAE decoder weights stay resident between requests. Low-memory mode unloads DiT weights between stages. VAE decode batches independent tiles when memory allows, retaining the original sequential path for low-memory and FP32 diagnostics. The two text-refiner blocks run once within each request; their result is reused across its six steps and discarded afterward. Each new request recomputes its prompt and reference conditioning.

Reference VAE encoding has a single-image fast path: it removes the two zero causal frames from each convolution, batches independent 256-pixel tiles (up to eight, bounded by free VRAM), and uses cuBLAS TF32 products with FP32 accumulation on Ampere or newer GPUs. Activations, normalization, residuals and posterior moments remain FP32. Low-memory mode and `--vae-precision fp32` keep the original FP32 encoder. This is checked with relative L2 error, so reference edits need not be pixel-identical to the original implementation. Gate-first FFNs use a fused CUDA activation with BF16 approximate exponential, retaining the SiLU-before-multiply rounding boundary and checking relative L2. FP16/FP32 keep the original exponential. FFNs use 4096-row TP chunks and 8192-row local SP chunks.

Native checkpoint details handled here include per-head interleaved QKV, gate-first FFNs, three modality AdaLN rows, one FP32 PDD head per step, velocity-form Euler updates, 2D VAE rotary positions and register tokens, overlapping 256-pixel VAE tiles, ImageNet pixel denormalization and truncating RGB quantization.

The implementation processes one request at a time. It supports:

- Text-to-image.
- Single-reference editing: change a background, change an object or restyle the whole image.
- Multiple-reference composition, identity/subject preservation and style/color transfer, with up to 16 references in upload order.

Use `<Picture 1>`, `<Picture 2>`, etc. in the prompt to identify references. Describe the desired complete scene; detailed English prompts usually work best. Reference edits use both the native Qwen3-VL vision encoder and the image VAE encoder. References remain fixed during denoising, with their own grids and timestep `0.999`; only the target rows are updated.

```bash
CUDA_VISIBLE_DEVICES=3 build/src/models/minimax_h3/minimax_h3_image_generate \
  --model /workspace2/hfmodels/MiniMax-H3-Image \
  --image subject.jpg --image scene.webp \
  --prompt 'Place the subject from <Picture 1> into the scene from <Picture 2>. Match the lighting and preserve the subject’s identity.' \
  --width 2048 --height 2048 --seed 42 --output composition.png
```

Output dimensions are 32–8192 per side, up to 20 Mi pixels. Generation uses a canvas rounded to multiples of 32, followed by Lanczos resize to the exact requested dimensions. HTTP defaults to `min_generation_area=4194304`, so small outputs are generated on an approximately 2K canvas. The CLI defaults to `1024` for direct small numerical fixtures; set `--min-generation-area 4194304` for the HTTP behavior. The demo offers 2K/4K and five aspect ratios. Both 2048² and 4096² generation were tested on the selected 72 GB card.

References accept 8-bit non-interlaced PNG, JPEG and WebP, 32–8192 per side, up to 20 Mi pixels each, aspect ratios 1:4–4:1 and 64 Mi pixels combined. HTTP requests are limited to 32 MiB including multipart overhead. The demo recompresses large PNGs losslessly before upload, so generated PNGs can be reused as references. The CLI accepts files up to 96 MiB. JPEG EXIF rotation is applied during decoding. Reference preparation uses the released model’s size bounds: a 1024-pixel short edge for vision conditioning, and 1024–2048 for VAE encoding.

The model has six distilled steps, guidance scale 1 and one output per request. Masks/inpainting regions, a strength slider, negative prompts/CFG and arbitrary step counts are not implemented. The optional official MiniMax-H3-Image-PE prompt expander is a separate model and is not included here.

## Low VRAM and host-memory operation

Both native executables accept `--gpu-memory-limit GiB` and `--host-memory-limit GiB`, also available in the image profile's launcher settings. These are service startup settings and apply to text generation and reference editing alike; individual HTTP jobs cannot change them.

```bash
minimax_h3_image_server --model /path/to/MiniMax-H3-Image \
  --gpu-memory-limit 4 --host-memory-limit 2 --port 8009
```

With no GPU limit (or `0`), startup checks free device memory: below 60 GiB, DiT and VAE weights stream layer by layer; otherwise the DiT stays resident across requests, including while the service is idle. Text GPU weights always stream. Cancelling or failing a request drops the prepared model state so that a later request can rebuild it safely. An explicit limit must exceed 1 GiB. It reserves a persistent CUDA pool of `limit - 1 GiB`, leaving 1 GiB for the CUDA context/library workspace. A limit below 60 GiB selects layer streaming. Reserved pages remain allocated while the service is idle, including after cancellation or failure, and are reused by later requests. Insufficient activation space produces a failed job; precision, resolution and the six-step sampler are never silently reduced. Larger canvases or many references can require a larger budget.

With no host limit (or `0`), a shared weight cache uses at most one quarter of available host/container memory minus a 2 GiB working allowance, capped at 64 GiB. Detection respects the process's cgroup v1/v2 quota and visible parent quotas, including sibling usage. Explicit host limits must be at least 2 GiB: `2` disables weight caching; larger values allow caching within the remaining quota. Text, vision, reference encoding, DiT and VAE share the quota and load uncached tensors in at most 16 MiB chunks without whole-weight mmap reads. VAE output tiles are decoded in bounded batches (one tile in low-memory mode) and combined a row at a time. The host option bounds **weight caching**, not total process RSS: image buffers, queued uploads, tokenizer, CUDA runtime and filesystem cache also need memory. Low budgets trade speed for capacity.

`GET /health` reports `gpu_memory_limit_gib`, `host_memory_limit_gib`, `gpu_memory_reserved`, `layer_streaming`, `host_weight_cache_gib`, `tensor_parallel_size` (text encoder), `sequence_parallel_size` and `dit_parallel_mode`. A reported limit of zero means automatic policy; the cache field reports its effective capacity. CLI VAE diagnostics use the same policy. Inference remains native C++/CUDA, without a Torch dependency.

## Web demo and HTTP API

Run from the repository root:

```bash
CUDA_VISIBLE_DEVICES=3 build/src/models/minimax_h3/minimax_h3_image_server \
  --model /workspace2/hfmodels/MiniMax-H3-Image \
  --host 127.0.0.1 --port 8009 --output ./h3-image-output \
  --demo tools/minimax_h3_image/demo.html
```

Open `http://127.0.0.1:8009/demo` for text generation, single-image editing and multi-reference composition. The shared studio includes drag-and-drop reference upload, reordering, prompt presets, 2K/4K sizes, seeds, progress, cancellation, recent jobs, a large preview, PNG download, API examples and reuse of a result as the next reference. An optional `--api-key KEY` protects all `/v1` endpoints with bearer authentication; enter it under the demo's connection settings. `/health` and the demo assets remain public. Bind a suitable address and configure authentication before sharing the service outside localhost.

- `POST /v1/images/jobs` accepts JSON for text generation or multipart data for reference edits; returns HTTP 202 and a job ID. Common fields: `prompt`, optional `model: "minimax-h3-image"`, `size`, `seed`, `n: 1`, `num_inference_steps: 6`, `guidance_scale: 1` and `min_generation_area`.
- `GET /v1/images/jobs` lists recent jobs; `GET /v1/images/jobs/{id}` reports progress.
- `POST /v1/images/jobs/{id}/cancel` requests cancellation at the next progress boundary. Cancelling a running stage may take time.
- `GET /v1/images/jobs/{id}/content` downloads a completed PNG.
- `POST /v1/images/generations` waits for completion and returns `data[0].url` (relative to this server) or `data[0].b64_json` when requested.
- `POST /v1/images/edits` accepts multipart `image[]` (or repeated `image`, consistently), with the same scalar fields, and waits for completion. It returns the same URL/base64 response as generations.
- `/openapi.json` describes the API; `/v1/models` reports the model ID.

For example, a two-reference synchronous edit (omit the auth header when no key is configured):

```bash
curl http://127.0.0.1:8009/v1/images/edits \
  -H "Authorization: Bearer $FASTLLM_API_KEY" \
  -F 'image[]=@subject.jpg' -F 'image[]=@scene.webp' \
  -F 'prompt=Place the subject from <Picture 1> into the scene from <Picture 2>.' \
  -F size=2048x2048 -F seed=42 -F response_format=b64_json
```

For background execution, send the same multipart body to `/v1/images/jobs` and poll the returned job ID. Job metadata includes the requested `size`, actual `canvas_size`, mode, reference count and each reference’s decoded dimensions. Image order is significant; do not mix `image` and `image[]` names in one request. Reference URLs, masks and unsupported parameters are rejected explicitly.

The queue allows up to four pending/running jobs. Up to 128 recent jobs are kept in memory. Completed PNGs and JSON metadata are saved atomically in the output directory and reloaded on restart; pending jobs do not survive process termination. Output files are retained until manually removed. Immutable weights and fixed timestep tables are reused according to the memory policy above. Prompt embeddings and reference encodings are recomputed for every request; they are not cached.

## ftllm launch

Launch the current source/wheel with `ftllm launch`. Under **启动服务 → 增加启动项**, choose **图像（MiniMax H3 Image）**, set the model directory, port, output directory and available GPUs, then start the profile. Click **图像 Demo**, or open **工作室 → 图像生成**, to generate, edit or combine images, monitor progress, cancel, preview and download. The image backend can run alongside chat and video services; its requests pass through the launcher's authenticated service proxy.

The launcher locates `minimax_h3_image_server` from the profile's binary field, `FTLLM_H3_IMAGE_SERVER`, the installed wheel, PATH or the source build directory, in that order. For a separate build directory:

```bash
FTLLM_H3_IMAGE_SERVER=/absolute/build/src/models/minimax_h3/minimax_h3_image_server ftllm launch
```

GPU indices in the launcher refer to its visible GPU list. If a container exposes physical GPU 3 only, select logical GPU 0. Select 1, 2, 4 or 8 GPUs; two-device inference has been validated. The backend selects SP or streamed TP automatically from the per-device memory budget. CUDA wheel packaging includes both image executables, OpenAPI, the standalone demo and its shared JS/CSS assets. A full release wheel has not been built in this experiment.

## Numerical validation

Tests require PyTorch, NumPy, Pillow, safetensors and a Diffusers checkout containing the MiniMax-H3-Image classes and converter. The native runtime itself does not need them. The reference used during development is Diffusers commit `f598e6e6a99528858b962c1399c45b59a6c25484`.

```bash
export CUDA_VISIBLE_DEVICES=3
export PYTHONPATH=/path/to/diffusers/src
python test/minimax_h3_image/compare_vae.py \
  --model /workspace2/hfmodels/MiniMax-H3-Image \
  --binary build/src/models/minimax_h3/minimax_h3_image_generate \
  --diffusers /path/to/diffusers --output /tmp/h3-image-vae --precision fp32 --tiled
python test/minimax_h3_image/compare_encoder.py \
  --model /workspace2/hfmodels/MiniMax-H3-Image \
  --binary build/src/models/minimax_h3/minimax_h3_image_generate \
  --diffusers /path/to/diffusers --output /tmp/h3-image-encoder
python test/minimax_h3_image/compare_dit.py \
  --model /workspace2/hfmodels/MiniMax-H3-Image \
  --binary build/src/models/minimax_h3/minimax_h3_image_generate \
  --diffusers /path/to/diffusers --output /tmp/h3-image-dit
```

The VAE test compares all 24 decoder blocks and an asymmetric output tile; `--tiled` also compares the complete overlapping-tile PNG path. The DiT test runs all 50 blocks and six different PDD heads with fixed text and initial noise, comparing the six-step trajectory to the official scheduler. It releases the reference model before starting the C++ process.

With `-DUNIT_TEST=ON`, build `minimax_h3_image_condition` and pass its executable as `--condition-binary` to `compare_dit.py`. This runs the real-weight 50-layer, six-step comparison with mixed text/vision tags and two asymmetric reference grids.

The native two-GPU checkpoint test compares same-input operators/blocks with relative L2
below `0.01`, and reports full-trajectory drift separately. Its SP mode also requires
bitwise equality between concurrent and serial head groups, including uneven shards,
dispatch boundaries, scratch release and allocator reuse:

```bash
CUDA_VISIBLE_DEVICES=4,5 build/src/models/minimax_h3/minimax_h3_image_parallel /path/to/MiniMax-H3-Image sp
CUDA_VISIBLE_DEVICES=4,5 build/src/models/minimax_h3/minimax_h3_image_parallel /path/to/MiniMax-H3-Image activation
```

The HTTP regression suite does not load a model or use a GPU. It checks authenticated history restoration beyond 128 results, invalid/incomplete sidecars and the packaged demo asset layout. The frontend state test checks exact output aspect ratios, overlapping uploads and submitting/reusing while references are being prepared.

```bash
python3 test/minimax_h3_image/test_server.py --binary build/src/models/minimax_h3/minimax_h3_image_server
node test/minimax_h3_image/test_workspace.mjs
# Also registered with CTest when UNIT_TEST and a Python interpreter are available:
ctest --test-dir build -R minimax_h3_image_http --output-on-failure
```

Diagnostics: `--encode-image FILE` writes sampled, FP16-rounded and normalized CHW reference latents as FP32; it deliberately omits reference resizing for encoder comparisons. `--trace DIR` saves text, initial noise, step inputs and latents as raw row-major FP32 with `.shape` sidecars. `--trace-blocks` additionally saves all DiT block outputs and can use substantial disk space. `--text-in` accepts `[tokens,5120]`, `--noise-in` accepts `[height/32*width/32,96]`, and `--skip-decode` stops after the six steps. `--decode-latents` turns normalized packed latents into a PNG; `--vae-tile` accepts unnormalized HWC latents and writes FP32 patch rows for decoder comparisons.
