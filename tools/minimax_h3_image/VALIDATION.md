# Validation on pro5000, 2026-11-11

Base FastLLM: `3528b3e65040889c90a18c4cb01317b5e3209292`.
Working branch: `master` (uncommitted changes, as requested).
Checkpoint: MiniMaxAI/MiniMax-H3-Image revision `00dd081ebdd33315adf2c4d84f64aa327ddf3039`.
Official reference: Diffusers `f598e6e6a99528858b962c1399c45b59a6c25484`.

GPU 3 was idle when selected. Tests exposed only that RTX PRO 5000 Blackwell 72 GB card, in container `fastllm-h3-image-gpu3-20261111`, with CUDA 13 and PyTorch 2.14.0+cu130 for reference comparisons. No other user's container was stopped or changed.

| Check | Result |
| --- | --- |
| Native C++/CUDA build, SM120 | Passed |
| Real-weight 24-layer VAE, FP32, asymmetric 128×192 output | Relative RMS `5.9784e-7`, max absolute `4.4405e-6` |
| Same VAE, BF16 | Relative RMS `0.00557091`, max absolute `0.05078125` |
| BF16 VAE complete 384×512 overlapping-tile PNG | Mean absolute RGB difference `0.28088/255`, maximum `6/255` |
| Real-weight 50-layer DiT, all six PDD steps, fixed text/noise | Final latent relative RMS `0.0157648`, cosine `0.999876` |
| Native text encoder → DiT → VAE, 2048×2048, seed 42 | PNG generated and visually inspected; no obvious tile seams |

BF16 paths are numerically close, not bitwise identical: packed native GEMMs and CUDA attention differ from the reference's separate projections and PyTorch attention. Small numerical fixtures validate the wiring; the 2K example validates the supported end-to-end path, not general image quality across prompts.

The 2K C++ process took **70.72 seconds**, including model loading, text encoding, six denoising steps, 121 VAE tiles and diagnostic file writes. The outer monitor measured 72.43 seconds and sampled a peak **41,919 MiB (40.94 GiB)** of GPU memory once per second. Checkpoint files had already been read by prior tests, so this is not a cold filesystem benchmark or a warmed resident-server benchmark.

Prompt:

> A red panda wearing a tiny blue scarf, sitting on a moss-covered rock beside a clear mountain stream. Golden autumn forest in the background, soft morning sunlight, fine detailed fur, natural colors, cinematic wildlife photograph, sharp focus, no text, no watermark.

Remote artifacts are under `/workspace2/experiments/minimax-h3-native/image-native-20261111`:

- `sample-2k/red-panda-2048.png`, `sample-2k/run.json`, `sample-2k/generate.log`.
- `vae-fp32/metrics.json`, `vae-tiled-bf16/metrics.json`, `dit-six-step/metrics.json`.
- `build/src/models/minimax_h3/minimax_h3_image_generate`.

Run again in the existing isolated container:

```bash
docker exec fastllm-h3-image-gpu3-20261111 \
  /work/build/src/models/minimax_h3/minimax_h3_image_generate \
  --model /model --prompt 'A red panda beside a mountain stream, autumn forest.' \
  --width 2048 --height 2048 --seed 42 --output /work/red-panda.png
```

## HTTP and launcher validation

- Built `minimax_h3_image_server` against the native C++/CUDA implementation.
- `test/api/test_launcher_media.py`: 15 tests passed, including image backend discovery, single-GPU validation, command construction, bearer-authenticated proxying, multipart reference order/bytes, cancellation and existing chat/video regressions. The optional Playwright test was skipped because its runtime was unavailable; the actual launcher was exercised through a browser separately.
- Started the image service from `ftllm launch`, opened its **图像 Demo**, submitted the 2048×2048 seed-42 prompt above, and obtained a completed PNG.
- Real HTTP validation rejected empty prompts, malformed images, unknown models, invalid dimensions, `n=2` and non-six-step generation with HTTP 400. Cancelling a queued job produced `cancelled` without generating an image.

The running experiment uses launcher port 18095 and image server port 18096, both bound to localhost. Its configuration is `/work/launcher-profiles.json`; generated images are in `/work/demo-output`. The launcher executable is `/work/bin/ftllm`, importing the current source via `/work/launcher-runtime`. Physical GPU 3 appears as logical GPU 0 inside the isolated container. The source backend is selected with `FTLLM_H3_IMAGE_SERVER=/work/build/src/models/minimax_h3/minimax_h3_image_server`.

## Reference editing and final regression checks

- Native single-reference edit, 2048×2048, seed 42: retained the original red panda pose/composition, changed the blue scarf to red and the autumn forest to snow. CLI time **54.8448 seconds**, including loading and trace output; artifact `edit-single/red-scarf-winter.png`.
- Browser → launcher authenticated multipart proxy → native server: JPEG autumn subject plus WebP winter subject, reordered in the UI and restored to the intended order, seed 44, output **2304×1792** on canvas **2336×1792**. Both pandas appear together with the requested blue/red scarves. Job `image_1794391544000851_1`; PNG and metadata retained in `demo-output`.
- Native HTTP text generation, seed 46, **4096×4096**, completed and visually checked. Job `image_1794391704611763_2`. The standalone demo successfully reuses this 4K PNG as a reference after browser-side lossless recompression.
- Tiled FP32 reference VAE encoder on a 272×400 input, including asymmetric tiles, posterior sampling, FP16 rounding and latent cropping: normalized latent relative RMS **0.000331235**, maximum absolute error **0.00305963** against Diffusers; `encoder-final/metrics.json`.
- Two asymmetric reference grids with mixed text/vision tags, fixed reference timestep and all 50 DiT blocks/six PDD heads: final target latent relative RMS **0.0207736**, cosine **0.999786**; `condition-final/metrics.json`.
- Text-only six-step regression after adding references: final latent relative RMS **0.0157648**, cosine **0.999876**, matching the previous result; `dit-edit-regression/metrics.json`.
- Real API checks: repeated ordered multipart inputs, JPEG EXIF orientation, PNG/JPEG/WebP decoding, 16-reference validation, malformed uploads, mixed field-name rejection, cancellation, unsupported masks/strength/negative-prompt rejection and JS/CSS asset routes.
- Synchronous multipart `/v1/images/edits`, `response_format=b64_json`, 128×192 diagnostic canvas, seed 45: valid PNG, **48.56 seconds** including HTTP overhead. This small canvas tests the interface, not image quality.
- Standalone and launcher studio manually exercised: mode selection, upload, reordering, presets, async generation, preview/history, 4K-result reuse and download URLs. The service card now groups management and editing controls, removes the duplicate disabled running button and prevents text wrapping within buttons.
- `ldd` on both CLI and HTTP executable contains the native FastLLM library, CUDA runtime/cuBLAS/cuBLASLt and system C/C++ libraries; **no Torch or Python runtime dependency**. JPEG/WebP use FFmpeg/ffprobe executables. Torch/Diffusers are used only by differential tests.

The launcher regression suite had one transient failure in an existing Pi test when its generated bridge token began with `-` and argparse treated it as an option; rerunning the unchanged suite passed all 15 enabled tests. This unrelated pre-existing issue was not modified.

Current coverage is native single-GPU text generation, single-reference editing and two-reference composition through CLI/API/demo. The interface accepts up to 16 references, but a 16-image quality/memory workload has not been tested. Tensor parallelism, smaller-memory GPU execution, masks/strength/negative prompts and the separate prompt-expander model are not implemented. CUDA wheel packaging includes the new binaries and shared demo assets; a complete release wheel was not built during this experiment.

## Commit-readiness cleanup

- Reformatted the new C++ and browser code and removed unused code/localization entries. A token comparison confirmed that model math, image decoding, CLI behavior and the numerical fixture were unchanged apart from formatting, namespace comments and reordered header `using` declarations.
- Fixed history restoration to retain the newest 128 completed results regardless of filesystem iteration order. Pruning affects only the in-memory index; result files remain on disk. A regression test with 140 shuffled completed records failed against the previous server and passes with the fix.
- Corrected the demo's 2K/4K presets to preserve the displayed aspect ratio exactly, with generation-canvas rounding handled separately. Serialized reference preparation and disabled conflicting upload, mode, reference-editing, reuse and submission controls until it finishes; failures restore the controls.
- Rebuilt the CLI, native HTTP server and conditioning fixture with CUDA 13 / SM120 successfully.
- Launcher regression suites: `test_launcher.py` **64 passed**; `test_launcher_media.py` **15 passed, 1 optional Playwright test skipped**; `test_ui_plugins.py` **25 passed**. No retry was needed in this cleanup run.
- New `test/minimax_h3_image/test_server.py`: **3 passed**, also registered as `minimax_h3_image_http` in CTest. It launches the real native executable without a GPU/checkpoint and covers history retention, authentication, content retrieval and packaged demo asset paths/MIME types.
- New `test/minimax_h3_image/test_workspace.mjs`: passed exact aspect-ratio checks, upload serialization, mode/submission/reuse guards and failure recovery using the actual workspace methods.
- `git diff --check` passed. Changes remain uncommitted on `master`; no commit or push was performed.

These checks support committing the documented single-GPU feature. They do not replace the release-wheel build or broaden the model workload coverage listed above.
