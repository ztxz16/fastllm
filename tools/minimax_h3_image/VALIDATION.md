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

At this initial checkpoint, coverage was native single-GPU text generation, single-reference editing and two-reference composition through CLI/API/demo. The interface accepts up to 16 references, but a 16-image quality/memory workload has not been tested. Multi-GPU TP/SP was added in the later dual-GPU validation below. Masks/strength/negative prompts and the separate prompt-expander model remain unimplemented. CUDA wheel packaging includes the new binaries and shared demo assets; a complete release wheel was not built during this experiment.

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

## Memory adaptation (2026-11-11)

Physical GPU 4 (RTX PRO 5000, 72 GiB), BF16 VAE, seed 42, prompt `A red panda beside a mountain stream, autumn forest`, full text encoding and 2048×2048 output:

| Policy | Peak GPU memory | Peak process RSS | Native time |
| --- | ---: | ---: | ---: |
| Previous resident implementation | 42,429 MiB | 96,549 MiB | 42.04 s |
| `--gpu-memory-limit 4 --host-memory-limit 2` | 3,613 MiB | 872 MiB | 72.47 s |

The text features, all six denoising latents, final latent and PNG are byte-identical. RSS includes mapped checkpoint pages in the old implementation; OS filesystem cache is not included in process RSS. GPU figures are sampled with `nvidia-smi`, RSS with `/proc/PID/status` (`VmHWM`), every 300 ms. These are measurements of this input, not universal minimum budgets.

Automatic mode was also tested without either memory flag in a container limited to **4 GiB RAM with no swap**, while an independent CUDA allocation left **5 GiB free VRAM** before model startup. `/health` selected `layer_streaming=true`, `host_weight_cache_gib=0`, and no explicit GPU pool reservation. Full 2K generation completed in 71.81 s with 808 MiB process RSS and produced the same PNG bytes. The temporary CUDA allocation was freed after testing; its memory is excluded from the table above.

The HTTP service at 4 GiB GPU / 2 GiB host budget also survived a deliberate pool exhaustion and an in-progress cancellation, then completed a two-reference 2048×2048 edit in the same 4 GiB RAM container (3,573 MiB peak GPU memory, 920 MiB peak RSS). No precision, step count or canvas fallback was used. Model-source/parser regression, HTTP memory flag validation, readiness fields and launcher budget argument/validation tests cover the integration. Inference binaries continue to link only the native CUDA/C++ runtime, with no Torch dependency.

Reproduction logs, traces and outputs are under `/workspace2/experiments/minimax-h3-native/image-native-20261111/image-memory-20261111` on the test machine.

## Memory and video workspace commit review

Reviewed on `master` at `5cc1d98cfdd7e123da55f86a87cee35670504f75`, with the video workspace/reference integration and image memory changes still uncommitted.

- Extracted host availability detection into an internal, CPU-testable helper. It resolves the process's cgroup v1/v2 membership through mount metadata and applies every visible ancestor quota, including sibling usage. Fixtures cover nested groups, bind-mounted roots, private namespaces, escaped mount paths, host limits and exhausted quotas; passed on Linux and macOS with compiler warnings treated as errors.
- Aligned GPU budget validation across native parsers and launcher backends: a nonzero budget must leave at least one byte after the 1 GiB reserve. Values that would round to a zero-byte pool now fail validation before allocation.
- Fresh CUDA 13 / SM120 build passed for image CLI/server and native tests. CTest passed `minimax_h3_image_memory`, `minimax_h3_image_http` (5 Python cases) and `minimax_h3_model_source`. Launcher suites passed 107 cases, with 1 optional Playwright case skipped. Image/video JavaScript suites passed. Shared studio assets are covered by the existing wheel package-data glob; a full release wheel was not built.
- Repeated automatic 2K generation using the final native build on physical GPU 4, in the 4 GiB RAM/no-swap container with initially 5 GiB free VRAM. Health selected layer streaming and zero host weight cache; output was byte-identical to the resident baseline, with 808 MiB peak process RSS and 64.46 s native runtime. The monitor's GPU total includes the separate temporary allocation used to constrain free VRAM and is not the model's memory usage.
- `ldd` confirmed no Torch/Python runtime dependency. Test resources were released, and the existing launcher service was left running. `git diff --check` passed; no commit or push was performed.

Review logs and the final patch/manifest are under `/workspace2/experiments/minimax-h3-native/image-native-20261111/commit-readiness-20261111`. These checks support committing the two features separately; workload and release-packaging limitations above still apply.

## Persistent inference and batched image VAE optimization (2026-11-11)

Base: `43758adf012978d42d9ce369c3d1204ddbc42dac`. Single RTX PRO 5000 72 GB, physical GPU 4, 2048×2048, six BF16 steps, seed 42, same prompts and reference as the prior SGLang comparison. Each case ran four times; the table reports the median of the last three requests. Initialization and PNG encoding/writes are excluded. Stage boundaries synchronize CUDA and include weight preparation/movement. These are warm persistent-process results, not cold CLI latency.

| Stage (seconds) | Before T2I | After T2I | Before edit | After edit |
| --- | ---: | ---: | ---: | ---: |
| Text encoding | 20.8589 | 0.9295 | 21.4042 | 1.0370 |
| Reference VAE encoding | 0.0000 | 0.0000 | 3.8583 | 4.1234 |
| AdaLN preparation | 3.2029 | 0.0000 | 3.2409 | 0.0000 |
| Six DiT steps / preparation | 10.5484 | 6.1116 | 20.5773 | 16.0591 |
| VAE decoding | 0.8305 | 0.3203 | 0.8193 | 0.3191 |
| Other | 0.3648 | 0.0043 | 0.5221 | 0.1555 |
| Total | 35.8276 | 7.3659 | 50.4109 | 21.7064 |

- Reuse the bounded CPU text-weight cache and fixed AdaLN tables across requests. With at least 60 GiB initially available (or an explicit budget of that size), keep DiT and VAE decoder weights resident; retain the existing stage/layer unloading path below that threshold. Text embeddings and reference latents are still recomputed on every request. Reuse the tokenizer as well.
- Batch independent VAE tile sequences, repack native interleaved QKV weights once, and use the existing native packed attention/QK-norm/RoPE kernels. Memory-aware batches are capped at 128 tiles. Low-memory and FP32 paths retain sequential tile decoding and the original overlap blending order. No new Torch or Triton runtime dependency.
- T2I improves from 35.8276 to 7.3659 seconds (4.86×); single-reference editing from 50.4109 to 21.7064 seconds (2.32×). The prior same-protocol SGLang default/lossless result was 9.4139 seconds, and strict/exact was 9.9875 seconds. SGLang was not rerun here. Its previously observed edit correctness failure means edit timings are not equivalent-quality cross-engine comparisons.
- Warm post-request GPU usage was 42,450 MiB for T2I and 43,222 MiB for editing, including the CUDA pool/context; weights remain allocated while the service is idle. Peak process RSS was 48,873/50,090 MiB, within the shared host-cache policy. Explicit memory limits remain the way to trade cached speed for capacity.
- Checkpoint-dependent `minimax_h3_image_decode` tests compare sequential and batched outputs for 16×16 and 5×11 latent tiles, batch sizes 2/3/8/128, all 24 decoder blocks for small batches, and changing tile shapes. Maximum relative L2 was 0.00619592, below the BF16 threshold 0.01. This is an operator/decoder check; pixel identity is not promised for arbitrary shapes.
- For both 2K benchmark inputs, all four optimized PNGs were byte-identical to their previous native baselines. The edit remains a snowy winter scene; visually checked.
- `minimax_h3_image_state` verifies T2I → different-seed/reference/dimensions → T2I reuse, cancellation in text/AdaLN/denoise/decode, release of accounted CPU caches after failure, and byte-identical output after recovery.
- One 4096×4096 generation completed with 441 tiles, including multiple batches and a tail batch. Cold process inference time was 63.7281 seconds; sampled peak GPU memory 45,115 MiB and process RSS 49,102 MiB. This is a capacity/correctness smoke test, not a warm 4K speed comparison.
- In a real 4 GiB RAM/no-swap container with an independent allocation leaving only 5 GiB initially free VRAM, automatic policy selected layer streaming and zero host weight cache. Native 2K inference completed in 65.7708 seconds with 809 MiB peak process RSS and a byte-identical PNG. The recorded 69,850 MiB GPU monitor peak includes the deliberate reservation and is not model memory usage.
- Under the same 4 GiB RAM container limit, the HTTP service with explicit 4 GiB GPU / 2 GiB host budgets correctly reported a deliberately oversized job as failed, handled cancellation, then completed a 2K two-reference edit. Sampled GPU peak was 3,573 MiB and process RSS 895 MiB; the service remained ready.
- Fresh CUDA 13 / SM120 build and CTest image-memory / native-HTTP / model-source checks passed (3/3, including five HTTP checks). The CLI/server still link no Torch or Python runtime. Full release-wheel build and 16-reference maximum workloads remain untested.

Model-dependent checks are explicit targets, rather than default tests requiring a 135 GB checkpoint:

```bash
cmake --build build --target minimax_h3_image_decode minimax_h3_image_state
build/src/models/minimax_h3/minimax_h3_image_decode /path/to/MiniMax-H3-Image
build/src/models/minimax_h3/minimax_h3_image_state /path/to/MiniMax-H3-Image reference.png
```

The existing image demo profile was switched to the new `speed-opt-20261111/build` server through the launcher API after verifying that no jobs were active. The launcher address/authentication and all five saved outputs were preserved; native health is ready. The old build remains available. First use after restart still includes weight loading.

## Reference-edit optimization (2026-11-11)

Continued on `master`, after the persistent-cache optimization above. The image encoder now
removes the two zero causal frames from single-frame convolutions, batches up to eight
independent tiles, and uses a private cuBLAS TF32 handle with FP32 accumulation. Normalization,
residuals and posterior moments remain FP32. Low-memory and FP32 diagnostic paths retain the
previous encoder. A separate image CUDA file fuses gate-first SiLU/multiply without changing
its rounding or the existing 4096-row FFN chunk limit. The 16384-row candidate was slower and
used more VRAM, so it was discarded.

Same physical RTX PRO 5000 72 GB, 2048-square input/output, six steps, seed 42; four consecutive
requests per case, first excluded, median of the remaining three. Stage timings include
weight preparation/transfers and CUDA synchronization, but exclude initialization/PNG writing.

| Stage, seconds | Previous optimized edit | Current edit |
| --- | ---: | ---: |
| Text | 1.0370 | 1.0266 |
| Reference VAE | 4.1234 | 1.2129 |
| DiT | 16.0591 | 15.6642 |
| VAE decode | 0.3191 | 0.2986 |
| Total | 21.7064 | 18.3817 |

Edit latency fell 15.32%; reference encoding is 3.40 times faster. T2I is 7.1240 seconds versus
7.3659 previously. This is warm-request performance, not cold process startup. Current warm
post-request VRAM is 41974 MiB for edits / 41522 MiB for T2I, including the retained CUDA pool;
peak process RSS is 50095 / 48889 MiB. SGLang was not rerun; its earlier edit-quality issue remains.

Validation:

- Real checkpoint encoder comparisons cover 32x48, 48x80, 256x256; batches 1/2/3/8; every emitted
  layer; overlapping tiles; unload/reuse; and low-memory fallback. Maximum relative L2 is
  `0.0011612 < 0.01`; blended posterior moments have relative L2 `0.000208622`.
- Gate fusion is exact for BF16, FP16 and FP32, including scalar/vector widths and large tensors.
- All four T2I PNGs match the previous version byte-for-byte. All four edits match one another;
  versus the previous edit, RGB relative L2 is `0.012239746`, MAE `0.987/255`. The acceptance
  threshold applies to operators/encoder tensors, not final RGB. Edits are not pixel-identical;
  visual inspection confirmed the intended snowy forest and preserved subject/composition.
- Original shared-video encoding, one and seventeen frames at 32x48, through both `encodeTile`
  and `encodeMoments`, produces byte-identical outputs with the old/new binaries.
- CTest passes image ops/memory/HTTP, model source and shared H3 fused ops (5/5).
- T2I/edit/T2I state recovery passes. Reference/text/prepare/denoise/decode cancellation releases
  host cache and allows a subsequent identical T2I request.
- Actual 4 GiB RAM/no-swap container, explicit 4 GiB GPU / 2 GiB host budget: oversized job fails
  cleanly, cancellation succeeds, then a two-reference 2K edit completes. Peak GPU 3573 MiB,
  peak server RSS 848 MiB; final health ready.
- Native CUDA13/SM120 builds and `ldd` checks pass; no Torch/Python runtime library dependency.
  Full release wheels, other GPU architectures and the maximum 16-reference workload were not tested.

Model-dependent encoder check: build `minimax_h3_image_encode` with `UNIT_TEST=ON`, then run it
with the checkpoint directory and a reference PNG. `minimax_h3_image_ops` is a CUDA CTest target.
Experiment records and drivers are in `image-native-20261111/edit-opt-20261111`; production sources
contain no timing hooks. Do not overwrite the existing experiment outputs when reproducing.

## Optimization commit review (2026-11-12)

Reviewed the combined changes against `43758adf012978d42d9ce369c3d1204ddbc42dac` on `master`.
Consolidated checkpoint-test build rules, clarified tile-shape and precision invariants, and
made encoder fallback/cache comparisons require exact equality. The checkpoint encoder test
now switches the same VAE instance from fast image encoding back to one- and seventeen-frame
video encoding, through both tile and tiled-moments entry points. Runtime cleanup changed only
comments, formatting and an explicit standard-library include; inference calculations and
memory policies are unchanged from the preceding optimization.

- A fresh CUDA 13 / SM120 build of CLI/server and all relevant test targets passed.
- CTest image ops/memory/HTTP, model source and shared H3 fused ops passed (5/5, 37.42 seconds).
- Real-checkpoint encoder comparisons passed (118 comparisons, maximum relative L2 `0.00120415`).
  Image-to-video switching, unload/reuse and low-memory fallback were exactly equal to their
  respective references. Decoder comparisons passed (80 comparisons, maximum relative L2
  `0.00619592`). Both are below the `0.01` operator tolerance; this does not promise lossless RGB.
- Pipeline T2I/edit/T2I recovery and cancellation at reference/text/prepare/denoise/decode all
  passed, including release of accounted host caches and identical output after recovery.
- Native dependency checks found no Torch/Python libraries or unresolved runtime libraries.
- The previous constrained-RAM/GPU and timing results remain the evidence for those workloads;
  they were not rerun for this non-numerical cleanup. Release-wheel builds, other architectures
  and the maximum sixteen-reference workload remain outside the tested scope.

No blocking issue was found for this scope. Suggested boundaries are persistent caches/batched
decoding, followed by reference-encoder/gate acceleration, each with its own build rules,
tests and documentation. No commit or push was made. Review logs are in
`image-native-20261111/commit-review-20261112`; the existing studio build was left running.

## Dual GPU and operator optimization (2026-11-12)

The preceding reviewed single-GPU changes were committed on local and remote `master` as
`8476d9c0` (persistent caches/batched decoding) and `9217c932` (reference encoder/gate).
The dual-GPU changes below are a subsequent, uncommitted candidate based on `9217c932`.
No push or live studio deployment was performed for this candidate.

Two physical RTX PRO 5000 72 GiB cards (4 and 5), PCIe NODE without NVLink, CUDA 13/SM120.
Same 2048x2048 inputs, BF16, six steps, seed 42. Each case ran four requests; discard the
first and report the median of three warm requests, excluding initialization and PNG
encoding/writes. Native timing synchronizes stage boundaries. SGLang uses its internal
`metrics.total_duration_ms`, not outer save-to-file wall time. SGLang revision is
`23a45fd33921164fa440fd7c28cc63341259c28c`, speed mode, no condition cache, torch.compile,
CUDA Graph or FSDP. Its fastest tested configuration is SP2/Ulysses2 with TP1.

| Configuration | T2I seconds | Single-reference edit seconds |
| --- | ---: | ---: |
| Native single GPU, FP32 row reduction | 7.134661 | 18.352077 |
| Initial native TP2, FP32 reduction | 4.159608 | 11.081382 |
| Initial native TP2, BF16 reduction | 3.795151 | 10.459433 |
| Final native SP2 / text TP2 | 3.288426 | 9.553525 |
| SGLang TP2 | 3.628137 | 11.020111 |
| SGLang SP2, initial | 3.260060 | 9.761293 |
| SGLang SP2, repeat | 3.244474 | 9.768764 |

The final candidate is 2.17x/1.92x faster than the native single-GPU baseline. Against the
SGLang SP2 repeat it is **1.35% slower for T2I and 2.20% faster for editing**: it has not
surpassed SGLang for every workload/operator. An earlier one-shot tuning version measured
3.2996/9.4869 seconds; these results are retained separately, not mixed with the final
version's best values. Three warm requests do not establish a confidence interval.
SGLang's edit still produced an oversaturated autumn scene instead of the requested winter
scene; native produced the requested snow scene. Thus edit timings compare execution time,
not equivalent output quality. The cause of the SGLang quality failure is not established.

The candidate adds a separate image parallel implementation, per-device GPU/shared host
budgets, automatic resident SP versus streaming TP, and CLI/API/launcher integration.
Optimizations include packed QKV, vector QK norm/RoPE, local K/V scatter into
exchange buffers with peer-only chunk copies, overlapping communication and grouped
attention, fused modulation/residual RMS, per-request refiner reuse, local final projection
before gathering 96-wide velocities, SP-specific BF16 cuBLASLt candidate tuning, parallel
VAE tiles, and GPU tile blending/RGB conversion. BF16 gate uses an approximate exponential;
FP16/FP32 gate arithmetic and BF16 SiLU-before-multiply rounding remain unchanged.

CUDA-event profiling covers the main DiT and decoder operators; text and reference encoder
are compared at stage granularity. Rank-0 totals for one warmed request include:

| Region, milliseconds | Native T2I | SGLang T2I | Native edit | SGLang edit |
| --- | ---: | ---: | ---: | ---: |
| DiT QKV | 731.542 | 723.839 | 1592.280 | 1630.773 |
| DiT QK norm / exchange / attention | 671.797 | 583.450 | 2251.585 | 2145.725 |
| DiT output projection | 240.283 | 241.515 | 520.454 | 563.437 |
| DiT FF up | 966.527 | 962.129 | 2046.597 | 2157.754 |
| DiT gate | 37.065 | 37.996 | 98.789 | 93.872 |
| DiT FF down | 500.423 | 507.258 | 1112.949 | 1138.113 |
| DiT RMS / modulation / residual | 31.441 | 31.911 | 79.961 | 76.795 |
| VAE QK norm / RoPE | 2.407 | 2.504 | 2.446 | 2.520 |
| VAE attention | 3.949 | 5.499 | 4.032 | 5.548 |

Fusion boundaries differ, so related regions are combined. These event intervals include
dependencies and submission gaps; communication overlaps attention, and intervals cannot
be added into wall latency. Small 1–3% single-profile differences are not conclusive.
Native local DiT token counts are 2053/4622; SGLang pads them to 2064/4632. VAE work matches
at 61/60 tiles across ranks. Full rank-0/1 tables are in `operator-comparison.csv`.

Validation:

- Native CUDA build, nine CTest targets, plus shared H3 fused ops with one visible GPU all
  passed. Image ops/shared parallel and the real-checkpoint SP comparison passed again
  after the final stable GEMM candidate-selection change. No-NCCL parallel compilation passed.
- SP and BF16 TP each passed 156 same-input checkpoint comparisons, maximum relative L2
  `0.000526287` / `0.00762061`, against the native single-GPU reference. Threshold `0.01`
  applies to individual blocks/operators with identical inputs, not accumulated final output.
- Text encoder relative L2 `0.000584518`; VAE batch 1/3/121 maximum `0.00236291`. The new
  vector VAE QK kernel's standalone maximum was `0.00000529499`, with V unchanged.
- Full 50-block accumulated velocity relative L2 is reported separately. For the three test
  cases SP gives `1.91e-7`, `4.57e-7`, `0.070501`; TP gives `0.134882`, `0.0147643`, `0.0401633`.
  The third case uses repeated checkpoint text rows, mixed tags and 4096 reference rows.
  These accumulated values are not presented as passing the `0.01` operator criterion.
- T2I/edit/T2I reuse and cancellation at reference/text/prepare/denoise/decode passed,
  including host-cache release and byte-identical output after recovery. Final stable-tuning
  benchmark PNGs are byte-identical to the preceding final-kernel version; native edits
  were visually checked for the requested snow, subject and composition.
- HTTP with 3 GiB/GPU and 2 GiB host budgets selected TP2, layer streaming and zero host
  weight cache. Budget exhaustion and cancellation were followed by a successful 256x256
  two-reference edit: sampled GPU peak 3127 MiB, process RSS 1557 MiB, final health ready.
  At 2 GiB/GPU the text temporary tensor correctly exceeded the pool. At 4 GiB/GPU the
  5120x4096 fixture intended to exhaust memory instead completed (GPU 4151 MiB/RSS 1865 MiB).
  Both earlier attempts are preserved. Limits reserve workspace but do not hard-cap total
  context/library usage. The dual-GPU container had a 256 GiB RAM limit: these are explicit
  budget tests, not repeats of the earlier 4 GiB cgroup/no-swap test.
- Launcher: 18 passed, one optional Playwright case skipped. Workspace JS, Python/JSON
  syntax and `git diff --check` passed. Native `ldd` has no Torch, C10 or Python runtime
  dependency and no missing libraries; inference remains C++/CUDA/cuBLAS/NCCL.

Only two SM120 GPUs were exercised; four/eight-GPU execution, other architectures, release
wheels and the maximum sixteen-reference workload remain untested. The existing GPU 3
studio stays on its previously validated build. Reproduction drivers, rejected experiments,
raw timing/profile/numerical logs, final patch and manifests are in
`/workspace2/experiments/minimax-h3-native/image-native-20261111/dual-gpu-20261112`.

## Dual-GPU attention pipeline follow-up (2026-11-12)

Native tensor pools now explicitly grant participating peers access. Dual SM120 SP
uses seven head groups and two persistent attention streams for padded local lengths
512–8192; events order K/V transfer, attention, and the output projection. Scratch is
reused within DiT and released before VAE batching. NCCL fallback buffers survive until
all attention consumers join. Common padded lengths keep both ranks' group selection
consistent at odd-sequence dispatch boundaries. Other configurations retain serial groups.

Same GPU 4/5, 2048² BF16 six-step seed-42 workloads as above. Final native measurements
use ABBA process order, four requests/process, discarding only request zero; medians cover
all six warm samples/version. No initialization or PNG writing is included. SGLang SP2
was rerun at the same pinned revision/settings, taking three warm internal-duration samples.

| Workload | Previous native (s) | Final native (s) | Reduction | SGLang SP2 (s) |
|---|---:|---:|---:|---:|
| T2I | 3.317096 | 3.240128 | 2.32% | 3.245344 |
| Reference edit | 9.598166 | 9.474461 | 1.29% | 9.758389 |

T2I is effectively tied with SGLang, not a conclusive win. The SGLang edit still has the
previously recorded winter-prompt quality failure (unchanged output hash); its timing is
not an equal-quality comparison. All 16 final native PNGs match the previous native output
byte-for-byte; all 16 interleaved baseline PNGs match as well.

Final Nsight 2026.5.1 sampling confirms two attention compute streams and direct peer
copies. Rank-0 main-DiT 300-block totals below use interval union for concurrent kernels,
or the span from the first preparation activity to the next projection for the whole region.
Baseline/SGLang operator traces are from the preceding `operator-gap-20261112` experiment;
these profiled intervals are not a decomposition of the unprofiled ABBA requests above.

| Workload / interval (ms) | Previous native | Final native | SGLang |
|---|---:|---:|---:|
| t2i / Attention core | 487.255 | 445.142 | 410.295 |
| t2i / Attention region incl. dependencies | 727.145 | 566.469 | 591.746 |
| edit / Attention core | 1887.426 | 1825.260 | 1998.596 |
| edit / Attention region incl. dependencies | 2333.502 | 2098.966 | 2197.953 |

This improves the attention/communication region, not every individual operator. T2I
attention core and QK/V preparation still have headroom. Summed concurrent kernel time
cannot be substituted for interval union or added to whole-request latency.

Validation: both direct peer and forced NCCL fallback pass 36 strict-equality pipeline
cases (eight replays, scratch release/reuse, odd shards and dispatch boundaries). Final
SP checkpoint comparisons, cancellation/state recovery and nine CTests pass without skips.
SP/TP same-input maxima remain `0.000526287`/`0.00762061`; accumulated velocity drift is
unchanged and is reported separately in the preceding section. TP, shared single-GPU H3
fused ops, and 3 GiB/GPU + 2 GiB host OOM/cancel/two-reference recovery passed before the
last SP-only lifetime guard; those paths are unchanged. Low-memory sampled peaks were
3127 MiB GPU and 1583.6 MiB RSS, with final health ready. Native `ldd` has no Torch/C10/Python
dependency or missing libraries. Only dual SM120 was exercised; the GPU 3 studio is unchanged.

Reproduction drivers, rejected/intermediate results, final traces, timing/numerical logs,
patches and manifests are in
`/workspace2/experiments/minimax-h3-native/image-native-20261111/attention-opt-20261112`.

## Pre-commit cleanup review (2026-11-12)

Cleanup clarifies asynchronous gather buffer ownership, rejects invalid parallel modes in
the state test, adds the launcher GPU-count validation translation, and normalizes formatting.
Production C++/CUDA tokens are unchanged from the preceding validated attention-pipeline
snapshot; the existing inference libraries were reused, while the server, generation CLI,
state test and parallel test were rebuilt successfully.

Fresh checks on GPU 4/5: direct peer and forced NCCL each pass 36 bitwise-equality pipeline
cases; all nine selected CTests pass without skips; the rebuilt server passes five HTTP
tests. Launcher media tests report 18 passes and one optional Playwright skip; the launcher
asset test, local Node workspace checks, Python/JSON parsing and `git diff --check` pass.
The native server dependency check still finds no Torch/C10/Python or missing libraries.
The container lacks Node, so JS checks ran locally on the identical source. Launcher tests
use `PYTHONPATH=/src/tools/ftllm_agent_runtime/src` for the bundled companion package.

Full-checkpoint numerics, cancellation/low-memory recovery and timings are reused from the
source-equivalent preceding snapshot, not rerun in this cleanup. Their accumulated-drift,
SGLang edit-quality and dual-SM120-only limitations above still apply. Review logs, the cleanup
patch, final source hashes and proposed commit groups are recorded separately in
`/workspace2/experiments/minimax-h3-native/image-native-20261111/dual-commit-review-20261112`.

## Native FP8 integration (2026-11-12)

The existing model path selects BF16 or FP8 by inspecting safetensors matrix dtypes.
No additional precision flag is required. Tests cover ambiguous checkpoints, missing
or ambiguous bases, and U8 Comfy metadata alongside actual FP8 weights. Native CUDA
fixtures cover original head-interleaved versus Comfy packed QKV layouts, scalar/per-channel scales, FC1 half ordering, padded and zero rows,
multiple model instances, cache release/reload and malformed inputs.

On physical GPUs 4/5 (SM120, 72 GiB each), 2048², six steps, seed 42, the same Unsloth
E4M3 weights were compared with SGLang revision `23a45fd33921164fa440fd7c28cc63341259c28c`.
Each workload has six warm samples per implementation (two processes, discard request
zero from four per process). Timings exclude initialization and PNG writing.

| Workload | Native FP8 | SGLang FP8 | Native latency difference |
|---|---:|---:|---:|
| Text-to-image | 2.305785 s | 1.956908 s | +17.83% |
| One-reference edit | 7.377160 s | 6.896687 s | +6.97% |

The gap is concentrated in denoising: 2.171874 vs 1.796475 s for text-to-image,
5.631709 vs 4.969281 s for editing. Native uses cuBLAS 13.2.1.1 to avoid the
multi-device FP8 issue; SGLang's Torch 2.14.0+cu130 environment uses cuBLAS 13.1.1.3
and its existing CUTLASS FP8 implementation. This compares the actual framework stacks.
SGLang runs SP2/TP1, speed mode, lossless quality, without compilation, graphs, FSDP
or conditioning caches. Both retain native auxiliary weights and quantize the same
200 main projections; SGLang receives a lossless layout conversion of the downloaded file.

An isolated SGLang copy required loader compatibility fixes for rowwise metadata,
unaltered AdaLN reconstruction, and one-dimensional QKV scales (which must be reordered
with the weight rows). Its compute kernels were unchanged. Initialization audits
recorded 200 FP8 layers per rank and exact payload/scale matches for the first and last
blocks. Failed loader attempts and outputs before the scale fix are excluded from timing.
SGLang text-to-image is visually coherent after the fix, but its edit still has severe
color distortion and does not follow the requested winter change. The edit latency is
therefore **not an equal-quality comparison**. Native produces the intended snowy edit.

Native FP8 outputs match the preceding validated FP8 experiment byte-for-byte; original
BF16 text-to-image and edit outputs also match the preceding BF16 baseline. The earlier
same-input quantized projection relative L2 maxima were 0.04564226 / 0.04475079; FP8 is
accepted as a separate approximate model and is not claimed to satisfy the 1% BF16 gate.

Eleven native CTests pass without skips. Launcher tests pass 19 checks with one optional
browser check disabled; workspace state and syntax checks pass. Actual SP cancellation
at reference/text/prepare/denoise/decode releases host cache and produces byte-identical
repeats afterward. A 3 GiB-per-device/2 GiB-host FP8 HTTP run recovers from budget
exhaustion and cancellation, then completes a two-reference 256² edit; sampled peak
GPU usage is 3407 MiB and RSS 1565.55 MiB (budgets constrain allocation pools/caches,
not all process overhead). Single-device constrained-memory generation also passes.
The downloaded LSDJesus scalar-scaled E4M3 file completes a separate 2048² dual-device
text-to-image smoke after honoring its packed-QKV metadata; its edit quality/performance was not evaluated. The earlier scalar-layout noise image is retained as a failed attempt. A final Unsloth rerun after this compatibility fix produces the same pixels as the timed build. Other architectures,
4/8 devices, FP8 TP and maximum-reference-count requests were not exercised.

Evidence: `fp8-mainline-20261112` under the existing image experiment directory on
pro5000. It includes all warm samples, stage timings, image hashes, source/binary
manifests, loader adapters, cancellation/low-memory logs and a standalone report.
The GPU3 live studio was not replaced during validation.

## FP8 projection optimization (2026-11-12)

SM120 builds now use a native CUTLASS FP8 GEMM with row/channel scales fused into
the BF16 output epilogue. Aligned BF16 loads, register reuse and paired FP8 stores
speed up activation quantization. Three shape-selected GEMM tiles are internal;
the checkpoint still determines precision automatically. Other architectures and
builds retain the cuBLAS path. No Torch dependency or user option was added.

Fresh interleaved old-native/new-native/SGLang runs used the same dual-SM120,
2048², six-step protocol and stack versions described above. Each number is the
median of six warm requests, excluding initialization and PNG writing.

| Workload | Previous native | Optimized native | SGLang | Native reduction |
|---|---:|---:|---:|---:|
| Text-to-image | 2.308882 s | 1.952967 s | 1.964085 s | 15.42% |
| One-reference edit | 7.390615 s | 6.595090 s | 6.921824 s | 10.76% |

Text-to-image is effectively tied with SGLang; editing takes 4.72% less time.
The SGLang edit output is unchanged and still fails the winter instruction, so
this remains an execution-speed comparison, not an equal-quality comparison.
Native outputs are repeatable and the winter edit remains visually coherent.

Four fresh Nsight captures cover all 300 main blocks on each rank. Rank 0's
activation-quantization kernel sum is 17.12/72.57 ms for text/edit versus SGLang's
21.96/95.48 ms. The 1,200 separate output-scaling launches per rank are eliminated.
The remaining output-projection gap is about 6.6/6.2 ms per request (kernel sums);
not every individual projection is faster. These sums are not additive latency
savings. Warm and cold-cache microbenchmarks of six tile candidates are retained.

Fusing scales removes an intermediate BF16 rounding step. Same-input comparisons
against the previous native FP8 path checked all 200 projections on both ranks
at the first denoising step for each workload: 800 checks, maximum relative L2
0.00415827 (text) and 0.00428721 (edit), below the 0.01 gate. This measures the
optimization's error against previous FP8, not FP8 error against original BF16.
Independent CPU E4M3 fixtures also cover all three vectorized input widths,
non-aligned row counts, zero/tiny values and channel scales.

Eleven native CTests pass without skips. The actual checkpoint passes SP
cancellation/repeatability, two-reference HTTP editing after budget exhaustion
and cancellation, and single-GPU constrained-memory generation. The HTTP run
uses 3 GiB per GPU and 2 GiB host budgets, with sampled peaks of 3407 MiB GPU and
1614.21 MiB RSS; budgets limit pools/caches rather than total process memory.
SM80 and SM89 fallback translation units compile, and the SM89 PTX fallback test
passes on SM120. This is not runtime validation on those older GPUs.
Compute Sanitizer reports zero errors for the FP8 fixtures and 36 full-width
GEMM cases covering all three tiles, row-boundary dispatch and output guards.

Evidence, source/binary hashes, raw trace locations and reproduction scripts:
`/workspace2/experiments/minimax-h3-native/image-native-20261111/fp8-opt-20261112`.

## FP8 cleanup and default-library validation (2026-11-12)

The SM120 CUTLASS path no longer requires an override to load cuBLAS 13.2.1.
Device validation and dispatch share the same backend predicate; only the actual
cuBLAS FP8 fallback retains the multi-Blackwell version guard. Fresh dual-SM120
runs with `LD_LIBRARY_PATH` and `LD_PRELOAD` unset load system cuBLAS 13.1.1 and
complete generation, editing and low-memory recovery. An independently compiled
fallback rejects 13.1.1 and accepts 13.2.1 as intended. GPU selection and the
launcher's existing custom-executable setting remain deployment options; there
are no FP8 tuning environment variables or separate FP8 model-path flags.

Cleanup removes the obsolete launcher FP8 field, duplicate pipeline path state,
the redundant packed-QKV argument, an unused quantization argument, duplicate
GEMM output strides and stale localization entries. New native files follow the
surrounding formatting. CUDA kernel bodies and the CUTLASS configuration are
token-equivalent to the preceding optimization, excluding the unused argument.

All eight 2048² FP8 generation/edit outputs match the preceding optimized build
byte-for-byte. Eleven CTests pass without skips; launcher tests pass 19 checks
with one optional browser test skipped, and the packaged-asset check passes.
Workspace state tests, JS syntax, Python/JSON parsing and `git diff --check` pass.
Compute Sanitizer reports zero errors for the FP8 fixtures and 36 GEMM shape
cases. The SM80/SM89 fallback translation units compile, and the SM89 PTX fixtures
pass on SM120 with cuBLAS 13.2.1; older GPU hardware was not exercised.

Five-stage cancellation/repeatability, two-reference editing after budget
exhaustion and cancellation, and single-GPU low-memory generation all pass in
the default library environment. The 3 GiB-per-GPU/2 GiB-host HTTP test samples
3127 MiB GPU and 1573.45 MiB RSS, ending ready. Native dependencies contain no
Torch, C10 or Python library. The preceding 800 same-input FP8 comparisons and
SGLang performance measurements are retained as prior evidence, not rerun here.

The cleanup patch, environment audit, source/binary hashes and commit-readiness
report are in `fp8-cleanup-20261112` under the same pro5000 experiment directory.


## Single-GPU quantized text-weight residency (2026-11-12)

The single-GPU FP8 path retains immutable text/vision checkpoint weights within
an automatic device-cache budget. It leaves room for the DiT and request
workspaces, rebuilds when the input-size bucket changes, and serializes its first
fill before prefetching. Low-memory streaming and the shared host-cache quota
remain in effect. There are no new inference flags, environment variables or
math changes.

The mainline FP8 before/after resident check produces eight byte-identical 256²
PNGs. Request-state tests pass after reference/long-prompt budget changes and
cancellation in all five stages. Device-cache capacity, retention and rebuilding,
TensorPool, norm/modulate, image ops, FP8/source loaders, HTTP contracts and host
memory/cgroup tests pass. With 3 GiB GPU / 2 GiB host settings, FP8 HTTP recovers
from controlled OOM and cancellation before completing a two-reference edit.
The HTTP process samples 2549 MiB GPU and 847.31 MiB peak RSS. Native dependencies
contain no Torch/C10/Python libraries.

The isolated NVFP4 driver exercises the same cache policy: single-GPU warm T2I
improves from 3.464526 to 2.570106 seconds and reference editing from 9.881654 to
9.224084 seconds, with byte-identical outputs. These are experimental FP4 results,
not mainline FP8 timings; FP4 is not exposed through the API or studio. Complete
BF16 model performance was not rerun. Source diffs, build commands, raw requests,
PNG hashes and regression logs are in `fp4-single-opt-20261112`; subsequent native
and SGLang dual-GPU measurements are in `fp4-dual-after-cache-20261112` and
`fp4-dual-sglang-20261112` under the pro5000 image-model experiment directory.
SGLang's reference-edit color/instruction failure remains an equal-quality
comparison limitation.

After code cleanup, a fresh CMake build of the image/video libraries, CLIs and
servers passes all 13 relevant CTests (the fused-op test requires one visible
GPU and is run separately). Mainline FP8 single/dual-GPU request-state tests pass,
including reference/long-prompt changes and five-stage cancellation/recovery.
The 3 GiB GPU / 2 GiB host HTTP test recovers from OOM and cancellation and
completes a two-reference edit, peaking at 2551 MiB GPU and 847.34 MiB RSS.
Four new resident 256² outputs match the eight preceding before/after PNGs
byte-for-byte. No Torch/C10/Python libraries are linked. This is a correctness
and build check, not a new performance run; logs and hashes are retained in
`image-cache-cleanup-20261112` under the same experiment directory.

## INT8 optimization and quantized mainline integration (2026-11-12)

Base `master`: `333ee804b57ab27c17f5ff6119550d69142d6623`. The previously isolated
INT8 and optimized NVFP4 implementations now use the normal `QuantizedWeights`
interface and separate native CUDA files. CLI, HTTP and the launcher's studio
select BF16/FP8/INT8/NVFP4 from checkpoint data, without extra precision flags or
tuning environment variables. Existing host/GPU budgets and quantized DiT SP
apply to both new formats. NVFP4 is W4A4; INT8 ConvRot is W8A8. Inference remains
C++/CUDA with no Torch, C10 or Python runtime library dependency.

INT8 now keeps H256 rotation and per-row activation quantization in one kernel,
applies activation/weight scales in the CUTLASS GEMM epilogue, and combines
resident FFN chunks up to 16384 rows. Streaming retains bounded chunks. INT8
Comfy and Unsloth layouts are supported; the latter's unmarked rotation is
verified against sampled BF16 rows of every projection before loading. Other
quantization layouts are rejected. NVFP4 retains the validated group-16 E2M1
packing, blocked E4M3 scales and native SM120 block-scaled GEMM.

Fresh timing runs used physical GPU 4, RTX PRO 5000 72 GB, CUDA 13, 2048x2048,
six steps, seed 42 and batch one. Each engine/task ran in two fresh processes,
four requests each; the table is the median of the six warm requests. Native
uses synchronized `Pipeline.generate`; SGLang uses `metrics.total_duration_ms`.
Initialization, first requests and PNG output encoding/writes are excluded.
Native reference-PNG reading is outside the timer; SGLang's is inside it.

| Task | Native before | Native mainline | SGLang rerun | Native reduction |
| --- | ---: | ---: | ---: | ---: |
| Text to image | 4.408871 s | 3.676742 s | 3.893854 s | 16.61% |
| One-reference edit | 13.222265 s | 11.522244 s | 12.078937 s | 12.86% |

Mainline latency is 5.58% / 4.61% below this SGLang run. SGLang is pinned to
`23a45fd33921164fa440fd7c28cc63341259c28c`, with the same prepared INT8 payload
and isolated loader adapters as the previous comparison. DiT/VAE are resident;
text weights use permanent layerwise residency, 32 layers per stack, prefetch two.
Conditioning caching, torch compilation and CUDA graphs are disabled. SGLang's
edit still has severe color artifacts and fails the winter-forest instruction,
so its edit timing is not an equivalent-quality comparison.

Nsight captures attribute all 300 main blocks, with no unmatched launches or
unexpected CUDA API errors. T2I/edit rotation-plus-quantization kernel sums fall
from 419/993 ms to 164/358 ms. Separate output scaling falls from 400/898 ms to
zero; GEMM launch counts fall from 1800/3000 to 1200/1200. Main-block GPU spans
fall from 4086/11427 ms to 3366/9707 ms; pinned SGLang traces show 3422/9942 ms.
Rotation/quantization, output projection and FFN down still lag SGLang; the
overall result also benefits from faster native attention. These are one warm
capture per engine/task, not the endpoint timing samples above. Native/SGLang
sequence padding differs slightly: 4106/4128 T2I rows and 9244/9264 edit rows.

The full-size repeatability check exposed reference-encoder batch changes from
eight tiles to seven when pending frees or retained CUDA-pool pages distorted
raw free-memory queries. `Runtime::availableMemory()` settles its stream and
counts reusable CUDA-pool pages within the active budget before the batch
decision. Exact-size host caches remain accounted as allocations. This restores
eight identical PNGs per INT8 task across both processes, matching the previous
native cold reference; four repeated FP8 edits are also identical. No blanket
stage synchronization or encoder math change was introduced.

Validation:

- Fresh native image/video library, CLI and server builds pass. Nine relevant
  CTests pass, plus final-binary INT8 and tensor-pool regressions.
- Independent dense-H256 INT8 fixtures cover widths 256/5376/7168/14336,
  row counts 1/17/129, both checkpoint layouts, FC1 row/scale permutations,
  zero inputs, destination reuse and repeated calls: relative L2 is zero.
- Independent C++ NVFP4 conversion/matmul and malformed-checkpoint fixtures
  pass. Nine separate NumPy fixtures have maximum relative L2 `9.57532e-5`,
  matching the frozen optimized FP4 behavior. Both formats pass CUDA memcheck
  with zero errors.
- Launcher media tests pass 19 cases, with one optional browser case skipped.
  Real API tests for each format cover deliberate budget exhaustion,
  cancellation, subsequent two-reference 256x256 editing, demo content and
  `dit_precision`. With 3 GiB GPU / 2 GiB host settings, both sample 2549 MiB
  peak GPU usage; peak RSS is 846.59 MiB for INT8 and 853.94 MiB for NVFP4.
- Real unmarked Unsloth INT8 loads, verifies rotation and generates a PNG.
  Four-request mainline NVFP4 runs remain identical within each case: warm
  single-GPU T2I/edit medians are 2.575564/9.212392 s. Dual-GPU INT8 T2I is
  2.012464 s and NVFP4 edit is 5.561719 s, using text TP2 and DiT SP2. These are
  integration checks, not new multi-GPU SGLang comparisons.

Source changes are in the local and pro5000 `master` working trees. The live
GPU 3 studio binary was not restarted. A release wheel, other GPU architectures,
and maximum-reference workloads were not tested. Commands, raw samples,
hashes, sanitizer logs, plots and the patch are retained in
`quantized-mainline-20261112` under the pro5000 image experiment directory.

## Quantized mainline cleanup and commit review (2026-11-12)

FP8 now reuses the bounded checkpoint reader already used by INT8/NVFP4. This
removes duplicate header, payload and transfer code while retaining dtype,
integral-shape, offset, byte-count and scale validation. The always-true INT8
ConvRot flag, unused internal workspace arguments, NVFP4 forwarding wrapper and
unused includes are removed. NVFP4's static scheduler and epilogue require zero
global workspace, so its unused 32 MiB allocation is removed with an explicit
workspace-size guard. The cleanup reduces implementation/test code by 94 lines;
the nine CUDA kernel bodies and CUTLASS math/tile configurations are unchanged.

Review also found that converting `group_size` to an integer accepted `16.5` as
16. NVFP4 now checks the numeric value exactly. The new regression fails against
the preceding frozen library and passes against the cleaned build.

The model sources have no runtime tuning environment-variable hooks. Backend
selection remains based on checkpoint data. Compile-time capability macros,
the launcher's executable override and standard CUDA/NCCL/library configuration
retain their existing purposes; benchmark/profiling scripts stay outside the
repository.

Validation used a separate fresh CUDA 13 build on the same SM120 machine:

- Image/video libraries, CLIs and servers build; all 13 relevant CTests pass.
  FP8, INT8 and NVFP4 each pass CUDA memcheck with zero errors.
- The no-CUTLASS FP8 SM89-PTX and INT8 SM80-PTX paths compile and pass their
  operator fixtures on SM120. The unavailable-NVFP4 translation unit compiles.
  This does not constitute testing older physical GPUs or older CUDA toolkits.
- Launcher media tests pass 19 cases, with one optional browser case skipped.
  JS/Python/JSON checks and `git diff --check` pass. Native dependencies contain
  no Torch, C10 or Python runtime library.
- All 24 full-size outputs match the preceding build byte-for-byte: FP8 single
  GPU editing; INT8 single GPU generation/editing; NVFP4 single GPU
  generation/editing and dual GPU editing. Each case includes four requests.
- FP8/INT8/NVFP4 HTTP services each recover from deliberate budget exhaustion
  and cancellation, then complete two-reference 256x256 editing. Demo content,
  health precision and PNG retrieval pass. At 3 GiB GPU / 2 GiB host settings,
  all sample 2549 MiB GPU peak; RSS peaks are 847.52/833.57/840.09 MiB.

Default-memory edit timings varied during the functional runs. A separate
before/after check fixes the existing host budget to 64 GiB (62 GiB cache cap),
with automatic GPU mode, on the same card. Each library runs four requests;
the final three INT8 2048-square edit requests have medians 11.420793 s before
and 11.483222 s after (+0.55%), with identical PNGs. This is a cleanup regression
check, not a new SGLang benchmark or a cold-loading comparison.

The reviewed working tree remains on `master`, without a commit or push.
Suggested boundaries are the independent reference-VAE memory accounting fix,
then quantized model integration with its tests/API/docs. A release wheel and
maximum-reference workloads remain untested. Build commands, regression logs,
before/after output hashes, environment/kernel audits and the cleanup patch are
in `quantized-cleanup-20261112` under the pro5000 image experiment directory.

## Initial native GGUF image support (2026-11-13)

All five Abiray image variants (Q2_K, Q3_K_M, Q4_K_S, Q4_K_M and Q5_K_M)
were downloaded at revision `fec7ec774db49d0c785bdc2174a488576b7396dd`.
Their 53,012,508,800 bytes match the upstream LFS SHA-256 hashes. Downloads
are separate from the original base and safetensors; those files were not modified.

An independent bounded GGUF reader validates metadata, tensor shapes, block
sizes, offsets and the image-specific six-step PDD schedule. All 208 projections
in the main and token-refiner blocks use GGUF. Comfy QKV/FC1 layouts were checked
against original BF16 rows. Protected/non-projection tensors, PDD heads, text
encoder and VAE still use the matching original base. CUDA decodes compressed
weights into at most 56 MiB of temporary BF16 storage, followed by cuBLAS FP32
accumulation. There is no persistent expanded weight cache, second weight
quantization, Torch runtime, or new precision/environment option. Existing
safetensors backends retain their math and format enum values.

Validation used a separate CUDA 13 build on RTX PRO 5000 Blackwell (SM120):

- Image/video libraries, CLIs and servers build; all 14 relevant CTests pass.
  GGUF CUDA memcheck reports zero errors. Parser rejection, ten tensor types,
  the 2048-row chunk boundary, output slices and cache release are covered.
- An independent gguf-py 0.19.0/NumPy oracle checks ten synthetic types and
  three projection samples from each of the five real checkpoints. All 25
  decoded BF16 comparisons have relative L2 error zero. This measures decoder
  agreement, not quantization error relative to the original model.
- Launcher media tests pass 20 cases with one optional browser case skipped,
  including direct GGUF file selection. JS/Python/JSON and whitespace checks pass.
- Seven whole-model GGUF cases produce 14 full-size 2048-square PNGs: all five
  variants generate, and Q4_K_M edits on one and two GPUs. Both requests within
  each case are byte-identical. Dual GPU uses text TP2 and DiT SP2.
- All 24 FP8/INT8/NVFP4 regression PNGs match the preceding build byte-for-byte.
  BF16 generation/editing each run twice on the old and new builds; the eight
  outputs match between builds.
- The GGUF HTTP service recovers from deliberate GPU budget exhaustion and
  cancellation, then completes two-reference 256-square editing. Demo content,
  health format and PNG retrieval pass. At 3 GiB GPU / 2 GiB host settings,
  sampled GPU peak is 2557 MiB and peak RSS is 847.38 MiB.

Functional timing samples include one cold and one warm request per GGUF case.
Q4_K_M warm generation/editing take 6.537394/18.661272 seconds on one GPU;
dual-GPU editing takes 10.010974 seconds. These are single warm samples, not
statistical benchmarks, and no new SGLang comparison was performed.

Q3_K_M, Q4_K_S, Q4_K_M and Q5_K_M sample images look normal. Q2_K produces
severe noise and detail loss despite passing the decoder oracle; its quality is
not accepted. The cause has not been independently reproduced in another
whole-model backend. Start with Q4_K_M; these samples do not establish quality
across prompts. Other physical GPUs, older CUDA, release wheels, maximum-reference
workloads, arbitrary auxiliary fine-tunes and GGUF text encoders are not covered.

Downloads, hashes, commands, oracle fixtures, regression
logs, sample timings, source patch and build dependency records are retained in
`gguf-native-20261113` under the pro5000 image experiment directory.

## Reused GGUF MMQ and automatic dispatch (2026-11-13)

The image backend now reuses the existing GGUF Q8 activation packing and
`mul_mat_q_xy` INT8 Tensor Core kernels for Q4_0, Q8_0 and Q2_K through Q6_K.
The shared quantizer was extracted without changing its arithmetic; its original
LLM translation unit also compiles. The MMQ tile/math headers remain unchanged.
A separate image adapter owns scratch through the image TensorPool, rather than
the LLM allocator. Q8 and FP32 staging are each capped at 64 MiB, including the
partial-row guard. There is no Torch dependency, converted checkpoint, persistent
expanded weight cache, new CLI flag or runtime environment variable.

Direct MMQ and Stream-K were compared at tile widths 32/64/128; an experimental
FMA-enabled build was also measured. None beat the existing BF16 path for the
4096-token H3 projections. Direct MMQ's QKV/out/FC1/FC2 times were
6.192/2.103/8.431/4.174 ms versus 4.472/1.472/5.937/3.152 ms for BF16 decoding
plus cuBLAS. Full 2048-square generation with forced MMQ took 7.813 seconds.
Conversely, the 32–512-token probes favored MMQ; 768/1024 were mixed and 2048
was slower. Production therefore selects MMQ only for 8–512 input rows with
256-aligned K. Other shapes and unquantized tensors retain bounded BF16/cuBLAS.
This cutoff is based on SM120 measurements, not a claim about every architecture.

Same Q4_K_M checkpoint, RTX PRO 5000 72 GB Blackwell, CUDA 13, seed 42, six
steps, 64 GiB host limit and automatic GPU budget. Each case uses a fresh process,
one cold request and three warm requests; the table reports warm medians and
excludes PNG encoding. Editing uses one reference of the stated canvas size.
Dual GPU uses text TP2 and DiT SP2. Small differences in the large-image cases
should not be attributed to MMQ kernel speed, since those GEMMs retain cuBLAS.

| Workload | Before, seconds | Automatic MMQ/BF16, seconds |
| --- | ---: | ---: |
| 512-square generation, one GPU | 0.924774 | 0.755216 |
| 512-square editing, one GPU | 4.707030 | 4.630997 |
| 2048-square generation, one GPU | 6.656760 | 6.373640 |
| 2048-square editing, one GPU | 18.303334 | 18.095412 |
| 2048-square editing, two GPUs | 10.106966 | 10.089118 |

Native validation executes 14 CTests successfully; the fixture-dependent video
fused-op test is skipped. GGUF memcheck reports zero errors across 29 MMQ
cases, a large-row BF16 fallback, ten exact decode oracles, partial input/output
tiles, a 64 MiB staging boundary, output views and cache reloads. Twenty real
projection comparisons across all five checkpoints use deterministic Gaussian
BF16 inputs; forced-MMQ relative L2 versus the previous GGUF backend is at most
0.013918205 (under the test's 0.02 threshold). This measures implementation
agreement, not checkpoint quality relative to the original BF16 model.
Launcher tests pass 20 cases with one optional browser case skipped. `ldd -r`
finds no unresolved symbols and no Torch, C10 or Python runtime dependency.

All 40 GGUF benchmark PNGs repeat byte-for-byte within their own case. The
32 BF16/FP8/INT8/NVFP4 regression PNGs also match the preceding build exactly.
Single-GPU generation/editing and dual-GPU editing complete; inspected
2048-square Q4 samples have no obvious new corruption. This is a limited sample check, not a
broad image-quality validation. The earlier Q2_K quality warning still applies.

The 2048-square Q4 low-memory probes still pass at 2.5 GiB GPU for generation
and 2.75 GiB for editing, both with a 2 GiB host setting and CPU weight cache
disabled. Sampled GPU peaks remain 2045/2301 MiB; RSS peaks are 873.3/879.3 MiB.
These are runtime budget tests, not a fresh hard-container-RAM minimum search.
A separate 3 GiB GPU / 2 GiB host HTTP process recovers from intentional budget
exhaustion and cancellation, then completes two-reference 256-square editing;
peak GPU/RSS are 2557/835.1 MiB. Low-memory checks ran on the second idle GPU
while the first checked legacy output hashes, after all GGUF timings finished.

Artifacts, source snapshots, operator/tile sweeps, per-request timings and PNGs
are in `gguf-mmq-20261113` under the pro5000 image experiment directory. The
initial overlapping tile-32/direct and editing runs were excluded from final
comparisons. No new SGLang benchmark was run. Validation is limited to SM120;
older GPUs/toolkits and release wheels were not exercised.

## GGUF cleanup and commit validation (2026-11-13)

The shared header contains only the Q8 packing kernel and its block-size
constant. The LLM-specific dynamic launcher stays in its original translation
unit; both its body and the shared kernel arithmetic match the pre-GGUF code.
The image adapter retains compile-time layout selection without linking GGML's
LLM error/allocator paths. All MMQ eligibility checks now live in the adapter,
including the unchanged 8–512-row cutoff. Redundant metadata-array state,
conditional declarations and an unused test-fixture field were removed.
No inference environment variables or CLI options were added; the image sources
have no runtime tuning environment-variable hooks.

CUDA 13 / SM120 validation passes all 15 selected native CTests. GGUF memcheck
reports zero errors across ten decode types and 32 GEMM cases, including the
512/513-row cutoff, bounded staging, output guards and metadata-array parsing.
Launcher media tests pass 20 cases, with one optional browser test skipped.
The original LLM MMQ translation unit compiles; `ldd -r` reports no unresolved
symbols and no Torch, C10 or Python runtime dependency.

Against the saved pre-cleanup library, 28 real projection comparisons across all
five downloaded variants and the dispatch boundary have relative L2 error zero.
Twelve Q4_K_M PNGs from 512-square generation/editing and two-GPU 2048-square
editing match their pre-cleanup outputs byte-for-byte. The existing BF16,
FP8, INT8 and NVFP4 math is unchanged; their native operator regressions pass.
Older GPUs/toolkits, release wheels and broader image-quality coverage remain
outside this validation. The earlier Q2_K quality limitation still applies.
Detailed commands, source snapshots, logs and the proposed two-commit grouping
are in `gguf-cleanup-20261113` under the pro5000 image experiment directory.

## Quantized image tensor parallelism (2026-11-14)

Implemented directly on `master`, starting from `e4261060b4c81db3d8b44ac8d92a71794a2fe2c2`.
Tests use only idle physical GPUs 6/7 (RTX PRO 5000 Blackwell, CUDA 13). The existing
GPU 3 image service was not restarted. FP8/INT8/NVFP4/GGUF weight types remain
automatically detected. Explicit `--parallel-mode tp` now shards image projections;
quantized `auto` retains SP.

Real-checkpoint tests cover all 50 blocks with identical oracle inputs, two
asymmetric reference grids and a two-block 4096-row FFN tail. Both ranks produce
identical outputs. Maximum relative L2 across the input/refiner and block stages:

| Image weights | TP2 activation reduction, resident | TP2 FP32 reduction, streamed |
| --- | ---: | ---: |
| Unsloth FP8 | 0.0403632 | 0.0403053 |
| Comfy INT8 ConvRot | 0.0170969 | 0.0169071 |
| LSDJesus NVFP4 | 0.00421647 | 0.00223572 |
| Abiray GGUF Q4_K_M | 0.0187920 | 0.00283423 |

All pass the video quantized tests' comparison bounds of 0.05 per stage and 0.01
for the output head. These are same-input comparisons against the same
single-GPU quantized checkpoint, not BF16 or full-trajectory equivalence.
The tests separately report accumulated drift; changing TP degree can change pixels.

NVFP4 attention output retains FP32 partials until after summation, even with
activation reduction selected; its FFN output retains the selected precision.
This avoids perturbing the next FFN's global activation scale. The original
BF16-partial attention variant reached 0.0656588 maximum stage relative L2.
No single-GPU/SP arithmetic or video projection behavior is changed.

The 12 selected H3 native CTests and the continuous parallel runtime test pass.
Video FP8/INT8 TP2 real-checkpoint regressions pass.
`ldd -r` finds no unresolved symbols or Torch/C10/Python runtime
dependency. The inference path remains C++/CUDA/NCCL.

HTTP checks pass repeated deterministic 512-square TP2 generation for all four
quantized formats, INT8/NVFP4 two-reference editing, and FP8 with an 8 GiB budget
per GPU and 2 GiB host weight budget (cache disabled). The constrained service
recovers after cancellation during preparation, then completes two identical
requests. The budget test is not a claim about minimum possible memory or total RSS.

Full 2048-square TP2 generation passes for FP8, INT8, NVFP4 and GGUF Q4_K_M.
The four samples were visually inspected, with coherent subjects/backgrounds
and no obvious tile seams; this is a smoke check, not a broad quality evaluation.
Legacy single-GPU BF16/FP8/INT8/NVFP4 reference-edit PNGs and the GGUF generation
PNG match their saved pre-change outputs byte-for-byte. FP8 and NVFP4 two-GPU
SP-mode generation also matches the saved pre-change library byte-for-byte.
FP8 projection fixtures also check original interleaved, Comfy packed and Unsloth
separate Q/K/V layouts on virtual 1/2/4/8-way shards, plus gate/value and scale order.

Commands, logs, checkpoint paths, error diagnostics and outputs are in
`image-quant-tp-20261114` under the pro5000 image experiment directory.
Only two-GPU execution on SM120 is validated. INT8/GGUF require H256/block-aligned
input shards and reject TP8; use TP2/TP4 or SP. Four/eight-GPU execution, older
architectures/toolkits and release wheels have not been exercised.

### Resident single-GPU, SP2 and TP2 timing

RTX PRO 5000 72 GB, physical GPUs 6/7 on the same NUMA node, PCIe 5 x16;
2048x2048, batch 1, six PDD steps, seed 42, BF16 VAE, 64 GiB host budget and
automatic GPU budgets. Prompt: `A red panda beside a mountain stream, autumn forest`.
Each fresh process runs one warmup request and three measured requests; the table
shows medians in seconds as **complete generation / DiT**. Text is encoded on
every request. Complete generation includes text, preparation, DiT, VAE and CPU
image processing, excluding initial loading, PNG encoding, file writes and HTTP.

| Image weights | Single GPU | SP2 | TP2 |
| --- | ---: | ---: | ---: |
| BF16 | 6.916 / 5.799 | 3.266 / 3.134 | 3.511 / 3.365 |
| Unsloth FP8 | 3.567 / 3.231 | 1.952 / 1.822 | 2.334 / 2.205 |
| Comfy INT8 ConvRot | 3.641 / 3.296 | 2.009 / 1.878 | 2.429 / 2.297 |
| LSDJesus NVFP4 | 2.555 / 2.317 | 1.463 / 1.333 | 2.218 / 2.086 |
| Abiray GGUF Q4_K_M | 6.352 / 6.113 | 3.629 / 3.498 | 3.668 / 3.530 |

These measurements favor SP for latency and support retaining the current auto
selection. TP reduces per-device weight storage: FP8's post-request device usage
falls from about 47 GiB with SP2 to 37 GiB with TP2. These are cache-inclusive
snapshots, not peak usage or minimum-memory requirements. Results apply to this
text-to-image workload; they do not establish equal quality between formats.
All repeated outputs within each mode are byte-identical. An untimed original
library cross-check measured FP8 TP2 at 2.352 s (0.76% from the timed run), with
identical PNG output. Reproduction scripts and full samples are in
`image-quant-tp-speed-20261114` under the pro5000 image experiment directory.

### Pre-commit cleanup and regression

The shared checkpoint reader now uses one bounded-buffer upload helper, the FFN
uses one quantized output branch, and parallel test targets share their CUDA
link configuration. CUDA copy ordering, synchronization and numerical operations
are unchanged. No experimental environment switch remains in the image native
code; existing GPU visibility and server executable lookup settings are retained.
HTTP regression now checks both explicit DiT rank-count fields.

The rebuilt sources pass 13 native CTests, the image workspace state test,
eight real-checkpoint image
TP2 cases and video FP8/INT8 TP2 regressions. All eight maximum stage-relative-L2
values above are unchanged. Seven complete 2K generations match pre-cleanup PNG
SHA256 values: BF16/FP8/INT8/NVFP4/GGUF TP2, FP8 single GPU and NVFP4 SP2.
Dependency checks again find no unresolved symbols or Torch/C10/Python runtime.

An initial FP8 retest had a 16.37 s transient after the first warmup. A paired
old/new-library check with two warmups and four measured requests gives medians
of 2.339 s before and 2.345 s after (+0.25%), with identical PNGs. NVFP4 TP2 remains
2.215 s versus 2.218 s in the earlier run. No steady-state regression is observed.
Full samples, environment audit, source hashes and proposed commit grouping are
in `image-tp-cleanup-20261114`.
The earlier multi-GPU-count, platform and release-wheel validation limits still apply.
