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
