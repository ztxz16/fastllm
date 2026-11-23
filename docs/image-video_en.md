# Image and video generation

[中文](image-video.md) · [Project home](../README_EN.md) · [Launcher and Studio](launcher_en.md)

FastLLM provides native C++ / CUDA inference for MiniMax-H3 / Pika H3 Flex audio-video generation, MiniMax-H3-Image generation and editing, and Qwen-Image-2.1 generation and editing. Model inference does not require PyTorch or Diffusers. Python manages the launcher, CLI, and request proxying.

## Models and entry points

| Model | Capabilities | Recommended entry point | Alternative |
| --- | --- | --- | --- |
| MiniMax-H3 | Text-to-audio/video, first/last frames, ordered image/video/audio references | Video backend in `ftllm launch` | `minimax_h3_server` / `minimax_h3_generate` |
| Pika H3 Flex | H3 base plus official adapters, fixed six-step audio/video generation | Same video backend, selecting the Pika directory | Same H3 executables, with adapter detection |
| MiniMax-H3-Image | Six-step image generation, single-image editing, multiple references | Image backend in `ftllm launch` | `minimax_h3_image_server` / `minimax_h3_image_generate` |
| Qwen-Image-2.1 | Image generation and single/multiple-reference editing | `ftllm server MODEL_PATH` and `/demo` | `ftllm qwen-image generate` |

The launcher and `ftllm server MODEL_PATH` detect the image or video service from local model metadata.

World-model deployments have separate guides: [HY-Nomas](hy_nomas.md) and [LingBot World](lingbot_world.md), currently in Chinese.

## Installation and preparation

This guide targets **Linux x86_64 with NVIDIA GPUs**. Release media backends target SM80+; FP8 and NVFP4 have additional requirements below. The LLM backend's support for older GPUs or other platforms does not imply the same media support.

```sh
python -m pip install -U ftllm
ftllm launch
```

- Use a CUDA wheel or portable package containing the media components. The 1.9.2 release includes all three native backends, demos, and API assets. For the portable package, use its bundled launcher; see [releases](https://github.com/ztxz16/fastllm/releases).
- Download weights separately. Keep the complete original model directory, including encoders, VAE, tokenizer, configuration, and required sampling components. A quantized DiT file usually cannot run alone.
- Use a driver compatible with the CUDA runtime. Multi-GPU execution needs NCCL. Video encoding and some reference-image decoding need `ffmpeg` / `ffprobe`. Portable packages include these runtime components; wheel users should check their environment's dependencies and executables. A full PyTorch installation is unnecessary.

For the direct H3 executable examples below, locate the package in the Python environment where the wheel is installed:

```sh
FTLLM_PACKAGE="$(python -c 'from pathlib import Path; import ftllm; print(Path(ftllm.__file__).resolve().parent)')"
```

Source builds are documented for [H3](../tools/minimax_h3/README.md#构建), [H3 Image](../tools/minimax_h3_image/README.md#build-and-run), and [Qwen-Image](../tools/qwen_image/README.md#构建). Building only the LLM library does not build these executables.

## Video: MiniMax-H3 / Pika H3 Flex

### Launcher

Add a launch profile and select the MiniMax H3 or Pika model path. Set the port, output directory, and available GPUs, then save and start. Release packages discover the executable automatically.

- For H3, select the complete original model directory.
- For Pika, select the official adapter directory containing `fl2va/manifest.json`, `fl2va/adapter.safetensors`, `ref2va/manifest.json`, and `ref2va/adapter.safetensors`. A matching complete H3 base is still required. Set its path in the profile or add a `base_model` symlink inside the Pika directory. See the [Pika base revision and layout](../tools/minimax_h3/README.md#pika-h3-flex).
- Once ready, open **Studio → Video generation** or the standalone demo. Pika uses six steps; H3 defaults to 50.

Keyframes and reference-media mode cannot be combined. Reference mode requires `transformer_ref/`; standalone audio references must accompany an image or video. See [H3 API details](../tools/minimax_h3/README.md#http-与-demo) for reference counts and duration limits.

### Native server and API

```sh
"$FTLLM_PACKAGE/minimax_h3_server" \
  --model /path/to/MiniMax-H3 --devices 0 \
  --host 127.0.0.1 --port 8009 --output ./h3-output \
  --demo "$FTLLM_PACKAGE/h3_assets/demo.html"
```

For Pika, change the model directory and add `--base-model /path/to/MiniMax-H3` if needed. Open `http://127.0.0.1:8009/demo`. `/health` reports initialization status; `/v1/models` reports the model ID.

```sh
curl http://127.0.0.1:8009/v1/videos \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3","prompt":"A cat plays piano. Soft piano music fills the room.","size":"832x480","seconds":"5","seed":42,"num_inference_steps":50}'
```

Pika requests use `"model":"pika-h3-flex"` and `"num_inference_steps":6`. Submission returns HTTP 202 and an `id`. Replace `VIDEO_ID` below; download after completion:

```sh
curl http://127.0.0.1:8009/v1/videos/VIDEO_ID
curl http://127.0.0.1:8009/v1/videos/VIDEO_ID/content -o video.mp4
curl -X POST http://127.0.0.1:8009/v1/videos/VIDEO_ID/cancel
```

Keyframe requests use multipart `input_reference` / `first_frame` and `last_frame`. Reference-media requests use repeated `reference` fields with explicit MIME types. Server-local paths and remote URLs are not accepted. Duration rounds up to a supported frame count; inspect the returned values.

## Images: MiniMax-H3-Image

Select the MiniMax H3 Image model path in Launcher and set the port, output directory, and GPUs. **Studio → Image generation** and the standalone demo support text prompts, ordered references, editing, progress, cancellation, and downloads.

BF16 uses the complete original directory. For supported FP8 / INT8 / NVFP4 / GGUF layouts, select the specific quantized file and keep the full original H3 Image base. A `base_model` directory/symlink inside the quantized directory avoids ambiguous discovery. The H3 video base cannot replace the image base.

```sh
"$FTLLM_PACKAGE/minimax_h3_image_server" \
  --model /path/to/MiniMax-H3-Image --devices 0 \
  --host 127.0.0.1 --port 8010 --output ./h3-image-output \
  --demo "$FTLLM_PACKAGE/h3_image_assets/demo.html"
```

Use asynchronous jobs for long requests:

```sh
curl http://127.0.0.1:8010/v1/images/jobs \
  -H 'Content-Type: application/json' \
  -d '{"model":"minimax-h3-image","prompt":"An orange cat by a window in soft morning light.","size":"2048x2048","seed":42,"n":1,"num_inference_steps":6}'

# Replace IMAGE_ID with the returned id.
curl http://127.0.0.1:8010/v1/images/jobs/IMAGE_ID
curl http://127.0.0.1:8010/v1/images/jobs/IMAGE_ID/content -o image.png
```

Synchronous text generation uses `POST /v1/images/generations`. For synchronous reference editing:

```sh
curl http://127.0.0.1:8010/v1/images/edits \
  -F 'image[]=@subject.png' -F 'image[]=@scene.png' \
  --form-string 'prompt=Place the subject from <Picture 1> into the scene from <Picture 2>.' \
  -F size=2048x2048 -F seed=42 -F response_format=url
```

`data[0].url` is relative to the service. Send the same form to `/v1/images/jobs` for asynchronous editing; cancel with `POST /v1/images/jobs/{id}/cancel`. Reference ordering defines `<Picture 1>` labels. Do not mix `image` and `image[]` fields. This model uses fixed six-step PDD; do not reuse another diffusion model's step or CFG settings. See the [H3 Image guide](../tools/minimax_h3_image/README.md).

## Images: Qwen-Image-2.1

### Server or direct generation

```sh
ftllm server /path/to/Qwen-Image-2.1 \
  --devices 0 --host 127.0.0.1 --port 8008 --output ./qwen-image-output
```

Open `http://127.0.0.1:8008/demo` for reference uploads, seeds, progress, cancellation, downloads, and API examples. The CLI locates packaged demo assets automatically.

```sh
ftllm qwen-image generate --model /path/to/Qwen-Image-2.1 --devices 0 \
  --prompt 'A red apple on a wooden table.' \
  --width 1024 --height 1024 --steps 40 --seed 1234 --output apple.png
```

Add `--image reference.png --reference-resolution 512` to edit a reference; repeat `--image` for multiple images. Use `ftllm server --help` / `generate --help` for complete options.

### HTTP API

```sh
curl http://127.0.0.1:8008/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"model":"qwen-image-2.1","prompt":"An orange cat by a window.","size":"1024x1024","n":1,"seed":1234,"num_inference_steps":40,"response_format":"url"}'

curl http://127.0.0.1:8008/v1/images/edits \
  -F model=qwen-image-2.1 -F 'image[]=@apple.png' \
  --form-string 'prompt=Change the apple to green. Keep the background.' \
  -F size=512x512 -F reference_resolution=512 \
  -F seed=1234 -F num_inference_steps=40 -F response_format=url
```

Asynchronous submission/status/cancellation also uses `/v1/images/jobs`. Currently `n=1`; width/height are 64–2048, in multiples of 32; steps are 2–100, default 40. HTTP reference uploads must be PNG; the demo can convert JPEG/WebP. Text/vision encoding is limited to 4096 tokens: lower `reference_resolution` or reduce references when necessary. Masks, negative prompts / CFG, LoRA, and remote URL inputs are not implemented.

### Quantized Qwen directories

Keep the original BF16 directory intact and create a separate model directory. Reuse its encoders, VAE, processor, scheduler, `model_index.json`, and `transformer/config.json`.

- FP8 / INT8: place or symlink the supported quantized file as `transformer/diffusion_pytorch_model.safetensors`.
- GGUF: place exactly one selected GGUF inside `transformer/`, without another Safetensors DiT.
- Pass the new root directory to `--model`; precision is detected automatically. FP8 / INT8 / GGUF remain experimental, with some regressions exceeding existing error thresholds. Qwen-Image NVFP4 is not implemented.

See [Qwen-Image formats and validation](../tools/qwen_image/README.md) for supported exports and their limitations.

## Quantization and multiple GPUs

| Model | Implemented DiT formats | Parallelism |
| --- | --- | --- |
| H3 / Pika | BF16, supported FP8, INT8 ConvRot, mixed NVFP4 / INT8 / BF16, GGUF Q4_K_M | `--parallel-mode auto\|tp\|sp`; quantized auto selects SP; INT8 / GGUF TP supports 2/4 GPUs, use SP for eight |
| H3 Image | BF16, supported FP8, INT8, NVFP4, GGUF | Same mode option; quantized auto selects SP; FP8 / NVFP4 TP supports 2/4/8, INT8 / GGUF TP supports 2/4 |
| Qwen-Image-2.1 | BF16; experimental FP8 / INT8 / GGUF | DiT TP supports 1/2/4/8; encoders and VAE run on the first GPU |

Physical multi-GPU validation primarily covers two GPUs. Supported partition dimensions are not guarantees of performance or numerical accuracy on every device count. TP shards weights; SP mainly partitions sequence work while retaining weight copies. Select supported TP explicitly when reducing per-GPU weight memory is the goal.

FP8 needs SM89+ FP8 Tensor Cores. H3-family NVFP4 needs SM120, CUDA 12.8+, and the corresponding build. INT8 / GGUF still include BF16 model stages. Quantization replaces selected projections, while auxiliary components retain base precision. Format support does not cover arbitrary export layouts or guarantee faster or identical output. Inspect image quality especially with low-bit H3 Image GGUFs.

H3 / Pika quantized directories use `h3_quantization.json` to bind `base_model`, `fl2va`, `ref2va`, and optional `pika`. Video and image DiTs are not interchangeable. See the [H3 quantization manifest](../tools/minimax_h3/README.md#量化视频权重).

```sh
# Logical devices 0 and 1 refer to physical GPUs 2 and 3 selected here.
CUDA_VISIBLE_DEVICES=2,3 ftllm server \
  /path/to/Qwen-Image-2.1 --devices 0,1 --port 8008
```

H3 executables also accept `--devices 0,1`; Launcher remaps selected GPUs automatically. H3 / H3 Image expose `--gpu-memory-limit GiB` per GPU and `--host-memory-limit GiB` shared by the service. Qwen-Image automatically selects residency or stage offload from free memory and the request. Model budgets are not hard isolation of all process allocations; allow space for CUDA, communication, activations, and media processing.

## Let an LLM call the video service

Start a tool-capable LLM and an H3 / Pika service. Install the Pi runtime in Studio, create a **workspace Agent**, and choose the chat model. Ask it to generate a five-second cat video and save it under `videos` in the workspace.

Launcher supplies a live generation-service catalog and a scoped helper. The Agent submits, polls, and downloads the video, which appears as an attachment. Ordinary chat does not execute local commands. This Agent integration currently targets video; use the image workspace or standalone demo for images. Services need separate ports and a shared memory plan if they use the same GPU. See the [Launcher guide](launcher_en.md).

## Status, output, and troubleshooting

- Check `/health` and startup logs first. H3 reports actual precision and DiT parallel mode. Each server's `/openapi.json` is authoritative for request fields.
- Native services accept `--api-key KEY`; send `Authorization: Bearer KEY`. Service API keys and Launcher control tokens are separate. Open Launcher through the complete URL printed by `ftllm launch`.
- Multi-GPU inference does not mean multiple independent generation jobs execute simultaneously. Each service runs a task queue, currently at most four running/pending jobs; excess requests receive 429. Cancellation occurs at a stage or sampling boundary.
- Images are PNG and videos are MP4. Prefix relative download URLs with the service address. Output is saved in the configured directory; manage disk space separately. Interrupted unfinished jobs do not resume automatically.
- Missing native programs indicate an incomplete package/build. For missing `libnccl.so.2`, check the package runtime or environment's NCCL installation before restarting.
- A quantized file still needs matching original auxiliary components. Do not mix video/image bases or unrelated model revisions.

Numerical boundaries and regression entry points are documented for [H3](../tools/minimax_h3/README.md), [H3 Image](../tools/minimax_h3_image/README.md), and [Qwen-Image](../tools/qwen_image/README.md).
