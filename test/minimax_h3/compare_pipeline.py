"""Mixed-precision end-to-end differential test using identical native/reference noise tensors.

Run compare_text.py for this prompt first. Reference-only dependencies stay
outside the inference server. Both paths stream layers on the same selected GPU.
"""

import argparse
import gc
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from diffusers import (
    AutoencoderKLMiniMaxH3,
    AutoencoderKLMiniMaxH3Audio,
    MiniMaxH3Scheduler,
    MiniMaxH3Transformer3DModel,
)
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3RotaryPosEmbed
from diffusers.modular_pipelines.minimax_h3.before_denoise import (
    MiniMaxH3PrepareLayoutStep,
    MiniMaxH3SetTimestepsStep,
)
from safetensors import safe_open

assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Select the permitted GPU explicitly"
assert torch.cuda.device_count() == 1, "Expose only the permitted GPU"

p = argparse.ArgumentParser()
p.add_argument("--reference-image")
p.add_argument("--reference-image-short-edge", type=int, default=2048)
p.add_argument("--first-frame")
p.add_argument("--last-frame")
p.add_argument("--model", default="/model")
p.add_argument("--binary", required=True)
p.add_argument("--text-fixture")
p.add_argument("--precision", choices=["bf16", "fp32"], default="bf16")
p.add_argument("--vae-precision", choices=["fp16", "bf16", "fp32"])
p.add_argument("--output", required=True)
p.add_argument("--width", type=int, default=64)
p.add_argument("--height", type=int, default=64)
p.add_argument("--frames", type=int, default=124)
p.add_argument("--steps", type=int, default=4)
p.add_argument(
    "--prompt", default="A cat plays piano on a wooden stage. The audience applauds."
)
p.add_argument("--skip-reference", action="store_true")
p.add_argument("--attention-backend", choices=["auto", "math"], default="auto")
a = p.parse_args()
a.vae_precision = a.vae_precision or ("fp32" if a.precision == "fp32" else "fp16")
if a.attention_backend == "math":
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_cudnn_sdp(False)
root = Path(a.model)
out = Path(a.output)
out.mkdir(parents=True, exist_ok=True)
configuration = {
    "reference_image": a.reference_image,
    "reference_image_short_edge": a.reference_image_short_edge,
    "first_frame": a.first_frame,
    "last_frame": a.last_frame,
    "prompt": a.prompt,
    "width": a.width,
    "height": a.height,
    "frames": a.frames,
    "steps": a.steps,
    "attention_backend": a.attention_backend,
    "precision": a.precision,
    "vae_precision": a.vae_precision,
}
manifest = out / "reference-config.json"
if a.skip_reference:
    assert manifest.exists(), "Reference manifest missing; regenerate the reference"
    saved = json.loads(manifest.read_text())
    saved.setdefault("precision", "fp32")
    saved.setdefault("vae_precision", "fp32")
    assert saved == configuration, "Reference configuration mismatch"
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dtype = torch.float32 if a.precision == "fp32" else torch.bfloat16
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
text_dir = Path(a.text_fixture or f"/work/text-{a.precision}")
tokens = json.loads((text_dir / "tokens.json").read_text())
nt = len(tokens)
lt = (a.frames - 5) // 17 * 5 + 2
lh = a.height // 16
lw = a.width // 16
at = round(a.frames * 40 / 24)
np_rng = np.random.default_rng(225)
v = np_rng.standard_normal((lt * lh * lw // 4, 96), dtype=np.float32)
aud = np_rng.standard_normal((at * 2, 32), dtype=np.float32)
v.tofile(out / "initial_video.f32")
aud.tofile(out / "initial_audio.f32")


def save(name, tensor):
    tensor.detach().float().cpu().numpy().tofile(out / f"{name}.reference.f32")


if not a.skip_reference:
    text = torch.from_numpy(
        np.fromfile(text_dir / "layer.49.reference.f32", dtype=np.float32).reshape(
            1, nt, 5120
        )
    ).cuda()
    save("text", text)
    if a.reference_image:
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            MiniMaxH3Ref2VAPrepareLayoutStep,
        )
        from types import SimpleNamespace

        packed = MiniMaxH3Ref2VAPrepareLayoutStep.build_ref2va_packed_sequence(
            torch.tensor([0 if 151652 <= t <= 151656 else 1 for t in tokens]),
            [SimpleNamespace(kind="image", has_audio=False)],
            [
                torch.empty(
                    1,
                    24,
                    1,
                    a.reference_image_short_edge // 16,
                    a.reference_image_short_edge // 16,
                )
            ],
            [],
            lt,
            lh,
            lw,
            at,
            (1, 2, 2),
            2,
            2,
            0,
        )
    else:
        packed = MiniMaxH3PrepareLayoutStep.build_packed_sequence(
            torch.tensor([0 if 151652 <= t <= 151656 else 1 for t in tokens]),
            lt,
            lh,
            lw,
            at,
            (1, 2, 2),
            2,
            2,
            0,
            tuple(
                k for k, p in (("first", a.first_frame), ("last", a.last_frame)) if p
            ),
        )
    positions, tags, vi, ai, ti, _, _ = packed
    positions, tags, vi, ai, ti = [x.cuda() for x in (positions, tags, vi, ai, ti)]
    condition = []
    if a.first_frame or a.last_frame or a.reference_image:
        from PIL import Image
        from diffusers.modular_pipelines.minimax_h3.encoders import encode_vae_condition
        from diffusers.modular_pipelines.minimax_h3.before_denoise import (
            patchify_video_latents,
        )

        vae = (
            AutoencoderKLMiniMaxH3.from_pretrained(
                root / "vae", torch_dtype=torch.float32
            )
            .cuda()
            .eval()
        )
        generator = torch.Generator().manual_seed(0)
        with torch.inference_mode():
            for path in (a.first_frame, a.last_frame, a.reference_image):
                if not path:
                    continue
                image = Image.open(path).convert("RGB")
                if a.reference_image:
                    assert image.width == image.height, (
                        "This reference fixture uses a square image"
                    )
                    image = image.resize(
                        (a.reference_image_short_edge, a.reference_image_short_edge),
                        Image.Resampling.LANCZOS,
                    )
                elif not condition:
                    image = image.resize((a.width, a.height), Image.Resampling.LANCZOS)
                else:
                    scale = max(a.width / image.width, a.height / image.height)
                    w, h = round(image.width * scale), round(image.height * scale)
                    image = image.resize((w, h), Image.Resampling.LANCZOS)
                    image = image.crop(
                        (
                            (w - a.width) // 2,
                            (h - a.height) // 2,
                            (w + a.width) // 2,
                            (h + a.height) // 2,
                        )
                    )
                pixels = (
                    torch.from_numpy(np.array(image))
                    .cuda()
                    .permute(2, 0, 1)[None, :, None]
                )
                z = encode_vae_condition(
                    vae, pixels, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                )
                noise = torch.randn(z.shape, generator=generator)
                z = 0.999 * z + (1 - 0.999) * noise
                condition.append(patchify_video_latents(z, (1, 2, 2)))
        del vae, pixels, z
        gc.collect()
        torch.cuda.empty_cache()
    condition_rows = sum(x.shape[0] for x in condition)
    if condition:
        save("condition_video", torch.cat(condition))
    dit_directory = root / ("transformer_ref" if a.reference_image else "transformer")
    cfg = json.loads((dit_directory / "config.json").read_text())
    index = json.loads(
        (dit_directory / "diffusion_pytorch_model.safetensors.index.json").read_text()
    )["weight_map"]
    with torch.device("meta"):
        transformer = MiniMaxH3Transformer3DModel.from_config(cfg)

    def load(module, prefix):
        state = {}
        for name in module.state_dict():
            key = prefix + name
            with safe_open(
                str(dit_directory / index[key]), framework="pt", device="cpu"
            ) as f:
                target = (
                    torch.float32
                    if any(
                        s in key
                        for s in MiniMaxH3Transformer3DModel._keep_in_fp32_modules
                    )
                    else dtype
                )
                state[name] = f.get_tensor(key).to(device="cuda", dtype=target)
        module.load_state_dict(state, assign=True)

    for name, child in transformer.named_children():
        if name in ("rope", "time_proj"):
            continue
        if name == "transformer_blocks":
            for i, block in enumerate(child):
                block.register_forward_pre_hook(
                    lambda m, args, i=i: load(m, f"transformer_blocks.{i}.")
                )
                block.register_forward_hook(
                    lambda m, args, result: m.to_empty(device="meta") and None
                )
        else:
            load(child, name + ".")
    transformer.rope = MiniMaxH3RotaryPosEmbed(
        cfg["rope_freq_dim"], cfg["rope_theta"]
    ).cuda()
    transformer.eval()
    video = torch.cat(condition + [torch.from_numpy(v)]).cuda()
    audio = torch.from_numpy(aud).cuda()
    vs = MiniMaxH3Scheduler(12)
    aas = MiniMaxH3Scheduler(3)
    vs.set_timesteps(a.steps, device="cuda")
    aas.set_timesteps(a.steps, device="cuda")
    start = time.monotonic()
    with torch.inference_mode():
        for i, (t, ta) in enumerate(zip(vs.timesteps, aas.timesteps)):
            times, indices = MiniMaxH3SetTimestepsStep.build_row_timesteps(
                vi.cpu(),
                ai.cpu(),
                condition_rows,
                0,
                nt,
                float(t),
                float(ta),
                0.999,
                1.0,
            )
            pred = transformer(
                video[None],
                audio[None],
                text,
                times.cuda(),
                indices.cuda(),
                tags,
                positions,
                vi,
                ai,
                ti,
            )
            generated = vs.step(
                pred.sample[0, condition_rows:], t, video[condition_rows:]
            ).prev_sample
            video = torch.cat((video[:condition_rows], generated))
            audio = aas.step(pred.audio_sample[0], ta, audio).prev_sample
            save(f"video_step.{i}", video)
            save(f"audio_step.{i}", audio)
            print(
                f"reference denoise {i + 1}/{a.steps - 1}: {time.monotonic() - start:.1f}s",
                flush=True,
            )
    z = (
        video[condition_rows:]
        .reshape(lt, lh // 2, lw // 2, 24, 2, 2)
        .permute(3, 0, 1, 4, 2, 5)
        .reshape(1, 24, lt, lh, lw)
    )
    az = audio.reshape(2, at, 32).permute(0, 2, 1)
    del transformer, child, block, pred, text, video, audio
    gc.collect()
    torch.cuda.empty_cache()
    vae = (
        AutoencoderKLMiniMaxH3.from_pretrained(root / "vae", torch_dtype=torch.float32)
        .cuda()
        .eval()
    )
    mean = torch.tensor(vae.config.latents_mean, device="cuda").view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    video_dtype = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }[a.vae_precision]
    with (
        torch.inference_mode(),
        torch.autocast("cuda", dtype=video_dtype, enabled=a.vae_precision != "fp32"),
    ):
        pixels = vae.decode(z * std + mean).sample
        pixels = (
            pixels
            * torch.tensor([0.229, 0.224, 0.225], device="cuda").view(1, 3, 1, 1, 1)
            + torch.tensor([0.485, 0.456, 0.406], device="cuda").view(1, 3, 1, 1, 1)
        ).clamp(0, 1)
        rgb = (
            (pixels * 255)
            .round()
            .byte()
            .permute(0, 2, 3, 4, 1)
            .contiguous()
            .cpu()
            .numpy()
        )
    rgb.tofile(out / "output.reference.rgb")
    del vae, z, pixels
    gc.collect()
    torch.cuda.empty_cache()
    avae = (
        AutoencoderKLMiniMaxH3Audio.from_pretrained(
            root / "audio_vae", torch_dtype=torch.float32
        )
        .cuda()
        .eval()
    )
    am = torch.tensor(avae.config.latents_mean, device="cuda").view(1, -1, 1)
    ast = torch.tensor(avae.config.latents_std, device="cuda").view(1, -1, 1)
    with torch.inference_mode():
        samples = avae.decode(az * ast + am).sample[:, 0].T.contiguous().cpu().numpy()
    samples.tofile(out / "output.audio.reference.f32")
    del avae, az
    gc.collect()
    torch.cuda.empty_cache()

    manifest.write_text(json.dumps(configuration, indent=2))

native = out / "native"
native.mkdir(exist_ok=True)
subprocess.run(
    [
        a.binary,
        "--precision",
        a.precision,
        "--vae-precision",
        a.vae_precision,
        "--model",
        str(root),
        "--prompt",
        a.prompt,
        "--width",
        str(a.width),
        "--height",
        str(a.height),
        "--frames",
        str(a.frames),
        "--steps",
        str(a.steps),
        "--trace",
        str(native),
        "--initial-video",
        str(out / "initial_video.f32"),
        "--initial-audio",
        str(out / "initial_audio.f32"),
        "--output",
        str(out / "native.mp4"),
    ]
    + (["--first-frame", a.first_frame] if a.first_frame else [])
    + (["--last-frame", a.last_frame] if a.last_frame else [])
    + (
        [
            "--reference-image",
            a.reference_image,
            "--reference-image-short-edge",
            str(a.reference_image_short_edge),
        ]
        if a.reference_image
        else []
    ),
    check=True,
)
report = {"configuration": configuration}
for ref in out.glob("*.reference.f32"):
    name = ref.name.replace(".reference.f32", ".f32")
    actual = np.fromfile(native / name, dtype=np.float32)
    expected = np.fromfile(ref, dtype=np.float32)
    diff = actual.astype(np.float64) - expected
    report[name] = {
        "relative_l2": float(
            np.linalg.norm(diff) / np.linalg.norm(expected.astype(np.float64))
        ),
        "max_abs": float(np.max(np.abs(diff))),
    }
rgb = np.fromfile(native / "output.rgb", dtype=np.uint8).astype(np.int16)
ref = np.fromfile(out / "output.reference.rgb", dtype=np.uint8).astype(np.int16)
report["rgb"] = {
    "max_abs": int(np.max(np.abs(rgb - ref))),
    "mean_abs": float(np.mean(np.abs(rgb - ref))),
    "equal_fraction": float(np.mean(rgb == ref)),
}
(out / "report.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
assert report[f"video_step.{a.steps - 2}.f32"]["relative_l2"] < 1e-3, report
assert report[f"audio_step.{a.steps - 2}.f32"]["relative_l2"] < 1e-3, report
assert report["output.audio.f32"]["relative_l2"] < 1e-3, report
assert report["rgb"]["max_abs"] <= 1, report
