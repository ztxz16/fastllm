"""Compare a native trace to Diffusers with identical initial latent noise."""

import argparse
import json
import os
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from diffusers import QwenImage21Pipeline

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--trace", required=True)
p.add_argument("--prompt", default="A red apple on a wooden table.")
p.add_argument("--size", type=int, default=128)
p.add_argument("--cpu-offload", action="store_true")
p.add_argument("--image", action="append", default=[])
p.add_argument("--reference-resolution", type=int, default=1024)
p.add_argument("--height", type=int)
p.add_argument("--steps", type=int, default=4)
a = p.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES") and torch.cuda.device_count() == 1
torch.set_num_threads(8)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
dtype = torch.float32
root = Path(a.trace)
pipe = QwenImage21Pipeline.from_pretrained(a.model, torch_dtype=dtype, local_files_only=True)
if a.cpu_offload:
    pipe.enable_model_cpu_offload(gpu_id=0)
else:
    pipe.to("cuda")
latent = np.fromfile(root / "initial_latent.f32", np.float32).reshape(1, -1, 64)
metrics = {}


def compare(name, ref, native):
    ref = np.asarray(ref, dtype=np.float64).ravel()
    native = np.asarray(native, dtype=np.float64).ravel()
    d = ref - native
    metrics[name] = {
        "relative_l2": float(np.linalg.norm(d) / max(np.linalg.norm(ref), 1e-12)),
        "max_abs": float(np.max(np.abs(d))),
        "rmse": float(np.sqrt(np.mean(d * d))),
    }
    print(name, metrics[name], flush=True)


def step(pipeline, index, t, values):
    current = values["latents"].detach().float().cpu().numpy()
    compare("step." + str(index), current, np.fromfile(root / ("latent.step.%d.f32" % index), np.float32))
    return values


images = [Image.open(path).convert("RGBA") for path in a.image]
from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

resized = []
for image in images:
    w, h, _ = calculate_dimensions(a.reference_resolution**2, image.width / image.height)
    resized.append(pipe.image_processor.resize(image, width=w, height=h))
with torch.inference_mode():
    features, _, _ = pipe.encode_prompt(a.prompt, image=resized or None)
    for i, image in enumerate(resized):
        pixels = pipe.image_processor.preprocess(image).unsqueeze(2).to(device="cuda", dtype=dtype)
        encoded = pipe._encode_vae_image(pixels, None).squeeze(2).permute(0, 2, 3, 1).float().cpu().numpy()
        compare("condition." + str(i), encoded, np.fromfile(root / ("condition.%d.f32" % i), np.float32))
    compare("prompt", features.float().cpu().numpy(), np.fromfile(root / "prompt.f32", np.float32))
    result = pipe(
        prompt=a.prompt,
        height=a.height or a.size,
        width=a.size,
        image=images or None,
        output_resolution=a.reference_resolution,
        num_inference_steps=a.steps,
        latents=torch.from_numpy(latent).to(device="cuda", dtype=dtype),
        callback_on_step_end=step,
    )
result.images[0].save(root / "reference.png")
native = np.array(Image.open(root / "native.png"))
reference = np.array(result.images[0])
compare("pixels", reference, native)
(root / "pipeline-report.json").write_text(json.dumps(metrics, indent=2))
limit = 1e-3
assert all(v["relative_l2"] < limit for k, v in metrics.items() if k != "pixels"), metrics
