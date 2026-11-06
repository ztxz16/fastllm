"""Development-only differential tests. PyTorch/Diffusers are not serving dependencies.

Run with CUDA_VISIBLE_DEVICES set to the single permitted GPU UUID. Each stage
uses identical input tensors, and releases the reference GPU weights before the
native call. Reports are saved even when an accuracy assertion fails.
"""

import argparse
import ctypes as C
import gc
import json
import os
from pathlib import Path

import numpy as np
import torch

p = argparse.ArgumentParser()
p.add_argument("--model", required=True)
p.add_argument("--library", required=True)
p.add_argument("--stage", choices=["text", "dit", "vae", "all"], default="all")
p.add_argument("--output", default="qwen-image-accuracy")
args = p.parse_args()
assert os.environ.get("CUDA_VISIBLE_DEVICES"), "Explicit GPU selection is required"
assert torch.cuda.device_count() == 1, "Expose only the permitted GPU"
out = Path(args.output)
out.mkdir(parents=True, exist_ok=True)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_num_threads(8)
dtype = torch.float32
torch.manual_seed(1234)
lib = C.CDLL(args.library)
ptr = C.POINTER(C.c_float)
lib.fastllm_qwen_image_create.argtypes = [C.c_char_p, C.c_int]
lib.fastllm_qwen_image_create.restype = C.c_void_p
lib.fastllm_qwen_image_destroy.argtypes = [C.c_void_p]
lib.fastllm_qwen_image_error.restype = C.c_char_p
lib.fastllm_qwen_image_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.POINTER(C.c_int), C.c_int]
lib.fastllm_qwen_image_text.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, ptr]
lib.fastllm_qwen_image_dit.argtypes = [C.c_void_p, ptr, ptr, C.c_int, C.c_int, C.c_int, C.c_float, ptr]
lib.fastllm_qwen_image_vae.argtypes = [C.c_void_p, ptr, C.c_int, C.c_int, ptr]
handle = lib.fastllm_qwen_image_create(args.model.encode(), 0)
if not handle:
    raise RuntimeError(lib.fastllm_qwen_image_error().decode())


def check(status):
    if status < 0:
        raise RuntimeError(lib.fastllm_qwen_image_error().decode())


def array(t):
    return np.ascontiguousarray(t.detach().float().cpu().numpy())


def ap(a):
    return a.ctypes.data_as(ptr)


report = {"precision": "fp32", "device": torch.cuda.get_device_name(0), "components": {}}


def compare(name, reference, actual):
    np.save(out / (name + ".reference.npy"), reference)
    np.save(out / (name + ".native.npy"), actual)
    a, b = reference.astype(np.float64).ravel(), actual.astype(np.float64).ravel()
    diff = a - b
    m = {
        "shape": list(reference.shape),
        "max_abs": float(np.abs(diff).max()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "relative_l2": float(np.linalg.norm(diff) / max(np.linalg.norm(a), 1e-12)),
        "cosine": float(np.dot(a, b) / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12)),
        "finite": bool(np.isfinite(actual).all()),
    }
    report["components"][name] = m
    (out / "report.json").write_text(json.dumps(report, indent=2))
    print(name, json.dumps(m), flush=True)


try:
    if args.stage in ("text", "all"):
        from transformers import AutoTokenizer, Qwen3VLForConditionalGeneration

        tokenizer = AutoTokenizer.from_pretrained(args.model + "/processor", local_files_only=True)
        prompts = [
            "a cat",
            "中文，数字123和标点！",
            "cafe\u0301\n\n hello's 42",
            "  雪山 🐈\t",
            "<|im_start|>user\nhello<|im_end|>\n",
        ]
        for prompt in prompts:
            buf = np.empty(4096, dtype=np.int32)
            n = lib.fastllm_qwen_image_tokenize(
                handle, prompt.encode(), buf.ctypes.data_as(C.POINTER(C.c_int)), len(buf)
            )
            check(n)
            assert buf[:n].tolist() == tokenizer.encode(prompt), (prompt, buf[:n], tokenizer.encode(prompt))
        text = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\nA red apple on a wooden table.<|im_end|>\n<|im_start|>assistant\n"
        ids = tokenizer.encode(text)
        model = (
            Qwen3VLForConditionalGeneration.from_pretrained(
                args.model + "/text_encoder", dtype=dtype, local_files_only=True, attn_implementation="sdpa"
            )
            .eval()
            .cuda()
        )
        norm = model.model.language_model.norm
        hook = norm.register_forward_hook(lambda m, a, o: a[0])
        with torch.inference_mode():
            ref = array(
                model.model.language_model(
                    input_ids=torch.tensor([ids], device="cuda"), use_cache=False
                ).last_hidden_state[0]
            )
        hook.remove()
        del model
        gc.collect()
        torch.cuda.empty_cache()
        tokens = np.asarray(ids, dtype=np.int32)
        actual = np.empty_like(ref)
        check(
            lib.fastllm_qwen_image_text(
                handle, tokens.ctypes.data_as(C.POINTER(C.c_int)), len(ids), ap(actual)
            )
        )
        compare("text", ref, actual)

    if args.stage in ("dit", "all"):
        from diffusers import QwenImage21Transformer2DModel

        model = (
            QwenImage21Transformer2DModel.from_pretrained(
                args.model + "/transformer", torch_dtype=dtype, local_files_only=True
            )
            .eval()
            .cuda()
        )
        h, w, nt = 8, 8, 12
        latent = torch.randn(1, h * w, 64, dtype=dtype, device="cuda")
        features = torch.randn(1, nt, 4096, dtype=dtype, device="cuda")
        mask = torch.cat(
            [
                torch.zeros(1, nt, dtype=torch.bool, device="cuda"),
                torch.ones(1, h * w // 4, dtype=torch.bool, device="cuda"),
            ],
            1,
        )
        with torch.inference_mode():
            ref = array(
                model(
                    latent, features, torch.tensor([0.75], device="cuda", dtype=dtype), [[(1, h, w)]], mask
                ).sample[0, -h * w :]
            )
        l, f = array(latent[0]), array(features[0])
        del model, latent, features
        gc.collect()
        torch.cuda.empty_cache()
        actual = np.empty_like(ref)
        check(lib.fastllm_qwen_image_dit(handle, ap(l), ap(f), nt, h, w, 0.75, ap(actual)))
        compare("dit", ref, actual)

    if args.stage in ("vae", "all"):
        from diffusers import AutoencoderKLQwenImage21

        model = (
            AutoencoderKLQwenImage21.from_pretrained(
                args.model + "/vae", torch_dtype=dtype, local_files_only=True
            )
            .eval()
            .cuda()
        )
        h, w = 4, 4
        latent = torch.randn(1, 64, 1, h, w, device="cuda", dtype=dtype)
        with torch.inference_mode():
            ref = array(model.decode(latent).sample[0, :, 0].permute(1, 2, 0)).reshape(-1, 4)
        l = array(latent[0, :, 0].permute(1, 2, 0)).reshape(-1, 64)
        del model, latent
        gc.collect()
        torch.cuda.empty_cache()
        actual = np.empty_like(ref)
        check(lib.fastllm_qwen_image_vae(handle, ap(l), h, w, ap(actual)))
        actual = np.clip(actual, -1, 1)
        compare("vae", ref, actual)
finally:
    lib.fastllm_qwen_image_destroy(handle)
    (out / "report.json").write_text(json.dumps(report, indent=2))

limit = 3e-4
assert all(v["finite"] and v["relative_l2"] < limit for v in report["components"].values()), report
