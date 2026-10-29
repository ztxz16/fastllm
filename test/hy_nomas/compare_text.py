"""Validate native Qwen tokenizer and HY-Nomas's selected hidden layer."""
import argparse
import ctypes as C
import json
import os
from pathlib import Path
import random
import runpy

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from compare_block import metrics


@torch.no_grad()
def main(args):
    os.environ.setdefault("FASTLLM_NOMAS_PRECISION", "bf16")
    lib = C.CDLL(str(args.library))
    lib.fastllm_nomas_error.restype = C.c_char_p
    lib.fastllm_nomas_text_create.argtypes = [C.c_char_p, C.c_int]
    lib.fastllm_nomas_text_create.restype = C.c_void_p
    lib.fastllm_nomas_text_destroy.argtypes = [C.c_void_p]
    lib.fastllm_nomas_tokenize.argtypes = [C.c_void_p, C.c_char_p, C.c_int, C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_int)]
    lib.fastllm_nomas_encode.argtypes = [C.c_void_p, C.POINTER(C.c_int), C.c_int, C.POINTER(C.c_float)]
    handle = lib.fastllm_nomas_text_create(str(args.weights).encode(), 0)
    if not handle:
        raise RuntimeError(lib.fastllm_nomas_error().decode())
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.weights, local_files_only=True)
        rng = random.Random(42)
        cases = ["Hello world!", "一个宇航员在月球上缓缓行走。", "It's IT'S we're WE'RE I'd 123456789", "\r\n \t  \nabc\n\n", "你好，世界！ 日本語 한국어\u3000🌏👩🏽‍🚀 café é"]
        alphabet = "ABCxyz你好あカ한012٣Ⅳ①_？! 😀é́\r\n\t \u3000\u00a0'"
        cases += ["".join(rng.choices(alphabet, k=rng.randint(1, 200))) for _ in range(500)]
        for text in cases:
            expected = tokenizer.encode(text, add_special_tokens=False)
            buf, crop = (C.c_int * 4096)(), C.c_int()
            count = lib.fastllm_nomas_tokenize(handle, text.encode(), 0, buf, len(buf), C.byref(crop))
            if count < 0:
                raise RuntimeError(lib.fastllm_nomas_error().decode())
            assert list(buf[:count]) == expected, (text, list(buf[:count]), expected)
        report = {"tokenizer_cases": len(cases), "tokenizer_passed": True}
        print(json.dumps(report), flush=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n")
        text = args.prompt
        buf, crop = (C.c_int * 1536)(), C.c_int()
        count = lib.fastllm_nomas_tokenize(handle, text.encode(), 1, buf, len(buf), C.byref(crop))
        if count<0:
            raise RuntimeError(lib.fastllm_nomas_error().decode())
        if args.official:
            template = runpy.run_path(str(args.official / "configs/constants.py"))["PROMPT_TEMPLATE"]["li-dit-encode-video-json"]["template"]
            messages = [{**m, "content": m["content"].format(text)} for m in template]
            reference_tokens = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
            if hasattr(reference_tokens, "keys"):
                reference_tokens = reference_tokens["input_ids"]
            assert list(buf[:count]) == list(reference_tokens), "video prompt template mismatch"
            report["video_template_passed"] = True
        args.report.write_text(json.dumps(report, indent=2)+"\n")
        if args.tokenizer_only:
            return
        torch.set_num_threads(2)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
        model = AutoModel.from_pretrained(args.weights, local_files_only=True, dtype=(torch.float32 if os.environ.get("FASTLLM_NOMAS_PRECISION") == "fp32" else torch.bfloat16), device_map="cuda")
        if hasattr(model, "language_model"):
            model = model.language_model
        ids = torch.tensor(list(buf[:count]), device="cuda")[None]
        expected = model(input_ids=ids, attention_mask=torch.ones_like(ids), output_hidden_states=True, use_cache=False).hidden_states[-3]
        actual = np.empty((count, 3584), np.float32)
        rc = lib.fastllm_nomas_encode(handle, buf, count, actual.ctypes.data_as(C.POINTER(C.c_float)))
        if rc:
            raise RuntimeError(lib.fastllm_nomas_error().decode())
        report["hidden_layer_26"] = metrics(actual[crop.value:], expected[0, crop.value:].float().cpu().numpy())
        fp32 = os.environ.get("FASTLLM_NOMAS_PRECISION") == "fp32"
        report["precision"] = "fp32" if fp32 else "bf16"
        report["hidden_passed"] = report["hidden_layer_26"]["relative_l2"] <= (0.001 if fp32 else 0.02) and report["hidden_layer_26"]["cosine"] >= (0.999999 if fp32 else 0.9998)
        report["tokens"], report["crop"] = count, crop.value
        np.save(args.report.with_suffix(".features.npy"), expected[0, crop.value:].float().cpu().numpy())
        print(json.dumps(report), flush=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n")
        if not report["hidden_passed"]:
            raise SystemExit("Text encoder precision comparison failed")
    finally:
        lib.fastllm_nomas_text_destroy(handle)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--weights", type=Path, required=True)
    p.add_argument("--library", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--official", type=Path)
    p.add_argument("--tokenizer-only", action="store_true")
    p.add_argument("--prompt",default="A red ball rolls slowly across a wooden floor.")
    main(p.parse_args())
