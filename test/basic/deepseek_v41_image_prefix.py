#!/usr/bin/env python3
"""Image-aware prefix regression using an existing V4.1 vision fixture/checkpoint.

Example: PYTHONPATH=build-fastllm/tools python test/basic/deepseek_v41_image_prefix.py
    --model /path/to/tiny_tp8 --tp 8
The fixture can be generated with deepseek_v41_vision_reference.py. TP8 requires
o_groups divisible by 8. Real vision weights exercise the production ViT shapes.
"""
import argparse
import ctypes
import json
import os
import sys
import time
import traceback

import numpy as np
from PIL import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--tp", default="8")
    args = parser.parse_args()
    from ftllm import llm
    from ftllm.deepseek_v41_multimodal import (
        build_deepseek_v41_multimodal_payload, expand_image_placeholders,
        get_deepseek_v41_vision_config,
    )
    from ftllm.util import make_normal_llm_model, make_normal_parser

    device = "cuda:0" if int(args.tp) == 1 else "multicuda:" + ",".join(map(str, range(int(args.tp))))
    fargs = make_normal_parser("image prefix regression").parse_args([
        "--path", args.model, "--device", device, "--moe_device", device, "--dtype", "float16",
        "--cache_history", "true", "--prefix_cache", "true", "--max_batch", "1",
        "--chunked_prefill_size", "128", "--threads", "16",
    ])
    fargs.tokens = 65536
    model = make_normal_llm_model(fargs)
    cfg = get_deepseek_v41_vision_config(model.config)
    encode = lambda text: llm.encode_hf_prompt(model.hf_tokenizer, text)
    prefix = encode("A reproducible picture question. " * 8)
    raw = prefix + [cfg["image_token_id"]] + encode(" Please describe the picture in detail. " * 8)
    a, b = Image.new("RGB", (588, 588), "blue"), Image.new("RGB", (588, 588), "yellow")
    vocab = model.config.get("text_config", model.config)["vocab_size"]
    prompt_length = len(expand_image_placeholders(raw, [a], cfg)["input_ids"])
    # An exact repeat leaves one token for forward. Compare against a cold
    # request split at the same position, so this checks cache restoration,
    # independently of numerical differences between full prefill and decode.
    # This fixture has ratio-2 compression and supports this one-token rewind.
    assert (prompt_length - 1) % 2 == 0, prompt_length
    model.set_chunked_prefill_size(prompt_length - 1)

    def run(name, raw_tokens, images, cache=True):
        model.set_save_history(cache)
        started = time.monotonic()
        params = (ctypes.c_int(1), ctypes.c_int(0), ctypes.c_bool(False),
                  ctypes.c_float(1), ctypes.c_int(1), ctypes.c_float(1),
                  ctypes.c_float(1), ctypes.c_bool(True), ctypes.c_int(0), None)
        if images:
            native = expand_image_placeholders(raw_tokens, images, cfg)
            desc, payload = build_deepseek_v41_multimodal_payload(native)
            ids = native["input_ids"]
            data = ctypes.create_string_buffer(payload)
            handle = llm.fastllm_lib.launch_response_llm_model_multimodal(
                model.model, len(ids), (ctypes.c_int * len(ids))(*ids),
                json.dumps(desc).encode(), data, *params)
        else:
            ids = raw_tokens
            handle = llm.fastllm_lib.launch_response_llm_model(
                model.model, len(ids), (ctypes.c_int * len(ids))(*ids), *params)
        buf = (ctypes.c_float * vocab)()
        tokens, logits, statistics = [], [], {}
        while True:
            token = llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, buf)
            if token < 0:
                break
            tokens.append(token)
            logits.append(np.ctypeslib.as_array(buf).copy())
            statistics = model.get_response_statistics(handle) or statistics
        assert tokens and np.isfinite(logits[0]).all(), (name, tokens)
        print(json.dumps(dict(case=name, seconds=time.monotonic()-started, tokens=tokens,
                              prompt_tokens=len(ids), **statistics)), flush=True)
        return logits[0], statistics

    def compare(a, b):
        cosine = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
        print("logit cosine", cosine, "max_abs", float(np.max(np.abs(a-b))), flush=True)
        assert cosine > 0.999, cosine

    cold, stats = run("cold_image_a", raw, [a])
    warm, stats = run("same_image_a", raw, [a])
    assert stats["cached_input_tokens"] == prompt_length - 1, stats
    compare(cold, warm)
    changed, stats = run("changed_image_b", raw, [b])
    assert stats["cached_input_tokens"] <= len(prefix), stats
    reference, _ = run("changed_image_b_uncached", raw, [b], cache=False)
    compare(changed, reference)
    # Use a distinct seed for the continuation test. Its cold chunk ends at
    # exactly the cached prefix, including any raw compression tail. Reusing
    # the exact-repeat seed above would mix different prefill partitions.
    append_seed = encode("Independent append-cache case. " * 8) + raw[len(prefix):]
    seed_length = len(expand_image_placeholders(append_seed, [a], cfg)["input_ids"])
    model.set_chunked_prefill_size(seed_length)
    _, stats = run("append_seed", append_seed, [a])
    assert stats["cached_input_tokens"] == 0, stats
    # The new image and subsequent text span multiple prefill chunks.
    appended = append_seed + encode(" A follow-up image: ") + [cfg["image_token_id"]] + encode(" More details please. " * 80)
    warm, stats = run("append_image", appended, [a, b])
    assert stats["cached_input_tokens"] == seed_length, stats
    reference, _ = run("append_image_uncached", appended, [a, b], cache=False)
    compare(warm, reference)

    # Pure text still uses its own native cache path after image requests.
    text_seed = encode("Text-only cache isolation case. " * 48)
    model.set_chunked_prefill_size(len(text_seed))
    _, stats = run("text_seed", text_seed, [])
    assert stats["cached_input_tokens"] == 0, stats
    text_followup = text_seed + encode(" Continue this text-only conversation. " * 16)
    warm, stats = run("text_followup", text_followup, [])
    assert stats["cached_input_tokens"] == len(text_seed), stats
    reference, _ = run("text_followup_uncached", text_followup, [], cache=False)
    compare(warm, reference)
    print("RESULT: PASS", flush=True)


if __name__ == "__main__":
    # The native worker pool is process-persistent; avoid Python teardown races.
    # Do the same on failure so an assertion does not leave a hung GPU process.
    status = 0
    try:
        main()
    except BaseException:
        traceback.print_exc()
        status = 1
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(status)
