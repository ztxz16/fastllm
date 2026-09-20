#!/usr/bin/env python3
"""Native logits correctness checks for >8K multimodal prefix reuse.

Uses the same model/TP4/FP8-KV setup as the server test. Requesting full logits
intentionally bypasses speculative token selection. The separate HTTP test
validates actual DFlash decoding with the production settings.
"""
import argparse
import base64
import copy
import ctypes
import hashlib
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import time
import traceback

import numpy as np
from PIL import Image
from test_qwen35_prefix_cache_e2e import MODEL, DRAFT, ROOT, LIBRARY, scenarios


PAGE_SIZE = 128
CHUNK_SIZE = 2048
ALIGNED_PROMPT_TOKENS = 5 * CHUNK_SIZE
PAGE_TAIL_PROMPT_TOKENS = 4 * CHUNK_SIZE + 2 * PAGE_SIZE
PAGE_TAIL_FOLLOWUP_TOKENS = 20
OUTPUT_TOKENS = 4
HOT_TOLERANCE = {"rtol": 1e-5, "atol": 1e-5}
DISABLED_TOLERANCE = {"rtol": 0.02, "atol": 0.05}


def output_vocab_size(model_dir, config):
    """Check the native output width before calling the unsized C fetch API."""
    model_dir = Path(model_dir)
    index = json.loads((model_dir / "model.safetensors.index.json").read_text())
    text_config = config.get("text_config", config)
    name = "lm_head.weight"
    if name not in index["weight_map"] and text_config.get("tie_word_embeddings"):
        name = "model.language_model.embed_tokens.weight"
    with (model_dir / index["weight_map"][name]).open("rb") as source:
        header_length = struct.unpack("<Q", source.read(8))[0]
        header = json.loads(source.read(header_length))
    shape = header[name]["shape"]
    if len(shape) != 2 or shape[0] != int(text_config["vocab_size"]):
        raise RuntimeError(f"Unexpected logits width: {name} shape={shape}, config={text_config['vocab_size']}")
    return shape[0]


def image_spans(ids, image_token_id):
    spans = []
    for index, token in enumerate(ids):
        if token != image_token_id:
            continue
        if spans and spans[-1][1] == index:
            spans[-1][1] = index + 1
        else:
            spans.append([index, index + 1])
    return spans


def comparison_details(actual, expected, tolerance):
    delta = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    worst = np.unravel_index(int(np.argmax(delta)), delta.shape)
    return {
        "max_abs": float(delta[worst]),
        "worst_step": int(worst[0]),
        "worst_token": int(worst[1]),
        "actual": float(actual[worst]),
        "expected": float(expected[worst]),
        "mismatched_values": int(np.count_nonzero(~np.isclose(actual, expected, equal_nan=False, **tolerance))),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--case", choices=("text", "first", "middle", "last", "inside", "aligned", "page-tail"))
    parser.add_argument("--request-timeout", type=float, default=900)
    args = parser.parse_args()
    output = ROOT / "build-acceptance/20260920-multimodal-prefix-fix"
    output.mkdir(parents=True, exist_ok=True)
    logfile, resultfile = output / (args.label + ".log"), output / (args.label + ".json")
    if logfile.exists() or resultfile.exists():
        raise RuntimeError("Choose a new label; preserving prior evidence")
    result = {"status": "failed", "stage": "initializing", "cases": [],
              "hot_vs_cold_tolerance": HOT_TOLERANCE,
              "vs_unchunked_tail_tolerance": DISABLED_TOLERANCE,
              "page_size": PAGE_SIZE, "chunk_size": CHUNK_SIZE,
              "aligned_prompt_tokens": ALIGNED_PROMPT_TOKENS,
              "page_tail_prompt_tokens": PAGE_TAIL_PROMPT_TOKENS,
              "page_tail_followup_tokens": PAGE_TAIL_FOLLOWUP_TOKENS,
              "output_tokens": OUTPUT_TOKENS}

    def save():
        resultfile.write_text(json.dumps(result, ensure_ascii=False, indent=2))

    try:
        gpu_status = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"], text=True)
        result["gpu_before"] = gpu_status
        for line in gpu_status.splitlines():
            index, used = map(int, line.split(","))
            if index < 4 and used > 64:
                raise RuntimeError("GPU 0-3 are occupied")
        print("Native logits test log:", logfile, flush=True)
        with logfile.open("wb") as sink:
            os.dup2(sink.fileno(), 1)
            os.dup2(sink.fileno(), 2)
        os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2,3", FASTLLM_ACTIVATE_NUMA="ON",
            FASTLLM_NUMA_THREADS="27", FASTLLM_CUDA_DFLASH_TP_BACKBONE="force",
            FASTLLM_CUDA_SERVING_WARMUP="1", FASTLLM_MULTIMODAL_PREFIX_CACHE="1")
        from ftllm import llm, util
        from ftllm.qwen35_multimodal_native import prepare_qwen35_multimodal_inputs, build_qwen35_multimodal_payload
        result.update(stage="model_loading", library_sha256=hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest())
        save()
        command = [MODEL, "--tp", "0,1,2,3", "--gpu_mem_ratio", "0.95", "--chunked_prefill_size", "2048",
            "--kv_cache_dtype", "fp8_e4m3", "--enable_thinking", "true", "--speculative_algorithm", "dflash",
            "--speculative_draft_model_path", DRAFT, "--draft_tokens", "6", "--multimodal"]
        result["model_arguments"] = command
        model = util.make_normal_llm_model(util.make_normal_parser("logits regression").parse_args(command))
        model.set_verbose(True)
        vocab = output_vocab_size(MODEL, model.config)
        result["vocab_size"] = vocab

        def prepare(messages, padded_length=None, extra_tail_tokens=0):
            messages = copy.deepcopy(messages)
            images = []
            for message in messages:
                if not isinstance(message["content"], list):
                    continue
                for item in message["content"]:
                    if item["type"] == "image_url":
                        with Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"].split(",", 1)[1]))) as image:
                            images.append(image.convert("RGB"))
                        item.clear()
                        item["type"] = "image"
            if images:
                native = prepare_qwen35_multimodal_inputs(model.hf_tokenizer, MODEL, model.config,
                    messages, images=images, enable_thinking=True, encode_fn=model.encode)
                ids = list(native["input_ids"])
                config, payload = build_qwen35_multimodal_payload(native, model.hf_tokenizer, model_config=model.config)
            else:
                ids = list(llm.encode_hf_prompt(model.hf_tokenizer,
                    model._render_qwen35_text_prompt(messages, enable_thinking=True)))
                config, payload = None, None
            if padded_length is not None:
                assert len(ids) <= padded_length, ("padded fixture exceeds target", len(ids), padded_length)
                filler = llm.encode_hf_prompt(model.hf_tokenizer, " a")
                assert len(filler) == 1, ("tail filler must encode to one text token", filler)
                ids += filler * (padded_length - len(ids) + extra_tail_tokens)
            assert len(ids) > 8192, ("prompt must exceed 8192 tokens", len(ids))
            spans = image_spans(ids, config["image_token_id"]) if config else []
            return ids, config, payload, spans

        def run(prepared, cache, record, phase):
            os.environ["FASTLLM_MULTIMODAL_PREFIX_CACHE"] = "1" if cache else "0"
            ids, config, payload, _ = prepared
            result["stage"] = record["name"] + "/" + phase
            record[phase] = {"status": "running"}
            save()
            print(result["stage"], flush=True)
            token_buffer = (ctypes.c_int * len(ids))(*ids)
            generation = (OUTPUT_TOKENS, OUTPUT_TOKENS, False, 1.0, 1, 1.0, 1.0, True, 0, None)
            if config is None:
                handle = llm.fastllm_lib.launch_response_llm_model(model.model, len(ids), token_buffer, *generation)
            else:
                buffer = ctypes.create_string_buffer(payload)
                handle = llm.fastllm_lib.launch_response_llm_model_multimodal(model.model, len(ids),
                    token_buffer, json.dumps(config).encode(), buffer, *generation)
            if handle < 0:
                raise RuntimeError(f"{result['stage']}: launch returned {handle}")
            logits, tokens, stats = [], [], None
            # The C ABI does not accept a buffer size or return a logits count.
            # Verify width against lm_head above; NaNs detect an empty/truncated
            # result and the guard detects a small unexpected padded output.
            values = (ctypes.c_float * (vocab + 64))()
            array = np.ctypeslib.as_array(values)
            start = time.monotonic()
            finished = False
            try:
                while True:
                    while not llm.fastllm_lib.can_fetch_response_llm_model(model.model, handle):
                        if time.monotonic() - start > args.request_timeout:
                            raise TimeoutError(f"{result['stage']}: request exceeded {args.request_timeout}s")
                        time.sleep(0.01)
                    stats = model.get_response_statistics(handle) or stats
                    array.fill(np.nan)
                    token = llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, values)
                    if token == -1:
                        finished = True
                        break
                    if token < 0 or token >= vocab:
                        raise RuntimeError(f"{result['stage']}: invalid fetched token {token}")
                    if len(tokens) >= OUTPUT_TOKENS:
                        raise RuntimeError(f"{result['stage']}: exceeded output limit {OUTPUT_TOKENS}")
                    if not np.isnan(array[vocab:]).all():
                        raise RuntimeError(f"{result['stage']}: native logits exceeded verified width {vocab}")
                    if not np.isfinite(array[:vocab]).all():
                        raise RuntimeError(f"{result['stage']}: missing, truncated, or nonfinite native logits")
                    tokens.append(token)
                    logits.append(array[:vocab].copy())
                    stats = model.get_response_statistics(handle) or stats
            finally:
                if not finished:
                    model.abort_handle(handle)
                record[phase].update(stats=stats, tokens=tokens, seconds=time.monotonic() - start)
                save()
            assert len(logits) == OUTPUT_TOKENS, (result["stage"], "unexpected output count", tokens)
            assert stats is not None, (result["stage"], "missing native response statistics")
            assert stats["cached_input_tokens"] + stats["missed_input_tokens"] == len(ids), (result["stage"], stats, len(ids))
            record[phase]["status"] = "ok"
            save()
            return np.stack(logits), tokens, stats

        def compare(actual, expected, actual_ids, expected_ids, tolerance, record, name):
            record[name] = comparison_details(actual, expected, tolerance)
            record[name]["tokens_equal"] = actual_ids == expected_ids
            save()
            if record[name]["mismatched_values"] or actual_ids != expected_ids:
                artifact = output / (args.label + "-" + record["name"].replace("/", "-") + "-" + name + ".npz")
                np.savez_compressed(artifact, actual=actual, expected=expected)
                record[name]["logits_artifact"] = str(artifact)
                save()
            try:
                np.testing.assert_allclose(actual, expected, equal_nan=False, err_msg=record["name"] + ": " + name, **tolerance)
                assert actual_ids == expected_ids, (record["name"], name, "generated tokens differ", actual_ids, expected_ids)
            except AssertionError as error:
                record[name]["error"] = str(error)
                save()
                return False
            return True

        cases = scenarios()
        aligned = copy.deepcopy(next(messages for name, messages, _ in cases if name == "inside/cold"))
        aligned[0]["content"] += "完整页对齐场景。"
        cases.append(("aligned/cold", aligned, False))
        page_tail = copy.deepcopy(next(messages for name, messages, _ in cases if name == "inside/cold"))
        page_tail[0]["content"] += "整页非整块尾部场景。"
        cases.extend((("page-tail/cold", copy.deepcopy(page_tail), False),
                      ("page-tail/followup", page_tail, True)))
        page_tail_prefix = None
        for name, messages, should_hit in cases:
            if name.endswith("/repeat") or (args.case and name.split("/", 1)[0] != args.case):
                continue  # Every measured case gets an immediate repeat below.
            padded_length = None
            extra_tail_tokens = 0
            if name.startswith("aligned/"):
                padded_length = ALIGNED_PROMPT_TOKENS
            elif name.startswith("page-tail/"):
                padded_length = PAGE_TAIL_PROMPT_TOKENS
                extra_tail_tokens = PAGE_TAIL_FOLLOWUP_TOKENS if name.endswith("/followup") else 0
            prepared = prepare(messages, padded_length, extra_tail_tokens)
            prompt, _, _, spans = prepared
            record = {"name": name, "status": "running", "prompt_tokens": len(prompt), "image_spans": spans}
            if name == "page-tail/cold":
                assert len(prompt) == PAGE_TAIL_PROMPT_TOKENS and len(prompt) % PAGE_SIZE == 0 and len(prompt) % CHUNK_SIZE != 0
                page_tail_prefix = list(prompt)
            elif name == "page-tail/followup":
                assert page_tail_prefix is not None and prompt[:-PAGE_TAIL_FOLLOWUP_TOKENS] == page_tail_prefix, (name, "followup must preserve the exact native prefix")
                assert len(prompt) == PAGE_TAIL_PROMPT_TOKENS + PAGE_TAIL_FOLLOWUP_TOKENS
                record["preserved_prefix_tokens"] = len(page_tail_prefix)
            result["cases"].append(record)
            initial, initial_ids, initial_stats = run(prepared, True, record, "initial")
            if name.endswith("/cold"):
                assert initial_stats["cached_input_tokens"] == 0, (name, "cold case reused state", initial_stats)
            if should_hit or name == "text/followup":
                assert initial_stats["cached_input_tokens"] > 0, (name, "expected prefix reuse", initial_stats)
            if name == "page-tail/followup":
                assert initial_stats["cached_input_tokens"] == 8192, (name, "must not reuse the 8448-token partial chunk recorded after a hot request", initial_stats)
            if name.endswith("/changed"):
                assert initial_stats["cached_input_tokens"] <= spans[0][0], (name, "reused changed image", initial_stats, spans)
            hot, hot_ids, hot_stats = run(prepared, True, record, "hot")
            cached = hot_stats["cached_input_tokens"]
            assert cached > 0 and cached % PAGE_SIZE == 0, (name, "expected complete cached pages", hot_stats)
            if spans:
                assert cached > spans[0][0], (name, "hit only text before the first image", hot_stats, spans)
            if name == "inside/cold":
                assert any(begin < cached < end for begin, end in spans), (name, "cache boundary must cut through image", cached, spans)
            if name.startswith("aligned/"):
                assert len(prompt) == ALIGNED_PROMPT_TOKENS and len(prompt) % CHUNK_SIZE == 0, (name, "prompt must fill complete prefill chunks", len(prompt))
                assert any(begin < 8192 < end for begin, end in spans), (name, "image must cross the 8192-token boundary", spans)
            if name.startswith("page-tail/"):
                assert cached == 8192, (name, "cache must stop at the original complete prefill chunk", hot_stats)
                assert any(begin < cached < end for begin, end in spans), (name, "cached chunk must end inside the image", cached, spans)
            comparisons_ok = compare(hot, initial, hot_ids, initial_ids, HOT_TOLERANCE, record, "hot_vs_initial")
            if spans:
                # This switch disables only multimodal prefix reuse. Text
                # cold/repeat/followup above exercise the unchanged text cache.
                disabled, disabled_ids, disabled_stats = run(prepared, False, record, "disabled")
                assert disabled_stats["cached_input_tokens"] == 0, (name, "disabled cache hit", disabled_stats)
                disabled_ok = compare(hot, disabled, hot_ids, disabled_ids, DISABLED_TOLERANCE, record, "hot_vs_disabled")
                comparisons_ok = comparisons_ok and disabled_ok
            assert comparisons_ok, (name, "logits or tokens differ; see saved comparison metrics and .npz artifacts")
            record["status"] = "ok"
            save()
            print(json.dumps(record, ensure_ascii=False), flush=True)
        assert result["cases"], "No cases selected"
        result.update(status="ok", stage="complete")
    except BaseException as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
