"""Opt-in real-model context regression; run under numactl on the test host.

Writes first-token logits and a JSON report for before/after and reference comparisons.
No model downloads or server processes are started by importing this module.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--python-root", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=-1)
    parser.add_argument("--rope", default="")
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--input-tokens", type=int, default=0)
    parser.add_argument("--kv", default="auto")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--mtp", type=int, default=0)
    parser.add_argument("--graph", type=int, default=0)
    parser.add_argument("--no-logits", action="store_true")
    parser.add_argument("--cuda-embedding", action="store_true")
    parser.add_argument("--image-case", action="store_true")
    parser.add_argument("--expect-capacity-error", action="store_true")
    args = parser.parse_args()
    import numpy as np

    sys.path.insert(0, args.python_root)
    from ftllm import llm, util
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    cli = ["--path", args.model, "--dtype", "auto", "--threads", "8",
           "--tokens", str(args.tokens), "--kv_cache_dtype", args.kv,
           "--mtp", str(args.mtp), "--max_batch", "1"]
    cli += ["--tp", str(args.tp)] if args.tp > 1 else ["--device", "cuda:0"]
    if args.cuda_embedding:
        cli += ["--cuda_embedding"]
    if args.context > 0:
        cli += ["--max_context_length", str(args.context)]
    if args.rope:
        cli += ["--rope_scaling", args.rope]
    os.environ["FASTLLM_CUDA_GRAPH"] = str(args.graph)
    output_logits = not (args.no_logits or args.mtp > 0)
    start = time.monotonic()
    report = {"arguments": vars(args), "cases": {}}
    try:
        model = util.make_normal_llm_model(util.make_normal_parser("context test").parse_args(cli))
    except RuntimeError as error:
        if not args.expect_capacity_error or "Context capacity insufficient" not in str(error):
            raise
        report.update(capacity_error=str(error), seconds=time.monotonic()-start)
        (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
        print("EXPECTED_CAPACITY_ERROR", str(error), flush=True)
        return
    try:
        if args.expect_capacity_error:
            raise AssertionError("Expected capacity rejection, but model started")
        report.update(load_seconds=time.monotonic()-start,
                      context=getattr(model, "context_config", None))
        from ftllm.openai_server.fastllm_model import FastLLmModel
        metadata = FastLLmModel("context-test", model)
        report["metadata"] = {key: getattr(metadata, key) for key in (
            "context_window", "model_context_window", "configured_context_window_limit", "kv_cache_token_limit")}
        if args.context > 0:
            assert metadata.context_window == args.context, report["metadata"]
        text_config = model.config.get("text_config", model.config)
        vocab = text_config.get("vocab_size") or llm.fastllm_lib.get_tokenizer_vocab_size(model.model)
        assert vocab > 0, "missing vocabulary size"
        stops, stop_ids = model.stop_token_ctypes(None)
        buffer = (ctypes.c_float * (vocab * 4))()
        prompts = [("math", "只输出结果：17乘以23等于多少？"),
                   ("text", "请用一句话解释什么是缓存。")]
        if args.image_case:
            prompts = [("image", "Describe the colors in this image briefly.")]
        if args.input_tokens:
            prompts = [("long", "口令是 K314159。记住它。\n")]
        for name, prompt in prompts:
            rendered = model.hf_tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            tokens = model.hf_tokenizer.encode(rendered, add_special_tokens=False)
            if args.input_tokens:
                long_prompt = model.hf_tokenizer.apply_chat_template(
                    [{"role": "user", "content": "用户提供的口令是 K314159。\n<PADDING>\n请直接输出最初的口令："}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False)
                before_text, after_text = long_prompt.split("<PADDING>")
                before = model.hf_tokenizer.encode(before_text, add_special_tokens=False)
                padding = model.hf_tokenizer.encode("这是用于测试上下文长度的无关内容。\n", add_special_tokens=False)
                after = model.hf_tokenizer.encode(after_text, add_special_tokens=False)
                count = args.input_tokens - len(before) - len(after)
                assert count > 0
                tokens = before + (padding * ((count + len(padding) - 1) // len(padding)))[:count] + after
            multimodal = None
            if args.image_case:
                from PIL import Image
                from ftllm.qwen35_multimodal_native import prepare_qwen35_multimodal_inputs, build_qwen35_multimodal_payload
                pixels = np.zeros((192, 192, 3), dtype=np.uint8)
                pixels[:, :, 0] = 220
                pixels[64:128, :, 2] = 220
                prepared = prepare_qwen35_multimodal_inputs(
                    tokenizer=model.hf_tokenizer, model_dir=model.model_path, model_config=model.config,
                    conversation=[{"role": "user", "content": prompt}], images=[Image.fromarray(pixels)],
                    enable_thinking=False, encode_fn=model.encode)
                config, payload = build_qwen35_multimodal_payload(prepared, model.hf_tokenizer, model.config)
                tokens = prepared["input_ids"]
                multimodal = (json.dumps(config).encode(), ctypes.create_string_buffer(payload))
            print("CASE_START", name, len(tokens), flush=True)
            start = time.monotonic()
            launch = (llm.fastllm_lib.launch_response_llm_model_multimodal if multimodal else
                      llm.fastllm_lib.launch_response_llm_model)
            handle = launch(model.model, len(tokens), (ctypes.c_int * len(tokens))(*tokens),
                *(multimodal or ()), 24, 0, False, 1.0, 1, 1.0, 1.0, output_logits, stops, stop_ids)
            output, first, statistics = [], None, None
            while True:
                token = (llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, buffer)
                         if output_logits else llm.fastllm_lib.fetch_response_llm_model(model.model, handle))
                if token < 0:
                    break
                if first is None:
                    first = time.monotonic()
                    if output_logits:
                        logits = np.ctypeslib.as_array(buffer)[:vocab].copy()
                        assert np.isfinite(logits).all(), "nonfinite logits"
                        np.save(out / (name + ".npy"), logits)
                output.append(token)
                statistics = model.get_response_statistics(handle) or statistics
            assert output, f"{name}: empty generation"
            report["cases"][name] = dict(prompt_tokens=len(tokens), token_ids=output,
                text=model.hf_tokenizer.decode(output, skip_special_tokens=True),
                ttft=first-start, seconds=time.monotonic()-start,
                statistics=statistics)
            (out / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
            print("CASE_RESULT", json.dumps(report["cases"][name], ensure_ascii=False), flush=True)
    finally:
        model.release_memory()


if __name__ == "__main__":
    main()
