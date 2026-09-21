#!/usr/bin/env python3
"""Cross-process SSD logits regression, using the production model settings.

Run write and restore as separate bounded processes with the same --label.
This validates same-dtype restoration. FP16-to-FP8 is a separate numerical
contract and must not be compared to a native-FP8 cold run as if lossless.
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
import signal
import time

import numpy as np
from PIL import Image
from test_qwen35_prefix_cache_e2e import MODEL, DRAFT, ROOT, LIBRARY, scenarios
from test_qwen35_prefix_logits import output_vocab_size
from test_qwen35_ssd_prefix_cache_e2e import assert_resources_idle


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--phase", choices=("write", "restore"), required=True)
    parser.add_argument("--case", choices=("text", "first", "middle", "last"), default="first")
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--prompt-tokens", default="0",
                        help="Comma-separated exact synthetic prompt lengths; 0 keeps the original fixture")
    args = parser.parse_args()
    lengths = [int(value) for value in args.prompt_tokens.split(",")]
    if not lengths or len(lengths) != len(set(lengths)) or any(value < 0 for value in lengths):
        raise ValueError("Provide distinct nonnegative prompt lengths")
    if not args.label.replace("-", "").replace("_", "").isalnum():
        raise ValueError("Use a filesystem-safe label")
    assert_resources_idle()
    signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(TimeoutError("SSD logits deadline")))
    signal.alarm(args.timeout)
    evidence = ROOT / "build-acceptance/20260920-ssd-prefix-cache" / ("logits-" + args.label)
    evidence.mkdir(parents=True, exist_ok=True)
    resultfile = evidence / (args.phase + ".json")
    if resultfile.exists():
        raise RuntimeError("Preserving previous evidence; choose a new label")
    cache = Path("/mnt/nvme/LLM/fastllm-kv-cache1") / ("logits-" + args.label)
    if args.phase == "write" and cache.exists():
        raise RuntimeError("Write phase requires a new cache directory")
    os.environ.update(CUDA_VISIBLE_DEVICES="0,1,2,3", FASTLLM_ACTIVATE_NUMA="ON",
        FASTLLM_NUMA_THREADS="27", FASTLLM_CUDA_DFLASH_TP_BACKBONE="force",
        FASTLLM_CUDA_SERVING_WARMUP="1", FASTLLM_MULTIMODAL_PREFIX_CACHE="1",
        FASTLLM_PREFIX_CACHE_DIR=str(cache), FASTLLM_PREFIX_CACHE_DISK_BYTES=str(256 << 30),
        FASTLLM_PREFIX_CACHE_RESTORE_POLICY="always")
    from ftllm import llm, util
    from ftllm.qwen35_multimodal_native import prepare_qwen35_multimodal_inputs, build_qwen35_multimodal_payload
    command = [MODEL,"--tp","0,1,2,3","--gpu_mem_ratio","0.95","--chunked_prefill_size","2048",
        "--kv_cache_dtype","fp8_e4m3","--enable_thinking","true","--speculative_algorithm","dflash",
        "--speculative_draft_model_path",DRAFT,"--draft_tokens","6","--multimodal"]
    result = {"status":"running","phase":args.phase,"case":args.case,"command":command,
        "library_sha256":hashlib.sha256(Path(LIBRARY).read_bytes()).hexdigest(),"scope":"same-dtype SSD full logits", "cases":[]}
    resultfile.write_text(json.dumps(result,indent=2))
    model = util.make_normal_llm_model(util.make_normal_parser("SSD logits").parse_args(command))
    model.set_verbose(True)
    vocab = output_vocab_size(MODEL,model.config)
    def run_one(target_tokens):
        messages = copy.deepcopy(next(messages for name,messages,_ in scenarios() if name == args.case+"/cold"))
        messages[0]["content"] = "SSD logits="+args.label+("-"+str(target_tokens) if target_tokens else "")+"。"+messages[0]["content"]
        images = []
        for message in messages:
            if not isinstance(message["content"],list): continue
            for item in message["content"]:
                if item["type"] == "image_url":
                    with Image.open(io.BytesIO(base64.b64decode(item["image_url"]["url"].split(",",1)[1]))) as image:
                        images.append(image.convert("RGB"))
                    item.clear(); item["type"] = "image"
        config, payload = None, None
        if images:
            native = prepare_qwen35_multimodal_inputs(model.hf_tokenizer,MODEL,model.config,messages,
                images=images,enable_thinking=True,encode_fn=model.encode)
            ids = list(native["input_ids"])
            config,payload = build_qwen35_multimodal_payload(native,model.hf_tokenizer,model_config=model.config)
        else:
            ids = list(llm.encode_hf_prompt(model.hf_tokenizer,model._render_qwen35_text_prompt(messages,enable_thinking=True)))
        if target_tokens:
            assert target_tokens >= len(ids), (target_tokens, len(ids))
            filler = list(llm.encode_hf_prompt(model.hf_tokenizer, " a"))
            assert len(filler) == 1
            # A synthetic long-context numerical fixture. Preserve the closing
            # template tokens so the insertion remains before generation starts.
            ids[-16:-16] = filler * (target_tokens - len(ids))
        assert len(ids) > 8192
        record = {"status":"running", "target_tokens":target_tokens}
        result["cases"].append(record)
        suffix = "-"+str(target_tokens) if target_tokens else ""
        print("SSD logits:",args.phase,"tokens=",len(ids),flush=True)
        token_buffer = (ctypes.c_int*len(ids))(*ids)
        generation = (4,4,False,1.0,1,1.0,1.0,True,0,None)
        if payload is None:
            handle = llm.fastllm_lib.launch_response_llm_model(model.model,len(ids),token_buffer,*generation)
        else:
            buffer = ctypes.create_string_buffer(payload)
            handle = llm.fastllm_lib.launch_response_llm_model_multimodal(model.model,len(ids),token_buffer,
                json.dumps(config).encode(),buffer,*generation)
        assert handle >= 0
        values = (ctypes.c_float*(vocab+64))()
        array = np.ctypeslib.as_array(values)
        tokens, logits, stats = [], [], None
        try:
            while True:
                while not llm.fastllm_lib.can_fetch_response_llm_model(model.model,handle): time.sleep(0.01)
                stats = model.get_response_statistics(handle) or stats
                array.fill(np.nan)
                token = llm.fastllm_lib.fetch_response_logits_llm_model(model.model,handle,values)
                if token == -1: break
                assert 0 <= token < vocab and len(tokens) < 4
                assert np.isnan(array[vocab:]).all() and np.isfinite(array[:vocab]).all()
                tokens.append(token); logits.append(array[:vocab].copy())
            assert len(tokens) == 4 and stats
            assert model.wait_persistent_prefix_cache(timeout=120)
            current = np.stack(logits)
            np.savez(evidence/(args.phase+suffix+".npz"),tokens=tokens,logits=current)
            record.update(tokens=tokens,usage=stats,prompt_tokens=len(ids),ssd=model.get_persistent_prefix_cache_statistics())
            if args.phase == "write":
                assert stats["cached_input_tokens"] == 0
                assert record["ssd"]["committed"] > 0
            else:
                reference = np.load(evidence/("write"+suffix+".npz"))
                assert stats["cached_input_tokens"] > 0 and record["ssd"]["restored_tokens"] > 0
                record["max_abs"] = float(np.abs(current-reference["logits"]).max())
                assert np.array_equal(tokens,reference["tokens"])
                np.testing.assert_allclose(current,reference["logits"],rtol=1e-5,atol=1e-5)
            record["status"] = "ok"
        finally:
            resultfile.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    try:
        for length in lengths:
            run_one(length)
        result["status"] = "ok"
    except BaseException as error:
        result["status"] = "failed"
        result["error"] = repr(error)
        raise
    finally:
        resultfile.write_text(json.dumps(result,ensure_ascii=False,indent=2))
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
    signal.alarm(0)


if __name__ == "__main__": main()
