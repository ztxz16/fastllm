#!/usr/bin/env python3
"""Real-checkpoint regression for Qwen4 prefix snapshots, with and without MTP.

Run with the built ftllm package on PYTHONPATH. Each mode uses a fresh process;
uncached references and cached requests use the same weights and sampling.
"""

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import time


def emit(event, **values):
    print(json.dumps({"event": event, **values}), flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mtp", default="0,3")
    parser.add_argument("--moe-device", default="numa")
    parser.add_argument("--cache-gib", type=int, default=50)
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--graph", choices=("on", "off"), default="on")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_tokens < 8 or args.cache_gib < 0:
        parser.error("output-tokens must be >= 8 and cache-gib must be >= 0")
    return args


def worker(args):
    os.environ["FASTLLM_QWEN4_ENABLE_MTP"] = str(args.worker)
    os.environ["FASTLLM_PREFIX_CACHE"] = "0"
    os.environ["FASTLLM_QWEN4_PREFIX_CACHE_DEBUG"] = "1"
    from ftllm import llm
    from ftllm.benchmark import _encode_prompt

    assert llm.set_cuda_graph(args.graph == "on")
    llm.set_moe_cuda_cache(args.cache_gib << 30)
    llm.set_cpu_threads(args.threads)
    llm.set_cuda_slab(225)
    llm.set_gpu_mem_ratio(0.98)
    llm.set_cuda_shared_expert(True)
    llm.set_max_tokens(16384)
    llm.set_device_map("cuda")
    llm.set_device_map(args.moe_device, is_moe=True)
    library = Path(llm.fastllm_lib._name).resolve()
    emit("configuration", mtp=args.worker, graph=args.graph,
         cache_gib=args.cache_gib, model=str(Path(args.model).resolve()),
         library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    model = llm.model(args.model, dtype="auto", tokenizer_type="auto")
    model.enable_thinking = False
    model.set_atype("float16")
    model.set_moe_atype("float16")
    model.set_max_batch(1)
    lib = llm.fastllm_lib
    results = []

    def request(name, tokens, chunk, cached, expected=None, restore=0, count=None):
        os.environ["FASTLLM_PREFIX_CACHE"] = "1" if cached else "0"
        model.set_chunked_prefill_size(chunk)
        count = args.output_tokens if count is None else count
        emit("request_start", name=name, cached=cached, expected_restore=restore)
        stop_len, stop_list = model.stop_token_ctypes(None)
        started = time.perf_counter()
        handle = lib.launch_response_llm_model(
            model.model, len(tokens), (ctypes.c_int * len(tokens))(*tokens),
            ctypes.c_int(count), ctypes.c_int(count), ctypes.c_bool(False),
            ctypes.c_float(1.0), ctypes.c_int(1), ctypes.c_float(1.0),
            ctypes.c_float(1.0), ctypes.c_bool(False), stop_len, stop_list)
        generated = []
        first_token = None
        while True:
            if time.perf_counter() - started > args.timeout:
                raise TimeoutError(name)
            if not lib.can_fetch_response_llm_model(model.model, handle):
                time.sleep(0.0002)
                continue
            token = lib.fetch_response_llm_model(model.model, handle)
            if token < 0:
                assert token == -1, (name, "unexpected finish", token)
                break
            if first_token is None:
                first_token = time.perf_counter() - started
            generated.append(int(token))
        assert len(generated) == count, (name, len(generated), count)
        row = {"name": name, "cached": cached, "input_tokens": len(tokens),
               "chunk": chunk, "expected_restore": restore,
               "prompt_sha256": hashlib.sha256(json.dumps(tokens).encode()).hexdigest(),
               "generated": generated, "first_token_seconds": first_token,
               "seconds": time.perf_counter() - started}
        results.append(row)
        emit("request_done", **row)
        if expected is not None:
            assert generated == expected, (name, "cached output differs from uncached reference")
        return generated

    def prompt(length):
        instruction = (
            "只继续补全下面的 Python 代码，不要解释，不要使用省略号。保持完全相同的格式，继续生成尽可能多的条目：\n"
            "def square_table():\n    return [\n" +
            "".join(f"        ({i}, {i*i}),\n" for i in range(1, 17)))
        terminal = _encode_prompt(model, model.get_prompt(instruction))
        filler = _encode_prompt(model, f"# Prefix regression family {length}.\ndef identity(value):\n    return value\n\n")
        remaining = length - len(terminal)
        assert remaining > 0 and filler
        return (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining] + terminal

    request("warmup", _encode_prompt(model, model.get_prompt("Write a Python list of squares.")),
            4096, False, count=8)
    families = []
    # Full-group boundaries plus a multi-chunk prompt. The latter records an
    # 8192-token intermediate snapshot and resumes with an uncached suffix.
    for length, chunk in ((4095, 4096), (4096, 4096), (4097, 4096), (8193, 2048)):
        base = prompt(length)
        name = f"n{length}_chunk{chunk}"
        reference = request(name + "_reference", base, chunk, False)
        suffixes = {"one": reference[:1], "seven": reference[:7]}
        branch = _encode_prompt(model, "        (100, 10000),\n")
        assert branch != reference[:len(branch)]
        suffixes["branch"] = branch
        expected = {"base": reference}
        for tag, suffix in suffixes.items():
            expected[tag] = request(name + "_" + tag + "_reference", base + suffix, chunk, False)
        families.append((name, base, chunk, suffixes, expected))

    for name, base, chunk, suffixes, expected in families:
        # 4097 and 8193 retain a chunk-boundary snapshot one token shorter
        # than the full prompt. Other families save the complete prompt.
        snapshot_length = len(base) if len(base) < 4097 else len(base) - 1
        request(name + "_seed", base, chunk, True, expected["base"])
        for repeat in range(2):
            request(name + f"_repeat{repeat}", base, chunk, True, expected["base"],
                    snapshot_length if snapshot_length < len(base) else 0)
        for tag, suffix in suffixes.items():
            request(name + "_" + tag, base + suffix, chunk, True,
                    expected[tag], snapshot_length)
        # Revisit an earlier branch after other continuations have modified
        # their request state, to detect accidental writes into shared snapshots.
        request(name + "_one_again", base + suffixes["one"], chunk, True,
                expected["one"], snapshot_length)

    model.release_memory()
    out = Path(args.output_dir) / f"mtp{args.worker}.json"
    out.write_text(json.dumps({"mtp": args.worker, "requests": results}, indent=2) + "\n")
    emit("worker_done", mtp=args.worker, requests=len(results))


def main():
    args = parse_args()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if args.worker is not None:
        worker(args)
        return
    modes = [int(value) for value in args.mtp.split(",")]
    assert modes and all(0 <= mode <= 8 for mode in modes)
    summary = []
    for mode in modes:
        result_path = out / f"mtp{mode}.json"
        assert not result_path.exists(), f"Use a fresh output directory: {result_path}"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", str(mode)]
        log_path = out / f"mtp{mode}.log"
        with log_path.open("w") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, timeout=args.timeout)
        assert run.returncode == 0 and result_path.exists(), f"Worker failed; see {log_path}"
        result = json.loads(result_path.read_text())
        current = None
        restores = {}
        for line in log_path.read_text(errors="replace").splitlines():
            if line.startswith('{"event": "request_start"'):
                current = json.loads(line)["name"]
            match = re.search(r"\[qwen4-prefix-cache\] restore tokens=(\d+)", line)
            if match:
                restores.setdefault(current, []).append(int(match.group(1)))
        for row in result["requests"]:
            observed = restores.get(row["name"], [])
            assert observed == ([row["expected_restore"]] if row["expected_restore"] else []), (
                row["name"], "unexpected prefix restores", observed, row["expected_restore"])
        record = {"mtp": mode, "requests": len(result["requests"]),
                  "restored_requests": len(restores), "tokens_match": True}
        summary.append(record)
        emit("mode_pass", **record)
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
