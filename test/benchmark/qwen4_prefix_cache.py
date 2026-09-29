#!/usr/bin/env python3
"""Real-checkpoint regression for Qwen4 prefix snapshots, with and without MTP.

Run with the built ftllm package on PYTHONPATH. Each mode uses a fresh process;
uncached references and cached requests use the same weights and sampling.
TP runs also cover concurrent startup, graph restore, the default snapshot
interval and enabling MTP after a target-only snapshot.
"""

import argparse
import concurrent.futures
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
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--ngram-device", default="cpu")
    parser.add_argument("--families", default="4095:4096,4096:4096,4097:4096,8193:2048",
                        help="Comma-separated prompt-length:prefill-chunk pairs")
    parser.add_argument("--threads", type=int, default=64)
    parser.add_argument("--output-tokens", type=int, default=64)
    parser.add_argument("--graph", choices=("on", "off"), default="on")
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.output_tokens < 8 or args.cache_gib < 0 or args.tp < 1:
        parser.error("output-tokens must be >= 8, cache-gib >= 0 and tp >= 1")
    try:
        args.families = [tuple(map(int, pair.split(":"))) for pair in args.families.split(",")]
        if not all(len(pair) == 2 and min(pair) > 0 for pair in args.families):
            raise ValueError
        args.mtp = [int(mode) for mode in args.mtp.split(",")]
        if not all(0 <= mode <= 8 for mode in args.mtp):
            raise ValueError
    except ValueError:
        parser.error("families must be positive length:chunk pairs; mtp modes must be 0..8")
    return args


def worker(args):
    os.environ["FASTLLM_QWEN4_ENABLE_MTP"] = str(args.worker)
    os.environ["FASTLLM_PREFIX_CACHE"] = "0"
    os.environ["FASTLLM_QWEN4_PREFIX_CACHE_DEBUG"] = "1"
    # Own the test configuration instead of inheriting a shell's TP setup or
    # snapshot interval. The default 16-page interval is checked separately.
    os.environ["FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES"] = "1"
    os.environ.pop("FASTLLM_TP", None)
    if args.tp > 1:
        os.environ["FASTLLM_TP"] = ",".join(map(str, range(args.tp)))
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
    llm.set_ngram_device(args.ngram_device)
    library = Path(llm.fastllm_lib._name).resolve()
    emit("configuration", mtp=args.worker, graph=args.graph, tp=args.tp,
         ngram_device=args.ngram_device,
         moe_device=args.moe_device, cache_gib=args.cache_gib, model=str(Path(args.model).resolve()),
         library=str(library), library_sha256=hashlib.sha256(library.read_bytes()).hexdigest())
    model = llm.model(args.model, dtype="auto", tokenizer_type="auto")
    model.enable_thinking = False
    model.set_atype("float16")
    model.set_moe_atype("float16")
    model.set_max_batch(1)
    lib = llm.fastllm_lib
    results = []

    def generate(name, tokens, chunk, cached, expected=None, restore=0, count=None):
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

    def request(name, tokens, chunk, cached, *args, **kwargs):
        os.environ["FASTLLM_PREFIX_CACHE"] = "1" if cached else "0"
        model.set_chunked_prefill_size(chunk)
        return generate(name, tokens, chunk, cached, *args, **kwargs)

    def prompt(length):
        instruction = "继续补全当前助手消息中的 Python 代码，只输出后续条目，保持相同格式，尽可能多地生成，不要重复已有内容。"
        # Prefill the assistant's code block: a free choice between starting
        # a fence and continuing raw code is sensitive to small CPU/GPU MoE
        # rounding differences, independently of prefix-cache correctness.
        code = ("```python\ndef square_table():\n    return [\n" +
            "".join(f"        ({i}, {i*i}),\n" for i in range(1, 17)))
        terminal = _encode_prompt(model, model.get_prompt(instruction) + code)
        filler = _encode_prompt(model, f"# Prefix regression family {length}.\ndef identity(value):\n    return value\n\n")
        remaining = length - len(terminal)
        assert remaining > 0 and filler
        return (filler * ((remaining + len(filler) - 1) // len(filler)))[:remaining] + terminal

    warmup = _encode_prompt(model, model.get_prompt("Write a Python list of squares."))
    if args.tp > 1:
        # The second lookup can overlap the first Forward's TP initialization.
        # This short prompt does not record a snapshot, but caching must be
        # enabled so lookup enters the TP path instead of returning early.
        os.environ["FASTLLM_PREFIX_CACHE"] = "1"
        model.set_chunked_prefill_size(4096)
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(generate, "warmup", warmup, 4096, True, count=8)
            time.sleep(0.05)
            second = pool.submit(generate, "warmup_concurrent", warmup, 4096, True, count=8)
            assert first.result() == second.result(), "concurrent startup changed output"
    else:
        request("warmup", warmup, 4096, False, count=8)
    families = []
    # Full-group boundaries plus a multi-chunk prompt. The default final
    # family records 8192 tokens and resumes with an uncached suffix.
    for length, chunk in args.families:
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
        # These families end either at a full chunk or shortly after one;
        # the final short chunk is below the configured snapshot interval.
        snapshot_length = len(base) if len(base) <= chunk else (len(base) // chunk) * chunk
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

    if args.tp > 1:
        # Restore into fresh graph storage, then revisit after a branch has
        # mutated its own recurrent/KV buffers.
        assert llm.set_cuda_graph(True)
        name, base, chunk, suffixes, _ = families[-1]
        extended = base + suffixes["branch"]
        reference = request("graph_reference", extended, chunk, False)
        restore = len(base) if len(base) <= chunk else (len(base) // chunk) * chunk
        request("graph_cached", extended, chunk, True, reference, restore)
        request("graph_cached_again", extended, chunk, True, reference, restore)
        assert llm.set_cuda_graph(args.graph == "on")
        # Also cover the default 16-page interval, without relying only on
        # the short interval used to make the boundary cases inexpensive.
        os.environ["FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES"] = "16"
        default_base = prompt(2049)
        default_reference = request("default_interval_reference", default_base, 2048, False)
        request("default_interval_seed", default_base, 2048, True, default_reference)
        request("default_interval_cached", default_base, 2048, True, default_reference, 2048)
        os.environ["FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES"] = "1"
        if args.worker > 0:
            # A new prompt family avoids the MTP-capable snapshots above.
            length = ((max(len(f[1]) for f in families) // 512) + 1) * 512 + 1
            base = prompt(length)
            reference = request("switch_base_reference", base, 512, False)
            extended = base + reference[:1]
            expected = request("switch_reference", extended, 512, False)
            os.environ["FASTLLM_QWEN4_ENABLE_MTP"] = "0"
            request("switch_target_seed", base, 512, True, reference)
            os.environ["FASTLLM_QWEN4_ENABLE_MTP"] = str(args.worker)
            # The first hit lacks draft state, so the whole TP request must
            # fall back to recomputation and replace it with a complete hit.
            request("switch_recompute", extended, 512, True, expected)
            request("switch_restored", extended, 512, True, expected, length - 1)

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
    summary = []
    for mode in args.mtp:
        result_path = out / f"mtp{mode}.json"
        assert not result_path.exists(), f"Use a fresh output directory: {result_path}"
        command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", str(mode)]
        log_path = out / f"mtp{mode}.log"
        with log_path.open("w") as log:
            run = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                 stdin=subprocess.DEVNULL, timeout=args.timeout)
        assert run.returncode == 0 and result_path.exists(), (
            f"Worker exited with code {run.returncode}; see {log_path}")
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
