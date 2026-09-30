"""Real-model regression for TP DFlash decode at the KV pool boundary.

Run with a Qwen3.5/Qwen3.8 model, --tp 0,1, --draft PATH, --tokens 4096,
--max_context_length 4096, --max_batch 1, and --prefix_cache true.
Compare with --tokens 4224 to give the same requests one spare KV page.
Use --sampling to exercise rejection sampling instead of greedy acceptance.
The parent captures native logs so repeated prefill fails without a timing
threshold. Both parent and worker inherit the caller's CPU/NUMA affinity.
"""

import ctypes
import json
import re
import subprocess
import sys
import time


def worker():
    from ftllm import llm
    from ftllm.util import make_normal_llm_model, make_normal_parser

    parser = make_normal_parser("TP DFlash decode near KV capacity")
    parser.add_argument("--sampling", action="store_true")
    args = parser.parse_args()
    if args.max_batch != 1 or args.max_context_length < 2048:
        parser.error("requires --max_batch 1 and --max_context_length >= 2048")
    if args.tokens < args.max_context_length:
        parser.error("--tokens must cover --max_context_length")
    if not args.speculative_draft_model_path or args.mtp:
        parser.error("requires a DFlash --draft model and --mtp 0")
    model = make_normal_llm_model(args)
    # The parent also verifies the actual native TP and KV pool logs, so a
    # silently disabled draft or a larger-than-requested pool cannot pass.
    print("KV_CAPACITY_CONFIG", json.dumps(dict(
        tokens=args.tokens, max_context_length=args.max_context_length,
        sampling=args.sampling)), flush=True)
    model.set_verbose(True)
    tokenizer = model.hf_tokenizer
    encode = lambda text: llm.encode_hf_prompt(tokenizer, text)
    prefix = encode("<|im_start|>user\n以下是参考资料：\n")
    unit = encode("The research notes describe language models, their training data, "
                  "evaluation methods, and practical applications.\n")
    suffix = encode("\n请写一篇不少于1500字的文章，解释大语言模型的训练与推理过程。"
                    "直接开始正文。<|im_end|>\n<|im_start|>assistant\n"
                    "<think>\n\n</think>\n\n")
    input_count = args.max_context_length - 17
    remaining = input_count - len(prefix) - len(suffix)
    assert remaining > 0 and unit
    inputs = prefix + (unit * ((remaining + len(unit) - 1) // len(unit)))[:remaining] + suffix
    input_buffer = (ctypes.c_int * len(inputs))(*inputs)

    def run(name, maximum):
        print("KV_CAPACITY_BEGIN", name, flush=True)
        started = time.monotonic()
        handle = llm.fastllm_lib.launch_response_llm_model(
            model.model, len(inputs), input_buffer, maximum, maximum,
            args.sampling, 1.0, 20 if args.sampling else 1,
            .95 if args.sampling else 1.0, 1.0, False, 0, None)
        assert handle >= 0, "request launch failed"
        tokens, times = [], []
        first_stats = last_stats = None
        finished = False
        try:
            while True:
                if time.monotonic() - started > 180:
                    raise TimeoutError("KV capacity request timed out: " + name)
                if not llm.fastllm_lib.can_fetch_response_llm_model(model.model, handle):
                    time.sleep(.0005)
                    continue
                stats = model.get_response_statistics(handle)
                if stats:
                    last_stats = stats
                token = llm.fastllm_lib.fetch_response_llm_model(model.model, handle)
                if token < 0:
                    assert token == -1, ("request failed", token)
                    finished = True
                    break
                if not tokens:
                    first_stats = stats
                tokens.append(token)
                times.append(time.monotonic() - started)
        finally:
            if not finished:
                model.abort_handle(handle)
        elapsed = time.monotonic() - started
        assert len(tokens) == min(maximum, 17), (name, len(tokens))
        result = dict(name=name, token_ids=tokens, input_tokens=len(inputs),
                      first_stats=first_stats, last_stats=last_stats,
                      decode_tps=(len(tokens) - 1) / (elapsed - times[0])
                      if len(tokens) > 1 else None)
        print("KV_CAPACITY_RESULT", json.dumps(result), flush=True)

    try:
        run("cold", 64)
        run("cached", 64)
        run("cached_repeat", 64)
    finally:
        model.release_memory()


def main():
    if "--worker" in sys.argv:
        sys.argv.remove("--worker")
        worker()
        return
    completed = subprocess.run(
        [sys.executable, __file__, "--worker", *sys.argv[1:]],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=600)
    print(completed.stdout, end="", flush=True)
    assert completed.returncode == 0, ("worker failed", completed.returncode)
    if "--help" in sys.argv or "-h" in sys.argv:
        return
    cases = {}
    config = None
    current = None
    for line in completed.stdout.splitlines():
        if line.startswith("KV_CAPACITY_CONFIG "):
            config = json.loads(line.split(" ", 1)[1])
        elif line.startswith("KV_CAPACITY_BEGIN "):
            current = line.split()[1]
            cases[current] = dict(restores=0, result=None)
        elif current and "prefix cache restored:" in line:
            cases[current]["restores"] += 1
        elif line.startswith("KV_CAPACITY_RESULT "):
            result = json.loads(line.split(" ", 1)[1])
            cases[result["name"]]["result"] = result
    assert config is not None, "missing worker configuration"
    pool = re.search(
        r"KV Cache Token limit: (\d+) tokens \(totalPages=(\d+), pageLen=(\d+)\)",
        completed.stdout)
    assert pool, "missing actual KV pool capacity"
    capacity, pages, page_len = map(int, pool.groups())
    assert capacity == pages * page_len == config["tokens"], (pool.groups(), config)
    assert config["max_context_length"] % page_len == 0, "context must end at a page boundary"
    assert 0 <= capacity - config["max_context_length"] <= page_len, "pool has more than one spare page"
    tp = re.search(r"\[Qwen3\.5 DFlash2\] enabled: [^\n]*tp_devices=(\d+)", completed.stdout)
    assert tp and int(tp.group(1)) >= 2, "DFlash TP was not enabled"
    assert set(cases) == {"cold", "cached", "cached_repeat"}, cases
    for name in ("cold", "cached", "cached_repeat"):
        case = cases[name]
        expected_restores = 0 if name == "cold" else 1
        assert case["restores"] == expected_restores, (name, "repeated prefix restore / re-prefill", case)
        result = case["result"]
        assert result is not None
        first, last = result["first_stats"], result["last_stats"]
        assert first and last, (name, result)
        assert (first["cached_input_tokens"] > 0) == (name != "cold"), (name, result)
        for field in ("cached_input_tokens", "missed_input_tokens"):
            assert first[field] == last[field], (name, "request was rebuilt", result)
    print("PASS: TP speculative decode at KV capacity without repeated prefill", flush=True)


if __name__ == "__main__":
    main()
