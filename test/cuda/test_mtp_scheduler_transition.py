"""Real-model regression for MTP proposal state across 1 -> 2 -> 1 requests.

Sampling requests use random drafts and full-distribution rejection by default.
Covers sampling pairs and mixed sampling/greedy pairs on the same model.
Requires a CUDA Qwen3.5/Qwen3.8 checkpoint with MTP and --max_batch 2.
"""
import ctypes
import json
import time

from ftllm import llm
from ftllm.util import make_normal_llm_model, make_normal_parser


def run_pair(model, mixed=False):
    prompts = [
        "Write Python to merge closed intervals. Validate every input pair, "
        "accept generators, preserve the input, and explain the implementation.",
        "Implement an O(1) Python LRU cache. Reject invalid capacities, support "
        "None as a cached value, and include usage examples and tests.",
    ]
    requests = []
    started = time.monotonic()
    returned_to_single = False

    def launch(index):
        prompt = ("<|im_start|>user\n" + prompts[index]
                  + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
        if model.hf_tokenizer is not None:
            ids = model.hf_tokenizer.encode(prompt, add_special_tokens=False)
        else:
            ids = model.encode(prompt)
        assert ids, "prompt encoding produced no tokens"
        buf = (ctypes.c_int * len(ids))(*ids)
        sampling = not (mixed and index == 1)
        # Keep the first request alive long enough to join the second request.
        maximum = 768 if index == 0 else 256
        handle = llm.fastllm_lib.launch_response_llm_model(
            model.model, len(ids), buf, ctypes.c_int(maximum), ctypes.c_int(maximum),
            ctypes.c_bool(sampling), ctypes.c_float(1.0), ctypes.c_int(20 if sampling else 1),
            ctypes.c_float(.95 if sampling else 1.0), ctypes.c_float(1.0),
            ctypes.c_bool(False), 0, None)
        assert handle >= 0, "request launch failed"
        requests.append(dict(handle=handle, maximum=maximum, count=0, done=False))
        print("MTP_TRANSITION_LAUNCH", index, "sampling" if sampling else "greedy", flush=True)

    launch(0)
    try:
        while not (len(requests) == 2 and all(r["done"] for r in requests)):
            if time.monotonic() - started > 180:
                raise TimeoutError("MTP scheduler transition exceeded 180 seconds")
            for row in requests:
                if row["done"] or not llm.fastllm_lib.can_fetch_response_llm_model(model.model, row["handle"]):
                    continue
                token = llm.fastllm_lib.fetch_response_llm_model(model.model, row["handle"])
                if token < 0:
                    assert token == -1, ("request failed", token)
                    assert row["count"] == row["maximum"], ("truncated request", row)
                    row["done"] = True
                else:
                    row["count"] += 1
            if len(requests) == 1 and requests[0]["count"] >= 64:
                assert not requests[0]["done"], "first request ended before overlap"
                launch(1)
            if len(requests) == 2 and requests[1]["done"] and not requests[0]["done"]:
                returned_to_single = True
            time.sleep(.0005)
        assert returned_to_single, "requests never returned from overlap to a live single request"
    finally:
        for row in requests:
            if not row["done"]:
                llm.fastllm_lib.abort_response_llm_model(model.model, row["handle"])
    print("MTP_TRANSITION_PASS", json.dumps(dict(mixed=mixed, requests=requests)), flush=True)


def main():
    parser = make_normal_parser("MTP single/batch proposal-state regression")
    args = parser.parse_args()
    if args.max_batch != 2 or args.mtp < 2:
        parser.error("this regression requires --max_batch 2 and --mtp >= 2")
    model = make_normal_llm_model(args)
    try:
        run_pair(model)
        run_pair(model, mixed=True)
    finally:
        model.release_memory()
    print("PASS: MTP sampling and mixed requests across scheduler transitions", flush=True)


if __name__ == "__main__":
    main()
