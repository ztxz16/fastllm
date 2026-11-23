"""Compare dense 8-rank prefill logits on the shipped Naive SM120 EP8 geometry.

Run against a built ftllm package on eight 110-SM GPUs in paired NUMA order:
    python test/cuda/test_sm120_ep_prefill_logits.py --model /path/to/mxfp4-model
"""
import argparse
import ctypes
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path


def measure(path):
    from ftllm import llm, util

    args = util.make_normal_parser("SM120 prefill logits").parse_args([
        path, "--tp", "8", "--ep", "8", "--attn-dp-size", "8", "--atype", "bfloat16",
        "--kv_cache_dtype", "bfloat16", "--cuda_embedding", "--cache_history", "false",
        "--tokens", "32768", "--max_batch", "16", "--max_context_length", "4096",
        "--chunked_prefill_size", "2048", "--prefill_batch_tokens", "16384",
        "--prefill_max_requests", "8", "--prefill_batch_requests", "8",
        "--prefill_max_wait_ms", "100", "--prefill_dp_balance", "true",
        "--speculative_algorithm", "off", "--dspark", "0",
    ])
    model = util.make_normal_llm_model(args)
    model.set_verbose(True)
    lib = llm.fastllm_lib
    config = json.loads((Path(path) / "config.json").read_text())
    vocab = config.get("text_config", config)["vocab_size"]
    scores = (ctypes.c_float * vocab)()
    rng = random.Random(480)

    def launch(length):
        ids = (ctypes.c_int * length)(*(rng.randrange(200, 8000) for _ in range(length)))
        handle = lib.launch_response_llm_model(model.model, length, ids, 1, 0, False,
                                               1., 1, 1., 1., True, 0, None)
        assert handle >= 0
        return handle

    # Queue all eight requests while the warm-up prompt occupies the scheduler.
    warmup = launch(2048)
    time.sleep(.12)
    handles = [launch(480) for _ in range(8)]
    digest = hashlib.sha256()
    for handle in [warmup, *handles]:
        token = lib.fetch_response_logits_llm_model(model.model, handle, scores)
        assert 0 <= token < vocab
        if handle != warmup:
            assert max(scores) > min(scores)
            digest.update(bytes(scores))
        assert lib.fetch_response_logits_llm_model(model.model, handle, scores) == -1
    print("DENSE_LOGITS", digest.hexdigest(), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        measure(args.model)
        return

    digests = []
    for mode, backend in [("native", "flashinfer-sm120-native-cubin"),
                          ("auto", "sm120-auto-grouped-and-native")]:
        env = dict(os.environ, FASTLLM_SM120_EP_BACKEND=mode, FASTLLM_SM120_EP_PREFILL_PIPELINE="1")
        child = subprocess.run([sys.executable, __file__, "--worker", "--model", args.model],
                               env=env, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        output = child.stdout
        if child.returncode:
            raise RuntimeError(f"{mode} worker exited with {child.returncode}\n{output[-3000:]}")
        assert "backend=" + backend in output, "Expected " + backend
        assert re.search(r"\[Prompt\] 3840 Tokens\..*Batch: 8\.", output), "Expected 8 requests of 480 tokens"
        digests.append(re.search(r"DENSE_LOGITS ([0-9a-f]{64})", output).group(1))
    assert digests[0] == digests[1], "Native and auto prefill logits differ"
    print("Native and auto prefill logits match", digests[0])


if __name__ == "__main__":
    main()
