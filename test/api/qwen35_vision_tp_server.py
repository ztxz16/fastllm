"""Start the normal server while recording original generated token IDs.

For regression only; arguments after this script are normal ftllm CLI arguments:
  PYTHONPATH=/path/to/runtime python3 test/api/qwen35_vision_tp_server.py \
      server /path/to/model --tp 3 --mtp 3 --multimodal --port 8082

The wrapper observes the existing token fetch call. It leaves sampling, MTP,
payload preparation and output_logits unchanged. Each finished request writes
one JSON record to stdout; qwen35_vision_tp_probe.py can require and compare it.
"""

import json
from pathlib import Path
import runpy
import sys
import threading

# test/api/openai.py is a client example, not the installed OpenAI SDK.
test_dir = Path(__file__).resolve().parent
sys.path = [entry for entry in sys.path if Path(entry or ".").resolve() != test_dir]
from ftllm import llm


def main():
    original = llm.fastllm_lib.fetch_response_llm_model
    original_batch = llm.fastllm_lib.fetch_response_tokens_batch_llm_model
    pending = {}
    lock = threading.Lock()

    def record(model, handle, tokens, end=None):
        key = (int(model), int(handle))
        with lock:
            if tokens:
                pending.setdefault(key, []).extend(tokens)
            if end is not None:
                print("[Vision TP probe tokens] " + json.dumps({
                    "model": key[0], "handle": key[1], "end": end,
                    "token_ids": pending.pop(key, [])}), flush=True)

    def fetch(model, handle):
        token = original(model, handle)
        record(model, handle, [token] if token >= 0 else [], token if token < 0 else None)
        return token

    def fetch_batch(model, handle, buffer, capacity):
        count = original_batch(model, handle, buffer, capacity)
        record(model, handle, [buffer[i] for i in range(max(0, count))],
               count if count < 0 else None)
        return count

    llm.fastllm_lib.fetch_response_llm_model = fetch
    llm.fastllm_lib.fetch_response_tokens_batch_llm_model = fetch_batch
    try:
        runpy.run_module("ftllm.cli", run_name="__main__")
    finally:
        llm.fastllm_lib.fetch_response_llm_model = original
        llm.fastllm_lib.fetch_response_tokens_batch_llm_model = original_batch


if __name__ == "__main__":
    main()
