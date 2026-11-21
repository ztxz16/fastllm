"""Exercise the original Inkling 2 Mini checkpoint through the FastLLM request scheduler."""

import argparse
import ctypes

import numpy as np
from ftllm import llm
from tokenizers import Tokenizer


def generate(model, prompts, steps, vocab, sample=False):
    lib = llm.fastllm_lib
    handles = []
    for ids in prompts:
        tokens = (ctypes.c_int * len(ids))(*ids)
        handles.append(lib.launch_response_llm_model(
            model.model, len(ids), tokens, steps, 0, sample, 0.95 if sample else 1,
            40 if sample else 1, 1, 1, True, 0, None))
    results = []
    for handle in handles:
        ids, scores = [], []
        buffer = (ctypes.c_float * vocab)()
        while True:
            token = lib.fetch_response_logits_llm_model(model.model, handle, buffer)
            if token == -1:
                break
            assert token >= 0, token
            logits = np.ctypeslib.as_array(buffer).copy()
            assert 0 <= token < vocab and np.isfinite(logits).all()
            assert np.count_nonzero(logits > logits[token]) < (40 if sample else 1)
            ids.append(token)
            scores.append(logits)
        assert len(ids) == steps
        results.append((ids, scores))
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--experts", choices=["cuda:0", "cpu", "numa"], required=True)
    parser.add_argument("--window", action="store_true", help="Also exercise the 1024-token KV wrap")
    parser.add_argument("--graph", action="store_true", help="Capture the pure-GPU single-request decode graph")
    parser.add_argument("--sample", action="store_true", help="Use top-k/top-p sampling and verify the sampled prefix")
    args = parser.parse_args()
    llm.set_cuda_graph(args.graph)
    llm.set_device_map({"cuda:0": 1})
    llm.set_device_map({args.experts: 1}, is_moe=True)
    llm.set_max_tokens(4096)
    model = llm.model(args.model, dtype="bfloat16", tokenizer_type="fastllm")
    model.set_max_batch(2)
    tokenizer = Tokenizer.from_file(args.model + "/tokenizer.json")
    vocab = model.config["text_config"]["unpadded_vocab_size"]
    texts = ["The capital of France is", "1 + 1 ="]
    prompts = [tokenizer.encode(text, add_special_tokens=False).ids for text in texts]
    # Reference greedy continuations from transformers add-inkling2-mini, 6173a2f.
    expected = [[152717, 13, 68014, 32], [220, 18, 68014, 371]]
    # Serial requests exercise graph reuse across independent caches.
    batch = ([generate(model, [p], 4, vocab, args.sample)[0] for p in prompts] if args.graph
             else generate(model, prompts, 4, vocab, args.sample))
    for prompt, (tokens, scores), reference in zip(prompts, batch, expected):
        if not args.sample:
            assert tokens == reference, (tokens, reference)
        replay = generate(model, [prompt + tokens[:2]], 1, vocab, args.sample)[0]
        if not args.sample:
            assert replay[0][0] == tokens[2]
        cosine = np.dot(replay[1][0], scores[2]) / (np.linalg.norm(replay[1][0]) * np.linalg.norm(scores[2]))
        assert cosine > 0.98, cosine
    if args.window:
        length = model.config["text_config"]["sliding_window_size"] + 1
        prompt = (prompts[0] * ((length + len(prompts[0]) - 1) // len(prompts[0])))[:length]
        tokens, decoded = generate(model, [prompt], 2, vocab, args.sample)[0]
        _, prefetched = generate(model, [prompt + tokens[:1]], 1, vocab, args.sample)[0]
        cosine = np.dot(decoded[1], prefetched[0]) / (np.linalg.norm(decoded[1]) * np.linalg.norm(prefetched[0]))
        assert cosine > 0.98, cosine
        print("window boundary cosine", cosine)
    print(args.experts, "PASS")


if __name__ == "__main__":
    main()
