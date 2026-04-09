import json
import os
import subprocess
import sys
import time

from transformers import AutoTokenizer


WORKSPACE_ROOT = os.path.dirname(os.path.abspath(__file__))
FASTLLM_TOOLS_PATH = os.path.join(WORKSPACE_ROOT, "build", "tools")

DENSE_MODEL_PATH = "/home/use/gemma-4-31B-it"
MOE_MODEL_PATH = "/home/use/gemma-4-26B-A4B-it"

FASTLLM_CHILD_FLAG = "--fastllm-child"
FASTLLM_MAX_NEW_TOKENS = 16

FIXED_MESSAGES = [
    {"role": "user", "content": "What is the capital of France? Answer concisely."},
]


def load_fastllm_llm():
    if FASTLLM_TOOLS_PATH not in sys.path:
        sys.path.insert(0, FASTLLM_TOOLS_PATH)
    from ftllm import llm

    return llm


def print_banner(title):
    print(f"\n{'=' * 80}")
    print(title)
    print(f"{'=' * 80}")


def build_hf_input(model_path, messages):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    return tokenizer, prompt, input_ids


def compare_hf_and_fastllm_tokenizer(label, model_path, messages):
    llm = load_fastllm_llm()

    hf_tokenizer, hf_prompt, hf_ids = build_hf_input(model_path, messages)
    fastllm_tokenizer = llm.tokenizer(model_path)
    fastllm_prompt = fastllm_tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    fastllm_ids = fastllm_tokenizer.encode(fastllm_prompt)

    print_banner(f"{label} tokenizer comparison")
    print(f"model_path: {model_path}")
    print(f"hf_prompt == fastllm_prompt: {hf_prompt == fastllm_prompt}")
    print(f"hf_ids == fastllm_ids: {hf_ids == fastllm_ids}")
    print(f"prompt repr: {hf_prompt!r}")
    print(f"hf_ids: {hf_ids}")
    print(f"fastllm_ids: {fastllm_ids}")

    if hf_ids != fastllm_ids:
        limit = min(len(hf_ids), len(fastllm_ids))
        mismatch_idx = None
        for idx in range(limit):
            if hf_ids[idx] != fastllm_ids[idx]:
                mismatch_idx = idx
                break
        if mismatch_idx is None:
            mismatch_idx = limit
        print(f"first_mismatch_idx: {mismatch_idx}")
        print(f"hf_slice: {hf_ids[max(0, mismatch_idx - 8):mismatch_idx + 8]}")
        print(f"fastllm_slice: {fastllm_ids[max(0, mismatch_idx - 8):mismatch_idx + 8]}")

    return hf_tokenizer, hf_prompt, hf_ids


def extract_marker(text, marker):
    pos = text.rfind(marker)
    if pos == -1:
        return None
    end = text.find("\n", pos)
    if end == -1:
        end = len(text)
    return text[pos + len(marker):end].strip()


def tail_lines(text, count=20):
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    return "\n".join(lines[-count:])


def run_fastllm_child(model_path, input_ids):
    command = [
        sys.executable,
        __file__,
        FASTLLM_CHILD_FLAG,
        model_path,
        json.dumps(input_ids),
    ]
    return subprocess.run(command, capture_output=True, text=True)


def print_fastllm_child_summary(label, result):
    print_banner(f"{label} fastllm fixed-input run")
    print(f"child_exit_code: {result.returncode}")

    load_sec = extract_marker(result.stdout, "FASTLLM_DEBUG_LOAD_SEC=")
    gen_sec = extract_marker(result.stdout, "FASTLLM_DEBUG_GEN_SEC=")
    new_ids = extract_marker(result.stdout, "FASTLLM_DEBUG_NEW_IDS=")
    decoded = extract_marker(result.stdout, "FASTLLM_DEBUG_DECODED=")

    if load_sec is not None:
        print(f"load_sec: {load_sec}")
    if gen_sec is not None:
        print(f"generate_sec: {gen_sec}")
    if new_ids is not None:
        print(f"new_ids: {json.loads(new_ids)}")
    if decoded is not None:
        print(f"decoded: {json.loads(decoded)!r}")

    debug_lines = [line for line in result.stdout.splitlines() if "[Gemma4 debug]" in line]
    if debug_lines:
        print("\nGemma4 debug lines:")
        for line in debug_lines:
            print(line)

    if result.returncode != 0 or new_ids is None or decoded is None:
        print("\nstdout tail:")
        print(tail_lines(result.stdout, count=40))
        if result.stderr.strip():
            print("\nstderr tail:")
            print(tail_lines(result.stderr, count=20))


def fastllm_child_main(model_path, input_ids):
    llm = load_fastllm_llm()
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    t0 = time.time()
    model = llm.model(model_path, dtype="float16")
    load_sec = time.time() - t0

    t1 = time.time()
    output_ids = model.generate(
        [input_ids.copy()],
        max_new_tokens=FASTLLM_MAX_NEW_TOKENS,
        do_sample=False,
        top_k=1,
        top_p=1.0,
        temperature=1.0,
    )[0]
    gen_sec = time.time() - t1

    new_ids = output_ids[len(input_ids):]
    decoded = tokenizer.decode(new_ids, skip_special_tokens=False)

    print(f"\nFASTLLM_DEBUG_LOAD_SEC={load_sec:.6f}", flush=True)
    print(f"FASTLLM_DEBUG_GEN_SEC={gen_sec:.6f}", flush=True)
    print("FASTLLM_DEBUG_NEW_IDS=" + json.dumps(new_ids), flush=True)
    print("FASTLLM_DEBUG_DECODED=" + json.dumps(decoded, ensure_ascii=False), flush=True)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


def main():
    models = [
        ("Gemma4 31B dense", DENSE_MODEL_PATH),
        ("Gemma4 26B A4B MoE", MOE_MODEL_PATH),
    ]

    token_results = {}
    for label, model_path in models:
        _, _, input_ids = compare_hf_and_fastllm_tokenizer(label, model_path, FIXED_MESSAGES)
        token_results[label] = input_ids

    dense_ids = token_results["Gemma4 31B dense"]
    moe_ids = token_results["Gemma4 26B A4B MoE"]

    print_banner("Cross-model token check")
    print(f"dense_ids == moe_ids: {dense_ids == moe_ids}")
    print(f"shared_input_ids: {dense_ids}")

    if dense_ids != moe_ids:
        raise RuntimeError("Dense and MoE model tokenizers produced different input_ids. Stop here first.")

    for label, model_path in models:
        result = run_fastllm_child(model_path, dense_ids)
        print_fastllm_child_summary(label, result)


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == FASTLLM_CHILD_FLAG:
        fastllm_child_main(sys.argv[2], json.loads(sys.argv[3]))
    else:
        main()
