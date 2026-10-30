"""Exact token-ID comparison against the official UMT5 text preprocessing."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--official', type=Path, required=True)
    parser.add_argument('--tokenizer', type=Path, required=True)
    parser.add_argument('--library', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location('official_tokenizer', args.official/'wan/modules/tokenizers.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    reference = module.HuggingfaceTokenizer(str(args.tokenizer), seq_len=512, clean='whitespace')
    lib = ctypes.CDLL(str(args.library))
    lib.fastllm_lingbot_error.restype = ctypes.c_char_p
    lib.fastllm_lingbot_tokenizer_create.argtypes = [ctypes.c_char_p]
    lib.fastllm_lingbot_tokenizer_create.restype = ctypes.c_void_p
    lib.fastllm_lingbot_tokenizer_destroy.argtypes = [ctypes.c_void_p]
    lib.fastllm_lingbot_tokenize.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    handle = lib.fastllm_lingbot_tokenizer_create(str(args.tokenizer).encode())
    if not handle:
        raise RuntimeError(lib.fastllm_lingbot_error().decode())
    prompts = ['', 'A tree beside a lake.', '房间里有一张桌子，窗外是晴朗的天空。',
        '  a\t b\n\nc  ', 'cafe\u0301 déjà vu', '猫 🐈 and 🧑🏽‍🚀',
        '&amp; &lt;tag&gt; &quot;yes&quot;', '&amp;amp; &#x4e2d;&#25991;',
        '“你好”—‘hello’', 'ＡＢＣ１２３　中文', 'office ﬃ', 'a\u00a0b\u2003c',
        '<pad>Hello</s>world', 'one <extra_id_0> two', '🌈'*20, 'hello '*600]
    results = []
    try:
        for prompt in prompts:
            ids, mask = reference(prompt, return_mask=True, add_special_tokens=True)
            expected = ids[0, :int(mask.sum())].tolist()
            output = (ctypes.c_int*512)()
            count = lib.fastllm_lingbot_tokenize(handle, prompt.encode(), output, 512)
            if count<0:
                raise RuntimeError(lib.fastllm_lingbot_error().decode())
            actual = list(output[:count])
            result = dict(prompt=prompt[:100], exact=actual==expected, native=actual, reference=expected)
            results.append(result)
            print(prompt[:60].encode('unicode_escape').decode(), result['exact'], flush=True)
    finally:
        lib.fastllm_lingbot_tokenizer_destroy(handle)
    args.output.write_text(json.dumps(dict(results=results, passed=all(r['exact'] for r in results)), ensure_ascii=False, indent=2))
    if not all(r['exact'] for r in results):
        raise SystemExit('Tokenizer mismatch')


if __name__ == '__main__':
    main()
