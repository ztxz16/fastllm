"""CPU-only differential check of the reused Qwen tokenizer with H3 assets."""

import argparse
import ctypes

from transformers import AutoTokenizer

p = argparse.ArgumentParser()
p.add_argument("--model", default="/model/tokenizer")
p.add_argument("--library", required=True)
a = p.parse_args()
lib = ctypes.CDLL(a.library)
lib.h3_error.restype = ctypes.c_char_p
lib.h3_tokenize.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
reference = AutoTokenizer.from_pretrained(a.model)
prompts = [
    "A cat plays piano on a wooden stage. The audience applauds.",
    "一只橘猫在舞台上弹钢琴，镜头缓缓拉近。观众鼓掌。",
    "中文 English mixed 1234567890!\n\nSecond line\t🌊🎹",
    "  leading spaces, trailing spaces  ",
    "café naïve Straße — 你好！",
    "<|im_start|>user\nHello<|im_end|>",
    "",
]
for prompt in prompts:
    output = (ctypes.c_int * 4096)()
    count = lib.h3_tokenize(a.model.encode(), prompt.encode(), output, len(output))
    assert count >= 0, lib.h3_error().decode()
    assert list(output[:count]) == reference.encode(prompt, add_special_tokens=False), prompt
print(f"{len(prompts)} multilingual tokenizer cases match the reference exactly")
