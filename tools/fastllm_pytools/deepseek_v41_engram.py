"""
DeepSeek-V4.1 Engram 元数据生成。

Engram 层按"tokenizer 归一化后的压缩 token id"做 n-gram 哈希，压缩映射依赖
HuggingFace tokenizers 的 normalizer（NFKC / NFD / 去重音 / 小写 / 空白折叠），
C++ 侧不重复实现，而是由本模块预先计算并写成 ``engram_meta.json``：

    {
      "token_map": [压缩 id ...],            # 长度 = tokenizer 词表大小
      "compressed_vocab_size": 99092,
      "pad_token_id": 2,
      "multipliers": [["..."], ["..."]]      # 每个 engram 层一行，int64 以字符串保存
    }

素数桶布局（primes / offsets）由 C++ 侧按同样的算法推导，不写入文件。

用法：
    python -m ftllm.deepseek_v41_engram /path/to/DeepSeek-V4.1-Flash [--output engram_meta.json]

与官方 ``inference/engram.py`` 中的 ``build_compressed_token_map`` /
``compute_hash_multipliers`` 逐位一致。
"""

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Tuple


def _load_text_config(model_dir: str) -> Dict[str, Any]:
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    text = config.get("text_config")
    return text if isinstance(text, dict) else config


class _TokenizerBackend:
    """只依赖 ``tokenizers``（不引入 transformers / torch，避免与已加载的 fastllm 动态库冲突）。"""

    def __init__(self, tokenizer_dir: str):
        from tokenizers import Tokenizer

        path = os.path.join(tokenizer_dir, "tokenizer.json")
        if not os.path.isfile(path):
            raise FileNotFoundError("tokenizer.json not found in " + tokenizer_dir)
        self.backend_tokenizer = Tokenizer.from_file(path)

    def __len__(self) -> int:
        return self.backend_tokenizer.get_vocab_size(with_added_tokens=True)


def build_compressed_token_map(tokenizer) -> Tuple[List[int], int]:
    from tokenizers import Regex, normalizers

    sentinel = ""
    normalizer = normalizers.Sequence(
        [
            normalizers.NFKC(),
            normalizers.NFD(),
            normalizers.StripAccents(),
            normalizers.Lowercase(),
            normalizers.Replace(Regex(r"[ \t\r\n]+"), " "),
            normalizers.Replace(Regex(r"^ $"), sentinel),
            normalizers.Strip(),
            normalizers.Replace(sentinel, " "),
        ]
    )
    backend = tokenizer.backend_tokenizer
    key_to_new: Dict[str, int] = {}
    lookup = [0] * len(tokenizer)
    for token_id in range(len(tokenizer)):
        text = backend.decode([token_id], skip_special_tokens=False)
        if "�" in text:
            key = backend.id_to_token(token_id)
        else:
            normalized = normalizer.normalize_str(text)
            key = normalized if normalized else text
        new_id = key_to_new.get(key)
        if new_id is None:
            new_id = len(key_to_new)
            key_to_new[key] = new_id
        lookup[token_id] = new_id
    return lookup, len(key_to_new)


def compute_hash_multipliers(layer_ids: List[int], max_ngram_size: int, tokenizer_vocab_size: int) -> List[List[int]]:
    import numpy as np

    max_long = np.iinfo(np.int64).max
    multiplier_bound = max(1, (max_long // tokenizer_vocab_size) // 2)
    rows = []
    for layer_id in layer_ids:
        generator = np.random.default_rng(10007 * layer_id)
        values = generator.integers(low=0, high=multiplier_bound, size=(max_ngram_size,), dtype=np.int64)
        rows.append([int(v) * 2 + 1 for v in values])
    return rows


def build_engram_meta(model_dir: str, tokenizer_dir: str = None) -> Dict[str, Any]:
    text = _load_text_config(model_dir)
    layer_ids = list(text.get("engram_layer_ids", []))
    if not layer_ids:
        raise ValueError("config has no engram_layer_ids; this is not a DeepSeek-V4.1 checkpoint")
    max_ngram = int(text.get("engram_max_ngram_size", 4))
    pad_token_id = int(text.get("engram_pad_token_id", 2))
    expected_compressed = int(text.get("engram_compressed_vocab_size", 0))

    tokenizer = _TokenizerBackend(tokenizer_dir or model_dir)
    token_map, compressed_vocab = build_compressed_token_map(tokenizer)
    if expected_compressed and compressed_vocab != expected_compressed:
        raise ValueError(
            "compressed vocab size mismatch: computed %d, config says %d "
            "(tokenizer / tokenizers version differs from training?)" % (compressed_vocab, expected_compressed)
        )
    multipliers = compute_hash_multipliers(layer_ids, max_ngram, compressed_vocab)
    return {
        "token_map": token_map,
        "compressed_vocab_size": compressed_vocab,
        "pad_token_id": pad_token_id,
        "engram_layer_ids": layer_ids,
        "multipliers": [[str(v) for v in row] for row in multipliers],
    }


def default_meta_path(model_dir: str) -> str:
    return os.path.join(model_dir, "engram_meta.json")


def ensure_engram_meta(model_dir: str, quiet: bool = False) -> str:
    """若 engram_meta.json 不存在则生成；返回文件路径（可能位于缓存目录）。"""
    path = default_meta_path(model_dir)
    if os.path.exists(path):
        return path
    meta = build_engram_meta(model_dir)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        return path
    except OSError:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "fastllm", "engram")
        os.makedirs(cache_dir, exist_ok=True)
        path = os.path.join(cache_dir, os.path.basename(os.path.abspath(model_dir.rstrip("/"))) + "_engram_meta.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(meta, f)
        if not quiet:
            print("[ftllm] model directory is read-only, engram meta written to", path)
        return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate DeepSeek-V4.1 engram_meta.json")
    parser.add_argument("model_dir")
    parser.add_argument("--tokenizer", default=None, help="tokenizer directory (default: model_dir)")
    parser.add_argument("--output", default=None, help="output path (default: <model_dir>/engram_meta.json)")
    args = parser.parse_args(argv)
    meta = build_engram_meta(args.model_dir, args.tokenizer)
    output = args.output or default_meta_path(args.model_dir)
    with open(output, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    print("compressed vocab size:", meta["compressed_vocab_size"])
    print("multipliers:", meta["multipliers"])
    print("written:", output)


if __name__ == "__main__":
    sys.exit(main())
