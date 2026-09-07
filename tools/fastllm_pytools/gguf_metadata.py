"""Small, dependency-free helpers for reading GGUF model metadata.

Only requested values are materialized. This keeps model discovery cheap even
when a GGUF embeds hundreds of thousands of tokenizer tokens and BPE merges.
"""

import copy
import os
import struct
from functools import lru_cache


_GGUF_UINT8 = 0
_GGUF_INT8 = 1
_GGUF_UINT16 = 2
_GGUF_INT16 = 3
_GGUF_UINT32 = 4
_GGUF_INT32 = 5
_GGUF_FLOAT32 = 6
_GGUF_BOOL = 7
_GGUF_STRING = 8
_GGUF_ARRAY = 9
_GGUF_UINT64 = 10
_GGUF_INT64 = 11
_GGUF_FLOAT64 = 12

_SCALAR_FORMATS = {
    _GGUF_UINT8: "B",
    _GGUF_INT8: "b",
    _GGUF_UINT16: "H",
    _GGUF_INT16: "h",
    _GGUF_UINT32: "I",
    _GGUF_INT32: "i",
    _GGUF_FLOAT32: "f",
    _GGUF_BOOL: "?",
    _GGUF_UINT64: "Q",
    _GGUF_INT64: "q",
    _GGUF_FLOAT64: "d",
}


def _read_exact(stream, size):
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def _read_number(stream, fmt):
    return struct.unpack("<" + fmt, _read_exact(stream, struct.calcsize(fmt)))[0]


def _read_string(stream, materialize=True):
    length = _read_number(stream, "Q")
    if materialize:
        return _read_exact(stream, length).decode("utf-8")
    stream.seek(length, os.SEEK_CUR)
    return None


def _read_scalar(stream, value_type, materialize):
    if value_type == _GGUF_STRING:
        return _read_string(stream, materialize)
    fmt = _SCALAR_FORMATS.get(value_type)
    if fmt is None:
        raise ValueError("unsupported GGUF metadata type %d" % value_type)
    size = struct.calcsize(fmt)
    if not materialize:
        stream.seek(size, os.SEEK_CUR)
        return None
    return struct.unpack("<" + fmt, _read_exact(stream, size))[0]


def _read_value(stream, value_type, materialize):
    if value_type != _GGUF_ARRAY:
        return _read_scalar(stream, value_type, materialize)
    element_type = _read_number(stream, "I")
    count = _read_number(stream, "Q")
    if element_type == _GGUF_ARRAY:
        raise ValueError("nested GGUF metadata arrays are invalid")
    if not materialize and element_type != _GGUF_STRING:
        fmt = _SCALAR_FORMATS.get(element_type)
        if fmt is None:
            raise ValueError("unsupported GGUF array type %d" % element_type)
        stream.seek(struct.calcsize(fmt) * count, os.SEEK_CUR)
        return None
    values = [] if materialize else None
    for _ in range(count):
        value = _read_scalar(stream, element_type, materialize)
        if materialize:
            values.append(value)
    return values


def read_gguf_metadata(path, keys):
    """Return selected GGUF key/value pairs without loading tensor data."""
    wanted = set(keys)
    if not wanted:
        return {}
    result = {}
    with open(path, "rb") as stream:
        if _read_exact(stream, 4) != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        version = _read_number(stream, "I")
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version %d" % version)
        _read_number(stream, "Q")  # tensor count
        metadata_count = _read_number(stream, "Q")
        for _ in range(metadata_count):
            key = _read_string(stream)
            value_type = _read_number(stream, "I")
            materialize = key in wanted
            value = _read_value(stream, value_type, materialize)
            if materialize:
                result[key] = value
                if wanted.issubset(result):
                    break
    return result


def read_gguf_tensor_names(path):
    """Read the tensor directory, skipping metadata values and tensor data."""
    names = set()
    with open(path, "rb") as stream:
        if _read_exact(stream, 4) != b"GGUF":
            raise ValueError("not a GGUF file: %s" % path)
        version = _read_number(stream, "I")
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version %d" % version)
        tensor_count = _read_number(stream, "Q")
        metadata_count = _read_number(stream, "Q")
        for _ in range(metadata_count):
            _read_string(stream, materialize=False)
            _read_value(stream, _read_number(stream, "I"), materialize=False)
        for _ in range(tensor_count):
            names.add(_read_string(stream))
            dimensions = _read_number(stream, "I")
            if not 1 <= dimensions <= 4:
                raise ValueError("invalid GGUF tensor dimensions")
            _read_exact(stream, dimensions * 8 + 4 + 8)  # shape, type, offset
    return names


_ARCHITECTURE_CONFIGS = {
    "llama": ("llama", "LlamaForCausalLM"),
    "qwen2": ("qwen2", "Qwen2ForCausalLM"),
    "qwen3": ("qwen3", "Qwen3ForCausalLM"),
    "qwen3moe": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    "qwen3_moe": ("qwen3_moe", "Qwen3MoeForCausalLM"),
    "qwen35": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    "qwen3_5": ("qwen3_5", "Qwen3_5ForConditionalGeneration"),
    "qwen35moe": ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
    "qwen3_5_moe": ("qwen3_5_moe", "Qwen3_5MoeForConditionalGeneration"),
    "deepseek2": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "deepseek_v2": ("deepseek_v2", "DeepseekV2ForCausalLM"),
    "deepseek_v3": ("deepseek_v3", "DeepseekV3ForCausalLM"),
    "glm-dsa": ("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
    "glm_moe_dsa": ("glm_moe_dsa", "GlmMoeDsaForCausalLM"),
    "minimax_m2": ("minimax_m2", "MiniMaxM2ForCausalLM"),
}


@lru_cache(maxsize=8)
def _get_gguf_model_config_cached(path, _file_size, _mtime_ns):
    """Build and cache the small HF-shaped config used by Python startup."""
    sampling_keys = {
        "general.sampling.top_k": "top_k",
        "general.sampling.top_p": "top_p",
        "general.sampling.temp": "temperature",
        "general.sampling.penalty_repeat": "repetition_penalty",
    }
    metadata = read_gguf_metadata(
        path, {"general.architecture", *sampling_keys})
    gguf_arch = str(metadata.get("general.architecture", ""))
    if not gguf_arch:
        raise ValueError("GGUF has no general.architecture metadata: %s" % path)
    model_type, architecture = _ARCHITECTURE_CONFIGS.get(
        gguf_arch, (gguf_arch.replace("-", "_"), ""))
    config = {
        "model_type": model_type,
        "gguf_architecture": gguf_arch,
    }
    if architecture:
        config["architectures"] = [architecture]
    generation_config = {
        target: metadata[source]
        for source, target in sampling_keys.items()
        if source in metadata
    }
    if generation_config:
        config["_fastllm_generation_config"] = generation_config
    return config


def get_gguf_model_config(path):
    """Read model discovery data, invalidating the cache if the file changes."""
    path = os.path.realpath(path)
    file_stat = os.stat(path)
    return copy.deepcopy(_get_gguf_model_config_cached(
        path, file_stat.st_size, file_stat.st_mtime_ns))


_GPT2_PRETOKENIZER_PATTERNS = {
    "qwen35": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        r"[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
        r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|"
        r"\s+(?!\S)|\s+"
    ),
    "qwen2": (
        r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|"
        r"[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}|"
        r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|"
        r"\s+(?!\S)|\s+"
    ),
    "gpt-2": (
        r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+|"
        r" ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
    ),
}


def _gguf_merge_pair(value):
    separator = value.find(" ", 1)
    if separator < 0:
        return None
    return value[:separator], value[separator + 1:]


def try_load_gguf_tokenizer(path):
    """Reconstruct a fast tokenizer entirely from GGUF metadata.

    The native tokenizer remains available as a fallback. Using the Rust
    tokenizer backend here also reproduces Unicode pre-tokenizer presets such
    as qwen35, which cannot be inferred from BPE merges alone.
    """
    try:
        from tokenizers import AddedToken, Regex, Tokenizer
        from tokenizers import decoders, models, normalizers, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
    except Exception:
        return None

    keys = {
        "tokenizer.ggml.model",
        "tokenizer.ggml.pre",
        "tokenizer.ggml.tokens",
        "tokenizer.ggml.scores",
        "tokenizer.ggml.token_type",
        "tokenizer.ggml.merges",
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.unknown_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.chat_template",
    }
    try:
        metadata = read_gguf_metadata(path, keys)
        tokens = metadata.get("tokenizer.ggml.tokens") or []
        if not tokens:
            return None
        tokenizer_model = metadata.get("tokenizer.ggml.model", "")
        tokenizer_pre = metadata.get("tokenizer.ggml.pre", "")
        vocab = {token: index for index, token in enumerate(tokens)}

        if tokenizer_model == "gpt2":
            merges = []
            for value in metadata.get("tokenizer.ggml.merges") or []:
                pair = _gguf_merge_pair(value)
                if pair is not None:
                    merges.append(pair)
            backend = Tokenizer(models.BPE(vocab=vocab, merges=merges))
            pattern = _GPT2_PRETOKENIZER_PATTERNS.get(tokenizer_pre)
            if pattern is not None:
                backend.pre_tokenizer = pre_tokenizers.Sequence([
                    pre_tokenizers.Split(
                        Regex(pattern), behavior="isolated", invert=False),
                    pre_tokenizers.ByteLevel(
                        add_prefix_space=False,
                        trim_offsets=False,
                        use_regex=False),
                ])
            else:
                backend.pre_tokenizer = pre_tokenizers.ByteLevel(
                    add_prefix_space=False,
                    trim_offsets=False,
                    use_regex=True)
            if tokenizer_pre == "qwen35":
                backend.normalizer = normalizers.NFC()
            backend.decoder = decoders.ByteLevel(
                add_prefix_space=False, trim_offsets=False, use_regex=False)
        elif tokenizer_model in ("llama", "t5"):
            scores = metadata.get("tokenizer.ggml.scores") or []
            if len(scores) != len(tokens):
                return None
            unknown_id = metadata.get("tokenizer.ggml.unknown_token_id")
            backend = Tokenizer(models.Unigram(
                list(zip(tokens, scores)),
                unk_id=unknown_id,
                byte_fallback=True))
            backend.pre_tokenizer = pre_tokenizers.Metaspace(
                replacement="▁", prepend_scheme="always")
            backend.decoder = decoders.Metaspace(
                replacement="▁", prepend_scheme="always")
        elif tokenizer_model == "bert":
            unknown_id = metadata.get("tokenizer.ggml.unknown_token_id")
            unknown_token = (
                tokens[unknown_id]
                if isinstance(unknown_id, int) and 0 <= unknown_id < len(tokens)
                else "[UNK]")
            backend = Tokenizer(models.WordPiece(
                vocab=vocab, unk_token=unknown_token))
            backend.pre_tokenizer = pre_tokenizers.BertPreTokenizer()
        else:
            return None

        token_types = metadata.get("tokenizer.ggml.token_type") or []
        special_tokens = []
        for index, token_type in enumerate(token_types[:len(tokens)]):
            if token_type in (3, 4):
                special_tokens.append(AddedToken(
                    tokens[index], normalized=False, special=True))
        if special_tokens:
            backend.add_special_tokens(special_tokens)

        def token_for_id(key):
            token_id = metadata.get(key)
            if isinstance(token_id, int) and 0 <= token_id < len(tokens):
                return tokens[token_id]
            return None

        wrapper = PreTrainedTokenizerFast(
            tokenizer_object=backend,
            bos_token=token_for_id("tokenizer.ggml.bos_token_id"),
            eos_token=token_for_id("tokenizer.ggml.eos_token_id"),
            unk_token=token_for_id("tokenizer.ggml.unknown_token_id"),
            pad_token=token_for_id("tokenizer.ggml.padding_token_id"),
            clean_up_tokenization_spaces=False,
        )
        wrapper.chat_template = metadata.get("tokenizer.chat_template") or None
        wrapper.name_or_path = path
        return wrapper
    except Exception as error:
        print("Load tokenizer from GGUF metadata failed: %s" % error)
        return None
