"""Content identities and startup configuration for native prefix persistence.

Identity is computed once per model construction, from tensor bytes rather than
paths or mtimes. The native model copies the exported settings at construction;
request processing never hashes model files or consults another model's env.
"""
from contextlib import contextmanager
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import copy
import ctypes
import hashlib
from importlib import metadata
import json
import math
import os

try:
    from .prefix_cache_logging import (
        configure_prefix_cache_logging, prefix_cache_log_level,
        prefix_cache_info, prefix_cache_debug,
    )
except ImportError:
    from prefix_cache_logging import (
        configure_prefix_cache_logging, prefix_cache_log_level,
        prefix_cache_info, prefix_cache_debug,
    )
from pathlib import Path
import re
import struct
import threading
import time

FORMAT_VERSION = 4
# Storage-format axes are independent and must never share one constant
# (SSD_CACHE_REWRITE_PLAN.md §10.1.1 / §13.1):
#   storage_format_version   = on-disk v4 record format
#   storage_manifest_schema  = startup manifest structure version
#   STATE_ABI["version"]     = model state semantics (stays 4)
STORAGE_FORMAT_V4 = 4
STORAGE_MANIFEST_SCHEMA_V4 = 2
# Bits 0-1 are retired; do not reuse them for a different capability.
STORAGE_CAP_READ_V4 = 1 << 2
STORAGE_CAP_WRITE_V4 = 1 << 3
STORAGE_CAP_CONFIG = 1 << 4
DEFAULT_DISK_BYTES = 256 << 30
_READ_BYTES = 8 << 20
_HASH_WORKERS = 8
_ENV_LOCK = threading.RLock()
_INPUT_ENV = ("FASTLLM_PREFIX_CACHE_DIR", "FASTLLM_PREFIX_CACHE_DISK_BYTES",
              "FASTLLM_PREFIX_CACHE_RESTORE_POLICY", "FASTLLM_PREFIX_CACHE_RESTORE_TIMEOUT_MS",
              "FASTLLM_PREFIX_CACHE_L2_HOST_KV", "FASTLLM_PREFIX_CACHE_L2_HOST_KV_MB",
              "FASTLLM_PREFIX_CACHE_LOG_LEVEL")
_DERIVED_ENV = tuple("FASTLLM_PREFIX_CACHE_" + suffix for suffix in (
    "IDENTITY", "FAMILY_IDENTITY", "TARGET_IDENTITY", "ENCODER_IDENTITY",
    "DRAFT_IDENTITY"))
# Startup capacity controls do not change model math or the bytes represented by
# a prefix checkpoint.  In particular, --multimodal only uses this budget to
# reserve the vision workspace before KV sizing; including it split otherwise
# compatible text and image checkpoints into separate SSD namespaces.
SCOPE_VERSION = 1
TARGET_MATH_VERSION = "qwen35-target-v2"
STATE_ABI = {"version": 4, "target": "ordinary_kv+linear_state+speculative"}
# Development namespace for the request-gated restore contract.  It keeps
# pre-repair checkpoints out of the new lookup family without deleting or
# migrating the old data in the existing directory.
CACHE_NAMESPACE = "qwen35-ssd-request-restore-v1"


def _fields(names, scope, reason, category=None):
    return {name: {"scope": scope, "reason": reason, "normalizer_version": 1,
                   "category": category} for name in names.split()}


# This is a closed, versioned schema, not a catch-all exclusion list. New CLI,
# checkpoint or mathematical fields require an explicit classification and a
# corresponding normalizer. test_persistent_prefix checks parser/schema drift.
EXECUTION_FIELDS = {
    **_fields("dtype atype dtype_config", "base", "Effective target computation precision", 2),
    **_fields("rope_scaling max_context_length", "base", "Effective RoPE, not the admission limit", 3),
    **_fields("kv_cache_dtype", "encoding", "KV representation; explicit converters may bridge it", 4),
    **_fields("chat_template enable_thinking", "prefix", "Only the resulting token plan participates", 5),
    **_fields("lora custom ori mmproj", "base", "Unhashed weight/graph overrides are rejected", 1),
    # storage-config 子组本身必须登记，否则 execution 记录被再次分类时会报
    # “Unclassified ... storage_config”。它只装 §13.1 的两个 runtime-only 参数。
    **_fields("storage_config", "runtime-only",
             "Storage version; subgroup of execution", None),
    **_fields("moe_cache_half_life moe_cache_update_interval moe_cache_max_replacements "
              "moe_cache_max_bytes moe_cache_min_heat moe_cache_margin moe_cache_factor "
              "moe_cache_min_residence moe_cache_prefill_prior moe_cache_rank_by_bytes",
              "runtime-only", "GGUF expert placement policy; persistent targets must be dense"),
    **_fields("mtp_min_p", "runtime-only",
              "Qwen4 draft stopping policy; does not change dense Qwen3.5 target state"),
    **_fields("kvmem", "runtime-only",
              "Only disabled sparse attention is compatible; checked before identity construction"),
    **_fields("help command version model path threads low low_gpu_mem moe_dtype moe_atype cuda_embedding "
              "kv_cache_limit max_batch chunked_prefill_size fast_prefill device vision_device multimodal "
              "tp pp moe_device moe_device_layers moe_cuda_cache moe_cpu_cache image_embedding_cache "
              "ngram_device moe_experts cache_history cache_fast cuda_shared_expert enable_amx tokens "
              "page_size prefix_cache prefix_cache_log_level prefix_cache_dir prefix_cache_disk_gb "
              "prefix_cache_restore_policy prefix_cache_restore_timeout_ms "
              "prefix_cache_l2 prefix_cache_l2_gb "
              "prefix_cache_snapshot_interval_pages "
              "prefix_cache_snapshot_max_per_request prefix_cache_snapshot_max_records "
              "prefix_cache_checkpoint_interval prefix_cache_checkpoint_jobs prefix_cache_host_limit_mb "
              "prefix_cache_storage_version "
              "prefix_cache_v5 prefix_cache_v5_trim prefix_cache_v5_request_hold prefix_cache_restore_drain "
              "prefix_cache_v5_read_threads prefix_cache_v5_drain_ms prefix_cache_v5_interval_tokens "
              "prefix_cache_mm_tail_align "
              "prefix_cache_ledger_bound prefix_cache_ledger_max_page_refs prefix_cache_ledger_max_records host_malloc_trim_seconds "
              "prefix_cache_v4_segments prefix_cache_min_context_tokens prefix_cache_write_min_reuse "
              "prefix_cache_drain_timeout_s prefix_cache_write_workers prefix_cache_write_mode "
              "prefix_cache_pack_workers prefix_cache_pinned_slots prefix_cache_d2_staging_mb "
              "prefix_cache_gpu_pin_limit_mb prefix_cache_host_limit_max_mb prefix_cache_hash_mode "
              "prefix_cache_hash_pool_mb "
              "gpu_mem_ratio cuda_slab mtp mtp_fp8_draft_head dspark speculative_algorithm "
              "speculative_draft_model_path draft_tokens speculative_num_draft_tokens "
              "speculative_dspark_block_size speculative_dspark_confidence_threshold triton cache_dir "
              "tool_call_parser model_name host port api_key temperature top_p top_k repeat_penalty "
              "think hide_input dev_mode startup_progress env", "runtime-only",
              "Deployment, scheduling, decoding or speculative state; not dense target mathematics"),
    **_fields("_auto_explicit auto auto_disk_cache_ratio auto_gpu_memory_limit auto_host_memory_limit "
              "force_ddr_limit autotune autotune_budget autotune_profile autotune_prompt",
              "runtime-only", "Automatic resource planning and launch tuning; not target mathematics"),
    **_fields("pd prefill_batch_requests prefill_batch_tokens prefill_max_requests prefill_max_wait_ms "
              "prefill_min_decode_requests", "runtime-only",
              "KV manager and prefill batching schedule; not target mathematics"),
    **_fields("attn_dp_size ep draft_gpu_resident indexer_dtype indexer_tp_storage", "runtime-only",
              "DeepSeek-V4.1/MiMo MoE/Naive DSA placement options; unused by the dense Qwen3.5 target"),
}
TEXT_FIELDS = {
    **_fields("hidden_size num_hidden_layers num_layers n_layer num_attention_heads n_head "
              "num_key_value_heads head_dim linear_num_key_heads linear_num_value_heads "
              "linear_key_head_dim linear_value_head_dim rms_norm_eps", "base", "Native target geometry", 1),
    **_fields("rope_theta partial_rotary_factor rope_parameters rope_scaling max_position_embeddings "
              "mrope_section mrope_interleaved", "base", "Effective positional mathematics", 3),
    **_fields("bos_token_id eos_token_id pad_token_id image_token_id video_token_id "
              "vision_start_token_id vision_end_token_id", "prefix", "Actual token/media plan", 5),
    **_fields("model_type architectures dtype torch_dtype transformers_version _name_or_path "
              "_commit_hash auto_map tokenizer_class use_cache return_dict output_attentions "
              "output_hidden_states tie_word_embeddings initializer_range attention_dropout "
              "attention_bias attn_output_gate output_gate_type hidden_act intermediate_size "
              "vocab_size full_attention_interval layer_types linear_conv_kernel_dim mamba_ssm_dtype "
              "mtp_num_hidden_layers mtp_use_dedicated_embeddings num_experts num_local_experts "
              "num_experts_per_tok norm_topk_prob shared_expert_intermediate_size", "runtime-only",
              "Dense loader derives this from tensors or uses fixed inference operations; speculation excluded"),
}
MODEL_FIELDS = {
    **TEXT_FIELDS,
    **_fields("text_config", "base", "Classified recursively; never hash the whole dictionary", 1),
    **_fields("vision_config", "prefix", "Encoder identity only, after the first affected media", 5),
    **_fields("quantization_config", "base", "Source weight decoding is validated separately", 2),
    **_fields("language_model_only", "runtime-only", "Factory/vision availability, not text math"),
}
VISION_FIELDS = {
    **_fields("depth hidden_size num_heads intermediate_size patch_size temporal_patch_size "
              "spatial_merge_size out_hidden_size num_position_embeddings deepstack_visual_indexes",
              "prefix", "Effective native encoder geometry", 5),
    **_fields("model_type hidden_act in_channels initializer_range dtype torch_dtype",
              "runtime-only", "Native RGB encoder uses fixed activation and channel operations"),
}
ROPE_FIELDS = _fields("type rope_type rope_theta partial_rotary_factor factor "
                      "original_max_position_embeddings beta_fast beta_slow attention_factor "
                      "mrope_section mrope_interleaved full_attention", "base", "Effective RoPE", 3)
QUANTIZATION_FIELDS = _fields("quant_method activation_scheme fmt weight_block_size "
                              "modules_to_not_convert", "runtime-only",
                              "Dense FP8 loader resolves storage and scales from tensor headers")
PROCESSOR_FIELDS = {
    **_fields("patch_size temporal_patch_size merge_size image_mean image_std size", "prefix",
              "Effective native preprocessing, confined to affected media", 5),
    **_fields("image_processor_type video_processor_type processor_class do_resize do_rescale "
              "do_normalize do_convert_rgb resample rescale_factor min_pixels max_pixels fps "
              "min_frames max_frames", "runtime-only", "Not read by the native Qwen35 preprocessor"),
}
GENERATION_FIELDS = {
    **_fields("bos_token_id eos_token_id pad_token_id decoder_start_token_id", "prefix",
              "Actual generated/request tokens", 5),
    **_fields("_from_model_config transformers_version do_sample temperature top_k top_p min_p "
              "typical_p repetition_penalty encoder_repetition_penalty max_length max_new_tokens "
              "min_length min_new_tokens num_beams num_beam_groups early_stopping length_penalty "
              "no_repeat_ngram_size encoder_no_repeat_ngram_size bad_words_ids force_words_ids "
              "forced_bos_token_id forced_eos_token_id remove_invalid_values renormalize_logits "
              "suppress_tokens begin_suppress_tokens sequence_bias diversity_penalty "
              "penalty_alpha epsilon_cutoff eta_cutoff guidance_scale use_cache cache_implementation "
              "cache_config return_legacy_cache output_attentions output_hidden_states output_scores "
              "output_logits return_dict_in_generate num_return_sequences max_time stop_strings "
              "token_healing watermarking_config constraints forced_decoder_ids exponential_decay_length_penalty",
              "runtime-only", "Generation policy does not change an existing target prefix"),
}

# Exact names only: neither environment-prefix scans nor implementation hashes
# can create a namespace. These flags select equivalent kernels/runtime policy.
ENV_FIELDS = _fields("""
CUDA_VISIBLE_DEVICES FT_THREADS FASTLLM_ACTIVATE_NUMA FASTLLM_NUMA_THREADS FASTLLM_TP
FASTLLM_CUDA_DFLASH_TP_BACKBONE FASTLLM_DFLASH_BLOCK_SIZE FASTLLM_DFLASH_MODEL_PATH
FASTLLM_DSPARK_MODEL_PATH FASTLLM_QWEN35_ENABLE_MTP FASTLLM_QWEN35_MM_MAX_PATCHES
FASTLLM_CUDA_GRAPH FASTLLM_CUDA_TRITON FASTLLM_CUDA_SERVING_WARMUP
FASTLLM_PREFIX_CACHE FASTLLM_MULTIMODAL_PREFIX_CACHE FASTLLM_IMAGE_EMBEDDING_CACHE_BYTES
FASTLLM_PREFIX_CACHE_LOG_LEVEL
FASTLLM_PREFIX_CACHE_ENCODER_IDENTITY
FASTLLM_PREFIX_CACHE_SNAPSHOT_INTERVAL_PAGES FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_PER_REQUEST
FASTLLM_PREFIX_CACHE_SNAPSHOT_MAX_RECORDS FASTLLM_QWEN35_VISION_DEVICE
FASTLLM_CUDA_CUTLASS_LINEAR_FP8_GDN_INPUT_RMSNORM_QUANT_TP
FASTLLM_CUDA_CUTLASS_LINEAR_FP8_INPUT_RMSNORM_QUANT_TP
FASTLLM_CUDA_DFLASH_DIRECT_KV_CACHE FASTLLM_CUDA_DFLASH_EXACT_VERIFY
FASTLLM_CUDA_DFLASH_FP16_LINEAR_WEIGHTS FASTLLM_CUDA_DFLASH_FUSED_CONV
FASTLLM_CUDA_DFLASH_FUSED_GATEUP_PREPARE FASTLLM_CUDA_DFLASH_FUSED_KV_COMPACT
FASTLLM_CUDA_DFLASH_FUSED_KV_MATERIALIZE FASTLLM_CUDA_DFLASH_FUSED_KV_PROJECTION
FASTLLM_CUDA_DFLASH_FUSED_LINEAR FASTLLM_CUDA_DFLASH_FUSED_QKV_PREPARE
FASTLLM_CUDA_DFLASH_IMPLICIT_ATTENTION FASTLLM_CUDA_DFLASH_PREALLOCATE_KV_CACHE
FASTLLM_CUDA_DFLASH_TP_MIN_WEIGHT_MB FASTLLM_CUDA_DFLASH_TP_MLP
FASTLLM_CUDA_FREEZE_AFTER_WARMUP FASTLLM_CUDA_GRAPH_MOE_RECAPTURE_WITHOUT_MARKERS
FASTLLM_CUDA_LINEAR_FP16_PATH_NATIVE FASTLLM_CUDA_QWEN35_FUSED_ATTN_PREFILL
FASTLLM_CUDA_QWEN35_FUSED_ATTN_PREFILL_MIN_BATCH FASTLLM_CUDA_TP_FUSIONS
FASTLLM_CUDA_QWEN35_GDN_FUSED_CUMSUM_DECAY_MASK FASTLLM_CUDA_QWEN35_GDN_FUSED_MUL_CAUSAL_MASK
FASTLLM_CUDA_QWEN35_GDN_FUSED_NEG_MUL_CAUSAL_MASK FASTLLM_CUDA_QWEN35_GDN_FUSED_RMSNORM_POSTCONV
FASTLLM_CUDA_TRITON_CHUNK_GDN_POSTCONV FASTLLM_GPU_TOKEN_HANDOFF
FASTLLM_IDLE_PREFILL_BATCH_QUIET_US FASTLLM_IDLE_PREFILL_BATCH_WAIT_US
FASTLLM_MTP_FP8_DRAFT_HEAD FASTLLM_MTP_GPU_CHAIN FASTLLM_MTP_GUMBEL FASTLLM_MTP_PREFIX_GRAPH
FASTLLM_PRINT_LOGITS FASTLLM_QWEN35_CUDA_GRAPH_MAX_BATCH
FASTLLM_QWEN35_CUDA_GRAPH_PRECAPTURE_DENSE FASTLLM_QWEN35_CUDA_GRAPH_PRECAPTURE_START_BATCH
FASTLLM_QWEN35_DISABLE_BATCH_PREFILL FASTLLM_QWEN35_DISABLE_SHARDED_GREEDY
FASTLLM_QWEN35_DISABLE_SINGLE_PREFILL_FUSED_CONV FASTLLM_QWEN35_DISABLE_TRANSPOSED_BATCH_DECODE
FASTLLM_QWEN35_MOE_DISABLE_FUSED_MOE FASTLLM_QWEN3_MOE_DISABLE_FUSED_MOE
FASTLLM_QWEN35_MTP_BATCHED_STATE_COPY FASTLLM_QWEN35_MTP_BATCHED_STATE_RESTORE
FASTLLM_QWEN35_MTP_COMBINED_Z_GATE FASTLLM_QWEN35_MTP_FUSED_LINEAR_SEQ
FASTLLM_QWEN35_MTP_PROFILE FASTLLM_QWEN35_MTP_TOKEN_MAJOR_CONV FASTLLM_QWEN35_MTP_TP
FASTLLM_QWEN35_MTP_VERIFY_CUDA_GRAPH FASTLLM_QWEN35_MTP_WARMUP
FASTLLM_QWEN35_MTP_WORKER_PROFILE FASTLLM_QWEN35_SKIP_INTERMEDIATE_CHUNK_HEAD
FASTLLM_QWEN35_THREAD_TP FASTLLM_QWEN35_TRACE_DIR
""", "runtime-only", "Kernel/deployment policy under the explicit target math ABI")
FIELD_SCOPES = {"execution": EXECUTION_FIELDS, "text_config": TEXT_FIELDS,
                "model_config": MODEL_FIELDS, "vision_config": VISION_FIELDS,
                "rope": ROPE_FIELDS, "quantization_config": QUANTIZATION_FIELDS,
                "processor": PROCESSOR_FIELDS, "generation_config": GENERATION_FIELDS, "env": ENV_FIELDS}


def _classified(value, group):
    if not isinstance(value, dict):
        raise ValueError("Persistent-cache %s must be an object" % group)
    unknown = set(value) - FIELD_SCOPES[group].keys()
    if unknown:
        raise ValueError("Unclassified persistent-cache %s fields: %s" % (group, ", ".join(sorted(unknown))))
    return value


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, allow_nan=False).encode()


def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()


def _stat_identity(stat):
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _check_unchanged(path, handle, before):
    if (_stat_identity(before) != _stat_identity(os.fstat(handle.fileno())) or
            _stat_identity(before) != _stat_identity(path.stat())):
        raise ValueError("Model file changed while building persistent-cache identity")


def _file_record(path):
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        for data in iter(lambda: handle.read(_READ_BYTES), b""):
            digest.update(data)
        _check_unchanged(path, handle, before)
    return {"bytes": before.st_size, "sha256": digest.hexdigest()}


def _draft_identity(draft_path, execution, target_speculative_weights=None):
    algorithm = str(execution.get("speculative_algorithm") or "").strip().lower()
    options = {key: execution.get(key) for key in (
        "draft_tokens", "speculative_num_draft_tokens", "mtp", "dspark",
        "speculative_dspark_block_size", "speculative_dspark_confidence_threshold")}
    active = algorithm in ("dflash", "mtp", "dspark") or any(
        int(options.get(key) or 0) > 0 for key in ("mtp", "dspark"))
    if not draft_path:
        if not active:
            return "none"
        return "builtin:" + _digest({"algorithm": algorithm, "options": options,
                                      "weights": target_speculative_weights or {}})
    path = Path(draft_path).expanduser().resolve()
    files = {}
    if path.is_file():
        files[path.name] = _file_record(path)
        config = path.parent / "config.json"
        if config.is_file():
            files["config.json"] = _file_record(config)
    elif path.is_dir():
        for candidate in sorted(path.rglob("*")):
            relative = candidate.relative_to(path)
            if (not candidate.is_file() or any(part.startswith(".") for part in relative.parts) or
                    (candidate.suffix not in (".json", ".safetensors") and
                     candidate.name not in ("merges.txt", "vocab.txt"))):
                continue
            files[relative.as_posix()] = _file_record(candidate)
    else:
        raise ValueError("Speculative draft checkpoint does not exist: " + str(path))
    if not files:
        raise ValueError("Speculative draft checkpoint has no identifiable files: " + str(path))
    return _digest({"kind": "external", "algorithm": algorithm,
                    "options": options, "files": files})


def _is_encoder_tensor(name):
    return any(part in name.split(".") for part in
               ("visual", "vision_model", "vision_tower"))


def _is_speculative_tensor(name):
    """Target-side draft components are not part of target forward state."""
    return any(part in name.split(".") for part in ("mtp", "dflash", "dspark"))


def _target_config(text_config):
    _classified(text_config, "text_config")
    defaults = {"hidden_size": 4096, "num_hidden_layers": 28, "num_attention_heads": 32,
                "rms_norm_eps": 1e-6, "linear_num_key_heads": 16, "linear_num_value_heads": 32,
                "linear_key_head_dim": 128, "linear_value_head_dim": 128}
    aliases = {"num_hidden_layers": ("num_layers", "n_layer"), "num_attention_heads": ("n_head",)}
    result = {}
    for key, default in defaults.items():
        result[key] = text_config.get(key, next((text_config[a] for a in aliases.get(key, ())
                                                if a in text_config), default))
    result["head_dim"] = text_config.get("head_dim", result["hidden_size"] // result["num_attention_heads"])
    result["num_key_value_heads"] = text_config.get("num_key_value_heads", result["num_attention_heads"])
    for key, value in result.items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError("Invalid target geometry: " + key)
        if key != "rms_norm_eps" and int(value) != value:
            raise ValueError("Noninteger target geometry: " + key)
        result[key] = _float32(value) if key == "rms_norm_eps" else int(value)
    return result


def _float32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _rope_object(value):
    if value is None or value == "":
        return {}
    if isinstance(value, str):
        value = {"rope_type": "yarn"} if value == "yarn" else json.loads(value)
    value = copy.deepcopy(_classified(value, "rope"))
    if "full_attention" in value:
        if len(value) != 1:
            raise ValueError("RoPE full_attention must be the only group")
        return _rope_object(value["full_attention"])
    if "type" in value:
        if "rope_type" in value and value["type"] != value["rope_type"]:
            raise ValueError("RoPE type conflicts with rope_type")
        value["rope_type"] = value.pop("type")
    return value


def _effective_rope(text, execution):
    """Mirror ResolveContextOptions, storing only parameters used in arithmetic."""
    legacy, modern = _rope_object(text.get("rope_scaling")), _rope_object(text.get("rope_parameters"))
    override = _rope_object(execution.get("rope_scaling"))
    rope = dict(legacy)
    for key, value in modern.items():
        if key in rope and rope[key] != value and key not in override:
            raise ValueError("Conflicting checkpoint RoPE field: " + key)
        rope[key] = value
    if "rope_type" in override and override["rope_type"] != rope.get("rope_type", "default"):
        rope = {key: value for key, value in rope.items() if key in
                ("rope_theta", "partial_rotary_factor", "mrope_section", "mrope_interleaved")}
    rope.update(override)
    kind = rope.get("rope_type", "default") or "default"
    if kind not in ("default", "linear", "yarn", "dynamic"):
        raise ValueError("Unsupported persistent-cache RoPE type: " + str(kind))
    declared = text.get("max_position_embeddings", 0)
    requested = execution.get("max_context_length", -1)
    if requested is None:
        requested = -1
    if type(requested) is not int or (requested != -1 and requested <= 0):
        raise ValueError("max_context_length must be -1 or positive")
    effective = requested if requested > 0 else declared
    geometry = _target_config(text)
    theta = _float32(rope.get("rope_theta", text.get("rope_theta", 10000)))
    partial = rope.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25))
    rotary = geometry["head_dim"] * partial
    if (not math.isfinite(theta) or theta <= 1 or not 0 < partial <= 1 or
            int(rotary) != rotary or int(rotary) % 2):
        raise ValueError("Invalid effective RoPE geometry")
    rotary = int(rotary)
    sections = rope.get("mrope_section", text.get("mrope_section", [11, 11, 10]))
    configured = bool(override) or requested > 0 or kind == "yarn"
    if (not isinstance(sections, list) or len(sections) != 3 or
            any(type(v) is not int or v < 0 for v in sections) or sum(sections) != rotary // 2):
        if configured:
            raise ValueError("Invalid effective mrope_section")
        half = rotary // 2
        sections = [half // 3, half // 3, half - 2 * (half // 3)]
    if rope.get("mrope_interleaved", True) is not True:
        raise ValueError("Only interleaved MRoPE is supported")
    result = {"rope_type": kind, "theta": theta, "rotary_dim": rotary,
              "mrope_section": sections, "mrope_interleaved": True}
    if kind == "dynamic":
        if override or effective > declared:
            raise ValueError("Dynamic RoPE override/extension is unsupported")
        # Unlike ordinary RoPE, NTK actually uses the original window in math.
        result.update(factor=_float32(rope.get("factor", 1)), original_max_position_embeddings=declared)
        return result
    if kind != "yarn" and any(key in rope for key in ("beta_fast", "beta_slow", "attention_factor")):
        raise ValueError("YaRN-only field on non-YaRN RoPE")
    original = rope.get("original_max_position_embeddings", 0)
    if kind == "yarn":
        if not original and declared == 262144 and theta == 10000000 and rotary == 64 and sections == [11, 11, 10]:
            original = 262144
        if type(original) is not int or original <= 0:
            raise ValueError("YaRN requires original_max_position_embeddings")
        factor = _float32(rope.get("factor", max(1, math.ceil(effective / original))))
        fast, slow = _float32(rope.get("beta_fast", 32)), _float32(rope.get("beta_slow", 1))
        attention = _float32(rope.get("attention_factor", 1 + 0.1 * math.log(factor)))
        if not 0 < slow <= fast or attention <= 0:
            raise ValueError("Invalid YaRN parameters")
        result.update(original_max_position_embeddings=original, beta_fast=fast, beta_slow=slow,
                      attention_factor=attention)
    else:
        factor = _float32(rope.get("factor", 1))
    if not math.isfinite(factor) or factor < 1 or (kind == "default" and factor != 1):
        raise ValueError("Invalid RoPE factor")
    supported = original * factor if kind != "default" and original else declared
    if declared and (effective > supported or effective > 1 << 24):
        raise ValueError("Requested context exceeds the configured RoPE window")
    # linear factor=1 is exactly the default positional arithmetic.
    result["rope_type"] = "default" if kind == "linear" and factor == 1 else kind
    result["factor"] = factor
    return result


def _dtype(value):
    value = str(value or "auto").strip().lower()
    aliases = {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "float": "float32",
               "fp8": "fp8_e4m3", "float8": "fp8_e4m3", "fp4": "fp4_e2m1", "nvfp4": "fp4_e2m1"}
    value = aliases.get(value, value)
    if value not in ("auto", "float16", "float32", "bfloat16", "fp8_e4m3", "fp4_e2m1", "int8", "int4") and not re.fullmatch(r"int[24]g(?:[1-9][0-9]*)?", value):
        raise ValueError("Unclassified effective precision: " + value)
    if value in ("int4g", "int2g"):
        value += "128"
    return value


def _canonical_weight_name(name):
    # Qwen3_5Model::GetTensorMap's text-only aliases apply to every source tensor,
    # including quantization scales. Do not guess aliases the loader lacks.
    if name in ("model.embed_tokens.weight", "model.norm.weight"):
        return "model.language_model." + name[len("model."):]
    if name.startswith("model.layers."):
        return "model.language_model.layers." + name[len("model.layers."):]
    return name


# Scope the tensors actually referenced by the dense target forward. Unknown
# names are a capability error, not an excuse to invent another base identity.
TARGET_TENSOR_ROLES = {
    "model.language_model.embed_tokens.weight": "embedding",
    "model.language_model.norm.weight": "normalization",
    "lm_head.weight": "linear",
}
TARGET_LAYER_ROLES = {
    **{name: "normalization" for name in ("input_layernorm.weight", "post_attention_layernorm.weight",
                                          "self_attn.q_norm.weight", "self_attn.k_norm.weight",
                                          "linear_attn.norm.weight")},
    **{name: "linear" for name in (
        "self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
        "self_attn.o_proj.weight", "self_attn.mergeqkv.weight", "self_attn.W_pack.weight",
        "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight", "mlp.gateup_proj.weight",
        "linear_attn.in_proj_qkv.weight", "linear_attn.in_proj_z.weight", "linear_attn.in_proj_b.weight",
        "linear_attn.in_proj_a.weight", "linear_attn.in_proj_qkvz.weight", "linear_attn.in_proj_ba.weight",
        "linear_attn.out_proj.weight")},
    **{name: "bias" for name in ("self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias",
                                 "self_attn.o_proj.bias", "self_attn.mergeqkv.bias", "self_attn.W_pack.bias",
                                 "mlp.down_proj.bias", "linear_attn.conv1d.bias", "linear_attn.dt_bias")},
    "linear_attn.A_log": "recurrent_parameter",
    "linear_attn.conv1d.weight": "convolution",
    "linear_attn.in_proj_qkvzba.weight": "fused_recurrent_parameter",
}
_NON_FORWARD_TENSOR_PREFIXES = ("optimizer.", "optimizer_state.", "training.", "metadata.", "unused_metadata.")


def _target_tensor_role(name, layers):
    if name.startswith(_NON_FORWARD_TENSOR_PREFIXES) or name in ("global_step", "training_step"):
        return "runtime-only"
    if name in TARGET_TENSOR_ROLES:
        return TARGET_TENSOR_ROLES[name]
    match = re.fullmatch(r"model\.language_model\.layers\.([0-9]+)\.(.+)", name)
    if match and match.group(2) in TARGET_LAYER_ROLES:
        return TARGET_LAYER_ROLES[match.group(2)] if int(match.group(1)) < layers else "runtime-only"
    raise ValueError("Unclassified target tensor role: " + name)


def _scale_candidates(name):
    candidates = [name + "_scale_inv", name + "_scale"]
    if name.endswith(".weight"):
        candidates.extend((name[:-7] + ".scale_inv", name[:-7] + ".scale", name[:-7] + ".weight_scale"))
    return candidates


def _semantic_weights(raw, geometry):
    result, auxiliaries = {}, set()
    for name in raw:
        auxiliaries.update(candidate for candidate in _scale_candidates(name) if candidate in raw)
    for name, record in raw.items():
        if _is_speculative_tensor(name) or name in auxiliaries:
            continue
        canonical = _canonical_weight_name(name)
        if not _is_encoder_tensor(name) and _target_tensor_role(canonical, geometry["num_hidden_layers"]) == "runtime-only":
            continue
        if canonical in result:
            raise ValueError("Ambiguous native tensor alias: " + canonical)
        result[canonical] = record
        # Native FindSafeTensorScaleTensorName consumes the first existing alias.
        # Other recognized aliases are unused; only the effective scale belongs
        # to the numerical tensor. Normalize that association across checkpoints.
        selected = (next((candidate for candidate in _scale_candidates(name) if candidate in raw), None)
                    if record["dtype"] == "F8_E4M3" else None)
        if selected is not None:
            result[canonical + "_scale_inv"] = raw[selected]
    return result


_LINEAR_NAME = re.compile(r"(?:lm_head|model\.language_model\.layers\.[0-9]+\.(?:"
    r"self_attn\.(?:q_proj|k_proj|v_proj|o_proj|mergeqkv|W_pack)|"
    r"mlp\.(?:gate_proj|up_proj|down_proj|gateup_proj)|"
    r"linear_attn\.(?:in_proj_qkv|in_proj_z|in_proj_b|in_proj_a|in_proj_qkvz|in_proj_ba|out_proj)))\.weight$")
_VISION_LINEAR_NAME = re.compile(r"model\.visual\.(?:patch_embed\.proj|blocks\.[0-9]+\."
    r"(?:attn\.(?:qkv|proj)|mlp\.linear_fc[12])|(?:merger|deepstack_merger_list\.[0-9]+)\.linear_fc[12])\.weight$")


def _weight_precision(weights, execution):
    requested = _dtype(execution.get("dtype"))
    rules = execution.get("dtype_config") or []
    if isinstance(rules, str):
        rules = json.loads(rules)
    if not isinstance(rules, list):
        raise ValueError("dtype_config must be an ordered list of key/dtype rules")
    parsed = []
    for rule in rules:
        if not isinstance(rule, dict) or set(rule) != {"key", "dtype"} or not isinstance(rule["key"], str):
            raise ValueError("Unclassified dtype_config rule")
        # Restrict to the common Python/ECMAScript regex subset, rather than
        # claiming Python-only constructs have the native std::regex meaning.
        if "(?" in rule["key"] or re.search(r"\\[^.\[\](){}*+?^$|\\]", rule["key"]):
            raise ValueError("dtype_config regex requires native semantic classification")
        parsed.append((re.compile(rule["key"]), _dtype(rule["dtype"])))
    result = {}
    for name, record in weights.items():
        if not (_LINEAR_NAME.fullmatch(name) or _VISION_LINEAR_NAME.fullmatch(name)):
            continue
        selected = requested
        for pattern, dtype in parsed:
            if pattern.search(name):
                selected = dtype  # Native ParseDataType uses the last match.
        scales = (name + "_scale_inv", name + "_scale", name[:-7] + ".scale_inv",
                  name[:-7] + ".scale", name[:-7] + ".weight_scale")
        if selected == "auto":
            selected = "fp8_e4m3" if record["dtype"] == "F8_E4M3" and any(k in weights for k in scales) else "float16"
        if selected == "fp8_e4m3" and record["dtype"] != "F8_E4M3":
            selected = "float16"
        if selected == "fp4_e2m1":
            raise ValueError("Packed FP4 weight precision needs a semantic decoder before persistence")
        result[name] = selected
    return result


def _tensor_layout(path):
    """Validate raw tensor ranges once, keeping no weight payload in memory."""
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        prefix = handle.read(8)
        if len(prefix) != 8:
            raise ValueError("Invalid safetensors header in persistent-cache model")
        header_bytes = struct.unpack("<Q", prefix)[0]
        if not 0 < header_bytes <= min(100_000_000, before.st_size - 8):
            raise ValueError("Invalid safetensors header length")
        raw_header = handle.read(header_bytes)
        header = json.loads(raw_header)
        if not isinstance(header, dict):
            raise ValueError("Invalid safetensors tensor table")
        tensors = []
        for name, descriptor in header.items():
            if name == "__metadata__":
                continue
            if not isinstance(descriptor, dict):
                raise ValueError("Invalid safetensors tensor descriptor")
            offsets = descriptor.get("data_offsets")
            shape = descriptor.get("shape")
            dtype = descriptor.get("dtype")
            if (not isinstance(offsets, list) or len(offsets) != 2 or
                    any(type(value) is not int for value in offsets) or
                    not isinstance(shape, list) or
                    any(type(value) is not int or value < 0 for value in shape) or
                    not isinstance(dtype, str) or not dtype):
                raise ValueError("Invalid safetensors tensor metadata")
            start, end = offsets
            if not 0 <= start <= end <= before.st_size - 8 - header_bytes:
                raise ValueError("Safetensors tensor is outside its data buffer")
            tensors.append((start, end, name, shape, dtype))
        previous_end = 0
        for start, end, name, shape, dtype in sorted(tensors):
            if start < previous_end:
                raise ValueError("Overlapping safetensors tensor ranges")
            previous_end = end
        _check_unchanged(path, handle, before)
    return before, 8 + header_bytes, tensors


def _model_plan(model_path):
    root = Path(model_path).expanduser().resolve()
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    index_path = root / "model.safetensors.index.json"
    weight_map = None
    if index_path.is_file():
        weight_map = json.loads(index_path.read_text(encoding="utf-8")).get("weight_map")
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("Persistent cache requires a nonempty safetensors weight_map")
        shards = sorted(set(weight_map.values()))
    else:
        shards = ["model.safetensors"]
    layouts, names = [], set()
    for name in shards:
        shard = Path(name)
        if shard.is_absolute() or ".." in shard.parts:
            raise ValueError("Checkpoint shard path must be relative to the model directory")
        path = root / shard
        before, offset, tensors = _tensor_layout(path)
        present = {tensor[2] for tensor in tensors}
        if weight_map is not None:
            expected = {key for key, value in weight_map.items() if value == name}
            if not expected.issubset(present):
                raise ValueError("Safetensors index references a missing tensor")
        # Native SafeTensors also loads tensors absent from the index.
        if names & present:
            raise ValueError("Duplicate tensor names in checkpoint shards")
        names.update(present)
        layouts.append((path, before, offset, tensors))
    if not names:
        raise ValueError("Persistent cache requires model weight tensors")
    return {"root": root, "config": config, "layouts": layouts, "weights": {}}


def _hash_tensor(task, buffers, buffer_bytes, progress):
    model_index, path, before, offset, tensor = task
    start, end, name, shape, dtype = tensor
    if not hasattr(buffers, "data"):
        buffers.data = bytearray(buffer_bytes)
    view = memoryview(buffers.data)
    digest = hashlib.sha256()
    # Each task has its own file offset; never seek a shared handle. The same
    # reusable buffer feeds the complete tensor SHA-256 in original byte order.
    with path.open("rb", buffering=0) as handle:
        _check_unchanged(path, handle, before)
        handle.seek(offset + start)
        remaining = end - start
        while remaining:
            count = handle.readinto(view[:min(remaining, len(view))])
            if not count:
                raise ValueError("Truncated model tensor while hashing identity")
            digest.update(view[:count])
            remaining -= count
            if progress:
                progress(model_index, read_bytes=count)
        _check_unchanged(path, handle, before)
    return model_index, name, {"dtype": dtype, "shape": shape,
                              "bytes": end - start, "sha256": digest.hexdigest()}


def _hash_models(plans, *, progress=False):
    tasks = [(index, path, before, offset, tensor)
             for index, plan in enumerate(plans)
             for path, before, offset, tensors in plan["layouts"] for tensor in tensors]
    # One pool for target, vision and draft. Large tensors go first so a large
    # embedding or the single-file draft cannot become a serial tail.
    tasks.sort(key=lambda task: (-(task[4][1] - task[4][0]), task[0], str(task[1]), task[4][2]))
    workers = min(_HASH_WORKERS, len(tasks))
    buffer_bytes = min(_READ_BYTES, max(task[4][1] - task[4][0] for task in tasks))
    buffers = threading.local()
    started = time.monotonic()
    last_report = started
    hashed_bytes, completed = [0] * len(plans), [0] * len(plans)
    totals = [sum(len(layout[3]) for layout in plan["layouts"]) for plan in plans]
    progress_lock = threading.Lock()

    def report(index=0, *, read_bytes=0, finished=0, force=False):
        nonlocal last_report
        with progress_lock:
            hashed_bytes[index] += read_bytes
            completed[index] += finished
            now = time.monotonic()
            if force or now - last_report >= 5:
                for index, plan in enumerate(plans):
                    prefix_cache_debug("[Prefix SSD] hashing %s: %.2f GiB read, tensors=%d/%d, elapsed=%.1fs" %
                          (plan["root"].name, hashed_bytes[index] / (1 << 30), completed[index],
                           totals[index], now - started))
                last_report = now

    if progress:
        prefix_cache_debug("[Prefix SSD] tensor hashing: models=%d, workers=%d, max_in_flight=%d, read_buffers=%.2f MiB" %
                           (len(plans), workers, 2 * workers, workers * buffer_bytes / (1 << 20)))
    iterator, pending = iter(tasks), set()
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="prefix-hash") as pool:
        def submit_next():
            task = next(iterator, None)
            if task is not None:
                pending.add(pool.submit(_hash_tensor, task, buffers, buffer_bytes, report if progress else None))

        for _ in range(2 * workers):
            submit_next()
        try:
            while pending:
                finished, pending = wait(pending, return_when=FIRST_COMPLETED)
                for future in finished:
                    index, name, record = future.result()
                    plans[index]["weights"][name] = record
                    if progress:
                        report(index, finished=1)
                for _ in finished:
                    submit_next()
        except BaseException:
            for future in pending:
                future.cancel()
            raise
    if progress:
        report(force=True)
    # Catch changes to an already-finished shard while other shards were read.
    for plan in plans:
        for path, before, _, _ in plan["layouts"]:
            if _stat_identity(path.stat()) != _stat_identity(before):
                raise ValueError("Model file changed while building persistent-cache identity")


def _model_files(root):
    files = {}
    # Includes tokenizer, templates, processor configuration and local model code.
    # Shard filenames/packing are not semantic; the tensor names and raw bytes are.
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root)
        if (any(part.startswith(".") for part in relative.parts) or
                not path.is_file() or path.name == "model.safetensors.index.json"):
            continue
        if (path.suffix in (".json", ".jinja", ".jinja2", ".py", ".model", ".tiktoken") or
                path.name in ("merges.txt", "vocab.txt")):
            files[relative.as_posix()] = _file_record(path)
    return files


def _runtime_record(library_path):
    root = Path(__file__).resolve().parent
    implementations = {name: _file_record(root / name) for name in (
        "qwen35_multimodal_native.py", "llm.py")}
    triton_override = os.environ.get("FASTLLM_CUDA_TRITON_SERVER_SCRIPT")
    if triton_override:
        implementations["triton_server_override"] = _file_record(
            Path(triton_override).expanduser())
    else:
        for directory in (root, root.parent):
            script = directory / "fastllm_triton_server.py"
            if script.is_file():
                implementations["triton_server"] = _file_record(script)
                break
    versions = {}
    for name in ("numpy", "Pillow", "transformers", "tokenizers", "jinja2",
                 "safetensors", "triton", "nvidia-cublas-cu12", "nvidia-cuda-runtime-cu12"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return {"library": _file_record(Path(library_path)),
            "python_implementations": implementations, "dependencies": versions}


def build_identity(model_path, execution, library_path, draft_path=None, *, progress=False,
                   storage_format=STORAGE_FORMAT_V4):
    if storage_format != STORAGE_FORMAT_V4:
        raise ValueError("Persistent prefix cache supports only storage format v4")
    target_plan = _model_plan(model_path)
    config = _classified(target_plan["config"], "model_config")
    text_config = config.get("text_config", config)
    if (config.get("model_type") not in ("qwen3_5", "qwen3_5_text") or
            not isinstance(text_config, dict) or
            any(scope.get(key, 0) for scope in (config, text_config)
                for key in ("num_experts", "num_local_experts"))):
        raise ValueError("Persistent prefix v4 requires dense Qwen3.5/Qwen3.8 layout")
    execution = copy.deepcopy(_classified(execution, "execution"))
    _classified(execution.get("env", {}), "env")
    for key in ("lora", "custom", "ori", "mmproj"):
        if execution.get(key):
            raise ValueError("Persistent cache cannot safely identify this weight/graph override: " + key)
    if os.environ.get("FASTLLM_CUDA_TRITON_SERVER_SCRIPT"):
        raise ValueError("Persistent cache cannot classify a custom Triton implementation")
    # A top-level math override and its nested counterpart otherwise follow two
    # different native initialization paths. Refuse conflicts, not guess a hash.
    if "text_config" in config:
        text_config = copy.deepcopy(_classified(text_config, "text_config"))
        for key, value in config.items():
            if key in TEXT_FIELDS and TEXT_FIELDS[key]["scope"] == "base":
                if key in text_config and value != text_config[key]:
                    raise ValueError("Conflicting root/text_config mathematical field: " + key)
                text_config[key] = value
    else:
        text_config = {key: value for key, value in config.items() if key in TEXT_FIELDS}
    quantization = _classified(config.get("quantization_config") or {}, "quantization_config")
    if quantization.get("quant_method", "fp8") not in ("fp8", None):
        raise ValueError("Persistent semantic weight decoding currently supports unquantized/FP8 checkpoints")
    generation_file = target_plan["root"] / "generation_config.json"
    if generation_file.is_file():
        # Native LoadHF copies every generation_config key into weight.dicts
        # before InitParams. Do not let hidden_size/RoPE or a future field hide
        # in what is normally a decoding-only file and bypass classification.
        _classified(json.loads(generation_file.read_text(encoding="utf-8")), "generation_config")
    _hash_models([target_plan], progress=progress)
    target_speculative_weights = {
        name: record for name, record in target_plan["weights"].items()
        if _is_speculative_tensor(name)}
    draft_identity = _draft_identity(draft_path, execution, target_speculative_weights)
    geometry = _target_config(text_config)
    weights = _semantic_weights(target_plan["weights"], geometry)
    runtime = _runtime_record(library_path)
    target_weights = {key: value for key, value in weights.items() if not _is_encoder_tensor(key)}
    encoder_weights = {key: value for key, value in weights.items() if _is_encoder_tensor(key)}
    atype = _dtype(execution.get("atype"))
    atype = "float16" if atype == "auto" else atype
    if atype not in ("float16", "bfloat16", "float32"):
        raise ValueError("Unsupported activation precision")
    kv_dtype = _dtype(execution.get("kv_cache_dtype"))
    kv_dtype = atype if kv_dtype == "auto" else kv_dtype
    if kv_dtype not in ("float16", "fp8_e4m3"):
        raise ValueError("Persistent cache supported encodings are float16 and fp8_e4m3")
    # Both GPU and ordinary Forward derive an absent lm_head from embeddings;
    # tie_word_embeddings does not suppress an explicitly present native head.
    derived_head = "lm_head.weight" not in target_weights and "model.language_model.embed_tokens.weight" in target_weights
    if derived_head:
        target_weights["lm_head.weight"] = target_weights["model.language_model.embed_tokens.weight"]
    precision = _weight_precision(target_weights, execution)
    if derived_head:
        precision["lm_head.weight"] = atype
    reduction = str(execution.get("env", {}).get("FASTLLM_CUDA_CUSTOM_ALLREDUCE", "off")).strip().lower()
    reduction = "custom" if reduction in ("1", "true", "on", "yes") else "nccl"
    math_version = TARGET_MATH_VERSION + "-" + reduction
    target = {"tensors": target_weights, "config": geometry,
              "weight_precision": precision,
              "activation_precision": atype, "rope": _effective_rope(text_config, execution),
              "math_version": math_version}
    vision = _classified(config.get("vision_config") or {}, "vision_config")
    # The real preprocessing routine supplies effective defaults; whitespace,
    # irrelevant processor metadata and tokenizer files cannot split identities.
    try:
        from .qwen35_multimodal_native import get_qwen35_multimodal_config
    except ImportError:
        from qwen35_multimodal_native import get_qwen35_multimodal_config
    for filename in ("preprocessor_config.json", "video_preprocessor_config.json"):
        path = target_plan["root"] / filename
        if path.is_file():
            settings = _classified(json.loads(path.read_text(encoding="utf-8")), "processor")
            if "size" in settings and (not isinstance(settings["size"], dict) or
                    set(settings["size"]) - {"shortest_edge", "longest_edge"}):
                raise ValueError("Unclassified processor size configuration")
    processor = get_qwen35_multimodal_config(str(target_plan["root"]), config)
    vision_defaults = {"depth": 0, "hidden_size": 0, "num_heads": 0, "intermediate_size": 0,
                       "patch_size": 16, "temporal_patch_size": 2, "spatial_merge_size": 2,
                       "out_hidden_size": target["config"]["hidden_size"], "num_position_embeddings": 0,
                       "deepstack_visual_indexes": []}
    encoder = {
        "tensors": encoder_weights,
        "config": {key: vision.get(key, value) for key, value in vision_defaults.items()},
        "processor": processor, "math_version": "qwen35-vision-v1",
        "activation_precision": atype, "weight_precision": _weight_precision(encoder_weights, execution),
    }
    manifest = {
        "scope_version": SCOPE_VERSION,
        "format_version": STORAGE_FORMAT_V4,
        "target": target, "encoder": encoder,
        "draft": None if draft_identity == "none" else {"identity": draft_identity},
        "execution": {"runtime": runtime,
            "options": {key: value for key, value in execution.items() if key != "api_key"}},
        "encoding": {"kv_dtype": kv_dtype,
                     "container_version": STORAGE_FORMAT_V4,
                     "conversion_version": "fp16-to-fp8-v1"},
        "state_abi": copy.deepcopy(STATE_ABI),
    }
    manifest["storage_manifest_schema_version"] = STORAGE_MANIFEST_SCHEMA_V4
    manifest["storage_format_version"] = STORAGE_FORMAT_V4
    manifest["target_identity"] = _digest(target)
    manifest["encoder_identity"] = _digest({"encoder": encoder, "cache_namespace": CACHE_NAMESPACE})
    manifest["draft_identity"] = draft_identity
    effective_fields = {
        "target.tensors": ("base", 1, _digest(target_weights), "Canonical target tensor names and complete bytes"),
        "target.config": ("base", 1, target["config"], "Native effective geometry"),
        "target.precision": ("base", 2, {"weights": target["weight_precision"], "activation": atype}, "Effective computation precision"),
        "target.math_version": ("base", 2, math_version, "Target arithmetic revision and TP summation order"),
        "target.rope": ("base", 3, target["rope"], "Effective RoPE/MRoPE arithmetic"),
        "state_abi": ("base", 4, manifest["state_abi"], "Semantic state ABI"),
        "encoding": ("encoding", 4, manifest["encoding"], "Explicit storage/conversion variant"),
        "encoder": ("prefix", 5, manifest["encoder_identity"], "Only affects the prefix after media insertion"),
        "execution": ("runtime-only", None, manifest["execution"]["options"], "Diagnostic, not a cache namespace"),
    }
    manifest["fields"] = [{"name": name, "scope": scope, "category": category,
                           "effective_value": value, "reason": reason, "normalizer_version": 1}
                          for name, (scope, category, value, reason) in effective_fields.items()]
    return manifest


def identity_keys(manifest):
    semantic = {
        "target": manifest["target"],
        "state_abi": manifest["state_abi"],
        "cache_namespace": CACHE_NAMESPACE,
    }
    family_identity = _digest(semantic)
    # The native Qwen3.5 adapter derives its concrete target-store identity
    # from the semantic family and the persisted KV dtype. Keep the exported
    # identity identical so diagnostics and native catalog lookups describe the
    # same namespace; conversion provenance remains a checkpoint-level field.
    identity = hashlib.sha256(
        (family_identity + ":target-state-v4:" +
         manifest["encoding"]["kv_dtype"]).encode("utf-8")
    ).hexdigest()
    return {"identity": identity, "family_identity": family_identity,
            "target_identity": manifest["target_identity"],
            "encoder_identity": manifest["encoder_identity"],
            "draft_identity": manifest["draft_identity"]}


STORAGE_CONFIG_FIELDS = ("prefix_cache_storage_version",)


def _group_storage_config(execution):
    """归入 storage-config 子组（SSD_CACHE_REWRITE_PLAN.md §13.1 第 2 条）。

    存储版本是 runtime-only 配置：它必须出现在 execution
    记录里可审计，但不进入 identity_keys()/family_identity/模型数学 hash。
    """
    subgroup = {}
    for name in STORAGE_CONFIG_FIELDS:
        if name in execution:
            subgroup[name] = execution.pop(name)
    if subgroup:
        execution["storage_config"] = subgroup
    return execution


def _execution_record(args):
    values = _classified(vars(args), "execution")
    kvmem = values.get("kvmem")
    if kvmem is not None:
        options = json.loads(kvmem)
        if not isinstance(options, dict):
            raise ValueError("--kvmem must be a JSON object")
        enabled = options.get("enabled", True)
        if not isinstance(enabled, bool):
            raise ValueError("--kvmem enabled must be bool")
        # KVMem selects a sparse attention view and disables prefix caching.
        # Its state cannot share the ordinary full-attention SSD namespace.
        # Reject it before hashing weights or opening storage; do not silently
        # bypass either feature or mint an unverified compatibility identity.
        if enabled:
            raise ValueError("Persistent prefix cache cannot be combined with enabled --kvmem "
                             "sparse attention")
    execution = {key: copy.deepcopy(value) for key, value in values.items() if key != "api_key"}
    for key in ("dtype_config",):
        value = execution.get(key)
        if isinstance(value, str) and value and os.path.isfile(value):
            execution[key] = Path(value).read_text(encoding="utf-8")
    execution["env"] = {key: os.environ[key] for key in ENV_FIELDS if key in os.environ}
    return _group_storage_config(execution)


def _native_manifest_function(native_library):
    function = getattr(native_library, "set_persistent_prefix_manifest", None)
    if function is None:
        raise RuntimeError("Native persistent-cache v4 manifest metadata support is required")
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_int
    return function


def _storage_capability_function(native_library):
    function = getattr(native_library, "get_persistent_prefix_cache_storage_capabilities", None)
    if function is None:
        return None
    function.argtypes = []
    function.restype = ctypes.c_uint32
    return function


def _storage_config_function(native_library):
    function = getattr(native_library, "set_persistent_prefix_cache_storage_config", None)
    if function is None:
        return None
    function.argtypes = [ctypes.c_char_p]
    function.restype = ctypes.c_int
    return function


def _storage_config_error(native_library):
    function = getattr(native_library, "get_persistent_prefix_cache_storage_config_error", None)
    if function is None:
        return ""
    function.argtypes = []
    function.restype = ctypes.c_char_p
    value = function()
    return value.decode("utf-8", "replace") if value else ""


def _requested_storage(args):
    raw = getattr(args, "prefix_cache_storage_version", None)
    version = STORAGE_FORMAT_V4 if raw is None else int(raw)
    if version != STORAGE_FORMAT_V4:
        raise ValueError("prefix_cache_storage_version supports only v4")
    return version


def _negotiate_storage(args, native_library):
    """Require a v4 writer before hashing weights or opening a cache directory."""
    version = _requested_storage(args)
    capability = _storage_capability_function(native_library)
    config_function = _storage_config_function(native_library)
    if capability is None or config_function is None:
        raise RuntimeError("the loaded native library does not support the v4 prefix-cache writer "
                           "(missing storage capability/config ABI)")
    capabilities = int(capability())
    required = STORAGE_CAP_READ_V4 | STORAGE_CAP_WRITE_V4 | STORAGE_CAP_CONFIG
    if capabilities & required != required:
        raise RuntimeError("the loaded native library does not support the v4 prefix-cache writer "
                           "(storage capabilities=0x%x, need READ_V4|WRITE_V4|STORAGE_CONFIG)" % capabilities)
    return version, capabilities


def configure_persistent_prefix(args, is_qwen35_model, library_path, native_library=None):
    """Validate and export startup settings; use the context manager in loaders."""
    for name in _DERIVED_ENV:
        os.environ.pop(name, None)
    directory = getattr(args, "prefix_cache_dir", None)
    if directory is None:
        directory = os.environ.get("FASTLLM_PREFIX_CACHE_DIR", "")
    if not directory:
        os.environ.pop("FASTLLM_PREFIX_CACHE_DIR", None)
        return None
    prefix_cache = str(getattr(args, "prefix_cache", "") or
                       os.environ.get("FASTLLM_PREFIX_CACHE", "true")).lower()
    if prefix_cache in ("0", "off", "false"):
        # A cache-off control must not be forced through the persistent backend
        # merely because the parser supplied its default directory.
        os.environ.pop("FASTLLM_PREFIX_CACHE_DIR", None)
        return None
    if not is_qwen35_model:
        raise ValueError("Persistent prefix v4 requires dense Qwen3.5/Qwen3.8 target weights")
    if "FASTLLM_PREFIX_CACHE_FILE_BYTES" in os.environ:
        raise ValueError("FASTLLM_PREFIX_CACHE_FILE_BYTES is removed: v4 has no container file limit")
    storage_format, capabilities = _negotiate_storage(args, native_library)
    # Reject programmatic callers that bypass the CLI with removed v3 fields.
    execution = _execution_record(args)
    for name in ("lora", "custom", "ori", "mmproj"):
        if getattr(args, name, None):
            raise ValueError("Persistent cache cannot safely identify this weight/graph override: " + name)
    disk_gb = getattr(args, "prefix_cache_disk_gb", None)
    if disk_gb is not None:
        if (not math.isfinite(disk_gb) or disk_gb <= 0 or
                disk_gb > ((1 << 63) - 1) / (1 << 30)):
            raise ValueError("prefix_cache_disk_gb must fit a finite positive signed 64-bit byte quota")
        disk_bytes = int(disk_gb * (1 << 30))
    else:
        try:
            disk_bytes = int(os.environ.get("FASTLLM_PREFIX_CACHE_DISK_BYTES", DEFAULT_DISK_BYTES))
        except ValueError as error:
            raise ValueError("FASTLLM_PREFIX_CACHE_DISK_BYTES must be a positive integer") from error
    if not 0 < disk_bytes <= (1 << 63) - 1:
        raise ValueError("Persistent cache quota must fit a positive signed 64-bit byte count")
    policy = getattr(args, "prefix_cache_restore_policy", None)
    if policy is None:
        policy = os.environ.get("FASTLLM_PREFIX_CACHE_RESTORE_POLICY", "auto")
    if policy not in ("auto", "always", "never"):
        raise ValueError("prefix_cache_restore_policy must be auto, always or never")
    restore_timeout_ms = getattr(args, "prefix_cache_restore_timeout_ms", None)
    if restore_timeout_ms is None:
        restore_timeout_ms = os.environ.get("FASTLLM_PREFIX_CACHE_RESTORE_TIMEOUT_MS", "100000")
    try:
        restore_timeout_ms = int(restore_timeout_ms)
    except (TypeError, ValueError) as error:
        raise ValueError("prefix_cache_restore_timeout_ms must be a positive integer") from error
    if not 0 < restore_timeout_ms <= 2147483647:
        raise ValueError("prefix_cache_restore_timeout_ms must fit a positive native 32-bit millisecond value")
    l2_mode = str(getattr(args, "prefix_cache_l2", "off") or "off").lower()
    l2_gb = float(getattr(args, "prefix_cache_l2_gb", 4.0))
    if l2_mode not in ("on", "off") or not math.isfinite(l2_gb) or not (1 / 1024 <= l2_gb <= 2097151):
        raise ValueError("prefix_cache_l2 must be on/off and prefix_cache_l2_gb must be finite and between 1 MiB and 2097151 GiB")
    draft_path = getattr(args, "speculative_draft_model_path", "") or None
    # §13.1 order: capability/config preflight first, then the semantic manifest,
    # then the manifest setter and the storage-config setter. Native must not
    # open, recover or create any directory before all three have been accepted.
    config_function = _storage_config_function(native_library)
    root = Path(directory).expanduser().resolve()
    configure_prefix_cache_logging(args)
    prefix_cache_info("[Prefix SSD] Checking model files for prefix-cache compatibility...")
    prefix_cache_debug("[Prefix SSD] hashing target and speculative semantic contents for storage format v%d..." %
                       storage_format)
    manifest = build_identity(args.path, execution, library_path, draft_path,
                              progress=prefix_cache_log_level() == "debug", storage_format=storage_format)
    keys = identity_keys(manifest)
    if not _native_manifest_function(native_library)(_canonical(manifest)):
        raise RuntimeError("Native persistent-cache manifest initialization failed")
    if config_function is not None:
        storage_config = json.dumps({"writer_format": storage_format})
        # ctypes c_char_p 需要 bytes；传 str 会直接抛 ArgumentError。
        if not config_function(storage_config.encode("utf-8")):
            raise RuntimeError(
                "Native persistent-cache storage configuration rejected: " +
                (_storage_config_error(native_library) or "unknown"))
    exported = {"FASTLLM_PREFIX_CACHE_" + name.upper(): value for name, value in keys.items()}
    exported.update({"FASTLLM_PREFIX_CACHE_DIR": str(root),
                     "FASTLLM_PREFIX_CACHE_DISK_BYTES": str(disk_bytes),
                     "FASTLLM_PREFIX_CACHE_RESTORE_POLICY": policy,
                     "FASTLLM_PREFIX_CACHE_RESTORE_TIMEOUT_MS": str(restore_timeout_ms),
                     "FASTLLM_PREFIX_CACHE_L2_HOST_KV": "1" if l2_mode == "on" else "0",
                     "FASTLLM_PREFIX_CACHE_L2_HOST_KV_MB": str(int(l2_gb * 1024))})
    os.environ.update(exported)
    prefix_cache_info("[Prefix SSD] configured: directory=%s quota=%.1f GiB store=%s restore=%s L2=%s log_level=%s." %
                      (root, disk_bytes / (1 << 30), "v4" if os.environ.get("FASTLLM_PREFIX_V5") == "0" else "v5",
                       policy, l2_mode, prefix_cache_log_level()))
    prefix_cache_debug("[Prefix SSD] identity=%s, family=%s, manifest_format=v%d, directory=%s, quota=%d, "
          "restore=%s, restore_timeout_ms=%d, read=request_gated_background, capabilities=0x%x" %
          (keys["identity"], keys["family_identity"], storage_format, root, disk_bytes,
           policy, restore_timeout_ms, capabilities))
    return exported


@contextmanager
def persistent_prefix_environment(args, is_qwen35_model, library_path, native_library=None):
    """Scope native constructor env while preserving the user's raw settings."""
    with _ENV_LOCK:
        original = {name: os.environ.get(name) for name in _INPUT_ENV}
        try:
            yield configure_persistent_prefix(args, is_qwen35_model, library_path, native_library)
        finally:
            manifest_function = getattr(native_library, "set_persistent_prefix_manifest", None)
            if manifest_function is not None:
                _native_manifest_function(native_library)(None)
            for name in _DERIVED_ENV:
                os.environ.pop(name, None)
            for name, value in original.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
