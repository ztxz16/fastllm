"""Configuration and capacity accounting for DeepSeek V4.1 GGUF add-ons."""

from .gguf_metadata import read_gguf_metadata, read_gguf_tensor_info, _V41_GGUF_ALIASES


def dspark_config(path, target=None):
    keys = {"general.architecture", "tokenizer.ggml.mask_token_id"}
    keys.update("dflash." + key for key in _V41_GGUF_ALIASES.values())
    keys.update("dflash." + key for key in ("block_size", "target_layers"))
    meta = read_gguf_metadata(path, keys)
    tensors = read_gguf_tensor_info(path)
    if meta.get("general.architecture") != "dflash":
        raise ValueError("DeepSeek DSpark GGUF requires dflash architecture")
    layers = int(meta.get("dflash.block_count", 0))
    hidden = int(meta.get("dflash.embedding_length", 0))
    targets = meta.get("dflash.target_layers", [])
    markov = tensors.get("markov_w1.weight", ((), 0))[0]
    if (layers <= 0 or hidden <= 0 or len(markov) != 2 or
            tensors.get("markov_w2.weight", ((), 0))[0] != markov or
            tensors.get("fc.weight", ((), 0))[0] != (hidden, hidden * len(targets)) or
            int(meta.get("dflash.block_size", 0)) <= 0 or not targets or
            not 0 <= meta.get("tokenizer.ggml.mask_token_id", -1) < markov[0] or
            meta.get("dflash.attention.compress_ratios") != [0] * layers):
        raise ValueError("invalid DeepSeek DSpark GGUF configuration/tensor dimensions")
    for stage in range(layers):
        for name in ("attn_q_a", "attn_q_b", "attn_kv", "ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"):
            if "blk.%d.%s.weight" % (stage, name) not in tensors:
                raise ValueError("incomplete DeepSeek DSpark GGUF stage %d" % stage)
    config = dict(num_nextn_predict_layers=layers,
                  dspark_block_size=meta["dflash.block_size"],
                  dspark_target_layer_ids=targets,
                  dspark_noise_token_id=meta["tokenizer.ggml.mask_token_id"],
                  dspark_markov_rank=markov[1], vocab_size=markov[0],
                  dspark_n_routed_experts=meta.get("dflash.expert_count", 0),
                  dspark_num_experts_per_tok=meta.get("dflash.expert_used_count", 0))
    if target is not None:
        if target.get("gguf_architecture") != "deepseek41":
            raise ValueError("DeepSeek DSpark GGUF requires a DeepSeek V4.1 GGUF target")
        for key in ("hidden_size", "num_attention_heads", "head_dim", "qk_rope_head_dim",
                    "q_lora_rank", "o_lora_rank", "o_groups", "hc_mult", "moe_intermediate_size"):
            if target.get(key) != meta.get("dflash." + _V41_GGUF_ALIASES[key]):
                raise ValueError("DeepSeek DSpark GGUF target mismatch: " + key)
        if any(not 0 <= i < target["num_hidden_layers"] for i in targets):
            raise ValueError("DeepSeek DSpark target layer is out of range")
    return config


def draft_tensor_name(name, layers):
    # The planner only needs stage ownership and expert/shared classification.
    if name.startswith("blk."):
        return "mtp." + name[4:]
    stage = 0 if name in ("fc.weight", "enc.output_norm.weight") else layers - 1
    return "mtp.%d.%s" % (stage, name)


def vision_config(path, target):
    aliases = {"hidden_size": "embedding_length", "intermediate_size": "feed_forward_length",
               "num_hidden_layers": "block_count", "num_attention_heads": "attention.head_count",
               "patch_size": "patch_size", "rms_norm_eps": "attention.layer_norm_epsilon",
               "downsample_ratio": "projector.scale_factor", "image_size": "image_size"}
    meta = read_gguf_metadata(path, {"general.architecture", "clip.projector_type",
        "clip.vision.projection_dim", *("clip.vision." + v for v in aliases.values())})
    if (target.get("gguf_architecture") != "deepseek41" or
            meta.get("general.architecture") != "clip" or
            meta.get("clip.projector_type") != "deepseek41v" or
            meta.get("clip.vision.projection_dim") != target.get("hidden_size")):
        raise ValueError("DeepSeek V4.1 mmproj architecture/projection dimension mismatch")
    config = {k: meta["clip.vision." + v] for k, v in aliases.items() if "clip.vision." + v in meta}
    if any(config.get(k, 0) <= 0 for k in ("hidden_size", "num_hidden_layers", "patch_size", "downsample_ratio")):
        raise ValueError("invalid DeepSeek V4.1 vision configuration")
    return {"vision_config": config}
