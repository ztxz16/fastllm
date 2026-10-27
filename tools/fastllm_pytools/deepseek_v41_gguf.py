"""Configuration and capacity accounting for DeepSeek V4.1 GGUF vision modules."""

from .gguf_metadata import read_gguf_metadata


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
