"""Identify local service models from release metadata without loading weights."""
import json
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

from .gguf_metadata import read_gguf_metadata


class ModelService(NamedTuple):
    backend: str
    error: str = ""


CHAT = ModelService("llm")
H3 = ModelService("minimax_h3")
IMAGE = ModelService("minimax_h3_image")
RAW_VIDEO = ModelService(H3.backend, "视频 DiT 单文件需要配套组件，请选择含 h3_quantization.json 的模型目录。")
PIPELINES = {"MiniMaxH3ModularPipeline": H3, "MiniMaxH3ImagePipeline": IMAGE,
             "QwenImage21Pipeline": ModelService("qwen_image")}


def _json(path):
    try:
        with path.open("rb") as stream:
            data = json.loads(stream.read(1024 * 1024))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


@lru_cache(maxsize=128)
def _weight(path, size, mtime):
    if path.suffix == ".gguf":
        if read_gguf_metadata(path, ["general.architecture"]).get("general.architecture") != "wan":
            return CHAT
        meta = read_gguf_metadata(path, ["config", "comfy.gguf.orig_shape.adaln_t_table"])
        config = json.loads(meta.get("config", "{}")).get("transformer", {})
        if config.get("pdd_steps") and config.get("pdd_sigmas"):
            return IMAGE
        if "comfy.gguf.orig_shape.adaln_t_table" in meta:
            return RAW_VIDEO
        return ModelService("unsupported", "当前服务没有识别出这个 Wan GGUF 所需的完整模型。")
    with path.open("rb") as stream:
        length = int.from_bytes(stream.read(8), "little")
        if not 0 < length <= min(size - 8, 16 << 20):
            return CHAT
        header = json.loads(stream.read(length))
    if not isinstance(header, dict):
        return CHAT
    if "pdd_proj_out.0.weight" in header:
        return IMAGE
    if "adaln_t_table" in header and "video_patch_proj.weight" in header:
        shape = header.get("final_layer.video_out.weight", {}).get("shape")
        if shape == [576, 5376]:
            return IMAGE
        if shape == [96, 5376]:
            return RAW_VIDEO
    return CHAT


def _weight_service(path):
    try:
        stat = path.stat()
        return _weight(path, stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError, OverflowError, TypeError, AttributeError):
        return CHAT


def has_pika_adapter(root):
    return any((root / task / name).is_file()
               for task in ("fl2va", "ref2va") for name in ("manifest.json", "adapter.safetensors"))


def _bundle(root):
    if (root / "h3_quantization.json").is_file():
        return H3
    if _json(root / "release.json").get("schema") == "duo-release-v1" or (
            (root / "intent_adapter").is_dir() and (root / "realization").is_dir()):
        if not (root / "realization/diffusion_pytorch_model.safetensors.index.json").is_file():
            return ModelService("duo", "Duo 实时服务需要 FP8 Realization，请选择 distill-quant 模型根目录。")
        return ModelService("duo")
    pipeline = _json(root / "model_index.json").get("_class_name")
    if isinstance(pipeline, str) and pipeline in PIPELINES:
        return PIPELINES[pipeline]
    if pipeline:
        return ModelService("unsupported", f"当前版本没有 {pipeline} 的模型服务。")
    if (root / "pdd_config.json").is_file() and (root / "transformer").is_dir():
        return IMAGE
    transformer = _json(root / "transformer/config.json").get("_class_name")
    if transformer == "MiniMaxH3Transformer3DModel":
        return H3
    if transformer == "QwenImage21Transformer2DModel":
        return ModelService("qwen_image")
    if has_pika_adapter(root):
        return H3
    if (root / "DiT/distilled").is_dir() and (root / "VAE/PixelUMM_FT").is_dir():
        return ModelService("hy_nomas")
    if (root / "transformers/model.safetensors.index.json").is_file() and (root / "google/umt5-xxl").is_dir():
        return ModelService("lingbot_world")
    return None


def identify_model(model):
    if not model or not model.strip():
        return CHAT
    path = Path(model.strip()).expanduser()
    if not path.exists():
        return CHAT  # Remote LLM identifiers keep the normal download path.
    if path.name == "h3_quantization.json" and path.is_file():
        return H3
    components = {"realization", "intent", "intent_adapter", "codec", "speech", "DiT", "VAE",
                  "transformers", "transformer", "transformer_ref", "text_encoder", "vae", "fl2va", "ref2va"}
    for component in (path, path.parent, path.parent.parent):
        if component.name in components:
            root = component.parent
            bundle = _bundle(root)
            if bundle:
                return ModelService(bundle.backend, f"所选路径是模型组件，请选择完整目录 {root}。")
    if path.is_file():
        return _weight_service(path) if path.suffix in {".gguf", ".safetensors"} else CHAT
    bundle = _bundle(path)
    if bundle:
        return bundle
    if (path / "config.json").is_file() or (path / "model.safetensors.index.json").is_file():
        return CHAT
    try:
        files = [p for folder in (path, path / "diffusion_models") if folder.is_dir()
                 for p in folder.iterdir() if p.suffix in {".gguf", ".safetensors"} and p.is_file()]
    except OSError:
        return CHAT
    for weight in files:
        family = _weight_service(weight)
        if family == IMAGE and len(files) > 1:
            return ModelService(IMAGE.backend, "目录包含多个权重，请选择具体的 .gguf 或 .safetensors 文件。")
        if family != CHAT:
            return family
    return CHAT
