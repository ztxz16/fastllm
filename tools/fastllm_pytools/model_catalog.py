"""Curated browser downloads. Bundle paths are relative to the chosen model directory."""


MODELSCOPE_DOWNLOAD_GROUPS = (
    {
        "id": "deepseek-v41",
        "label": "DeepSeek · V4.1 Flash",
        "models": (
            {
                "preset": "deepseek-v41-flash-1017",
                "id": "deepseek-ai/DeepSeek-V4.1-Flash-1017",
                "label": "官方权重 · 1017",
                "downloadGB": 510,
                "meta": "混合推理",
                "include": "",
            },
            {
                "preset": "deepseek-v41-fastllm-3.0bit",
                "id": "huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-3.0bit",
                "label": "Fastllm · 3.0bit",
                "meta": "Fastllm 量化 · 完整仓库",
                "include": "",
            },
            {
                "preset": "deepseek-v41-fastllm-2.5bit",
                "id": "huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-2.5bit",
                "label": "Fastllm · 2.5bit",
                "meta": "Fastllm 量化 · 完整仓库",
                "include": "",
            },
            {
                "preset": "deepseek-v41-fastllm-2.0bit",
                "id": "huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-2.0bit",
                "label": "Fastllm · 2.0bit",
                "meta": "Fastllm 量化 · 完整仓库",
                "include": "",
            },
        ),
    },
    {
        "id": "qwen38-27b",
        "label": "Qwen3.8 · 27B",
        "models": (
            {
                "preset": "qwen38-27b-bf16",
                "id": "Qwen/Qwen3.8-27B",
                "label": "BF16",
                "meta": "原始质量",
                "include": "",
            },
            {
                "preset": "qwen38-27b-fp8",
                "id": "Qwen/Qwen3.8-27B-FP8",
                "label": "FP8",
                "downloadGB": 31,
                "meta": "",
                "include": "",
            },
            {
                "preset": "qwen38-27b-nvfp4",
                "id": "QUASAR-QAT/Qwen3.8-27B-QUASAR-NVFP4",
                "label": "QUASAR NVFP4",
                "meta": "显存友好",
                "include": "",
            },
        ),
    },
    {
        "id": "qwen38-27b-dflash",
        "label": "Qwen3.8-27B · DFlash2",
        "models": (
            {
                "preset": "qwen38-27b-dflash2",
                "id": "z-lab/Qwen3.8-27B-DFlash2",
                "label": "DFlash2",
                "meta": "推测解码草稿",
                "include": "",
            },
        ),
    },
    {
        "id": "qwen38-flash-gsq",
        "label": "Qwen3.8-Flash-Next · GSQ-RCO",
        "models": (
            {
                "preset": "flash-gsq-q2-0",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · Q2_0",
                "downloadGB": 66.4,
                "meta": "速度优先",
                "include": "Q2_0/*,mmproj-*.gguf",
            },
            {
                "preset": "flash-gsq-iq2-xs",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · IQ2_XS",
                "downloadGB": 68.0,
                "meta": "体积优先",
                "include": "IQ2_XS/*,mmproj-*.gguf",
            },
            {
                "preset": "flash-gsq-iq3-xxs",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · IQ3_XXS",
                "downloadGB": 75.8,
                "meta": "平衡",
                "include": "IQ3_XXS/*,mmproj-*.gguf",
            },
            {
                "preset": "flash-gsq-iq3-s",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF",
                "label": "GSQ-RCO · IQ3_S",
                "downloadGB": 83.6,
                "meta": "质量优先",
                "include": "IQ3_S/*,mmproj-*.gguf",
            },
        ),
    },
    {
        "id": "qwen38-flash-gsq-coder",
        "label": "Qwen3.8-Flash-Next · GSQ-RCO Coder",
        "models": (
            {
                "preset": "flash-gsq-coder-iq1-m",
                "id": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF",
                "label": "GSQ-RCO Coder · IQ1_M",
                "meta": "代码场景",
                "include": "IQ1_M/*,mmproj-*.gguf",
            },
        ),
    },
    {
        "id": "quick-check",
        "label": "Qwen3 · 0.6B",
        "models": (
            {
                "preset": "qwen3-0.6b",
                "id": "Qwen/Qwen3-0.6B",
                "label": "BF16",
                "meta": "快速验证",
                "include": "",
            },
        ),
    },
)


def source(repo, files, into="", rename=None, *, label="模型权重", required=""):
    return {"repo": repo, "label": label, "files": files.split(), "required": required.split(), "into": into, "rename": rename or {}}


def bundle(preset, label, sources, *, entry="", manifest=None, prepare="", note="", gb=None):
    return {"preset": preset, "id": sources[0]["repo"], "label": label, "sources": sources,
            "downloadGB": gb, "entry": entry, "manifest": manifest, "prepare": prepare, "meta": note}


H3 = "MiniMax/MiniMax-H3"
PIKA = "Pika-Labs/Pika-H3-Flex"
H3_IMAGE = "MiniMax/MiniMax-H3-Image"
QWEN_IMAGE = "Qwen/Qwen-Image-2.1"
COMMON = "model_index.json processor/* tokenizer/* text_encoder/* vae/*"
H3_FILES = COMMON + " scheduler/* audio_scheduler/* audio_vae/* transformer/* transformer_ref/*"
H3_IMAGE_FILES = COMMON + " pdd_config.json pdd_fused_heads.safetensors transformer/*"
QWEN_FILES = "model_index.json processor/* text_encoder/* vae/* scheduler/* transformer/config.json"
PIKA_FILES = "fl2va/manifest.json fl2va/adapter.safetensors ref2va/manifest.json ref2va/adapter.safetensors"

# A surviving config or README must not mask missing weights in a component directory.
ENCODER_REQUIRED = "text_encoder/config.json text_encoder/model.safetensors.index.json text_encoder/*.safetensors"
H3_REQUIRED = (ENCODER_REQUIRED + " tokenizer/tokenizer.json transformer/config.json "
               "transformer/*.safetensors.index.json transformer_ref/*.safetensors.index.json "
               "vae/*.safetensors.index.json audio_vae/*.safetensors")
IMAGE_REQUIRED = ENCODER_REQUIRED + " tokenizer/tokenizer.json transformer/config.json transformer/*.safetensors.index.json vae/*.safetensors"
QWEN_REQUIRED = ENCODER_REQUIRED + " processor/tokenizer.json vae/config.json vae/*.safetensors scheduler/scheduler_config.json"


def video_versions(pika):
    name = "pika-h3" if pika else "minimax-h3"
    adapters = [source(PIKA, PIKA_FILES, label="六步适配器")] if pika else []
    base = source(H3, H3_FILES, "base_model", label="BF16 基座", required=H3_REQUIRED)
    if pika:
        # Selected ModelScope files matching the H3 release pinned in the Pika manifests.
        base["checksum"] = "bc1c39dafc49c5b0d802aded0f152b64989292b7431eb76afb580d22d7fd824a"
    versions = [bundle(name + "-bf16", "BF16", adapters + [base if pika else source(H3, H3_FILES, required=H3_REQUIRED)], gb=215.9 if pika else 210.3)]
    for key, label, repo, names, gb in (
        ("fp8", "FP8", "Comfy-Org/MiniMax-H3",
         "diffusion_models/minimax_h3_{}_pruned_fp8_scaled.safetensors", (252.2, 257.8)),
        ("int8", "INT8 ConvRot", "Comfy-Org/MiniMax-H3",
         "diffusion_models/minimax_h3_{}_pruned_int8_convrot.safetensors", (252.3, 257.8)),
        ("nvfp4", "NVFP4 / INT8", "rockerBOO/minimax-h3-nvfp4-convrot",
         "minimax_h3_{}_pruned_nvfp4_convrot_int8.safetensors", (250.5, 256.0)),
        ("q4km", "GGUF Q4_K_M", "Abiray/MiniMax-H3-Pruned-GGUF",
         "MiniMax-H3-{}-Pruned-Q4_K_M.gguf", (233.5, 239.0)),
    ):
        tasks = ("FL2VA", "Ref2VA") if key == "q4km" else ("fl2va", "ref2va")
        files = {task.lower(): names.format(task) for task in tasks}
        manifest = {"format_version": 1, "base_model": "base_model", **files}
        if pika:
            manifest["pika"] = "."
        versions.append(bundle(name + "-" + key, label, adapters + [source(repo, " ".join(files.values()), label="量化 DiT"), base],
                               manifest=manifest, gb=gb[1 if pika else 0], note="DiT · 含完整 BF16 基座"))
    return versions


def image_versions():
    versions = [bundle("h3-image-bf16", "BF16", [source(H3_IMAGE, H3_IMAGE_FILES, required=IMAGE_REQUIRED)], gb=135.4)]
    for key, label, gb, repo, filename in (
        ("fp8", "FP8", 155.6, "unsloth/MiniMax-H3-Image-FP8", "MiniMax-H3-Image-FP8.safetensors"),
        ("int8", "INT8 ConvRot", 155.7, "unsloth/MiniMax-H3-Image-FP8", "MiniMax-H3-Image-INT8-ConvRot.safetensors"),
        ("q4km", "GGUF Q4_K_M", 147.0, "Abiray/MiniMax-H3-Image-GGUF", "MiniMax-H3-Image-Pruned-Q4_K_M.gguf"),
    ):
        versions.append(bundle("h3-image-" + key, label,
                               [source(repo, filename, label="量化 DiT"), source(H3_IMAGE, H3_IMAGE_FILES, "base_model", label="BF16 基座", required=IMAGE_REQUIRED)],
                               entry=filename, gb=gb, note="DiT · 含完整 BF16 基座"))
    # The native NVFP4 release is public on Hugging Face. Pin the LFS object and
    # commit so a moving branch can never be combined with an old partial file.
    filename = "minimax_h3_image_pruned_nvfp4.safetensors"
    quant = source("LSDJesus/MiniMax-H3-Image-fp8-nvfp4", filename, label="量化 DiT")
    quant.update(hub="Hugging Face", revision="1d810bd38052f3b28792f351e7b1988ac0cefee3",
                 bytes=12537612944, sha256="9afd227d6cf8594db996b7feafb4b1c6dc3ca4df2fd5b9d6bc2ebb212b8b7703")
    versions.append(bundle("h3-image-nvfp4", "NVFP4", [quant, source(H3_IMAGE, H3_IMAGE_FILES, "base_model", label="BF16 基座", required=IMAGE_REQUIRED)],
                           entry=filename, gb=147.9, note="DiT · 含完整 BF16 基座"))
    return versions


def qwen_image_versions():
    versions = [bundle("qwen-image-bf16", "BF16", [source(QWEN_IMAGE, QWEN_FILES, label="编码器、VAE 与配置", required=QWEN_REQUIRED),
                                                         source(QWEN_IMAGE, "transformer/*.safetensors*", label="BF16 DiT", required="transformer/*.safetensors.index.json")], gb=33.1)]
    for key, label, gb, repo, filename in (
        ("fp8", "FP8", 26.0, "unsloth/Qwen-Image-2.1-FP8", "Qwen-Image-2.1-FP8.safetensors"),
        ("int8", "INT8 ConvRot", 26.2, "Comfy-Org/Qwen-Image-2.1", "diffusion_models/qwen_image_2.1_int8_convrot.safetensors"),
        ("q4km", "GGUF Q4_K_M", 23.1, "unsloth/Qwen-Image-2.1-GGUF", "qwen-image-2.1-Q4_K_M.gguf"),
    ):
        target = filename if filename.endswith(".gguf") else "diffusion_pytorch_model.safetensors"
        versions.append(bundle("qwen-image-" + key, label,
                               [source(repo, filename, "transformer", {filename: target}, label="量化 DiT"),
                                source(QWEN_IMAGE, QWEN_FILES, label="编码器、VAE 与配置", required=QWEN_REQUIRED)], gb=gb, note="DiT · 原版编码器与 VAE"))
    return versions


MODELSCOPE_DOWNLOAD_GROUPS += (
    {"category": "image", "id": "qwen-image", "label": "Qwen-Image-2.1", "models": qwen_image_versions()},
    {"category": "image", "id": "h3-image", "label": "MiniMax H3 Image", "models": image_versions()},
    {"category": "video", "id": "minimax-h3", "label": "MiniMax H3", "models": video_versions(False)},
    {"category": "video", "id": "pika-h3", "label": "Pika H3 Flex", "models": video_versions(True)},
    {"category": "world", "id": "hy-nomas", "label": "HY-Nomas", "models": (
        bundle("hy-nomas-original", "FP32 DiT", [
            source("Tencent-Hunyuan/Hy-Nomas", "DiT/distilled/diffusion_pytorch_model_ema_0_995.safetensors "
                   "VAE/PixelUMM_FT/diffusion_pytorch_model.safetensors", label="DiT 与 VAE"),
            source("Qwen/Qwen2.5-VL-7B-Instruct", "tokenizer.json model.safetensors.index.json model-*.safetensors", "text_encoder", label="文本编码器")], gb=82.4, note="原版 DiT 与 VAE 为 FP32，推理使用 BF16。"),)},
    {"category": "world", "id": "lingbot-world", "label": "LingBot World 2.0 · 1.3B", "models": (
        bundle("lingbot-world-1.3b", "FP32 DiT · Causal Fast", [
            source("robbyant/lingbot-world-v2-1.3b-causal-fast", "model.safetensors.index.json model-*.safetensors", "transformers", label="1.3B DiT"),
            source("robbyant/lingbot-world-v2-14b-causal-fast", "google/umt5-xxl/tokenizer.json "
                   "models_t5_umt5-xxl-enc-bf16.pth Wan2.1_VAE.pth", label="分词器、T5 与 VAE")],
               prepare="lingbot", gb=18.7, note="下载后在 CPU 转换 T5 与 VAE，需要 torch 和 safetensors。"),)},
    {"category": "dialogue", "id": "duo", "label": "Duo 1.0", "models": (
        bundle("duo-distill-fp8", "Distill FP8", [
            source("inclusionAI/Duo-1.0-distill-quant", "release.json realization/*", label="FP8 Realization",
                   required="realization/diffusion_pytorch_model.safetensors.index.json realization/config.json"),
            source("inclusionAI/Duo-1.0-causal-base", "intent/* intent_adapter/* codec/* speech/*", label="Intent、Codec 与语音组件",
                   required="intent/model.safetensors.index.json intent/tokenizer.json intent_adapter/adapter_model.safetensors "
                            "codec/video_vae/model.safetensors codec/audio_vae/diffusion_pytorch_model.safetensors "
                            "speech/recognizer/model.safetensors.index.json speech/forced-aligner/model.safetensors")],
               gb=131.4, note="FP8 Realization · 原版 Intent、Codec 与语音组件"),)},
)

DOWNLOAD_BUNDLES = {model["preset"]: model for group in MODELSCOPE_DOWNLOAD_GROUPS
                    for model in group["models"] if "sources" in model}
