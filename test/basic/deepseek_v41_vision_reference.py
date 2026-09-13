#!/usr/bin/env python3
"""
DeepSeek-V4.1 图文（视觉输入）端到端数值对齐测试。

在 deepseek_v41_reference.py 的基础上：
  1. 迷你模型带上小规模视觉塔（ViT + aligner + image_start/end/newline），随机权重；
     ``--real-vision <model_dir>`` 时改用真实 checkpoint 的 vision.* / aligner.* / image_* 权重（文本侧仍为迷你模型），
     用于验证 ViT + aligner 与官方 vision.py 在真实权重下的一致性；
  2. 用程序合成的图片（不依赖任何二进制资源）构造带图像占位符的 prompt，
     官方 image_processor.prepare_vl_inputs 与 ftllm.deepseek_v41_multimodal 各自展开，先比较 token / patch 是否完全一致；
  3. 官方 model.py（纯 torch kernel shim）做图文 prefill + 贪心 decode，fastllm 走 launch_response_llm_model_multimodal，
     逐步比较 logits；``--dump-dir`` 时另外比较 aligner 输出与合并后的输入嵌入。

用法（21 上）：
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=build/tools python test/basic/deepseek_v41_vision_reference.py \
      --work-dir /root/v41-tiny-vision --tokenizer-dir /root/v41-tokenizer \
      --reference-dir /mnt/shared2/models/DeepSeek-V4.1-Flash/inference --no-fake-quant --regenerate
  加 ``--real-vision /mnt/shared2/models/DeepSeek-V4.1-Flash --image-size 640x480`` 验证真实视觉权重。
"""

import argparse
import ctypes
import io
import json
import os
import struct
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import deepseek_v41_reference as ref  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--reference-dir", required=True, help="官方 inference/ 目录")
    parser.add_argument("--decode", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--moe-device", default="cuda")
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--regenerate", action="store_true")
    parser.add_argument("--skip-reference", action="store_true")
    parser.add_argument("--skip-fastllm", action="store_true")
    parser.add_argument("--fastllm-only-load", action="store_true")
    parser.add_argument("--dump-dir", default="")
    parser.add_argument("--no-fake-quant", action="store_true")
    parser.add_argument("--experts", type=int, default=2)
    parser.add_argument("--activated", type=int, default=2)
    parser.add_argument("--index-topk", type=int, default=128)
    parser.add_argument("--candidate-topk-blocks", type=int, default=64)
    parser.add_argument("--chunked-prefill", type=int, default=-1)
    parser.add_argument("--quant-format", default="bf16", choices=["bf16"])
    parser.add_argument("--real-vision", default="", help="真实 checkpoint 目录，复制其 vision.*（ViT）权重；aligner / 分隔符嵌入仍为随机")
    parser.add_argument("--image-size", default="112x70,84x126",
                        help="合成图片尺寸列表（宽x高，逗号分隔），每张图对应 prompt 中一个占位符")
    parser.add_argument("--min-pixels", type=int, default=-1, help="覆盖 vision_min_pixels（默认迷你模式 0，真实视觉模式取官方值）")
    parser.add_argument("--bias-vl-boost", type=float, default=0.0,
                        help="生成后给每层 gate.bias_vl 的最后两个专家加上该值，使图像 token 必然选中它们，用于验证 bias_vl 路由")
    parser.add_argument("--ref-vision-fp32", action="store_true",
                        help="参考侧的 ViT / aligner 用 float32 计算（官方默认 bf16），用于区分实现偏差与 bf16 舍入噪声")
    return parser.parse_args()


TINY_VISION = dict(vision_n_layers=2, vision_dim=64, vision_n_heads=2, vision_inter_dim=96, vision_patch_size=14,
                   vision_rope_theta=10000.0, vision_downsample_ratio=3, vision_max_n_token=1024,
                   vision_min_pixels=0, vision_max_wh_ratio=None)
REAL_VISION = dict(vision_n_layers=32, vision_dim=1024, vision_n_heads=16, vision_inter_dim=2816, vision_patch_size=14,
                   vision_rope_theta=10000.0, vision_downsample_ratio=3, vision_max_n_token=1024,
                   vision_min_pixels=295936, vision_max_wh_ratio=None)
VISION_KEYS = tuple(TINY_VISION.keys()) + ("image_token_id",)


# ---------------- 官方模块（真实 vision / image_processor）----------------

def install_vision_shims(reference_dir):
    """先装 deepseek_v41_reference 的 kernel shim，再换回真实的 vision / image_processor 模块。"""
    import importlib
    ref.install_shims(reference_dir)
    encoding_dir = os.path.join(os.path.dirname(os.path.abspath(reference_dir.rstrip("/"))), "encoding")
    if os.path.isdir(encoding_dir) and encoding_dir not in sys.path:
        sys.path.insert(0, encoding_dir)
    for name in ("vision", "image_processor", "engram", "model"):
        sys.modules.pop(name, None)
    vision = importlib.import_module("vision")
    image_processor = importlib.import_module("image_processor")
    engram = importlib.import_module("engram")
    model = importlib.import_module("model")
    assert model.ViT is vision.ViT
    return engram, model, image_processor


_ORIG_HF_CONFIG = ref.hf_config_from_tiny


def hf_config_with_vision(t, engram_num_embeddings, compressed_vocab):
    cfg = _ORIG_HF_CONFIG(t, engram_num_embeddings, compressed_vocab)
    for key in VISION_KEYS:
        cfg[key] = t[key]
    return cfg


# ---------------- 合成图片 ----------------

def make_synthetic_image(width, height, seed):
    from PIL import Image, ImageDraw
    rng = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:height, 0:width]
    r = (255.0 * xx / max(1, width - 1)).astype(np.uint8)
    g = (255.0 * yy / max(1, height - 1)).astype(np.uint8)
    b = ((xx + yy) % 256).astype(np.uint8)
    image = Image.fromarray(np.stack([r, g, b], axis=-1), "RGB")
    draw = ImageDraw.Draw(image)
    for _ in range(4):
        x0, y0 = rng.randint(0, max(1, width - 8)), rng.randint(0, max(1, height - 8))
        x1, y1 = x0 + rng.randint(4, max(5, width // 2)), y0 + rng.randint(4, max(5, height // 2))
        draw.rectangle([x0, y0, min(x1, width - 1), min(y1, height - 1)],
                       fill=tuple(int(v) for v in rng.randint(0, 255, 3)))
    draw.ellipse([width // 4, height // 4, width * 3 // 4, height * 3 // 4], outline=(255, 255, 255), width=2)
    return image


def image_png_bytes(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


# ---------------- 真实视觉权重 ----------------

def read_safetensor_headers(model_dir):
    headers = {}
    for name in sorted(os.listdir(model_dir)):
        if not name.endswith(".safetensors"):
            continue
        path = os.path.join(model_dir, name)
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            header = json.loads(f.read(n))
        for key, info in header.items():
            if key == "__metadata__":
                continue
            headers[key] = (path, 8 + n, info)
    return headers


def load_real_vision_state(model_dir, wanted_prefixes=("vision.",)):
    """只取 ViT（vision.*）：aligner.* / image_* 的形状依赖文本侧 dim（真实 5120，迷你 512），仍用随机权重。"""
    import torch
    headers = read_safetensor_headers(model_dir)
    state = {}
    for key, (path, base, info) in headers.items():
        if not key.startswith(wanted_prefixes):
            continue
        assert info["dtype"] == "BF16", (key, info["dtype"])
        st, end = info["data_offsets"]
        with open(path, "rb") as f:
            f.seek(base + st)
            raw = f.read(end - st)
        arr = np.frombuffer(raw, dtype=np.uint16).reshape(info["shape"])
        state[key] = torch.from_numpy(arr.astype(np.int16)).view(torch.bfloat16).contiguous()
    assert state, "no vision tensors found in %s" % model_dir
    return state


def boost_bias_vl(work_dir, boost):
    from safetensors.torch import load_file, save_file
    path = os.path.join(work_dir, "model.safetensors")
    state = load_file(path)
    n = 0
    for key in state:
        if key.endswith(".ffn.gate.bias_vl"):
            state[key][-2:] += boost
            n += 1
    save_file(state, path)
    print("boosted the last two experts of gate.bias_vl by %.1f in %d layers" % (boost, n))


def inject_real_vision(work_dir, model_dir):
    from safetensors.torch import load_file, save_file
    path = os.path.join(work_dir, "model.safetensors")
    state = load_file(path)
    real = load_real_vision_state(model_dir)
    for key in list(state.keys()):
        if key.startswith("vision."):
            del state[key]
    state.update(real)
    save_file(state, path)
    print("injected %d real vision tensors from %s" % (len(real), model_dir))


# ---------------- 参考前向 ----------------

def run_reference(args, engram_mod, model_mod, image_processor, tokenizer, prompt_text, image_bytes_list):
    import torch
    from safetensors.torch import load_file

    with open(os.path.join(args.work_dir, "reference_config.json")) as f:
        cfg = json.load(f)
    for k in ("compress_ratios", "kv_source_layers", "index_source_layers", "engram_layer_ids",
              "engram_num_embeddings", "dspark_target_layer_ids"):
        cfg[k] = tuple(cfg[k])
    margs = model_mod.ModelArgs(**cfg)
    for fn in (model_mod.precompute_freqs_cis, model_mod.get_window_topk_idxs, model_mod.get_dspark_topk_idxs):
        fn.cache_clear()
    sys.modules["vision"].get_vision_cos_sin.cache_clear()
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    model = model_mod.Transformer(margs, tokenizer)
    torch.set_default_dtype(torch.float32)
    state = load_file(os.path.join(args.work_dir, "model.safetensors"), device="cuda")
    missing, unexpected = model.load_state_dict(state, strict=False)
    missing = [m for m in missing if not (m.startswith("engram_hash.") or "freqs_cis" in m or "_cache" in m
                                          or m.endswith("kv_state") or m.endswith("score_state"))]
    assert not missing and not unexpected, (missing, unexpected)

    def engram_forward(self, indices):
        flat = indices.reshape(-1)
        values = self.weight.view(torch.uint8)[flat].view(torch.float8_e4m3fn).float()
        scales = self.scale.view(torch.uint8)[flat].float() - 127.0
        values = values.unflatten(-1, (-1, self.block_size)) * torch.pow(2.0, scales).unsqueeze(-1)
        return values.flatten(-2).to(torch.bfloat16).view(*indices.shape, self.dim)
    model_mod.ParallelEngramEmbedding.forward = engram_forward
    if args.ref_vision_fp32:
        model.vision.float()
        model.aligner.float()
        orig_encode = model.encode_image

        def encode_image_fp32(patches, n_vit_h, n_vit_w):
            return orig_encode(patches.float(), n_vit_h, n_vit_w).to(torch.bfloat16)
        model.encode_image = encode_image_fp32

    records = [{"data": b} for b in image_bytes_list]
    tokens, token_types, image_inputs = image_processor.prepare_vl_inputs(prompt_text, records, tokenizer, margs)
    print("reference prompt tokens: %d, images: %s" % (
        len(tokens), [(img.start, img.n_vit_h, img.n_vit_w, int(img.types.numel())) for img in image_inputs]))

    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        counter = {"aligner": 0}

        def aligner_hook(module, inp, out):
            out.detach().float().cpu().numpy().astype(np.float32).tofile(
                os.path.join(args.dump_dir, "ref_image%d_embeds.bin" % counter["aligner"]))
            counter["aligner"] += 1
        model.aligner.register_forward_hook(aligner_hook)
        counter["vit"] = 0

        def stage_hook(name):
            def hook(module, inp, out):
                out.detach().float().cpu().numpy().astype(np.float32).tofile(
                    os.path.join(args.dump_dir, "ref_image%d_%s.bin" % (counter["vit"], name)))
                if name == "vit_out":
                    counter["vit"] += 1
            return hook
        model.vision.patch_embed.register_forward_hook(stage_hook("vit_patch"))
        for i, block in enumerate(model.vision.blocks):
            block.register_forward_hook(stage_hook("vit_block%d" % i))
        model.vision.norm.register_forward_hook(stage_hook("vit_out"))

        def gate_hook(name):
            def hook(module, inp, out):
                path = os.path.join(args.dump_dir, name + ".bin")
                if not os.path.exists(path):   # 只记录 prefill（首次调用）
                    out[1].detach().float().cpu().numpy().astype(np.float32).tofile(path)
            return hook
        def ffn_in_hook(name):
            def hook(module, inp):
                path = os.path.join(args.dump_dir, name + ".bin")
                if not os.path.exists(path):
                    inp[0].detach().float().cpu().numpy().astype(np.float32).tofile(path)
            return hook
        for i, layer in enumerate(model.layers):
            layer.ffn.gate.register_forward_hook(gate_hook("ref_layer%d_expert_idx" % i))
            layer.ffn.register_forward_pre_hook(ffn_in_hook("ref_layer%d_ffn_in" % i))
        orig_merge = model.merge_image_embeddings

        def merge_and_dump(images, h):
            orig_merge(images, h)
            h.detach().float().cpu().numpy().astype(np.float32).tofile(os.path.join(args.dump_dir, "ref_mm_embeds.bin"))
        model.merge_image_embeddings = merge_and_dump

    ids = torch.tensor([tokens], dtype=torch.long, device="cuda")
    types = torch.tensor([token_types], dtype=torch.long, device="cuda")
    logits_list, out_tokens = [], []
    with torch.inference_mode():
        _, logits, _ = model(ids, 0, images=[image_inputs], token_types=types)
        logits_list.append(logits[0].float().cpu())
        nxt = int(logits[0].argmax().item())
        out_tokens.append(nxt)
        pos = ids.size(1)
        for _ in range(args.decode):
            _, logits, _ = model(torch.tensor([[nxt]], device="cuda"), pos)
            logits_list.append(logits[0].float().cpu())
            nxt = int(logits[0].argmax().item())
            out_tokens.append(nxt)
            pos += 1
    torch.set_default_device("cpu")
    patches = [img.patches.float().reshape(img.patches.size(0), -1).cpu().numpy() for img in image_inputs]
    return logits_list, out_tokens, tokens, patches, token_types


# ---------------- fastllm 前向 ----------------

def run_fastllm(args, prompt_text, images, vocab_size, ref_tokens, ref_patches):
    from ftllm import llm
    from ftllm.deepseek_v41_multimodal import (build_deepseek_v41_multimodal_payload, expand_image_placeholders,
                                               get_deepseek_v41_vision_config)
    from ftllm.util import make_normal_llm_model, make_normal_parser

    os.environ.setdefault("FASTLLM_SKIP_WARMUP", "1")
    os.environ["FASTLLM_DSV41_ENGRAM_META"] = os.path.join(args.work_dir, "engram_meta.json")
    if args.dump_dir:
        os.makedirs(args.dump_dir, exist_ok=True)
        os.environ["FASTLLM_DSV41_DUMP_DIR"] = args.dump_dir
    parser = make_normal_parser("v41 vision reference")
    fastllm_argv = ["--path", args.work_dir, "--dtype", args.dtype, "--device", args.device,
                    "--moe_device", args.moe_device, "-t", str(args.threads)]
    if args.chunked_prefill > 0:
        fastllm_argv += ["--chunked_prefill_size", str(args.chunked_prefill)]
    fargs = parser.parse_args(fastllm_argv)
    if fargs.max_batch <= 0:
        fargs.max_batch = 1
    if fargs.tokens <= 0:
        fargs.tokens = 65536
    model = make_normal_llm_model(fargs)
    if args.fastllm_only_load:
        print("fastllm model loaded")
        return [], []

    # Python 侧展开占位符，与官方 prepare_vl_inputs 的结果逐项比较
    cfg = get_deepseek_v41_vision_config(model.config)
    prompt_tokens = llm.encode_hf_prompt(model.hf_tokenizer, prompt_text)
    native = expand_image_placeholders(prompt_tokens, images, cfg)
    print("fastllm prompt tokens: %d, image grid: %s" % (len(native["input_ids"]), native["image_grid"].tolist()))
    if ref_tokens is not None:
        assert native["input_ids"] == list(ref_tokens), "token expansion differs from the official image_processor"
        for i, (a, b) in enumerate(zip(native["image_patches"], ref_patches)):
            assert a.shape == b.shape and np.array_equal(a, b), "image %d patches differ from the official preprocessing" % i
        print("prompt tokens and image patches match the official preprocessing")
    payload_config, payload = build_deepseek_v41_multimodal_payload(native)
    input_ids = native["input_ids"]
    buf_payload = ctypes.create_string_buffer(payload)
    handle = llm.fastllm_lib.launch_response_llm_model_multimodal(
        model.model, len(input_ids), (ctypes.c_int * len(input_ids))(*input_ids),
        json.dumps(payload_config).encode(), buf_payload,
        ctypes.c_int(args.decode + 1), ctypes.c_int(0), ctypes.c_bool(False), ctypes.c_float(1.0), ctypes.c_int(1),
        ctypes.c_float(1.0), ctypes.c_float(1.0), ctypes.c_bool(True), ctypes.c_int(0), None)
    logits_list, tokens = [], []
    buf = (ctypes.c_float * vocab_size)()
    while True:
        token_id = llm.fastllm_lib.fetch_response_logits_llm_model(model.model, handle, buf)
        if token_id < 0:
            break
        tokens.append(int(token_id))
        logits_list.append(np.ctypeslib.as_array(buf).copy())
    return logits_list, tokens


def compare_vision_dumps(dump_dir, n_images):
    def load(name):
        path = os.path.join(dump_dir, name + ".bin")
        return np.fromfile(path, dtype=np.float32) if os.path.exists(path) else None

    def report(tag, a, b):
        if a is None or b is None or a.size != b.size:
            print("%-22s missing or size mismatch (%s vs %s)" % (tag, None if a is None else a.size, None if b is None else b.size))
            return None
        diff = np.abs(a - b)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        print("%-22s max|diff|=%.4f mean|diff|=%.5f rel=%.4f cos=%.6f" % (
            tag, diff.max(), diff.mean(), diff.mean() / (np.abs(a).mean() + 1e-9), cos))
        return cos

    ok = True
    for i in range(n_images):
        for name in ["vit_patch"] + ["vit_block%d" % l for l in range(64)] + ["vit_out"]:
            a, b = load("ref_image%d_%s" % (i, name)), load("fl_image%d_%s" % (i, name))
            if a is not None and b is not None:
                report("image%d %s" % (i, name), a, b)
        cos = report("image%d aligner out" % i, load("ref_image%d_embeds" % i), load("fl_image%d_embeds" % i))
        ok = ok and cos is not None and cos >= 0.999
    a, b = load("ref_mm_embeds"), load("fl_mm_embeds")
    if a is not None and b is not None and a.size != b.size:
        print("merged embeds: fastllm dump covers the first prefill chunk only (%d vs %d values), comparing that prefix" % (b.size, a.size))
        a = a[: b.size]
    cos = report("merged embeds", a, b)
    ok = ok and cos is not None and cos >= 0.999
    return ok


def compare_routing(dump_dir, n_layers, token_types, n_experts, boosted):
    """比较 prefill 时每层的专家集合，分别统计文本位置与图像位置的翻转数（图像位置使用 gate.bias_vl）；
    boosted 时另外统计图像位置是否都选中了被加偏置的最后两个专家。"""
    seqlen = len(token_types)
    image_pos = [i for i, t in enumerate(token_types) if t >= 0]
    text_pos = [i for i, t in enumerate(token_types) if t < 0]
    boosted_set = {n_experts - 2, n_experts - 1}
    ok = True
    for layer in range(n_layers):
        pr = os.path.join(dump_dir, "ref_layer%d_expert_idx.bin" % layer)
        pf = os.path.join(dump_dir, "fl_layer%d_expert_idx.bin" % layer)
        if not (os.path.exists(pr) and os.path.exists(pf)):
            continue
        r = np.fromfile(pr, dtype=np.float32).reshape(seqlen, -1).astype(int)
        f = np.fromfile(pf, dtype=np.float32)
        if f.size != r.size:
            print("layer%d routing: fastllm dump covers the first prefill chunk only, skipping" % layer)
            return ok
        f = f.reshape(seqlen, -1).astype(int)
        flips_text = sum(1 for t in text_pos if set(r[t]) != set(f[t]))
        flips_image = sum(1 for t in image_pos if set(r[t]) != set(f[t]))
        last_flip = set(r[seqlen - 1]) != set(f[seqlen - 1])
        line = "%-22s expert-set flips: text %d / %d, image %d / %d, last token %s" % (
            "layer%d routing" % layer, flips_text, len(text_pos), flips_image, len(image_pos), "FLIP" if last_flip else "same")
        if boosted:
            hit_ref = sum(1 for t in image_pos if set(r[t]) == boosted_set)
            hit_fl = sum(1 for t in image_pos if set(f[t]) == boosted_set)
            text_hit_fl = sum(1 for t in text_pos if set(f[t]) == boosted_set)
            line += " | boosted experts on image tokens: ref %d, fastllm %d (text %d)" % (hit_ref, hit_fl, text_hit_fl)
            ok = ok and hit_fl == len(image_pos) and hit_ref == len(image_pos) and text_hit_fl < len(text_pos)
        print(line)
    return ok


def main():
    args = parse_args()
    ref.TINY["n_routed_experts"] = args.experts
    ref.TINY["n_activated_experts"] = args.activated
    ref.TINY["index_topk"] = args.index_topk
    ref.TINY["candidate_topk_blocks"] = args.candidate_topk_blocks
    ref.TINY.update(REAL_VISION if args.real_vision else TINY_VISION)
    if args.min_pixels >= 0:
        ref.TINY["vision_min_pixels"] = args.min_pixels
    ref.hf_config_from_tiny = hf_config_with_vision
    if args.no_fake_quant:
        os.environ["V41_REF_NO_FAKE_QUANT"] = "1"
        os.environ["FASTLLM_DSV41_DISABLE_FAKE_QUANT"] = "1"

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_dir)
    engram_mod, model_mod, image_processor = install_vision_shims(args.reference_dir)

    if args.regenerate or not os.path.exists(os.path.join(args.work_dir, "model.safetensors")):
        ref.generate_checkpoint(args, engram_mod, model_mod, tokenizer)
        if args.real_vision:
            inject_real_vision(args.work_dir, args.real_vision)
        if args.bias_vl_boost != 0:
            boost_bias_vl(args.work_dir, args.bias_vl_boost)

    sizes = [tuple(int(v) for v in s.lower().split("x")) for s in args.image_size.split(",") if s]
    images = [make_synthetic_image(w, h, args.seed + i) for i, (w, h) in enumerate(sizes)]
    image_bytes = [image_png_bytes(img) for img in images]
    from encoding import IMAGE_PLACEHOLDER as placeholder  # 官方 encoding 目录（install_vision_shims 已加入 sys.path）
    parts = ["Describe the following picture in detail. "]
    for i in range(len(images)):
        parts.append(placeholder)
        parts.append(" Picture %d shows colored shapes; 图片 %d 里有几何图形。" % (i + 1, i + 1))
    prompt_text = tokenizer.bos_token + "".join(parts) + " Answer:"

    ref_logits, ref_tokens, ref_prompt, ref_patches, ref_types = [], [], None, None, None
    if not args.skip_reference:
        ref_logits, ref_tokens, ref_prompt, ref_patches, ref_types = run_reference(
            args, engram_mod, model_mod, image_processor, tokenizer, prompt_text, image_bytes)
        print("reference tokens:", ref_tokens)
        import gc
        import torch
        gc.collect()
        torch.cuda.empty_cache()
    if args.skip_fastllm:
        return
    fl_logits, fl_tokens = run_fastllm(args, prompt_text, images, ref.TINY["vocab_size"], ref_prompt, ref_patches)
    print("fastllm tokens:  ", fl_tokens)
    if not ref_logits:
        return
    ok = True
    for step in range(min(len(ref_logits), len(fl_logits))):
        a = ref_logits[step].numpy()
        b = fl_logits[step]
        diff = np.abs(a - b)
        cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
        same = int(a.argmax()) == int(b.argmax())
        order = np.argsort(-a)
        gap = float(a[order[0]] - a[order[1]])
        # 随机权重下参考侧前两名的差距小于两侧数值差时，贪心 token 翻转不代表实现错误；
        # 此后的 token 自然分叉，比较到此为止
        near_tie = (not same) and int(b.argmax()) == int(order[1]) and gap < diff.max() and cos > 0.999
        print("step %d: max|diff|=%.4f mean|diff|=%.5f cos=%.6f ref_argmax=%d fl_argmax=%d ref_top2_gap=%.4f %s" % (
            step, diff.max(), diff.mean(), cos, a.argmax(), b.argmax(), gap,
            "OK" if same else ("NEAR-TIE (tolerated)" if near_tie else "MISMATCH")))
        ok = ok and (same or near_tie) and cos > 0.999
        if not same:
            break
    if args.dump_dir:
        ok = compare_vision_dumps(args.dump_dir, len(images)) and ok
        if ref_types is not None:
            routing_ok = compare_routing(args.dump_dir, ref.TINY["n_layers"], ref_types, args.experts, args.bias_vl_boost != 0)
            if args.bias_vl_boost != 0:
                print("bias_vl routing check:", "PASS" if routing_ok else "FAIL")
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
