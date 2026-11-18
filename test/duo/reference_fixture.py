"""Compare the native chunk sampler against inclusionAI/Duo's actual modules.

Use PYTHONPATH=/path/to/Duo. Outputs a tiny mixed-precision checkpoint, exact
noise inputs, and three committed chunks. No model downloads are needed.
"""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from duo.data.sequence import spatial_axis
from duo.models.realization.modules.convert import (
    DUO_MEDIA_TEST_TRANSFORMER_CONFIG,
    get_transformer_key_plan,
)
from duo.models.realization.modules.model import DuoTransformer
from duo.models.realization.modules.weights import (
    configure_trainables,
    initialize_synthetic,
)
from duo.models.realization.schedulers.flow import add_noise, euler, grid, to_x0
from safetensors.torch import save_file


def sequence(reference, shared, body, block, clean=False, prefix=False):
    h, w = 4, 4
    nv, nt, na = (
        (h * w // 4, len(shared), 0) if prefix else (5 * h * w // 4, len(body), 58)
    )
    # Official prefix order is reference then shared; current chunks have text then media.
    ids = lambda x: torch.tensor(x, device="cuda", dtype=torch.long)
    text = list(range(nv, nv + nt)) if prefix else list(range(nt))
    video = list(range(nv)) if prefix else list(range(nt, nt + nv))
    audio = list(range(nt + nv, nt + nv + na))
    groups = []

    def group(kind, b, start, end):
        return {"kind": kind, "block": b, "start": start, "end": end, "sample": 0}

    if prefix:
        groups = [group("reference", -1, 0, nv), group("shared", -1, nv, nv + nt)]
    else:
        groups = [
            group("text", block, 0, nt),
            group("clean" if clean else "noisy", block, nt, nt + nv + na),
        ]
    positions = torch.zeros(nt + nv + na, 3, device="cuda", dtype=torch.float32)
    yy, xx = spatial_axis((h, w))
    for i, row in enumerate(video):
        t, y, x = i // (h * w // 4), i // (w // 2) % (h // 2), i % (w // 2)
        positions[row] = torch.tensor(
            [
                0 if prefix else (block * 17 + [0, 1, 5, 9, 13][t]) * 1000 / 24 * 0.04,
                yy[y],
                xx[x],
            ],
            device="cuda",
        )
    positions[text, 0] = -1 if prefix else block * 17 * 1000 / 24 * 0.04
    for i, row in enumerate(audio):
        positions[row, 0] = float(
            np.float32(block * 17 * 1000 / 24 * 0.04) + np.float32(i % 29)
        )
        positions[row, 2] = xx[0 if i < 29 else -1].item()
    tags = torch.zeros(nt + nv + na, device="cuda", dtype=torch.long)
    tags[text] = 1
    tags[audio] = 2
    times = torch.zeros_like(tags)
    if not prefix and not clean:
        times[video + audio] = 1
    refiner = shared if prefix else torch.cat((shared, body))
    refiner_groups = [group("shared", -1, 0, len(shared))]
    if not prefix:
        refiner_groups.append(group("text", block, len(shared), len(refiner)))
    return SimpleNamespace(
        length=nt + nv + na,
        device=torch.device("cuda"),
        groups=tuple(groups),
        refiner_groups=tuple(refiner_groups),
        refiner_text=refiner[None],
        text_selection=ids(range(nt) if prefix else range(len(shared), len(refiner))),
        text_indices=ids(text),
        video_indices=ids(video),
        audio_indices=ids(audio),
        positions=positions,
        tags=tags,
        time_ids=times,
        video=reference[None] if prefix else torch.zeros(1, nv, 96, device="cuda"),
        audio=torch.zeros(1, na, 32, device="cuda"),
    )


def write(path, tensor):
    tensor.detach().float().cpu().numpy().tofile(path)


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--lora", action="store_true")
    mode.add_argument("--fp8", action="store_true")
    parser.add_argument("--causal", action="store_true")
    parser.add_argument(
        "--production-head",
        action="store_true",
        help="Use the release's 128-wide attention head to exercise fused BF16 attention",
    )
    args = parser.parse_args()
    CONFIG = dict(DUO_MEDIA_TEST_TRANSFORMER_CONFIG)
    if args.production_head:
        CONFIG.update(attention_head_dim=128, hidden_size=256, ffn_dim=512)
    root = args.output
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(923)
    torch.backends.cuda.matmul.allow_tf32 = False
    model = DuoTransformer(**CONFIG)
    modules = [
        n
        for n, m in model.named_modules()
        if isinstance(m, torch.nn.Linear)
        and "adaln" not in n
        and "time_embedder" not in n
        and n != "norm_out.linear"
    ]
    configure_trainables(
        model,
        "lora" if args.lora else "full",
        rank=4,
        alpha=8,
        modules=modules if args.lora else None,
        trainable=False,
    )
    initialize_synthetic(model, seed=42)
    if args.lora:
        for n, p in model.named_parameters():
            if ".lora_" in n:
                p.copy_(torch.randn_like(p) * 0.03)
    state = {k.replace(".base_layer.", "."): v for k, v in model.state_dict().items()}
    raw = {}
    for source, targets in get_transformer_key_plan(CONFIG).items():
        if not targets:
            raw[source] = model.rope.inv_freq.clone()
            continue
        tensors = [state[key].cpu() for key, _ in targets]
        if source.endswith("qkv_proj.weight"):
            raw[source] = (
                torch.cat(
                    [
                        v.reshape(
                            CONFIG["num_attention_heads"],
                            CONFIG["attention_head_dim"],
                            -1,
                        )
                        for v in tensors
                    ],
                    1,
                )
                .reshape(-1, CONFIG["hidden_size"])
                .contiguous()
            )
        elif source.endswith(".mlp.fc1.weight"):
            a, b = tensors[0].chunk(2)
            raw[source] = torch.cat((b, a)).contiguous()
        else:
            raw[source] = tensors[0].contiguous()
    backbone = root / "backbone"
    backbone.mkdir(exist_ok=True)
    if args.fp8:
        # Direct ModelOpt per-output-channel FP8 with dynamic per-token input scale.
        for key in list(raw):
            if key.startswith("blocks.") and key.endswith(
                ("qkv_proj.weight", "out_proj.weight", "fc1.weight", "fc2.weight")
            ):
                value = raw[key].float()
                scale = (value.abs().amax(1) / 448).clamp_min(1e-12)
                raw[key] = (value / scale[:, None]).to(torch.float8_e4m3fn)
                raw[key.removesuffix(".weight") + ".weight_scale"] = scale
        raw["role_embeddings.weight"] = state["role_embeddings.weight"]
        import types

        from duo.models.realization.modules.convert import (
            convert_transformer_key,
            reorder_interleaved_qkv,
        )

        named = dict(model.named_modules())
        for key in list(raw):
            scale_key = key.removesuffix(".weight") + ".weight_scale"
            if not key.endswith(".weight") or scale_key not in raw:
                continue
            weight = raw[key].float() * raw[scale_key][:, None]
            if key.endswith("qkv_proj.weight"):
                weight = reorder_interleaved_qkv(
                    weight, CONFIG["num_attention_heads"], CONFIG["attention_head_dim"]
                )
            for target, value in convert_transformer_key(key, weight, CONFIG):
                layer = named[target.removesuffix(".weight")]
                layer._fixture_fp8_weight = value.cuda()

                def forward(self, x):
                    scale = (x.float().abs().amax(-1, keepdim=True) / 448).clamp_min(
                        1e-12
                    )
                    q = (
                        (x.float() / scale)
                        .clamp(-448, 448)
                        .to(torch.float8_e4m3fn)
                        .float()
                    )
                    return (
                        torch.nn.functional.linear(q, self._fixture_fp8_weight) * scale
                    ).to(x.dtype)

                layer.forward = types.MethodType(forward, layer)
    save_file(raw, backbone / "model.safetensors")
    (backbone / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {k: "model.safetensors" for k in raw}})
    )
    manifest = {
        "format": "duo-release-dcp-v1",
        "stage": "dmd",
        "inference": {"sampling_steps": 4, "flow_shift": 5},
        "identity": {"codec": {"mode": "duo_5_17", "frames_per_block": 17, "fps": 24}},
        "models": {
            "generator": {
                "mode": "lora",
                "role_embedding": False,
                "config": CONFIG,
                "base": {"format": "original_duo", "path": "backbone"},
                "lora": {"alpha": 8 if args.lora else 1, "rank": 4 if args.lora else 1},
            }
        },
    }
    steps, shift = (50, 1) if args.causal else (4, 5)
    manifest["stage"] = "causal" if args.causal else "dmd"
    manifest["inference"] = {"sampling_steps": steps, "flow_shift": shift}
    if args.fp8:
        manifest["format"] = "duo-release-modelopt-v1"
        manifest["models"]["generator"].update(
            mode="full",
            lora=None,
            storage={"format": "modelopt_fp8", "path": "backbone"},
        )
        (backbone / "diffusion_pytorch_model.safetensors.index.json").write_text(
            (backbone / "model.safetensors.index.json").read_text()
        )
    (root / "complete.json").write_text(json.dumps(manifest))
    adapter = root / "adapter"
    adapter.mkdir(exist_ok=True)
    adapters = {
        k: v for k, v in state.items() if ".lora_" in k or k == "role_embeddings.weight"
    }
    save_file(adapters, adapter / "adapter.safetensors")
    (adapter / "adapter.json").write_text(
        json.dumps(
            {
                "format": "fastllm-duo-adapter-v1",
                "source_manifest": manifest,
                "tensors": {k: list(v.shape) for k, v in adapters.items()},
            }
        )
    )
    model.cuda().eval()
    reference = torch.randn(4, 96, device="cuda")
    shared = torch.randn(3, 48, device="cuda", dtype=torch.bfloat16)
    write(root / "reference.f32", reference)
    write(root / "shared.f32", shared)
    _, cache = model(
        sequence(reference, shared, None, 0, prefix=True),
        torch.ones(2, device="cuda"),
        commit=True,
    )
    request = {
        "latent_height": 4,
        "latent_width": 4,
        "shared_rows": 3,
        "reference": "reference.f32",
        "shared": "shared.f32",
        "chunks": [],
    }
    for k in range(3):
        body = torch.randn(4, 48, device="cuda", dtype=torch.bfloat16)
        write(root / f"intent-{k}.f32", body)
        # Native history=2 means at most one previous clean block during prediction.
        if k == 2:
            cache = cache.evict(1)
        s = sequence(reference, shared, body, k)
        files = []

        def noise(shape, kind, step, *, block=k, filenames=files):
            t = torch.randn(shape, device="cuda")
            name = f"noise-{block}-{step}-{kind}.f32"
            write(root / name, t)
            filenames.append(name)
            return t

        v = noise((1, 20, 96), "video", 0)
        a = noise((1, 58, 32), "audio", 0)
        ts = grid(steps + 1, shift=shift, device="cuda")
        for i in range(steps):
            s.video, s.audio = v, a
            (pv, pa), _ = model(s, torch.stack((ts.new_tensor(1), ts[i])), cache=cache)
            if args.causal:
                v, a = euler(v, pv, ts[i], ts[i + 1]), euler(a, pa, ts[i], ts[i + 1])
            else:
                v, a = to_x0(v, pv, ts[i]), to_x0(a, pa, ts[i])
                if i + 1 < steps:
                    v, a = (
                        add_noise(v, noise(v.shape, "video", i + 1), ts[i + 1]),
                        add_noise(a, noise(a.shape, "audio", i + 1), ts[i + 1]),
                    )
        clean = sequence(reference, shared, body, k, clean=True)
        clean.video, clean.audio = v, a
        _, cache = model(clean, torch.ones(2, device="cuda"), cache=cache, commit=True)
        write(root / f"expected-{k}.video.f32", v)
        write(root / f"expected-{k}.audio.f32", a)
        request["chunks"].append(
            {"intent": f"intent-{k}.f32", "rows": 4, "noise": files}
        )
    (root / "request.json").write_text(json.dumps(request, indent=2))
    print(root)


if __name__ == "__main__":
    main()
