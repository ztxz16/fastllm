#!/usr/bin/env python3
"""Export an independent Torch DSpark backbone oracle for naive_n05_draft_test.

Uses the draft checkpoint's own dflash.py implementation and real weights.
The deterministic synthetic context crosses its sliding-window boundary;
FastLLM appends that context in two chunks and checks repeated proposals.
Run with the Torch/Transformers reference environment used by the logits tool.
"""
import argparse
import copy
import importlib.util
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--context", type=int, default=1030)
    args = parser.parse_args()
    import torch
    from safetensors.torch import load_file
    from transformers import Qwen3Config

    torch.set_num_threads(8)
    torch.manual_seed(42)
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    root = Path(args.draft).expanduser()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    module_spec = importlib.util.spec_from_file_location("naive_original_dflash", root / "dflash.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    config_data = json.loads((root / "config.json").read_text())
    config = Qwen3Config(**config_data)
    config._attn_implementation = "eager"
    with torch.device("meta"):
        model = module.DFlashDraftModel(config)
    weights = load_file(str(root / "model.safetensors"))
    backbone = {name: value for name, value in weights.items()
                if not name.startswith(("markov_head.", "confidence_head."))}
    model.load_state_dict(backbone, strict=True, assign=True)
    # RoPE frequencies are non-persistent buffers and are absent from the
    # checkpoint; materialize them after constructing parameters on meta.
    model.rotary_emb = module.Qwen3RotaryEmbedding(config)
    model = model.eval().to("cuda")
    length, block, hidden = args.context, config.block_size, config.hidden_size
    # Keep the original fusion layer in the oracle; its output is the context
    # interface cached by FastLLM between target and draft forwards.
    features = torch.randn(1, length, len(model.target_layer_ids) * hidden,
                           device="cuda", dtype=torch.bfloat16) * .1
    anchor = torch.randn(1, 1, hidden, device="cuda", dtype=torch.bfloat16) * .1
    noise = anchor.expand(1, block, hidden).clone()
    positions = torch.arange(length + block, device="cuda").unsqueeze(0)
    query_positions = positions[:, -block:]
    allowed = positions[:, None, :] > query_positions[:, :, None] - config.sliding_window
    mask = torch.zeros(1, 1, block, length + block, device="cuda", dtype=torch.bfloat16)
    mask.masked_fill_(~allowed[:, None], float("-inf"))
    with torch.inference_mode():
        context = model.hidden_norm(model.fc(features))
        expected = model(position_ids=positions, noise_embedding=noise, target_hidden=features,
                         attention_mask=mask, use_cache=False)
    # Report the rounding scale of this synthetic, uncorrelated context. A
    # second pass keeps the context interface fixed and runs the backbone in
    # FP32; this distinguishes BF16 accumulation drift from a wrong cache mask.
    exact = copy.deepcopy(model).float()
    exact.fc = torch.nn.Identity()
    exact.hidden_norm = torch.nn.Identity()
    with torch.inference_mode():
        precise = exact(position_ids=positions, noise_embedding=noise.float(),
                        target_hidden=context.float(), attention_mask=mask.float(), use_cache=False)
    rounding_rmse = (expected.float() - precise).square().mean().sqrt().item()
    print(f"torch_bf16_vs_fp32_rmse={rounding_rmse:.6f}", flush=True)

    tensors = []
    def write(name, tensor, fp32=False):
        tensor = tensor.detach().cpu().contiguous()
        tensor = tensor.float() if fp32 else tensor.bfloat16()
        filename = f"tensor_{len(tensors):03d}.bin"
        tensor.view(torch.uint8).numpy().tofile(destination / filename)
        tensors.append({"name": name, "shape": list(tensor.shape), "file": filename,
                        "dtype": "float32" if fp32 else "bfloat16"})

    for name, tensor in weights.items():
        # The C++ backbone test consumes projected context and shared embedding
        # input directly; it does not need the fusion or token prediction heads.
        if name.startswith("layers.") or name in {"mask_embedding", "norm.weight"}:
            write("dspark." + name, tensor, "norm.weight" in name)
    write("model.embed_tokens.weight", anchor.reshape(1, hidden))
    write("test.context", context)
    write("test.expected", expected, True)
    manifest = {"hidden_size": hidden, "layers": config.num_hidden_layers,
                "heads": config.num_attention_heads, "kv_heads": config.num_key_value_heads,
                "head_dim": config.head_dim, "block": block, "window": config.sliding_window,
                "eps": config.rms_norm_eps, "theta": config_data["rope_parameters"]["rope_theta"],
                "torch_bf16_vs_fp32_rmse": rounding_rmse,
                "tensors": tensors}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination, flush=True)


if __name__ == "__main__":
    main()
