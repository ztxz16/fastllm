"""Checkpoint storage descriptions shared by inventory and capacity planning.

Encoding-specific normalization and capabilities belong here. Planners operate
on logical tensors and must not infer storage support from the model name.
"""
from dataclasses import dataclass
import re

@dataclass(frozen=True)
class Tensor:
    name: str
    shape: tuple
    bytes: int
    kind: str

    @property
    def preserve_source(self):
        """Already packed weights retain their encoded bytes on load."""
        return self.kind == "EXL3" or self.kind.startswith("gguf:")

    @property
    def supports_disk(self):
        # Disk linear currently requires a directly addressable source tensor.
        return self.kind != "EXL3"

    @property
    def calibration_override(self):
        """Optional GEMV probe for encodings not covered by the GGUF probes."""
        if self.kind == "EXL3" and len(self.shape) == 2:
            n, k = self.shape
            return -((self.bytes - 2 * (n + k)) * 16 // (n * k))
        return None


def _exl3_inventory(tensors):
    """Budget logical matrices with their original trellises and rotation scales."""
    by_name = {t.name: t for t in tensors}
    removed, logical, groups = set(), [], {}
    for tensor in tensors:
        if not tensor.name.endswith(".trellis"):
            continue
        prefix = tensor.name[:-8]
        su, sv, marker = (by_name.get(prefix + suffix) for suffix in (".suh", ".svh", ".mul1"))
        if tensor.kind != "I16" or len(tensor.shape) != 3 or not all((su, sv, marker)):
            raise ValueError("EXL3: incomplete matrix or unsupported codebook: " + prefix)
        k, n, words = tensor.shape[0] * 16, tensor.shape[1] * 16, tensor.shape[2]
        if (not k or not n or k % 128 or n % 128 or words not in (16, 24, 32, 40, 48, 56, 64, 80, 96, 112, 128) or
                su.kind != "F16" or sv.kind != "F16" or su.shape != (k,) or sv.shape != (n,) or
                su.bytes != k * 2 or sv.bytes != n * 2 or marker.kind != "I32" or marker.bytes != 4 or
                tensor.bytes != k * n * words // 128):
            raise ValueError("EXL3: invalid dimensions or bitrate: " + prefix)
        removed.update(prefix + suffix for suffix in (".trellis", ".suh", ".svh", ".mul1", ".scale", ".weight_scale", ".weight"))
        match = re.fullmatch(r"(.*\.wo_a)\.slice\.(\d+)", prefix)
        if match:
            groups.setdefault(match[1] + ".weight", []).append((int(match[2]), n, k, words, tensor.bytes + su.bytes + sv.bytes))
        else:
            logical.append(Tensor(prefix + ".weight", (n, k), tensor.bytes + su.bytes + sv.bytes, "EXL3"))
    if not logical:
        raise ValueError("EXL3 checkpoint contains no supported matrices")
    for name, slices in groups.items():
        slices.sort()
        _, n, k, words, size = slices[0]
        if [(i, ni, ki) for i, ni, ki, _, _ in slices] != [(i, n, k) for i in range(len(slices))]:
            raise ValueError("EXL3: inconsistent grouped projection: " + name)
        removed.add(name)
        same_bits = all(w == words for _, _, _, w, _ in slices)
        logical.append(Tensor(name, (len(slices), n, k), len(slices) * size if same_bits else len(slices) * n * k * 2, "EXL3" if same_bits else "BF16"))
    return [t for t in tensors if t.name not in removed] + logical


def normalize_checkpoint(metadata, tensors):
    method = (metadata.get("quantization_config") or {}).get("quant_method")
    normalizer = {"exl3": _exl3_inventory}.get(method)
    return normalizer(tensors) if normalizer else tensors
