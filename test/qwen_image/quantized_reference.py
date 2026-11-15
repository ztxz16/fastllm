"""Development-only INT8/GGUF oracles; never imported by native inference."""
import json
import torch
from safetensors import safe_open


class Int8Linear(torch.nn.Module):
    def __init__(self, weight, scale):
        super().__init__()
        self.register_buffer('weight', weight.cuda())
        self.register_buffer('scale', scale.float().reshape(1, -1).cuda())
        # Independent dense H4 Kronecker product, not the CUDA butterfly.
        h = torch.ones(4, 4, device='cuda') - 2 * torch.eye(4, device='cuda').flip(1)
        rotation = h
        for _ in range(3):
            rotation = torch.kron(rotation, h)
        self.register_buffer('rotation', rotation / 16)

    def forward(self, x):
        shape = x.shape
        assert x.dtype == torch.bfloat16
        x = x.reshape(-1, shape[-1]).contiguous()
        rotated = (x.float().reshape(-1, 256) @ self.rotation).reshape(x.shape).bfloat16().float()
        scale = (rotated.abs().amax(-1, keepdim=True) / 127).clamp_min(1e-12)
        q = (rotated / scale).round().clamp(-128, 127).to(torch.int8)
        rows = max(32, (q.shape[0] + 15) // 16 * 16)
        padded = torch.nn.functional.pad(q, (0, 0, 0, rows - q.shape[0]))
        result = torch._int_mm(padded, self.weight.t()).float()[:x.shape[0]]
        return ((result * scale) * self.scale).bfloat16().reshape(*shape[:-1], self.weight.shape[0])


class GgufLinear(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.register_buffer('weight', torch.from_numpy(weight.copy()).cuda().bfloat16())

    def forward(self, x):
        # Decode independently with gguf-py and use BF16 GEMM. Native MMQ also
        # quantizes activations on small token counts: its error is measured,
        # not duplicated in the oracle or excluded from acceptance checks.
        return torch.nn.functional.linear(x, self.weight)


def install_quantized(model, checkpoint):
    def replace(prefix, weight, scale=None):
        def one(name, w, s):
            old = model.get_submodule(name)
            assert isinstance(old, torch.nn.Linear) and old.bias is None and tuple(w.shape) == tuple(old.weight.shape), name
            parent, _, child = name.rpartition('.')
            new = Int8Linear(w, s) if s is not None else GgufLinear(w)
            setattr(model.get_submodule(parent) if parent else model, child, new)
        if prefix.endswith('.gate_up'):
            half = weight.shape[0] // 2
            one(prefix.removesuffix('gate_up') + 'gate_layer', weight[:half], None if scale is None else scale[:half])
            one(prefix.removesuffix('gate_up') + 'proj', weight[half:], None if scale is None else scale[half:])
        else:
            one(prefix, weight, scale)
    count = 0
    with open(checkpoint, 'rb') as f:
        gguf = f.read(4) == b'GGUF'
    if gguf:
        from gguf import GGUFReader
        from gguf.quants import dequantize
        reader = GGUFReader(checkpoint)
        for tensor in reader.tensors:
            name = tensor.name.removeprefix('model.diffusion_model.')
            value = dequantize(tensor.data, tensor.tensor_type).reshape(tuple(reversed(tensor.shape.tolist())))
            if value.ndim == 2:
                replace(name.removesuffix('.weight'), value)
                count += 1
            else:
                target = model.get_parameter(name)
                assert tuple(value.shape) == tuple(target.shape), name
                target.data.copy_(torch.from_numpy(value.copy()).to(target.device, target.dtype))
    else:
        with safe_open(checkpoint, framework='pt', device='cpu') as weights:
            for name in weights.keys():
                if not name.endswith('.weight'):
                    continue
                weight = weights.get_tensor(name)
                if weight.dtype != torch.int8:
                    target = model.get_parameter(name)
                    target.data.copy_(weight.to(target.device, target.dtype))
                    continue
                prefix = name.removesuffix('.weight')
                marker = json.loads(bytes(weights.get_tensor(prefix + '.comfy_quant').tolist()))
                assert marker['format'] == 'int8_tensorwise' and marker['convrot'] and marker['convrot_groupsize'] == 256
                scale = weights.get_tensor(prefix + '.weight_scale')
                assert scale.numel() == weight.shape[0] and torch.isfinite(scale).all() and (scale > 0).all()
                replace(prefix, weight, scale)
                count += 1
    assert count
    torch.cuda.empty_cache()
    return count
