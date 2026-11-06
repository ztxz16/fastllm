"""独立单卡数学参考：读取完整权重，不使用 C++ 分片边界或 CUDA 输出。"""

import math
import struct

from run_qwen3_shards import decode


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def f32(value):
    return struct.unpack("<f", struct.pack("<f", value))[0]


def matrix(tensor):
    values = [half(x) for x in decode(tensor["data"], tensor["dtype"])]
    if len(tensor["shape"]) == 1:
        return values
    width = tensor["shape"][1]
    return [values[i:i + width] for i in range(0, len(values), width)]


def linear(x, weight, bias=None, rounded=True):
    result = []
    for row in x:
        values = []
        for index, coefficients in enumerate(weight):
            # 双精度求和作为独立高精度点积参考，最后转 FP32，再按阶段转 FP16。
            value = f32(math.fsum(a * b for a, b in zip(row, coefficients)))
            if rounded:
                value = half(value)
            if bias is not None:
                value = f32(value + bias[index])
                if rounded:
                    value = half(value)
            values.append(value)
        result.append(values)
    return result


def norm(row, weight, eps):
    scale = 1 / math.sqrt(math.fsum(value * value for value in row) / len(row) + eps)
    return [half(half(f32(value * scale)) * w) for value, w in zip(row, weight)]


def rotate(row, position, theta):
    size = len(row) // 2
    result = [0.0] * len(row)
    for index in range(size):
        angle = f32(position / f32(theta ** (2 * index / len(row))))
        cosine, sine = half(math.cos(angle)), half(math.sin(angle))
        a, b = row[index], row[index + size]
        result[index] = half(half(a * cosine) - half(b * sine))
        result[index + size] = half(half(b * cosine) + half(a * sine))
    return result


def residual(x, projected):
    return [[half(a + half(b)) for a, b in zip(row, delta)] for row, delta in zip(x, projected)]


def forward(config, tensors, task):
    t, h, d = task["tokens"], config["hidden_size"], config["head_dim"]
    qh, kh = config["num_attention_heads"], config["num_key_value_heads"]
    eps, theta = config.get("rms_norm_eps", 1e-6), config.get("rope_theta", 1000000.0)
    prefix = f"model.layers.{task['layer']}."
    weight = lambda name: matrix(tensors[prefix + name])
    bias = lambda name: weight(name) if prefix + name in tensors else None
    x = [[half(v) for v in task["input"][i * h:(i + 1) * h]] for i in range(t)]
    if task["stage"] != 2:
        normalized = [norm(row, weight("input_layernorm.weight"), eps) for row in x]
        projections = [linear(normalized, weight(f"self_attn.{name}_proj.weight"), bias(f"self_attn.{name}_proj.bias"))
                       for name in ("q", "k", "v")]
        q, k, v = [[[row[i * d:(i + 1) * d] for i in range(len(row) // d)] for row in values]
                   for values in projections]
        for token in range(t):
            for heads, name in ((q, "q_norm"), (k, "k_norm")):
                heads[token] = [rotate(norm(head, weight(f"self_attn.{name}.weight"), eps),
                                       task["position_begin"] + token, theta) for head in heads[token]]
        context = []
        for token in range(t):
            row = []
            for head in range(qh):
                kv = head // (qh // kh)
                scores = [math.fsum(a * b for a, b in zip(q[token][head], k[key][kv])) / math.sqrt(d)
                          for key in range(token + 1)]
                maximum = max(scores)
                exp = [math.exp(score - maximum) for score in scores]
                probabilities = [value / sum(exp) for value in exp]
                row += [half(math.fsum(probabilities[key] * v[key][kv][dim] for key in range(token + 1)))
                        for dim in range(d)]
            context.append(row)
        x = residual(x, linear(context, weight("self_attn.o_proj.weight"), bias("self_attn.o_proj.bias"), rounded=False))
    if task["stage"] != 1:
        normalized = [norm(row, weight("post_attention_layernorm.weight"), eps) for row in x]
        gate, up = [linear(normalized, weight(f"mlp.{name}_proj.weight")) for name in ("gate", "up")]
        activated = [[half(half(f32(a / (1 + math.exp(-a)))) * b) for a, b in zip(g, u)] for g, u in zip(gate, up)]
        x = residual(x, linear(activated, weight("mlp.down_proj.weight"), bias("mlp.down_proj.bias"), rounded=False))
    return [value for row in x for value in row]
