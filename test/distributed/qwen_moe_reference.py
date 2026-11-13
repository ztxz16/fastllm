"""Qwen2 MoE 独立 CPU 数学参考与固定种子小模型；不加载原生库或触发编译。"""
import math
import random

from qwen3_layer_reference import half, f32, matrix, linear, norm, rotate, residual
from run_qwen3_shards import encode, decode, save_fixture


def fixture(directory, *, dtype="F16", bias=True, normalize=False, experts=5, topk=2,
            tied=False, indexed=False, hotspot=False):
    """/** 创建全专家小模型；权重直接按 HF 名称生成，不依赖 C++ 分片计划。 */"""
    config = dict(model_type="qwen2_moe", hidden_size=8, num_hidden_layers=2,
                  num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                  intermediate_size=32, moe_intermediate_size=7, shared_expert_intermediate_size=11,
                  num_experts=experts, num_experts_per_tok=topk, norm_topk_prob=normalize,
                  decoder_sparse_step=1, rms_norm_eps=1e-6, rope_theta=10000.0,
                  hidden_act="silu", vocab_size=13, tie_word_embeddings=tied)
    tensors = {}
    rng = random.Random(20261113)

    def add(name, shape):
        values = [rng.uniform(-0.25, 0.25) for _ in range(math.prod(shape))]
        if "layernorm.weight" in name or name == "model.norm.weight":
            values = [1 + x * 0.1 for x in values]
        if hotspot and name.endswith("mlp.gate.weight"):
            values = [0.5 * (e + 1) if col == 0 else 0.0 for e in range(experts) for col in range(8)]
        tensors[name] = dict(shape=shape, dtype=dtype, data=encode(values, dtype))

    add("model.embed_tokens.weight", [13, 8])
    add("model.norm.weight", [8])
    if not tied:
        add("lm_head.weight", [13, 8])
    for layer in range(2):
        p = f"model.layers.{layer}."
        add(p + "input_layernorm.weight", [8])
        add(p + "post_attention_layernorm.weight", [8])
        for name, width in (("q_proj", 8), ("k_proj", 4), ("v_proj", 4)):
            add(p + "self_attn." + name + ".weight", [width, 8])
            if bias:
                add(p + "self_attn." + name + ".bias", [width])
        add(p + "self_attn.o_proj.weight", [8, 8])
        if bias:
            add(p + "self_attn.o_proj.bias", [8])
        add(p + "mlp.gate.weight", [experts, 8])
        add(p + "mlp.shared_expert_gate.weight", [1, 8])
        for name, width in [(f"experts.{e}", 7) for e in range(experts)] + [("shared_expert", 11)]:
            add(p + "mlp." + name + ".gate_proj.weight", [width, 8])
            add(p + "mlp." + name + ".up_proj.weight", [width, 8])
            add(p + "mlp." + name + ".down_proj.weight", [8, width])
    directory.mkdir()
    save_fixture(directory, config, tensors, indexed)
    return config, tensors


def weights_for_rank(config, tensors, rank, world):
    """/** 独立按 HF 张量名选择专家并融合 QKV/GateUp；返回目标布局及源字节计数。 */"""
    result = {}
    for name, tensor in tensors.items():
        if ".experts." in name and int(name.split(".experts.")[1].split(".")[0]) % world != rank:
            continue
        target, sources = name, [tensor]
        if name.endswith(("k_proj.weight", "v_proj.weight", "k_proj.bias", "v_proj.bias", "up_proj.weight")):
            continue
        if "self_attn.q_proj." in name:
            target = name.replace("q_proj", "mergeqkv")
            sources = [tensors[name.replace("q_proj", part)] for part in ("q_proj", "k_proj", "v_proj")]
        elif name.endswith(".gate_proj.weight"):
            target = name.replace("gate_proj", "gateup_proj")
            sources = [tensor, tensors[name.replace("gate_proj", "up_proj")]]
        shape = [sum(x["shape"][0] for x in sources), *tensor["shape"][1:]]
        values = [half(v) for x in sources for v in decode(x["data"], x["dtype"])]
        result[target] = dict(shape=shape, values=values, source_bytes=sum(len(x["data"]) for x in sources))
    if "lm_head.weight" not in tensors:
        result["lm_head.weight"] = dict(result["model.embed_tokens.weight"])
    return result


def route(logits, topk, normalize):
    """/** 对完整专家概率排序，选中后可选重归一化；返回专家 ID 与 FP16 权重。 */"""
    exp = [math.exp(x - max(logits)) for x in logits]
    probs = [x / sum(exp) for x in exp]
    selected = sorted(range(len(probs)), key=lambda i: (-probs[i], i))[:topk]
    denominator = sum(probs[i] for i in selected) if normalize else 1.0
    return selected, [half(probs[i] / denominator) for i in selected]


def forward(config, tensors, task, input_):
    """/** 完整、未分片的 Qwen2 MoE 层参考；独立计算各请求，不读取 GPU 路由或 rank 分片。 */"""
    h, d = config["hidden_size"], config["head_dim"]
    qh, kh = config["num_attention_heads"], config["num_key_value_heads"]
    eps, theta = config["rms_norm_eps"], config["rope_theta"]
    flat = input_["hidden"]
    t = len(flat) // h
    x = [[half(v) for v in flat[i*h:(i+1)*h]] for i in range(t)]
    p = f"model.layers.{task['layer']}."
    weight = lambda name: matrix(tensors[p + name])
    bias = lambda name: weight(name) if p + name in tensors else None
    flatten = lambda rows: [value for row in rows for value in row]
    result = dict(router_logits=[], routing_weights=[], selected_experts=[], routed_output=[], shared_output=[])
    if task["stage"] != 2:
        normalized = [norm(row, weight("input_layernorm.weight"), eps) for row in x]
        projections = [linear(normalized, weight(f"self_attn.{name}_proj.weight"), bias(f"self_attn.{name}_proj.bias"))
                       for name in ("q", "k", "v")]
        q, k, v = [[[row[i*d:(i+1)*d] for i in range(len(row)//d)] for row in values] for values in projections]
        for token in range(t):
            for heads in (q, k):
                heads[token] = [rotate(head, input_["position_begin"] + token, theta) for head in heads[token]]
        context = []
        for token in range(t):
            row = []
            for head in range(qh):
                kv = head // (qh // kh)
                scores = [math.fsum(a*b for a, b in zip(q[token][head], k[key][kv])) / math.sqrt(d) for key in range(token+1)]
                probs = [math.exp(value - max(scores)) for value in scores]
                probs = [value / sum(probs) for value in probs]
                row.extend(half(math.fsum(probs[key]*v[key][kv][dim] for key in range(token+1))) for dim in range(d))
            context.append(row)
        x = residual(x, linear(context, weight("self_attn.o_proj.weight"), bias("self_attn.o_proj.bias"), rounded=False))
    if task["stage"] != 1:
        normalized = [norm(row, weight("post_attention_layernorm.weight"), eps) for row in x]
        logits = linear(normalized, weight("mlp.gate.weight"))
        routes = [route(row, config["num_experts_per_tok"], config["norm_topk_prob"]) for row in logits]

        def expert(rows, name):
            gate, up = [linear(rows, weight(f"mlp.{name}.{projection}_proj.weight")) for projection in ("gate", "up")]
            activated = [[half(half(f32(a/(1+math.exp(-a)))) * b) for a, b in zip(g, u)] for g, u in zip(gate, up)]
            return linear(activated, weight(f"mlp.{name}.down_proj.weight"))

        routed = [[0.0] * h for _ in range(t)]
        for e in range(config["num_experts"]):
            selected = [i for i, (ids, _) in enumerate(routes) if e in ids]
            outputs = expert([normalized[i] for i in selected], f"experts.{e}")
            for i, output in zip(selected, outputs):
                ids, probs = routes[i]
                probability = probs[ids.index(e)]
                for col in range(h):
                    # 独立参考按全局专家顺序 FP32 累加；GPU 可按 EP 分组规约。
                    routed[i][col] = f32(routed[i][col] + half(output[col] * probability))
        shared = expert(normalized, "shared_expert")
        gates = linear(normalized, weight("mlp.shared_expert_gate.weight"))
        shared = [[half(value * half(1/(1+math.exp(-gate[0])))) for value in row] for row, gate in zip(shared, gates)]
        result.update(router_logits=flatten(logits), selected_experts=flatten([ids for ids, _ in routes]),
                      routing_weights=flatten([probs for _, probs in routes]), routed_output=flatten(routed), shared_output=flatten(shared))
        x = [[half(a + half(half(r) + s)) for a, r, s in zip(row, rrow, srow)] for row, rrow, srow in zip(x, routed, shared)]
    result["output"] = flatten(x)
    return result
