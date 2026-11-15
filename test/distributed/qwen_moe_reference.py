"""Qwen2 MoE 独立 CPU 数学参考与固定种子小模型；不加载原生库或触发编译。"""
import math
import random

from qwen3_layer_reference import half, f32, matrix, linear, norm, rotate, residual
from run_qwen3_shards import encode, decode, save_fixture


def fixture(directory, *, dtype="F16", bias=True, normalize=False, experts=5, topk=2,
            tied=False, indexed=False, hotspot=False, qkv_bias_rounding=False, mixed_qkv_bias=False,
            expert_accumulation=False):
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

    if mixed_qkv_bias:
        if not bias or qkv_bias_rounding:
            raise ValueError('mixed QKV bias fixture requires bias and ordinary seeded weights')
        # 首层无QKV bias、第二层有；检出只按首层选后端、未warmup后续层等错误。
        for name in ('q', 'k', 'v'):
            del tensors[f'model.layers.0.self_attn.{name}_proj.bias']

    if qkv_bias_rounding:
        if dtype != 'F16' or not bias or tied:
            raise ValueError('QKV bias rounding fixture requires F16, bias and a separate LM head')
        # 只保留第一层V投影和O投影，让一次/两次舍入产生可解析的残差差别。
        for name, tensor in tensors.items():
            unit = name in ('model.embed_tokens.weight', 'model.norm.weight') or name.endswith('layernorm.weight')
            value = 1.0 if unit else 0.0
            tensor['data'] = encode([value]*math.prod(tensor['shape']), dtype)
        # 单位输入经Norm得到1.0009765625；乘512.5再加(-513)为2^-11，提前转FP16则为0。
        # O投影放大64倍后，首通道残差分别为1.03125和1；其他层仅保留残差。
        prefix = 'model.layers.0.'
        tensors[prefix+'input_layernorm.weight']['data'] = encode([1.0009765625]*8, dtype)
        tensors[prefix+'self_attn.v_proj.weight']['data'] = encode([512.5 if i % 8 == 0 else 0.0 for i in range(32)], dtype)
        tensors[prefix+'self_attn.v_proj.bias']['data'] = encode([-513.0]*4, dtype)
        tensors[prefix+'self_attn.o_proj.weight']['data'] = encode([64.0] + [0.0]*63, dtype)
        tensors['lm_head.weight']['data'] = encode([4.0] + [0.0]*103, dtype)

    if expert_accumulation:
        if dtype != 'F16' or experts != 4 or topk != 4 or normalize or tied or qkv_bias_rounding or mixed_qkv_bias:
            raise ValueError('expert accumulation fixture requires F16, four experts/topk and ordinary bias layout')
        # 所有Token输入为全1，Attention及第二层MoE为零；第一层四专家的加权贡献
        # 按ID依次为[2048,1,-2048,1]。HF逐次FP16得1，FP32及Top-K顺序均得2。
        for name, tensor in tensors.items():
            unit = name in ('model.embed_tokens.weight', 'model.norm.weight') or name.endswith('layernorm.weight')
            tensor['data'] = encode([1.0 if unit else 0.0]*math.prod(tensor['shape']), dtype)
        prefix = 'model.layers.0.mlp.'
        # 非并列路由顺序为[2,0,3,1]，检出按槽位排序代替全局专家顺序的错误。
        tensors[prefix+'gate.weight']['data'] = encode(
            [value if col == 0 else 0.0 for value in (.5, 0, 1, .25) for col in range(8)], dtype)
        for expert, down in enumerate((8264.0, 6.6484375, -5012.0, 5.1796875)):
            name = prefix+f'experts.{expert}.'
            # FP16 SiLU(16)=16，与Up=1/16相乘得到精确的1；Down仅作用于首通道。
            tensors[name+'gate_proj.weight']['data'] = encode([16.0] + [0.0]*55, dtype)
            tensors[name+'up_proj.weight']['data'] = encode([.0625] + [0.0]*55, dtype)
            tensors[name+'down_proj.weight']['data'] = encode([down] + [0.0]*55, dtype)
        tensors['lm_head.weight']['data'] = encode([4.0] + [0.0]*103, dtype)

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
        # HF Linear的bias在输出FP16转换前相加；Dense参考的旧舍入语义保持独立。
        projections = []
        for name in ("q", "k", "v"):
            values = linear(normalized, weight(f"self_attn.{name}_proj.weight"),
                            bias(f"self_attn.{name}_proj.bias"), rounded=False)
            projections.append([[half(value) for value in row] for row in values])
        q, k, v = [[[row[i*d:(i+1)*d] for i in range(len(row)//d)] for row in values] for values in projections]
        for token in range(t):
            for heads in (q, k):
                heads[token] = [rotate(head, input_["position_begin"] + token, theta) for head in heads[token]]
        context = []
        for token in range(t):
            row = []
            for head in range(qh):
                kv = head // (qh // kh)
                scores = [half(half(math.fsum(a*b for a, b in zip(q[token][head], k[key][kv]))) / math.sqrt(d))
                          for key in range(token+1)]
                probs = [math.exp(value - max(scores)) for value in scores]
                probs = [half(value / sum(probs)) for value in probs]
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
                    # HF的FP16 index_add_每加入一个专家便舍入，按全局专家ID顺序迭代。
                    routed[i][col] = half(routed[i][col] + half(output[col] * probability))
        shared = expert(normalized, "shared_expert")
        gates = linear(normalized, weight("mlp.shared_expert_gate.weight"))
        shared = [[half(value * half(1/(1+math.exp(-gate[0])))) for value in row] for row, gate in zip(shared, gates)]
        result.update(router_logits=flatten(logits), selected_experts=flatten([ids for ids, _ in routes]),
                      routing_weights=flatten([probs for _, probs in routes]), routed_output=flatten(routed), shared_output=flatten(shared))
        x = [[half(a + half(half(r) + s)) for a, r, s in zip(row, rrow, srow)] for row, rrow, srow in zip(x, routed, shared)]
    result["output"] = flatten(x)
    return result


def full_logits(config, tensors, token_ids):
    """/** 从完整 Token 历史独立重算两层/多层模型；不读取原生 KV、路由或前缀状态。 */"""
    embedding = matrix(tensors['model.embed_tokens.weight'])
    h = config['hidden_size']
    hidden = [x for token in token_ids for x in embedding[token]]
    for layer in range(config['num_hidden_layers']):
        hidden = forward(config, tensors, dict(layer=layer, stage=0),
                         dict(hidden=hidden, position_begin=0))['output']
    row = norm(hidden[-h:], matrix(tensors['model.norm.weight']), config['rms_norm_eps'])
    head = tensors.get('lm_head.weight', tensors['model.embed_tokens.weight'])
    return linear([row], matrix(head), rounded=False)[0]
