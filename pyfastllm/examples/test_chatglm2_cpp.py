import sys
import pytest
import numpy as np
import fastllm

import pyfastllm
import gc

np.random.seed(42)
from fastllm import ops
# import ops

def to_tensor(data):
    return pyfastllm.from_numpy(data)

def to_numpy(data):
    return np.array(data, copy=False)


## 模型测试
def load_weights():
    file = "/home/pan/Public/Models/models-flm/chatglm2-6b.flm"
    state_dict = ops.load(file)
    # print(state_dict.keys())
    return state_dict

state_dict = load_weights()    

def get_sin_cos():
    base = 1e4
    dim = 128
    inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))

    t = np.arange(0, 32768)
    freqs = np.einsum('i,j->ij', t, inv_freq)

    emb = np.concatenate((freqs, freqs), axis=-1)
    return np.sin(emb), np.cos(emb)


def get_postion_id(seq_len):
    pos_id = np.zeros(shape=[2, seq_len])
    for i in range(seq_len):
        pos_id[0, i] = i
    pos_id[1, -1] = 1
    return pos_id

def get_mask(seq_len):
    attn_mask = np.zeros(shape=[seq_len, seq_len])
    for i in range(seq_len):
        attn_mask[i, -1] = 1
    
    for i in range(seq_len):
        for j in range(i+1, seq_len):
            attn_mask[i, j] = 1

    return attn_mask



from transformers import AutoModel, AutoTokenizer
import math
from typing import List, Tuple

def build_inputs(tokenizer, query: str, history: List[Tuple[str, str]] = None):
    prompt = tokenizer.build_prompt(query, history=history)
    inputs = tokenizer([prompt], return_tensors="np")
    return inputs

sin_data, cos_data = get_sin_cos()
sin_data = to_tensor(sin_data)
cos_data = to_tensor(cos_data)

def chatglm():
    query = "你好"
    model_path = "/home/pan/Public/Models/models-hf/chatglm2-6b"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    input_ids = build_inputs(tokenizer, query=query)['input_ids']
    print(input_ids)

    batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]
    input_ids = to_tensor(input_ids)
    pos_id = to_tensor(get_postion_id(seq_len))
    attn_mask = to_tensor(get_mask(seq_len))

    pastKeyValues = []
    for i in range(28):
        pastKey = pyfastllm.Tensor(fastllm.float32)
        pastValue = pyfastllm.Tensor(fastllm.float32)
        pastKeyValues.append([pastKey, pastValue])

    index = 0
    promptLen = seq_len - 2

    results = []
    while True:
        token, pastKeyValues  = forward(input_ids, attn_mask, pos_id, pastKeyValues)

        if token == 2:
            break

        results.append(token)
        ret = tokenizer.decode(results)
        print(ret)

        index += 1

        if index >= 256:
            break

        input_ids.copy_from(fastllm.Tensor(fastllm.float32, [1, 1], [token]))
        attn_mask = fastllm.Tensor(fastllm.float32)
        pos_id.copy_from(fastllm.Tensor(fastllm.float32, [2, 1], [promptLen + index + 1, (index + 1)]))

def warmup():
    bos_token_id = 64792
    input_ids = pyfastllm.Tensor(fastllm.float32, [1, 1], [bos_token_id, ])
    attn_mask = pyfastllm.Tensor(fastllm.float32, [1, 1], [0])
    pos_id = pyfastllm.Tensor(fastllm.float32, [2, 1], [0, 0])

    pastKeyValues = []
    for i in range(28):
        pastKey = pyfastllm.Tensor(fastllm.float32)
        pastValue = pyfastllm.Tensor(fastllm.float32)
        pastKeyValues.append([pastKey, pastValue])

    forward(input_ids, attn_mask, pos_id, pastKeyValues)

def mlp(inputs, i):     
    fcInKeyName = f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h"
    fcOutKeyName = f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h"

    middle = ops.linear(inputs, weights=state_dict[fcInKeyName+".weight"], bias=state_dict[fcInKeyName+".bias"])
    middle = ops.activation(middle, activate_type='swiglu')
    middle = ops.linear(middle, weights=state_dict[fcOutKeyName+".weight"], bias=state_dict[fcOutKeyName+".bias"])

    return middle

def forward(input_ids, 
            attn_mask,
            pos_id,
            pastkv
    ):

    batch = input_ids.shape[0]
    seq_len = input_ids.shape[1]

    input_ids = ops.permute(input_ids, [1, 0])
    input_embedding = ops.embedding(inputs=input_ids, embedding_weights=state_dict['transformer.embedding.word_embeddings.weight'])
    hidden_states = input_embedding

    rotary_dim = 128
    layer_num = 28
    num_attention_heads = 32
    embed_dim = 4096
    head_dim = embed_dim // num_attention_heads
    scale_attn = math.sqrt(head_dim)    
    
    for i in range(layer_num):
        print(i)
        inputRMSWeightName = f"transformer.encoder.layers.{i}.input_layernorm.weight"
        atten_input = ops.rms_norm(hidden_states, state_dict[inputRMSWeightName])
        # print(atten_input)

        qkv_weight_name = f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight"
        qkv_bias_name = f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias"
        qkv = ops.linear(atten_input, state_dict[qkv_weight_name], state_dict[qkv_bias_name])

        qLen = embed_dim
        kvLen = (qkv.shape[-1] - embed_dim) // 2
        q = ops.split(qkv, -1, 0, qLen)
        k = ops.split(qkv, -1, qLen, qLen + kvLen)
        v = ops.split(qkv, -1, qLen + kvLen, qLen + kvLen + kvLen)

        q.reshape([q.shape[0], q.shape[1], -1, embed_dim // num_attention_heads])
        k.reshape([k.shape[0], k.shape[1], -1, embed_dim // num_attention_heads])
        v.reshape([v.shape[0], v.shape[1], -1, embed_dim // num_attention_heads])

        q = pyfastllm.nearlyrotateposition2D(q, pos_id, sin_data, cos_data, rotary_dim)
        k = pyfastllm.nearlyrotateposition2D(k, pos_id, sin_data, cos_data, rotary_dim)

        # print(k)

        k.reshape([k.shape[0], k.shape[1] * k.shape[2], k.shape[3]])
        v.reshape([v.shape[0], v.shape[1] * v.shape[2], v.shape[3]])

        k = ops.permute(k, [1, 0, 2])
        v = ops.permute(v, [1, 0, 2])

        pastKey = pastkv[i][0]
        pastValue = pastkv[i][1]

        unitLen = 64
        while (
            (len(pastKey.shape) == 0 and (len(pastKey.expansionDims) == 0 or k.shape[1] > pastKey.expansionDims[1])) 
            or (len(pastKey.shape) > 0 and (len(pastKey.expansionDims) == 0 or pastKey.shape[1] + k.shape[1] > pastKey.expansionDims[1]))
            ):
            if pastKey.count(0) == 0 or len(pastKey.shape) == 0:
                newDims =[k.shape[0], int(((k.shape[1] - 1) / unitLen + 1) * unitLen), k.shape[2]]
            else:
                newDims = pastKey.shape 
                newDims[1] += int(((k.shape[1] - 1) / unitLen + 1) * unitLen)
            
            # print(newDims)
            pastKey.expansion(newDims)
        
        while (
            (len(pastValue.shape) == 0 and (len(pastValue.expansionDims) == 0 or v.shape[1] > pastValue.expansionDims[1])) 
            or (len(pastValue.shape) > 0 and (len(pastValue.expansionDims) == 0 or pastValue.shape[1] + v.shape[1] > pastValue.expansionDims[1]))
            ):
            if pastValue.count(0) == 0 or len(pastValue.shape) == 0:
                newDims =[v.shape[0], int(((v.shape[1] - 1) / unitLen + 1) * unitLen), v.shape[2]]
            else:
                newDims = pastValue.shape
                newDims[1] += int(((v.shape[1] - 1) / unitLen + 1) * unitLen)

            pastValue.expansion(newDims)
        
        pyfastllm.cat_direct(pastKey, k, 1)
        pyfastllm.cat_direct(pastValue, v, 1)
        
        q.reshape([q.shape[0], q.shape[1] * q.shape[2], q.shape[3]])
        q = ops.permute(q, [1, 0, 2])

        context = ops.attention(q, pastKey, pastValue, attn_mask, q.shape[0]//pastKey.shape[0], 1.0 / scale_attn)
        context.reshape([batch, num_attention_heads, seq_len, -1])
        context = ops.permute(context, [2, 0, 1, 3])
        context.reshape([context.dims[0], context.dims[1], embed_dim])
        
        denseWeightName = f"transformer.encoder.layers.{i}.self_attention.dense.weight"
        denseBiasName = f"transformer.encoder.layers.{i}.self_attention.dense.bias"
        attnOutput = ops.linear(context, state_dict[denseWeightName],state_dict[denseBiasName])
        hidden_states = ops.add(hidden_states, attnOutput, 1.0)

        temp = ops.mul(hidden_states, 1.0)
        postRMSWeightName = f"transformer.encoder.layers.{i}.post_attention_layernorm.weight"
        mlp_input = ops.rms_norm(hidden_states, state_dict[postRMSWeightName], 1e-5)
        mlp_output = mlp(mlp_input, i)
        hidden_states = ops.add(mlp_output, temp, 1.0)
        # hidden_states = mlp_output
        gc.collect()

    if seq_len > 1:
        hidden_states = ops.split(hidden_states, 0, seq_len - 1, seq_len)

    hidden_states = ops.rms_norm(hidden_states, state_dict["transformer.encoder.final_layernorm.weight"], 1e-5)
    logits = ops.linear(hidden_states, state_dict["transformer.output_layer.weight"])

    topk = ops.topk(logits, 1)
    # print(topk.size())
    # print(topk)
    print(topk)
    topk_np = np.array(topk, copy=False)
    token = int(topk_np[0, 0, 0] + 1e-3)
    print(token)
    return token, pastkv


if __name__ == "__main__":    
    # warmup()
    chatglm()

