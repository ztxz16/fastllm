import sys
import pytest
import numpy as np
import fastllm

import pyfastllm
np.random.seed(42)

from fastllm import ops
from fastllm.nn import Module, Linear, SwiGLU, NearlyRoPE, RMSNorm, Embedding
from typing import List, Tuple
import math

class ChatGLMConfig():
    model_type = "chatglm"
    def __init__(
        self,
        num_layers=28,
        padded_vocab_size=65024,
        hidden_size=4096,
        ffn_hidden_size=13696,
        kv_channels=128,
        num_attention_heads=32,
        seq_length=2048,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        layernorm_epsilon=1e-5,
        rmsnorm=True,
        apply_residual_connection_post_layernorm=False,
        post_layer_norm=True,
        add_bias_linear=False,
        add_qkv_bias=False,
        interleaved_qkv=False,
        bias_dropout_fusion=True,
        multi_query_attention=False,
        multi_query_group_num=1,
        apply_query_key_layer_scaling=True,
        attention_softmax_in_fp32=True,
        fp32_residual_connection=False,
        quantization_bit=0,
        pre_seq_len=None,
        prefix_projection=False,
        **kwargs
    ):
        self.num_layers = num_layers
        self.vocab_size = padded_vocab_size
        self.padded_vocab_size = padded_vocab_size
        self.hidden_size = hidden_size
        self.ffn_hidden_size = ffn_hidden_size
        self.kv_channels = kv_channels
        self.num_attention_heads = num_attention_heads
        self.seq_length = seq_length
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layernorm_epsilon = layernorm_epsilon
        self.rmsnorm = rmsnorm
        self.apply_residual_connection_post_layernorm = apply_residual_connection_post_layernorm
        self.post_layer_norm = post_layer_norm
        self.add_bias_linear = add_bias_linear
        self.add_qkv_bias = add_qkv_bias
        self.bias_dropout_fusion = bias_dropout_fusion
        self.multi_query_attention = multi_query_attention
        self.multi_query_group_num = multi_query_group_num
        self.apply_query_key_layer_scaling = apply_query_key_layer_scaling
        self.attention_softmax_in_fp32 = attention_softmax_in_fp32
        self.fp32_residual_connection = fp32_residual_connection
        self.quantization_bit = quantization_bit
        self.pre_seq_len = pre_seq_len
        self.prefix_projection = prefix_projection
        super().__init__(**kwargs)


class MLP(Module):
    def __init__(self, config:ChatGLMConfig) -> None:
        super().__init__()
        self.rms = RMSNorm()
        self.dense_h_to_4h = Linear(config.hidden_size, config.ffn_hidden_size * 2)
        self.dense_4h_to_h = Linear(config.ffn_hidden_size, config.hidden_size)
        self.act = SwiGLU()
    
    def forward(self, inputs):
        outputs = self.rms(inputs)
        # print("开始dense")
        outputs = self.dense_4h_to_h(self.act(self.dense_h_to_4h(outputs)))
        # print("开始add")
        outputs = ops.add(outputs, inputs)
        return outputs

class CoreAttn(Module):
    def __init__(self,) -> None:
        super().__init__()
        # self.embed_dim = config.hidden_size

    def _expand_dims(self, past_data, cur_data, unitLen=64):
        while (
            (len(past_data.size()) == 0 and (len(past_data.expansionDims) == 0 or cur_data.size(1) > past_data.expansionDims[1])) 
            or (len(past_data.size()) > 0 and (len(past_data.expansionDims) == 0 or past_data.size(1) + cur_data.size(1) > past_data.expansionDims[1]))
        ):
            if past_data.count(0) == 0 or len(past_data.size()) == 0:
                newDims =[cur_data.size(0), int(((cur_data.size(1) - 1) / unitLen + 1) * unitLen), cur_data.size(2)]
            else:
                newDims = past_data.size()
                newDims[1] += int(((cur_data.size(1) - 1) / unitLen + 1) * unitLen)
        
            # print(newDims)
            past_data.expansion(newDims)

        ops.cat_(past_data, cur_data, 1)

    def forward(self, q, k, v, attn_mask, pastkv):
        seq_len, batch, num_attention_heads, attn_dim = q.size()
        embed_dim = num_attention_heads * attn_dim

        k.reshape([k.size(0), k.size(1) * k.size(2), k.size(3)])
        v.reshape([v.size(0), v.size(1) * v.size(2), v.size(3)])

        k = ops.permute(k, [1, 0, 2])
        v = ops.permute(v, [1, 0, 2])

        pastKey = pastkv[0]
        pastValue = pastkv[1]
        self._expand_dims(past_data=pastKey, cur_data=k)
        self._expand_dims(past_data=pastValue, cur_data=v)

        q.reshape([q.size(0), q.size(1) * q.size(2), q.size(3)])
        q = ops.permute(q, [1, 0, 2])
        context = ops.attention(q, pastKey, pastValue, attn_mask, group=q.size(0)//pastKey.size(0), scale=1.0/math.sqrt(attn_dim))
        context.reshape([batch, num_attention_heads, seq_len, -1])
        context = ops.permute(context, [2, 0, 1, 3])
        context.reshape([context.size(0), context.size(1), embed_dim])
        return context

class Transformer(Module):
    def __init__(self, config:ChatGLMConfig) -> None:
        super().__init__()
        # print("开始构建transformer")
        self.config = config
        self.rms = RMSNorm(dim=config.hidden_size)
        self.qkv = Linear(in_dim=config.hidden_size, out_dim=4608, bias=True)
        self.rope = NearlyRoPE()
        self.attn = CoreAttn()
        # print("构建core attn结束")
        self.post_linear = Linear(in_dim=config.hidden_size, out_dim=config.hidden_size)
        self.mlp = MLP(config=config)
        
    def _split_qkv(self, qkv):
        embed_dim = self.config.hidden_size
        num_attention_heads = self.config.num_attention_heads

        qLen = embed_dim
        kvLen = (qkv.size(-1) - embed_dim) // 2
        q = ops.split(qkv, -1, 0, qLen)
        k = ops.split(qkv, -1, qLen, qLen + kvLen)
        v = ops.split(qkv, -1, qLen + kvLen, qLen + kvLen + kvLen)

        q.reshape([q.size(0), q.size(1), -1, embed_dim // num_attention_heads])
        k.reshape([k.size(0), k.size(1), -1, embed_dim // num_attention_heads])
        v.reshape([v.size(0), v.size(1), -1, embed_dim // num_attention_heads])

        return (q, k, v)

    def forward(self, inputs, pos_id, attn_mask, pastkv):
        # print("开始计算rms")
        atten_input = self.rms(inputs)
        # print("开始计算qkv")
        qkv = self.qkv(atten_input)
        # print("开始split")
        q, k, v = self._split_qkv(qkv)

        q = self.rope(q, pos_id)
        k = self.rope(k, pos_id)

        context = self.attn(q, k, v, attn_mask, pastkv)
        outputs = self.post_linear(context)
        outputs = ops.add(outputs, inputs)  # TODO: 实现Tensor += 

        # print("开始mlp")
        outputs = self.mlp(outputs) 

        return outputs 

class ChatGLM2(Module):
    def __init__(self, config: ChatGLMConfig) -> None:
        super().__init__()
        # print("开始初始化模型")
        self.config = config
        self.num_layers = config.num_layers
        self.rotary_dim = 64
        self.num_attention_heads = config.num_attention_heads
        self.embed_dim = config.hidden_size
        self.head_dim = self.embed_dim // self.num_attention_heads
        scale_attn = math.sqrt(self.head_dim)

        # print("构建embeding")
        self.embedding = Embedding(vocab_size=65024, embed_dim=4096)
        self.decoder = [Transformer(config) for i in range(self.num_layers)]
        # print("构建decoder结束")
        self.rms = RMSNorm(eps=1e-5)
        self.head = Linear(config.hidden_size , config.vocab_size)
    
    def _get_postion_id(self, seq_len):
        pos_id = np.zeros(shape=[2, seq_len])
        pos_id[0, :] = np.arange(0, seq_len)
        pos_id[1, -1] = 1
        return pos_id

    def _get_mask(self, seq_len):
        attn_mask = np.zeros(shape=[seq_len, seq_len])
        attn_mask[:, -1] = 1
        for i in range(seq_len):
            for j in range(i+1, seq_len):
                attn_mask[i, j] = 1

        return attn_mask

    def forward(
            self,
            input_ids,
            attn_mask,
            pos_id,
            pastkvs
        ):
        batch = input_ids.size(0)
        seq_len = input_ids.size(1)
        input_ids = ops.permute(input_ids, [1, 0])
        input_embedding = self.embedding(inputs=input_ids)
        hidden_states = input_embedding

        # hidden_states.to("cuda")
        # print(hidden_states)

        for i in range(self.num_layers):
            hidden_states = self.decoder[i].forward(hidden_states, pos_id, attn_mask, pastkv=pastkvs[i])

        if seq_len > 1:
            hidden_states = ops.split(hidden_states, 0, seq_len - 1, seq_len)

        hidden_states = self.rms(hidden_states)
        logits = self.head(hidden_states)

        topk = ops.topk(logits, 1)
        topk.to("cpu")
        print(topk)
        topk_np = ops.util.to_numpy(topk)
        token = int(topk_np[0, 0, 0] + 1e-3)
        return token, pastkvs
    

    def set_weights(model, state_dict=None):
        # state_dict = load_weights()
        # state_dict = load_weights()
        # print("加载权重完成")
        model.embedding.weights.value = state_dict[f"transformer.embedding.word_embeddings.weight"]
        model.head.weights.value = state_dict[f"transformer.output_layer.weight"]
        model.rms.weights.value = state_dict[f"transformer.encoder.final_layernorm.weight"]
        
        for i in range(model.num_layers):
            model.decoder[i].rms.weights.value = state_dict[f"transformer.encoder.layers.{i}.input_layernorm.weight"]
            model.decoder[i].qkv.weights.value = state_dict[f"transformer.encoder.layers.{i}.self_attention.query_key_value.weight"]
            model.decoder[i].qkv.bias.value = state_dict[f"transformer.encoder.layers.{i}.self_attention.query_key_value.bias"]
            model.decoder[i].post_linear.weights.value = state_dict[f"transformer.encoder.layers.{i}.self_attention.dense.weight"]
            model.decoder[i].mlp.rms.weights.value = state_dict[f"transformer.encoder.layers.{i}.post_attention_layernorm.weight"]
            model.decoder[i].mlp.dense_h_to_4h.weights.value = state_dict[f"transformer.encoder.layers.{i}.mlp.dense_h_to_4h.weight"]
            model.decoder[i].mlp.dense_4h_to_h.weights.value = state_dict[f"transformer.encoder.layers.{i}.mlp.dense_4h_to_h.weight"]


    def warmup(model):
        bos_token_id = 64792
        input_ids = pyfastllm.Tensor(fastllm.float32, [1, 1], [bos_token_id, ])
        attn_mask = pyfastllm.Tensor(fastllm.float32, [1, 1], [0])
        pos_id = pyfastllm.Tensor(fastllm.float32, [2, 1], [0, 0])

        pastKeyValues = []
        for i in range(28):
            pastKey = pyfastllm.Tensor(fastllm.float32)
            pastValue = pyfastllm.Tensor(fastllm.float32)
            pastKeyValues.append([pastKey, pastValue])
        
        model.forward(input_ids, attn_mask, pos_id, pastKeyValues)

    def build_inputs(model, tokenizer, query: str, history: List[Tuple[str, str]] = None):
        prompt = tokenizer.build_prompt(query, history=history)
        inputs = tokenizer([prompt], return_tensors="np")
        return inputs

    def stream_chat(model, query="", tokenizer=None):
        # query = "你好"
        # model_path = "/home/pan/Public/Models/models-hf/chatglm2-6b"
        # tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        input_ids = model.build_inputs(tokenizer, query=query)['input_ids']
        print(input_ids)

        batch = input_ids.shape[0]
        seq_len = input_ids.shape[1]
        input_ids = ops.util.to_tensor(input_ids)
        pos_id = ops.util.to_tensor(model._get_postion_id(seq_len))
        attn_mask = ops.util.to_tensor(model._get_mask(seq_len))

        pastKeyValues = []
        for i in range(28):
            pastKey = pyfastllm.Tensor(fastllm.float32)
            pastValue = pyfastllm.Tensor(fastllm.float32)
            pastKeyValues.append([pastKey, pastValue])

        index = 0
        promptLen = seq_len - 2

        results = []
        while True:
            token, pastKeyValues  = model(input_ids, attn_mask, pos_id, pastKeyValues)

            if token == 2:
                break

            results.append(token)
            ret = tokenizer.decode(results)
            print(ret)
            yield ret

            index += 1

            if index >= 2048:
                break

            input_ids.copy_from(fastllm.Tensor(fastllm.float32, [1, 1], [token]))
            attn_mask = fastllm.Tensor(fastllm.float32)
            pos_id.copy_from(fastllm.Tensor(fastllm.float32, [2, 1], [promptLen + index + 1, (index + 1)]))



