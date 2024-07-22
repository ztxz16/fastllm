from ftllm.llm import ComputeGraph
import math

class Qwen2Model(ComputeGraph):
    def build(self):
        weight, data, config = self.weight, self.data, self.config
        config["max_positions"] = 128000
        
        head_dim = config["hidden_size"] // config["num_attention_heads"]
        self.Embedding(data["inputIds"], weight["model.embed_tokens.weight"], data["hiddenStates"]);
        self.DataTypeAs(data["hiddenStates"], data["atype"])
        for i in range(config["num_hidden_layers"]):
            pastKey = data["pastKey."][i]
            pastValue = data["pastValue."][i]
            layer = weight["model.layers."][i]
            self.RMSNorm(data["hiddenStates"], layer[".input_layernorm.weight"], config["rms_norm_eps"], data["attenInput"])
            self.Linear(data["attenInput"], layer[".self_attn.q_proj.weight"], layer[".self_attn.q_proj.bias"], data["q"])
            self.Linear(data["attenInput"], layer[".self_attn.k_proj.weight"], layer[".self_attn.k_proj.bias"], data["k"])
            self.Linear(data["attenInput"], layer[".self_attn.v_proj.weight"], layer[".self_attn.v_proj.bias"], data["v"])
            self.ExpandHead(data["q"], head_dim)
            self.ExpandHead(data["k"], head_dim)
            self.ExpandHead(data["v"], head_dim)
            self.LlamaRotatePosition2D(data["q"], data["positionIds"], data["sin"], data["cos"], head_dim // 2)
            self.LlamaRotatePosition2D(data["k"], data["positionIds"], data["sin"], data["cos"], head_dim // 2)
            self.FusedAttention(data["q"], pastKey, pastValue, data["k"], data["v"], data["attenInput"], 
                                data["attentionMask"], data["attenOutput"], data["seqLens"], 1.0 / math.sqrt(head_dim))
            self.Linear(data["attenOutput"], layer[".self_attn.o_proj.weight"], layer[".self_attn.o_proj.bias"], data["attenLastOutput"]);
            self.AddTo(data["hiddenStates"], data["attenLastOutput"]);
            self.RMSNorm(data["hiddenStates"], layer[".post_attention_layernorm.weight"], config["rms_norm_eps"], data["attenInput"])
            self.Linear(data["attenInput"], layer[".mlp.gate_proj.weight"], layer[".mlp.gate_proj.bias"], data["w1"])
            self.Linear(data["attenInput"], layer[".mlp.up_proj.weight"], layer[".mlp.up_proj.bias"], data["w3"])
            self.Silu(data["w1"], data["w1"])
            self.MulTo(data["w1"], data["w3"])
            self.Linear(data["w1"], layer[".mlp.down_proj.weight"], layer[".mlp.down_proj.bias"], data["w2"])
            self.AddTo(data["hiddenStates"], data["w2"])
        self.SplitLastTokenStates(data["hiddenStates"], data["seqLens"], data["lastTokensStates"])
        self.RMSNorm(data["lastTokensStates"], weight["model.norm.weight"], config["rms_norm_eps"], data["lastTokensStates"])
        self.Linear(data["lastTokensStates"], weight["lm_head.weight"], weight["lm_head.bias"], data["logits"])

__model__ = Qwen2Model