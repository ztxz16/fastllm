from .base_module import Module
from .base_module import Parameter
from ..functions import fastllm_ops as F
from ..functions import util
import numpy as np

class Linear(Module):
    def __init__(self, in_dim, out_dim, bias=False) -> None:
        self.has_bias = bias
        self.weights = Parameter(shape=(out_dim, in_dim))
        self.bias = None

        if bias:
            self.bias = Parameter(shape=(out_dim, ))

        super().__init__()
    
    def forward(self, x):
        if self.has_bias:
            return F.linear(x, self.weights.value, self.bias.value)
        
        return F.linear(x, self.weights.value)

class SiLU(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, axis=-1):
        return F.activation(x, axis=axis, activate_type='silu')

class SwiGLU(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, inputs):
        return F.activation(inputs=inputs, activate_type="swiglu")

class Embedding(Module):
    def __init__(self, vocab_size, embed_dim) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.weights = Parameter(shape=[vocab_size, embed_dim])
    
    def forward(self, inputs):
        return F.embedding(inputs, self.weights.value)

class RMSNorm(Module):
    def __init__(self, dim=4096, eps=1e-5) -> None:
        super().__init__()
        self.weights = Parameter(shape=[dim, ])
        self.eps = eps

    def forward(self, inputs):
        return F.rms_norm(inputs, self.weights.value, eps=self.eps)

class Attention(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, q, k, v, mask, group, scale):
        return F.attention(q, k, v, mask, group=group, scale=scale, attentionType=0)

class RoPE(Module):
    def __init__(self, rotary_dim=128) -> None:
        super().__init__()
        self.rotary_dim = rotary_dim
        self.sin_data, self.cos_data = self._get_sin_cos_data()
        self.sin_data = util.to_tensor(self.sin_data)
        self.cos_data = util.to_tensor(self.cos_data)

    def _get_sin_cos_data(self, base=1e4, seq_len=32768, dim=128):
        inv_freq = 1.0 / (base ** (np.arange(0, dim, 2) / dim))
        t = np.arange(0, seq_len)
        freqs = np.einsum('i,j->ij', t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        return np.sin(emb), np.cos(emb)
    
    def forward(self, data, pos_id):
        return F.RotatePosition2D(data, pos_id, self.sin_data, self.cos_data, self.rotary_dim)

class NearlyRoPE(RoPE):
    def __init__(self, rotary_dim=64) -> None:
        super().__init__(rotary_dim)
    
    def forward(self, data, pos_id):
        outputs = F.NearlyRotatePosition2D(data, pos_id, self.sin_data, self.cos_data, self.rotary_dim)
        return outputs