from .base_module import Module
from ..functions import fastllm_ops as F

class Linear(Module):
    def __init__(self, input_size, output_size, bias=False) -> None:
        self.weight = None
        self.bias = None
        super().__init__()
    
    def forward(self, x):
        return F.linear(x, self.weight, self.bias)

class SiLU(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, x, axis=-1):
        return F.activation(x, axis=axis, activate_type='silu')

class SwiGLU(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, inputs):
        return F.activation(input=inputs, activate_type="swiglu")

class Embedding(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self):
        return F.embedding()

class Embedding(Module):
    def __init__(self, vocab_size, embed_dim) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.embedding_weights = None
    
    # def _init_weight(self):
    #     self.embedding_weights = to_tensor(np.random.random(size=[self.vocab_size, self.embed_dim]))
    
    def forward(self, inputs):
        return F.embedding(inputs, self.embedding_weights)

class RMSNorm(Module):
    def __init__(self) -> None:
        super().__init__()
        self.weights = None

    def _init_weight(self):
        return super()._init_weight()

    def forward(self, inputs):
        return F.rms_norm(inputs, self.weights, eps=self.eps)

class Attention(Module):
    def __init__(self) -> None:
        super().__init__()
    
    def forward(self, q, k, v, mask, group, scale):
        return F.attention(q, k, v, mask, group=group, scale=scale, attentionType=0)