import pyfastllm

def embedding(inputs: pyfastllm.Tensor, embedding_weights:pyfastllm.Tensor):
    output = pyfastllm.Tensor()
    pyfastllm.embedding(inputs, embedding_weights, output)
    return output

def rms_norm(inputs:pyfastllm.Tensor, weights: pyfastllm.Tensor, eps: float=1e-5):
    output = pyfastllm.Tensor()
    pyfastllm.rms_norm(inputs, weights, eps, output)
    return output

def layer_norm(inputs: pyfastllm.Tensor, 
               gamma: pyfastllm.Tensor, 
               beta: pyfastllm.Tensor, 
               axis:int=-1 ):
    output = pyfastllm.Tensor()
    pyfastllm.layer_norm(inputs, gamma, beta,axis, output)
    return output

def linear(inputs: pyfastllm.Tensor, 
           weights: pyfastllm.Tensor, 
           bias: pyfastllm.Tensor=None):
    output = pyfastllm.Tensor()
    # print(weights)
    if not bias:
        bias = pyfastllm.Tensor()

    pyfastllm.linear(inputs, weights, bias, output)
    return output

def matmul(inputs0: pyfastllm.Tensor, 
           inputs1: pyfastllm.Tensor, 
           alpha: pyfastllm.Tensor):
    output = pyfastllm.Tensor()
    pyfastllm.matmul(inputs0, inputs1, alpha, output)
    return output

def attention(q: pyfastllm.Tensor, 
              k: pyfastllm.Tensor, 
              v: pyfastllm.Tensor, 
              mask: pyfastllm.Tensor,
              group: int, 
              scale: float, 
              attentionType:int = 0):
    output = pyfastllm.Tensor()
    pyfastllm.attention(q, k, v, mask, group, scale, attentionType, output)
    return output

def activation(inputs: pyfastllm.Tensor, axis=-1, activate_type="silu"):
    assert activate_type in ("softmax", "silu", "gelu", "swiglu")
    func = getattr(pyfastllm, activate_type)

    output = pyfastllm.Tensor()
    if activate_type == "softmax":
        func(inputs, axis, output)
    else:
        func(inputs, output)
    return output

def cat_(inputs, cur_data, axis=1):
    pyfastllm.cat_direct(inputs, cur_data, axis)

def mul(inputs: pyfastllm.Tensor, v: int):
    output = pyfastllm.Tensor()
    pyfastllm.mul(inputs, v, output)
    return output

def add(input0: pyfastllm.Tensor, input1: pyfastllm.Tensor, v:int=1.0):
    output = pyfastllm.Tensor()
    output = pyfastllm.add(input0, input1, v)
    return output

def permute(inputs: pyfastllm.Tensor, dims=None):
    output = pyfastllm.Tensor()
    pyfastllm.permute(inputs, dims, output)
    # pyfastllm.permute_(inputs, dims)
    return output

def split(inputs: pyfastllm.Tensor, axis:int, start:int, end:int):
    output = pyfastllm.Tensor()
    pyfastllm.split(inputs, axis, start, end, output)
    return output

def topk(logits:pyfastllm.Tensor, axis:int = 1):
    output = pyfastllm.Tensor()
    pyfastllm.topk(logits, axis, output)
    return output

def load(filepath):
    state_dict = pyfastllm.WeightMap()
    state_dict.load(filepath)
    return state_dict

def AttentionMask():
    pass

def AlibiMask():
    pass

def RotatePosition2D(data, pos_id, sin_data, cos_data, rotary_dim):
    return pyfastllm.rotateposition2D(data, pos_id, sin_data, cos_data, rotary_dim)

def NearlyRotatePosition2D(data, pos_id, sin_data, cos_data, rotary_dim):
    return pyfastllm.nearlyrotateposition2D(data, pos_id, sin_data, cos_data, rotary_dim)

def LlamaRotatePosition2D():
    pass

def RepeatPenalty():
    pass
