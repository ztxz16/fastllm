import pyfastllm as fastllm


def embedding(data: fastllm.Tensor, ):
    # some check
    return fastllm.embedding(data, )

def rms_norm(input:fastllm.Tensor, weight: fastllm.Tensor, eps: float, output: fastllm.Tensor=None):
    output = fastllm.rms_norm(input, weight, eps)
    return output

def layer_norm(input: fastllm.Tensor, 
               gamma: fastllm.Tensor, 
               beta: fastllm.Tensor, 
               axis:int=-1 ):
    output = fastllm.layer_norm(input, gamma, beta,axis)
    return output

def linear(input: fastllm.Tensor, 
           weight: fastllm.Tensor, 
           bias: fastllm.Tensor):
    output = fastllm.linear(input, weight, bias)
    return output

def matmul(input0: fastllm.Tensor, 
           input1: fastllm.Tensor, 
           alpha: fastllm.Tensor):
    output = fastllm.matmul(input0, input1, alpha)
    return output

def attention(q: fastllm.Tensor, 
              k: fastllm.Tensor, 
              v: fastllm.Tensor, 
              mask: fastllm.Tensor,
              group: int, 
              scale: float, 
              attentionType: int):
    output = fastllm.attention(q, k, v, mask, group, scale, attentionType)
    return output

def activation(input: fastllm.Tensor, axis=-1, activate_type="silu"):
    assert activate_type in ("softmax", "silu", "gelu", "swiglu")
    func = getattr(fastllm, activate_type)
    if activate_type == "softmax":
        return func(input, axis)
    return func(input)

def mul(input: fastllm.Tensor, v: int):
    output = fastllm.mul(input, v)
    return output

def matmul_transB():
    pass

def add(input0: fastllm.Tensor, input1: fastllm.Tensor):
    output = fastllm.add(input0, input1)
    return output

def AttentionMask():
    pass

def AlibiMask():
    pass

def topk():
    pass

def RotatePosition2D():
    pass

def NearlyRotatePosition2D():
    pass

def LlamaRotatePosition2D():
    pass

def RepeatPenalty():
    pass
