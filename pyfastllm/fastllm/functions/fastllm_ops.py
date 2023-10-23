import pyfastllm


def embedding(data: pyfastllm.Tensor, ):
    # some check
    return pyfastllm.embedding(data, )

def rms_norm(input:pyfastllm.Tensor, weight: pyfastllm.Tensor, eps: float, output: pyfastllm.Tensor=None):
    output = pyfastllm.rms_norm(input, weight, eps)
    return output

def layer_norm(input: pyfastllm.Tensor, 
               gamma: pyfastllm.Tensor, 
               beta: pyfastllm.Tensor, 
               axis:int=-1 ):
    output = pyfastllm.layer_norm(input, gamma, beta,axis)
    return output

def linear(input: pyfastllm.Tensor, 
           weight: pyfastllm.Tensor, 
           bias: pyfastllm.Tensor):
    output = pyfastllm.linear(input, weight, bias)
    return output

def matmul(input0: pyfastllm.Tensor, 
           input1: pyfastllm.Tensor, 
           alpha: pyfastllm.Tensor):
    output = pyfastllm.matmul(input0, input1, alpha)
    return output

def attention(q: pyfastllm.Tensor, 
              k: pyfastllm.Tensor, 
              v: pyfastllm.Tensor, 
              mask: pyfastllm.Tensor,
              group: int, 
              scale: float, 
              attentionType: int):
    output = pyfastllm.attention(q, k, v, mask, group, scale, attentionType)
    return output

def activation(input: pyfastllm.Tensor, axis=-1, activate_type="silu"):
    assert activate_type in ("softmax", "silu", "gelu", "swiglu")
    func = getattr(pyfastllm, activate_type)
    if activate_type == "softmax":
        return func(input, axis)
    return func(input)

def mul(input: pyfastllm.Tensor, v: int):
    output = pyfastllm.mul(input, v)
    return output

def matmul_transB():
    pass

def add(input0: pyfastllm.Tensor, input1: pyfastllm.Tensor):
    output = pyfastllm.add(input0, input1)
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
