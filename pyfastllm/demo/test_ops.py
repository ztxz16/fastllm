import fastllm
import numpy as np

def np_rms_norm(inputs, weights, eps):
    channel = inputs.shape[-1]
    sqrt_mean = np.sqrt(np.sum(inputs**2)/channel + eps)
    return inputs / sqrt_mean *weights


def np_layer_norm(inputs, gamma, beta, axis=-1):
    assert axis < len(inputs.shapes), "axis should less than inputs dims"
    channel = inputs.shape[axis]
    mean = np.mean(inputs, axis=axis)
    var = np.var(inputs, axis=axis)

    output = (inputs - mean) / var * gamma + beta
    return output

def np_linear(inputs, weights, bias):
    output = np.matmul(inputs, weights.T) + bias
    return output

def np_softmax(inputs, axis=None):
    maxv = inputs.max(axis, keepdims=True)
    exp_v = np.exp(inputs - maxv)
    exp_sum = np.sum(exp_v, axis=axis)
    return exp_v / exp_sum

def np_silu(inputs, ):
    return inputs / (1 + np.exp(-inputs))
    
def np_attention(q, k, v, mask=None, group=None, scale=None):
    qk = np_softmax(q @ k.T * scale, axis=-1)
    attn = qk @ v
    return attn

def test_linear():
    inputs = np.array([[1, 2]])
    weight = np.array([[3, 4, 5, 5, 6, 7]]).reshape([3, 2])
    bias = np.array([0, 1, 1])
    np_output = np_linear(inputs, weight, bias)
    print(np_output)

    input = fastllm.Tensor(fastllm.float32, [1, 2], [1, 2])
    weights = fastllm.Tensor(fastllm.float32, [3, 2], [3, 4, 5, 5, 6, 7])
    bias = fastllm.Tensor(fastllm.float32, [3], [0, 1, 1])
    out = fastllm.ops.linear(input, weights, bias)
    print(out)

def test_rms_norm():
    inputs = np.array([1, 5]).reshape([1, 2])
    weights = np.array([1, 3]).reshape([1, 2])
    eps = 1e-6

    np_out = np_rms_norm(inputs, weights, eps)
    print(np_out)

    input = fastllm.Tensor(fastllm.float32, [1, 2], [1, 5])
    weights = fastllm.Tensor(fastllm.float32, [1, 2], [1, 3])
    out = fastllm.Tensor()
    out = fastllm.ops.rms_norm(input, weights, eps=1e-6)
    print(out)

def test_silu():
    inputs = np.array([1, 5]).reshape([1, 2])
    output = np_softmax(inputs)
    # output = np_silu(inputs)
    print(output)

    inputs = fastllm.Tensor(fastllm.float32, [1, 2], [1, 5])
    out = fastllm.ops.activation(input=inputs, activate_type="softmax")
    # out = fastllm.ops.activation(input=inputs, activate_type="silu")
    print(out)

def test_attention():
    q = np.array([1, 2, 3, 4, 5, 6]).reshape([2, 3])
    k = np.array([5, 6, 7, 8, 9, 10]).reshape([2, 3])
    v = np.array([1, 1, 1, 2, 1, 3]).reshape([2, 3])
    scale = 1 / np.sqrt(q.shape[-1])
    output = np_attention(q, k, v, scale=scale)
    print(output)

    q = fastllm.Tensor(fastllm.float32, [1, 2, 3], [1, 2, 3, 4, 5, 6])
    k = fastllm.Tensor(fastllm.float32, [1, 2, 3], [5, 6, 7, 8, 9, 10])
    v = fastllm.Tensor(fastllm.float32, [1, 2, 3], [1, 1, 1, 2, 1, 3])
    mask = fastllm.Tensor()
    output = fastllm.ops.attention(q, k, v, mask, group=1, scale=scale, attentionType=0)
    print(output)

test_attention()
test_silu()
test_linear()
test_rms_norm()
