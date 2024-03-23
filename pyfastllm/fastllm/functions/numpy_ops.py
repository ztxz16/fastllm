import numpy as np
from numba import cuda
from numba import jit

@jit(nopython=True)
def rms_norm(inputs, weights, eps):
    channel = inputs.shape[-1]
    sqrt_mean = np.sqrt(np.sum(inputs**2)/channel + eps)
    return inputs / sqrt_mean *weights

@jit(nopython=True)
def layer_norm(inputs, gamma, beta, axis=-1):
    assert axis < len(inputs.shapes), "axis should less than inputs dims"
    channel = inputs.shape[axis]
    mean = np.mean(inputs, axis=axis)
    var = np.var(inputs, axis=axis)

    output = (inputs - mean) / var * gamma + beta
    return output

# @jit
def softmax(inputs, axis=None):
    maxv = inputs.max(axis, keepdims=True)
    exp_v = np.exp(inputs - maxv)
    exp_sum = np.sum(exp_v, axis=axis)
    return exp_v / exp_sum

@jit(nopython=True)
def silu(inputs, ):
    return inputs / (1 + np.exp(-inputs))

@jit
def swiglu(inputs, ):
    dim = inputs.shape[1] // 2 
    for batch in range(inputs.shape[0]):
        return inputs[batch, :dim] / (1 + np.exp(-inputs[batch, :dim])) * inputs[batch, dim:]

# @jit
def linear(inputs, weights, bias):
    if len(inputs.shape) == 2:
        inputs = inputs[None, :]
        weights = weights[None, :]
    
    output = np.zeros(shape=[inputs.shape[0], inputs.shape[1], weights.shape[0]])
    for batch in range(inputs.shape[0]):
        output[batch] = np.matmul(inputs[batch], weights.T)

        if bias:
            output[batch] += bias[batch]

    return output

# @jit
def attention(q, k, v, mask=None, group=None, scale=None):
    print("shape:", q.shape)
    if len(q.shape) == 2:
        q = q[None, :]
        k = k[None, :]
        v = v[None, :]
        # mask = mask[None, :]
    
    attn = np.zeros_like(q)
    for batch in range(q.shape[0]):
        qk = softmax(q[batch] @ k[batch].T * scale, axis=-1)
        attn[batch, :, :] = qk @ v[batch]
    return attn