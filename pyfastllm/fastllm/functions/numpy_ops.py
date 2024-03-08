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

@jit(nopython=True)
def softmax(inputs, axis=None):
    maxv = inputs.max(axis, keepdims=True)
    exp_v = np.exp(inputs - maxv)
    exp_sum = np.sum(exp_v, axis=axis)
    return exp_v / exp_sum

@jit(nopython=True)
def silu(inputs, ):
    return inputs / (1 + np.exp(-inputs))

@jit(nopython=True)
def linear(inputs, weights, bias):
    output = np.matmul(inputs, weights.T) + bias
    return output

@jit(nopython=True)    
def np_self_attention(q, k, v, mask=None, group=None, scale=None):
    qk = softmax(q @ k.T * scale, axis=-1)
    attn = qk @ v
    return attn