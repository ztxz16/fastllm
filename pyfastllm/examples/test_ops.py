import sys
import pytest
import numpy as np
import fastllm

import pyfastllm
import gc

import np_ops
import ops as flm_ops

# from fastllm import ops as flm_ops
# from fastllm import np_ops

np.random.seed(42)

def diff(dataA, dataB):
    # print(dataA)
    # print(dataB)
    mae = np.max(np.abs(dataA - dataB))
    print('max abs err is ', mae)
    return mae

def to_tensor(data):
    return pyfastllm.from_numpy(data)

def to_numpy(data):
    # return data.numpy()
    return np.array(data, copy=False, order='C')

def test_rms_norm(inputs=None, weights=None, eps=1e-6):
    if not inputs:
        inputs =  np.random.random(size=[1, 256])
        weights = np.random.random(size=[1, 256])

    np_out = np_ops.rms_norm(inputs, weights, eps)
    flm_out = flm_ops.rms_norm(to_tensor(inputs), to_tensor(weights), eps)
    mae = diff(np_out, to_numpy(flm_out))
    assert mae <= 1e-6
    return flm_out

def test_swiglu(inputs=None):
    if not inputs:
        inputs = np.random.random(size=[1, 256])
    
    np_out = np_ops.swiglu(inputs)
    out = flm_ops.activation(inputs=to_tensor(inputs), activate_type="swiglu")
    mae = diff(np_out, to_numpy(out))
    assert mae <= 1e-6
    return out
    
def test_attention(q=None, k=None, v=None, mask=None, group=1, scale=1.0):
    if q is None:
        q = np.random.random(size=[12, 1, 4096])
        k = np.random.random(size=[12, 1, 4096])
        v = np.random.random(size=[12, 1, 4096])
        scale = 1 / np.sqrt(q.shape[-1])

    np_out = np_ops.attention(q, k, v, scale=scale)

    mask = fastllm.Tensor()
    flm_out = flm_ops.attention(to_tensor(q), to_tensor(k), to_tensor(v), mask, group=group, scale=scale, attentionType=0)

    mae = diff(np_out, to_numpy(flm_out))
    assert mae <= 1e-6
    return flm_out


def test_linear(inputs=None, 
           weights=None,
           bias=None):
    
    if not inputs:
        inputs = np.random.random(size=[1, 12, 4096])
        weights = np.random.random(size=[256, 4096])
        
    np_out = np_ops.linear(inputs=inputs, weights=weights, bias=None)

    if not bias:
        bias = fastllm.Tensor()

    output = flm_ops.linear(to_tensor(inputs), to_tensor(weights), bias)
    mae = diff(np_out, to_numpy(output))

    assert mae <= 1e-3
    return output


if __name__ == "__main__":    
    test_rms_norm()
    test_attention()
    test_linear()
    test_swiglu()


