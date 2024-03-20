import numpy as np
import pyfastllm

def diff(dataA, dataB):
    mae =  np.max(np.abs(dataA - dataB))
    print('max abs err is ', mae)
    return mae

def to_tensor(data):
    if not isinstance(data, np.ndarray):
        return None
    return pyfastllm.from_numpy(data)

def to_numpy(data):
    if not isinstance(data, pyfastllm.Tensor):
        return None
    
    return np.array(data, copy=False)