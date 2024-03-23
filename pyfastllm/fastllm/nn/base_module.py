import pyfastllm
from typing import Any
from abc import abstractmethod

class Module():
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **kwds)
    
    @abstractmethod
    def forward(self, ):
        pass

    def _init_weight(self, ):
        pass

import numpy as np
from typing import Union, Sequence
from pyfastllm import Tensor
from ..functions import util

class Parameter(object):
    _DEFAULT_DTYPE = pyfastllm.float32

    def __init__(self,
                 value: Union[np.ndarray, None] = None,
                 shape: Sequence[int] = None,
                 dtype: Union[pyfastllm.DataType, None] = None):
        dtype = self._DEFAULT_DTYPE if dtype is None else dtype
        if value is None:
            assert isinstance(shape, (list, tuple))
            self._value = pyfastllm.Tensor()
            """
            value = np.zeros(shape=shape, dtype=np.float32)
        
            if len(shape) == 2:
                v_range = np.sqrt(6) / np.sqrt(shape[0] + shape[1])
            else:
                v_range = 0.1

            # value ~ U[-1, 1]
            value = np.random.random(size=shape) * 2 - 1
            value = np.array(value, dtype=np.float32)
            # value ~ U[-v_range, v_range]
            value *= v_range
            """
        else:
            self._value = util.to_tensor(value)

    @property
    def value(self) -> Tensor:
        if isinstance(self._value, np.ndarray):
            self._value = util.to_tensor(self._value)

        return self._value

    @value.setter
    def value(self, v: np.ndarray):
        assert isinstance(v, np.ndarray) or isinstance(v, pyfastllm.Tensor)
        # assert v.shape == self._value.shape, \
        #     ('The value updated is not the same shape as the original. ', \
        #     f'Updated: {v.shape}, original: {self._value.shape}')
        self._value = v
