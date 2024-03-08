import pyfastllm
from typing import Any
from abc import abstractmethod

class Module():
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **args)
    
    @abstractmethod
    def forward(self, ):
        pass

    def _init_weight(self, ):
        pass
