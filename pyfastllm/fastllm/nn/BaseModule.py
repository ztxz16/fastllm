from typing import Any


class Module():
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.forward(*args, **args)
    
    @classmethod
    def forward(self, ):
        pass

    def _init_weight(self, ):
        pass

    
