import json
import os

class FastllmEnv:
    _instance = None
    _build_info = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def build_info(self):
        if self._build_info is None:
            info_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "build_info.json")
            try:
                with open(info_path, "r") as f:
                    self._build_info = json.load(f)
            except:
                self._build_info = {}
        return self._build_info

    def has_build_flag(self, flag: str) -> bool:
        return self.build_info.get(flag, False)

    @property
    def use_cuda(self) -> bool:
        return self.has_build_flag("USE_CUDA")

    @property
    def use_rocm(self) -> bool:
        return self.has_build_flag("USE_ROCM")

    @property
    def use_numa(self) -> bool:
        return self.has_build_flag("USE_NUMA")

    @property
    def use_numas(self) -> bool:
        return self.has_build_flag("USE_NUMAS")

    @property
    def use_tfacc(self) -> bool:
        return self.has_build_flag("USE_TFACC")

    @property
    def use_tops(self) -> bool:
        return self.has_build_flag("USE_TOPS")

    @property
    def use_ivcorex(self) -> bool:
        return self.has_build_flag("USE_IVCOREX")

env = FastllmEnv()
