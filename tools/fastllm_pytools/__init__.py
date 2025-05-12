__all__ = ["llm"]

from importlib.metadata import version
try:
    __version__ = version("ftllm")  # 从安装的元数据读取
except:
    __version__ = version("ftllm-rocm")  # 从安装的元数据读取