"""兼容旧脚本路径；实现由 FastLLM Python 包统一提供。"""
import sys
from fastllm_pytools import qwen3_process_session as _implementation

sys.modules[__name__] = _implementation
