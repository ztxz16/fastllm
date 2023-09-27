import os
import sys
import ctypes
import glob

_BASE_DIR = os.path.dirname(__file__)
sys.path.append(_BASE_DIR)
# libs = glob.glob("*.so")
# for lib in libs: _cdll = ctypes.cdll.LoadLibrary(lib)

from pyfastllm import *
from . import utils
from . import functions as ops

