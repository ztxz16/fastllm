import os
import sys
SCRIPT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test", "multimodal")
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from fastllm_multimodal_check import main


if __name__ == "__main__":
    main()
