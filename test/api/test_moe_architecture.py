import os
import sys
import unittest


TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools.util import _is_moe_architecture


class MoeArchitectureTest(unittest.TestCase):
    def test_laguna_architecture_is_moe(self):
        self.assertTrue(
            _is_moe_architecture("LagunaForCausalLM")
        )

    def test_laguna_model_type_is_moe(self):
        self.assertTrue(
            _is_moe_architecture("UnknownArchitecture", "laguna")
        )

    def test_kimi_k3_architecture_is_moe(self):
        self.assertTrue(
            _is_moe_architecture("KimiK3ForConditionalGeneration")
        )

    def test_kimi_k3_model_type_is_moe(self):
        self.assertTrue(
            _is_moe_architecture("UnknownArchitecture", "kimi_k3")
        )

    def test_dense_model_is_not_moe(self):
        self.assertFalse(
            _is_moe_architecture("LlamaForCausalLM", "llama")
        )


if __name__ == "__main__":
    unittest.main()
