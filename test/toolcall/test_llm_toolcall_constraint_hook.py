import json
import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


_FASTLLM_PYTOOLS = REPO_ROOT / "tools" / "fastllm_pytools"
_NATIVE_LIBRARY_NAMES = (
    "libfastllm_tools.so",
    "libfastllm_tools-cu11.so",
    "libfastllm_tools-cpu.so",
    "fastllm_tools.dll",
    "libfastllm_tools.dylib",
)


def _import_llm_or_skip():
    if not any((_FASTLLM_PYTOOLS / name).exists()
               for name in _NATIVE_LIBRARY_NAMES):
        raise unittest.SkipTest(
            "fastllm native library is not available for llm.py hook tests")
    try:
        from tools.fastllm_pytools import llm as llm_module
    except SystemExit as exc:
        raise unittest.SkipTest(
            "fastllm native library could not be loaded for llm.py hook tests"
        ) from exc
    return llm_module


class _NativeSetter:
    def __init__(self, result=True):
        self.result = result
        self.calls = []

    def __call__(self, model_id, payload):
        self.calls.append((model_id, payload))
        return self.result


class _FakeNativeLib:
    def __init__(self, setter=None):
        if setter is not None:
            self.set_tool_call_constraint_llm_model = setter


class LlmToolCallConstraintHookTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.llm_module = _import_llm_or_skip()

    def _model(self):
        model = self.llm_module.model.__new__(self.llm_module.model)
        model.model = 123
        return model

    def test_native_tool_call_constraint_setter_receives_json(self):
        setter = _NativeSetter(result=True)
        original_lib = self.llm_module.fastllm_lib
        self.llm_module.fastllm_lib = _FakeNativeLib(setter)
        constraint = {
            "name_constraint": {
                "type": "tool_name_enum",
                "allowed_names": ["get_weather"],
            }
        }
        try:
            applied = self._model()._apply_tool_call_constraint_to_native(
                constraint)
        finally:
            self.llm_module.fastllm_lib = original_lib

        self.assertTrue(applied)
        self.assertEqual(len(setter.calls), 1)
        model_id, payload = setter.calls[0]
        self.assertEqual(model_id, 123)
        self.assertEqual(json.loads(payload.decode()), constraint)

    def test_native_tool_call_constraint_setter_clears_on_none(self):
        setter = _NativeSetter(result=True)
        original_lib = self.llm_module.fastllm_lib
        self.llm_module.fastllm_lib = _FakeNativeLib(setter)
        try:
            applied = self._model()._apply_tool_call_constraint_to_native(None)
        finally:
            self.llm_module.fastllm_lib = original_lib

        self.assertTrue(applied)
        self.assertEqual(len(setter.calls), 1)
        model_id, payload = setter.calls[0]
        self.assertEqual(model_id, 123)
        self.assertIsNone(json.loads(payload.decode()))

    def test_missing_native_setter_reports_not_applied(self):
        original_lib = self.llm_module.fastllm_lib
        self.llm_module.fastllm_lib = _FakeNativeLib()
        try:
            with self.assertLogs(level="DEBUG") as logs:
                applied = self._model()._apply_tool_call_constraint_to_native(
                    {"name_constraint": {"allowed_names": ["get_weather"]}})
        finally:
            self.llm_module.fastllm_lib = original_lib

        self.assertFalse(applied)
        self.assertIn("does not expose set_tool_call_constraint_llm_model",
                      "\n".join(logs.output))


if __name__ == "__main__":
    unittest.main()
