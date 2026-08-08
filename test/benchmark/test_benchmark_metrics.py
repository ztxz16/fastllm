import importlib
import json
import sys
import tempfile
import types
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
TOOLS_DIR = REPO_ROOT / "tools"
if str(TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(TOOLS_DIR))

benchmark = importlib.import_module("fastllm_pytools.benchmark")


class FakeFastllmLib:
    def __init__(self):
        self.next_handle = 1
        self.finished = set()
        self.fetches = {}

    def launch_response_llm_model(self, *args):
        handle = self.next_handle
        self.next_handle += 1
        self.fetches[handle] = [100 + handle, -1]
        return handle

    def can_fetch_response_llm_model(self, model, handle):
        return True

    def fetch_response_llm_model(self, model, handle):
        token = self.fetches[handle].pop(0)
        if token <= -1:
            self.finished.add(handle)
        return token


class FakeModel:
    model = 7

    def __init__(self, fastllm_lib):
        self.fastllm_lib = fastllm_lib

    def stop_token_ctypes(self, stop_token_ids):
        return 0, None

    def get_response_statistics(self, handle):
        if handle in self.fastllm_lib.finished:
            return None
        return {
            "cached_input_tokens": handle - 1,
            "missed_input_tokens": 3 - (handle - 1),
            "output_tokens": 1,
        }


def test_run_batch_records_native_cache_statistics():
    fake_llm = types.ModuleType("fastllm_pytools.llm")
    fake_llm.fastllm_lib = FakeFastllmLib()
    model = FakeModel(fake_llm.fastllm_lib)
    original_llm = sys.modules.get("fastllm_pytools.llm")
    sys.modules["fastllm_pytools.llm"] = fake_llm
    try:
        result = benchmark._run_batch(
            model,
            [1, 2, 3],
            output_tokens=1,
            batch=2,
            generation_args={
                "do_sample": False,
                "top_p": 1.0,
                "top_k": 1,
                "temperature": 1.0,
                "repeat_penalty": 1.0,
            },
        )
    finally:
        if original_llm is None:
            sys.modules.pop("fastllm_pytools.llm", None)
        else:
            sys.modules["fastllm_pytools.llm"] = original_llm

    assert result["response_statistics_available"] is True
    assert result["cached_input_tokens"] == 1
    assert result["missed_input_tokens"] == 5
    assert result["native_output_tokens"] == 2
    assert result["requests"][0]["response_statistics"] == {
        "cached_input_tokens": 0,
        "missed_input_tokens": 3,
        "output_tokens": 1,
    }


def test_write_result_json_is_machine_readable():
    result = {"input_tokens": 3, "requests": [{"finish_code": -1}]}
    with tempfile.TemporaryDirectory() as temp_dir:
        output = Path(temp_dir) / "result.json"
        benchmark._write_result_json(output, result)
        assert json.loads(output.read_text(encoding="utf-8")) == result
        assert not (Path(str(output) + ".tmp")).exists()


if __name__ == "__main__":
    test_run_batch_records_native_cache_statistics()
    test_write_result_json_is_machine_readable()
    print("benchmark metrics tests passed")
