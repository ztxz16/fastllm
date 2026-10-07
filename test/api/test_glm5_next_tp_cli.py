"""GLM resident expert layers inherit the complete requested TP group."""
import io
import json
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tools"))
from fastllm_pytools.util import make_normal_llm_model, make_normal_parser


class Glm5NextTpCliTest(unittest.TestCase):
    def test_layered_experts_use_all_tp_devices_and_keep_explicit_placement(self):
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, "config.json").write_text(json.dumps({"model_type": "glm5_next"}))
            for tp in ("0,1", "1,0"):
                for layered in (False, True):
                    with self.subTest(tp=tp, layered=layered):
                        placement = "numa" if layered else '{"cuda:0":9,"cuda:1":6,"numa":30}'
                        argv = [directory, "--device", "cuda:0", "--tp", tp,
                                "--moe_device", placement, "--mtp", "0"]
                        if layered:
                            argv += ["--moe_device_layers", "30"]
                        args = make_normal_parser("test").parse_args(argv)
                        llm = MagicMock()
                        llm.model.return_value.get_max_input_len.return_value = 4096
                        llm.model.return_value.get_max_batch.return_value = 1
                        with patch.dict(os.environ, {}, clear=True), \
                                patch.dict(sys.modules, {"ftllm": SimpleNamespace(llm=llm)}), \
                                redirect_stdout(io.StringIO()):
                            make_normal_llm_model(args)
                        if layered:
                            llm.set_device_map.assert_any_call("cuda:" + tp, True)
                            llm.set_layered_moe_device_map.assert_called_once_with("numa")
                            llm.set_moe_device_layers.assert_called_with(30)
                        else:
                            llm.set_device_map.assert_any_call(json.loads(placement), True)
                            llm.set_layered_moe_device_map.assert_not_called()


if __name__ == "__main__":
    unittest.main()
