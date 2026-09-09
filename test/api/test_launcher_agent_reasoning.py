import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from fastllm_pytools.launcher_agent_models import reasoning_options, with_model_metadata
from fastllm_pytools.launcher_codex import CodexRuntime
from fastllm_pytools.launcher_harness import HarnessRuntime
from fastllm_pytools.launcher_opencode import OpenCodeRuntime
from fastllm_pytools.openai_server.fastllm_model import FastLLmModel
from test_model_context_metadata import _FakeModel


SERVICE = {"modelName": "my-alias", "endpoint": "http://127.0.0.1:18001", "contextWindowTokens": 32768}


class AgentReasoningTest(unittest.TestCase):
    def test_model_discovery_uses_authenticated_metadata_and_exact_model_id(self):
        metadata = FastLLmModel("my-alias", _FakeModel(32768, 32768, model_type="qwen3_5")).response
        metadata["data"].insert(0, {"id": "other-model", "supported_reasoning_efforts": ["high"]})
        with patch("urllib.request.urlopen", return_value=io.BytesIO(json.dumps(metadata).encode())) as opening:
            service = with_model_metadata(SERVICE, "test-api-key")
        request = opening.call_args.args[0]
        self.assertEqual(request.full_url, SERVICE["endpoint"] + "/v1/models")
        self.assertEqual(request.get_header("Authorization"), "Bearer test-api-key")
        self.assertEqual(reasoning_options(service), (["low", "medium", "xhigh"], "xhigh"))
        self.assertNotIn("modelMetadata", SERVICE)

    def test_absent_or_unusable_metadata_does_not_invent_efforts(self):
        for payload in ({"data": [{"id": "other", "supported_reasoning_efforts": ["high"]}]}, [], "broken"):
            with self.subTest(payload=payload), patch("urllib.request.urlopen",
                    return_value=io.BytesIO(json.dumps(payload).encode())):
                self.assertEqual(reasoning_options(with_model_metadata(SERVICE, "")), ([], None))
        with patch("urllib.request.urlopen", side_effect=TimeoutError):
            self.assertEqual(with_model_metadata(SERVICE, ""), SERVICE)
        for metadata in ({}, {"supported_reasoning_efforts": "high"},
                         {"supported_reasoning_efforts": ["unknown", None, {}, []]}):
            self.assertEqual(reasoning_options({"modelMetadata": metadata}), ([], None))

    def test_object_presets_are_normalized_and_invalid_default_is_replaced(self):
        self.assertEqual(reasoning_options({"modelMetadata": {
            "supportedReasoningEfforts": [{"reasoningEffort": "max"}, {"effort": "low"}, "low"],
            "defaultReasoningEffort": "medium",
        }}), (["low", "max"], "max"))

    def test_native_agent_catalogs_offer_only_the_models_native_efforts(self):
        for model_type, expected, default in (("qwen3_5", ["low", "medium", "xhigh"], "xhigh"),
                ("kimi_k3", ["low", "high", "max"], "max"), (None, [], None)):
            with self.subTest(model_type=model_type), tempfile.TemporaryDirectory() as directory:
                service = dict(SERVICE, modelMetadata=FastLLmModel("my-alias",
                    _FakeModel(32768, 32768, model_type=model_type)).response["data"][0])
                codex = CodexRuntime(directory)
                args = codex._configure(service, directory)
                catalog = json.loads((Path(directory) / "models.json").read_text())["models"][0]
                self.assertEqual([p["effort"] for p in catalog["supported_reasoning_levels"]], expected)
                self.assertEqual(catalog["default_reasoning_level"], default)
                if default:
                    self.assertIn('model_reasoning_effort="' + default + '"', args)
                opencode = OpenCodeRuntime.configuration(service)["provider"]["fastllm"]["models"]["my-alias"]
                self.assertEqual(opencode["reasoning"], bool(expected))
                self.assertEqual({key: value for key, value in opencode["variants"].items() if not value.get("disabled")},
                                 {effort: {"reasoningEffort": effort} for effort in expected})
                if model_type == "qwen3_5":
                    self.assertEqual(opencode["variants"]["high"], {"disabled": True})
                harness = {p["id"]: p for p in HarnessRuntime._patch(service, "127.0.0.1") if "id" in p}
                provider = harness["llm-pi-ai"]["config"]["providers"]["fastllm"]
                self.assertEqual(provider.get("reasoning"), default)
                self.assertEqual(provider["models"][0]["reasoningEfforts"],
                    {effort: effort for effort in expected} if expected else False)
                self.assertEqual(harness["agent-default-model"]["config"].get("reasoningEffort"), default)

    def test_codex_forwards_supported_effort_and_rejects_invalid_or_stale_selections(self):
        with tempfile.TemporaryDirectory() as directory:
            runtime = CodexRuntime(directory)
            runtime._service = dict(SERVICE, modelMetadata={"supported_reasoning_efforts": ["low", "medium", "xhigh"]})
            connection = Mock()
            with patch.object(runtime, "_ready", return_value=connection):
                runtime.rpc("turn/start", {"threadId": "thread-a", "text": "Hello", "effort": "xhigh"})
                self.assertEqual(connection.call.call_args.args, ("turn/start", {
                    "threadId": "thread-a", "effort": "xhigh",
                    "input": [{"type": "text", "text": "Hello", "text_elements": []}]}))
                for effort in ("high", "max", "", None, {}):
                    with self.subTest(effort=effort), self.assertRaises(RuntimeError):
                        runtime.rpc("turn/start", {"threadId": "thread-a", "text": "Hello", "effort": effort})
                runtime._service = SERVICE
                with self.assertRaisesRegex(RuntimeError, "not supported"):
                    runtime.rpc("turn/start", {"threadId": "thread-a", "text": "Hello", "effort": "xhigh"})
                runtime.rpc("turn/start", {"threadId": "thread-a", "text": "Hello"})
                self.assertNotIn("effort", connection.call.call_args.args[1])
