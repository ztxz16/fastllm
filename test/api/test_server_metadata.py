import asyncio
import os
import sys
import unittest

from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.requests import Request


TEST_API_DIR = os.path.abspath(os.path.dirname(__file__))
sys.path = [path for path in sys.path
            if os.path.abspath(path or os.getcwd()) != TEST_API_DIR]
TOOLS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "tools")
)
if TOOLS_DIR not in sys.path:
    sys.path.insert(0, TOOLS_DIR)

from fastllm_pytools import server


class FakeCompletion:
    def __init__(self, active=0):
        self.active = active

    def get_active_conversations(self):
        return [{"id": i} for i in range(self.active)]


class ServerMetadataTest(unittest.TestCase):
    def setUp(self):
        self.old_info = dict(server.runtime_server_info)
        self.old_completion = getattr(server, "fastllm_completion", None)
        server.fastllm_completion = FakeCompletion(active=1)
        server.runtime_server_info.update({
            "ready": True,
            "model": "qwen-test",
            "max_batch": 2,
            "token_pool": 524288,
            "kv_cache_dtype": "turbo3",
            "activation_dtype": "float16",
            "default_max_tokens": 16384,
        })

    def tearDown(self):
        server.runtime_server_info.clear()
        server.runtime_server_info.update(self.old_info)
        if self.old_completion is None:
            try:
                del server.fastllm_completion
            except AttributeError:
                pass
        else:
            server.fastllm_completion = self.old_completion

    def test_health_version_and_props_payloads(self):
        health = server.build_health_payload()
        self.assertEqual(health, {
            "status": "ok",
            "ready": True,
            "accepting": True,
            "active_requests": 1,
            "queued_requests": 0,
            "model": "qwen-test",
        })
        self.assertEqual(server.build_version_payload(), {
            "name": "fastllm", "version": "unknown", "build": "unknown"})
        self.assertEqual(server.build_props_payload(), {
            "model": "qwen-test",
            "max_batch": 2,
            "token_pool": 524288,
            "kv_cache_dtype": "turbo3",
            "activation_dtype": "float16",
            "default_max_tokens": 16384,
            "backend": "fastllm",
        })

    def test_not_ready_is_not_accepting(self):
        server.runtime_server_info["ready"] = False
        health = server.build_health_payload()
        self.assertEqual(health["status"], "loading")
        self.assertFalse(health["accepting"])

    def test_normalized_404_and_405(self):
        async def invoke(status, method, path, headers=None):
            scope = {
                "type": "http",
                "method": method,
                "path": path,
                "raw_path": path.encode(),
                "query_string": b"",
                "headers": [],
                "server": ("test", 80),
                "client": ("test", 1),
                "scheme": "http",
            }
            request = Request(scope)
            return await server.normalized_http_error(
                request,
                StarletteHTTPException(
                    status_code=status, headers=headers))

        not_found = asyncio.run(invoke(404, "GET", "/missing"))
        self.assertEqual(not_found.status_code, 404)
        self.assertIn(b'"code":"not_found"', not_found.body)
        self.assertIn(b'"type":"invalid_request_error"', not_found.body)

        method = asyncio.run(invoke(
            405, "POST", "/health", {"Allow": "GET"}))
        self.assertEqual(method.status_code, 405)
        self.assertEqual(method.headers["allow"], "GET")
        self.assertIn(b'"code":"method_not_allowed"', method.body)


if __name__ == "__main__":
    unittest.main()
