import json
from pathlib import Path
import urllib.error
import urllib.request

import pytest

from ftllm_agent_runtime.runtime import (
    PiAgentError,
    PiAgentRuntime,
    _WebToolBridge,
    _normalize_api_base,
    _safe_name,
)


def _fake_pi_binary(tmp_path: Path, events=()) -> Path:
    binary = tmp_path / "pi"
    encoded_events = json.dumps(list(events))
    binary.write_text(
        f"""#!/usr/bin/env python3
import json
import sys

if "--version" in sys.argv:
    print("0.84.4")
else:
    json.loads(sys.stdin.readline())
    for event in json.loads({encoded_events!r}):
        print(json.dumps(event), flush=True)
""",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    return binary


def test_normalize_api_base():
    assert _normalize_api_base("http://localhost:8000") == "http://localhost:8000/v1"
    assert _normalize_api_base("http://localhost:8000/v1/") == "http://localhost:8000/v1"


def test_safe_name_removes_directories_and_special_characters():
    assert _safe_name("../../hello world.py", "fallback") == "hello_world.py"


def test_project_files_are_copied_to_numbered_snapshot(tmp_path: Path):
    manifest = PiAgentRuntime._copy_project_files(
        tmp_path,
        [{"name": "../unsafe.py", "text": "print('ok')\n", "size": 12}],
    )
    assert manifest[0]["index"] == 1
    assert manifest[0]["name"] == "unsafe.py"
    copied = Path(manifest[0]["path"])
    assert copied.parent == tmp_path / "project"
    assert copied.read_text(encoding="utf-8") == "print('ok')\n"


def test_project_files_require_at_least_one_file(tmp_path: Path):
    with pytest.raises(ValueError, match="at least one"):
        PiAgentRuntime._copy_project_files(tmp_path, [])


def test_runtime_keeps_api_key_out_of_process_arguments(tmp_path: Path):
    binary = _fake_pi_binary(tmp_path)
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        api_key="top-secret",
        binary=str(binary),
    )

    command = runtime._command("system prompt", "off", ["runtime_info"])

    assert "--api-key" not in command
    assert "top-secret" not in command


def test_runtime_can_finish_on_the_configured_turn_limit(tmp_path: Path):
    binary = _fake_pi_binary(tmp_path, [
        {"type": "turn_start"},
        {
            "type": "message_update",
            "assistantMessageEvent": {"type": "text_delta", "delta": "Done"},
        },
        {"type": "turn_end"},
        {"type": "agent_end"},
    ])
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        max_turns=1,
        binary=str(binary),
    )

    events = list(runtime.stream(
        "Inspect the project.",
        [{"name": "app.py", "text": "answer = 42\n"}],
        "Use the read-only project tools.",
    ))

    assert events[-2:] == [
        {"type": "text_delta", "text": "Done"},
        {"type": "done", "turns": 1},
    ]


def test_runtime_rejects_an_empty_agent_response(tmp_path: Path):
    binary = _fake_pi_binary(tmp_path, [
        {"type": "turn_start"},
        {"type": "turn_end"},
        {"type": "agent_end"},
    ])
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        binary=str(binary),
    )

    with pytest.raises(PiAgentError, match="without assistant output"):
        list(runtime.stream(
            "Inspect the project.",
            [{"name": "app.py", "text": "answer = 42\n"}],
            "Use the read-only project tools.",
        ))


def test_web_tool_bridge_registers_sources_and_reads_by_index():
    class FakeBackend:
        def search(self, query, limit=6):
            self.query = query
            self.limit = limit
            return [{
                "title": "Game result",
                "url": "https://example.com/game",
                "snippet": "China won 89-87",
            }]

        def read_page(self, url, limit=7000):
            self.url = url
            self.read_limit = limit
            return "Official final score: 89-87"

    backend = FakeBackend()
    bridge = _WebToolBridge(backend)

    def request(payload, token):
        req = urllib.request.Request(
            bridge.url + "/tool",
            data=json.dumps(payload).encode(),
            headers={
                "Authorization": f"Bearer {token}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=2) as response:
            return json.load(response)

    try:
        with pytest.raises(urllib.error.HTTPError) as forbidden:
            request({"action": "search", "query": "game"}, "wrong-token")
        assert forbidden.value.code == 403
        search = request({
            "action": "search", "query": "China Lebanon 2026", "limit": 4,
        }, bridge.token)
        assert search["results"][0]["index"] == 1
        assert backend.query == "China Lebanon 2026"
        assert backend.limit == 4
        page = request({"action": "read", "source": 1}, bridge.token)
        assert page["content"] == "Official final score: 89-87"
        assert backend.url == "https://example.com/game"
        assert bridge.public_sources() == search["results"]
    finally:
        bridge.close()
