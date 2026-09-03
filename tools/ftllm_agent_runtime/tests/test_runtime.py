import base64
import json
from pathlib import Path
import threading
import time
import urllib.error
import urllib.request

import pytest

from ftllm_agent_runtime.runtime import (
    PiAgentCancelled,
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


def test_workspace_mode_enables_coding_tools_and_project_context(tmp_path: Path):
    binary = _fake_pi_binary(tmp_path)
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        binary=str(binary),
    )

    command = runtime._command(
        "system prompt",
        "off",
        ["read", "bash", "edit", "write", "grep", "find", "ls"],
        workspace_mode=True,
    )

    assert "--no-builtin-tools" not in command
    assert "--no-context-files" not in command
    assert "--approve" in command
    assert "--append-system-prompt" in command
    assert "--system-prompt" not in command
    assert command[command.index("--tools") + 1] == (
        "read,bash,edit,write,grep,find,ls"
    )


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


def test_runtime_streams_tool_identity_arguments_updates_and_result(tmp_path: Path):
    binary = _fake_pi_binary(tmp_path, [
        {"type": "turn_start"},
        {
            "type": "tool_execution_start",
            "toolCallId": "call-42",
            "toolName": "bash",
            "args": {"command": "printf tool-output-42"},
        },
        {
            "type": "tool_execution_update",
            "toolCallId": "call-42",
            "toolName": "bash",
            "partialResult": {
                "content": [{"type": "text", "text": "tool-output"}],
            },
        },
        {
            "type": "tool_execution_end",
            "toolCallId": "call-42",
            "toolName": "bash",
            "result": {
                "content": [{"type": "text", "text": "tool-output-42"}],
            },
            "isError": False,
        },
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
        binary=str(binary),
    )

    events = list(runtime.stream(
        "Run the command.", [], "Use the coding tools.",
        working_directory=str(tmp_path),
    ))

    assert [event for event in events if event["type"].startswith("tool_")] == [
        {
            "type": "tool_start",
            "id": "call-42",
            "name": "bash",
            "arguments": {"command": "printf tool-output-42"},
            "arguments_truncated": False,
        },
        {
            "type": "tool_update",
            "id": "call-42",
            "name": "bash",
            "result": "tool-output",
            "result_truncated": False,
        },
        {
            "type": "tool_end",
            "id": "call-42",
            "name": "bash",
            "is_error": False,
            "result": "tool-output-42",
            "result_truncated": False,
        },
    ]


def test_runtime_runs_workspace_agent_from_selected_directory(tmp_path: Path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    binary = tmp_path / "pi"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, os, pathlib, sys\n"
        "json.loads(sys.stdin.readline())\n"
        "pathlib.Path('agent-output.txt').write_text(os.getcwd())\n"
        "events = [\n"
        " {'type': 'turn_start'},\n"
        " {'type': 'message_update', 'assistantMessageEvent': "
        "{'type': 'text_delta', 'delta': 'Updated'}},\n"
        " {'type': 'turn_end'},\n"
        " {'type': 'agent_end'},\n"
        "]\n"
        "for event in events: print(json.dumps(event), flush=True)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        binary=str(binary),
    )

    events = list(runtime.stream(
        "Update the project.",
        [],
        "Use the coding tools.",
        working_directory=str(workspace),
    ))

    assert events[-1] == {"type": "done", "turns": 1}
    assert (workspace / "agent-output.txt").read_text() == str(workspace)


def test_runtime_sends_image_attachments_in_the_rpc_prompt(tmp_path: Path):
    workspace = tmp_path / "project"
    workspace.mkdir()
    image = tmp_path / "sample.png"
    image.write_bytes(b"small-png-fixture")
    captured = tmp_path / "request.json"
    binary = tmp_path / "pi"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        f"pathlib.Path({str(captured)!r}).write_text(sys.stdin.readline())\n"
        "events = [\n"
        " {'type': 'turn_start'},\n"
        " {'type': 'message_update', 'assistantMessageEvent': "
        "{'type': 'text_delta', 'delta': 'Seen'}},\n"
        " {'type': 'turn_end'},\n"
        " {'type': 'agent_end'},\n"
        "]\n"
        "for event in events: print(json.dumps(event), flush=True)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        binary=str(binary),
    )

    events = list(runtime.stream(
        "Inspect the image.",
        [],
        "Use the image.",
        working_directory=str(workspace),
        images=[{"path": str(image), "mime_type": "image/png"}],
    ))

    request = json.loads(captured.read_text(encoding="utf-8"))
    assert request["images"] == [{
        "type": "image",
        "data": base64.b64encode(image.read_bytes()).decode("ascii"),
        "mimeType": "image/png",
    }]
    assert events[-1] == {"type": "done", "turns": 1}


def test_runtime_rejects_invalid_working_directory(tmp_path: Path):
    binary = _fake_pi_binary(tmp_path)
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        binary=str(binary),
    )

    with pytest.raises(ValueError, match="working directory is unavailable"):
        list(runtime.stream(
            "Inspect it.",
            [],
            "Use the coding tools.",
            working_directory=str(tmp_path / "missing"),
        ))


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


def test_runtime_cancellation_terminates_a_running_process(tmp_path: Path):
    binary = tmp_path / "pi"
    binary.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys, time\n"
        "json.loads(sys.stdin.readline())\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    binary.chmod(0o755)
    runtime = PiAgentRuntime(
        api_base="http://localhost:8000/v1",
        model="demo",
        binary=str(binary),
    )
    cancelled = threading.Event()
    timer = threading.Timer(0.15, cancelled.set)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(PiAgentCancelled, match="cancelled"):
            list(runtime.stream(
                "Inspect the project.",
                [{"name": "app.py", "text": "answer = 42\n"}],
                "Use the read-only project tools.",
                cancel_event=cancelled,
            ))
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3


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
