import importlib.util
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools" / "fastllm_pytools" / "openai_server" / "fastllm_completion.py"


def _ensure_stub(module_name: str, module_obj):
    if module_name not in sys.modules:
        sys.modules[module_name] = module_obj


_ensure_stub("shortuuid", types.SimpleNamespace(random=lambda: "test"))

fastapi_stub = types.ModuleType("fastapi")


class _Request:  # Minimal stub for typing only
    pass


fastapi_stub.Request = _Request
_ensure_stub("fastapi", fastapi_stub)

starlette_stub = types.ModuleType("starlette")
starlette_background_stub = types.ModuleType("starlette.background")


class _BackgroundTask:  # Minimal stub used in stream handling
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


starlette_background_stub.BackgroundTask = _BackgroundTask
_ensure_stub("starlette", starlette_stub)
_ensure_stub("starlette.background", starlette_background_stub)

pydantic_stub = types.ModuleType("pydantic")


class _BaseModel:
    def __init__(self, **data):
        for key, value in data.items():
            setattr(self, key, value)

    def model_dump(self, exclude_none: bool = False):
        data = self.__dict__.copy()
        if exclude_none:
            data = {k: v for k, v in data.items() if v is not None}
        return data


def _Field(*args, default=None, default_factory=None, **kwargs):
    if default_factory is not None:
        return default_factory()
    return default


pydantic_stub.BaseModel = _BaseModel
pydantic_stub.Field = _Field
_ensure_stub("pydantic", pydantic_stub)

openai_stub = types.ModuleType("openai")
openai_types_stub = types.ModuleType("openai.types")
openai_chat_stub = types.ModuleType("openai.types.chat")


class _ChatCompletionContentPartParam(dict):
    pass


class _ChatCompletionRole(str):
    pass


openai_chat_stub.ChatCompletionContentPartParam = _ChatCompletionContentPartParam
openai_chat_stub.ChatCompletionRole = _ChatCompletionRole
openai_types_stub.chat = openai_chat_stub
openai_stub.types = openai_types_stub
_ensure_stub("openai", openai_stub)
_ensure_stub("openai.types", openai_types_stub)
_ensure_stub("openai.types.chat", openai_chat_stub)

fastllm_pytools_pkg = types.ModuleType("fastllm_pytools")
fastllm_pytools_pkg.__path__ = [str(REPO_ROOT / "tools" / "fastllm_pytools")]
_ensure_stub("fastllm_pytools", fastllm_pytools_pkg)

openai_server_pkg = types.ModuleType("fastllm_pytools.openai_server")
openai_server_pkg.__path__ = [
    str(REPO_ROOT / "tools" / "fastllm_pytools" / "openai_server")
]
_ensure_stub("fastllm_pytools.openai_server", openai_server_pkg)

spec = importlib.util.spec_from_file_location(
    "fastllm_pytools.openai_server.fastllm_completion",
    MODULE_PATH,
)
fastllm_completion = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(fastllm_completion)

ConversationMessage = fastllm_completion.ConversationMessage
FastLLmCompletion = fastllm_completion.FastLLmCompletion


class DummyModel:
    def __init__(self):
        pass


def _create_completion():
    return FastLLmCompletion(
        model_name="dummy-model",
        model=DummyModel(),
        think=False,
        hide_input=False,
    )


def test_parse_chat_message_content_with_input_text():
    completion = _create_completion()

    messages, pending = completion._parse_chat_message_content(
        role="user",
        content=[
            {"type": "text", "text": "hello"},
            {"type": "input_text", "text": "world"},
        ],
    )

    assert pending == []
    assert len(messages) == 1
    assert isinstance(messages[0], ConversationMessage)
    assert messages[0].role == "user"
    assert messages[0].content == "hello\nworld"


def test_parse_chat_message_content_with_image_url():
    completion = _create_completion()

    messages, _ = completion._parse_chat_message_content(
        role="assistant",
        content=[
            {
                "type": "image_url",
                "image_url": {"url": "https://example.com/cat.png"},
            }
        ],
    )

    assert len(messages) == 1
    assert messages[0].role == "assistant"
    assert messages[0].content == "[Image: https://example.com/cat.png]"


def test_parse_chat_message_content_with_unknown_type():
    completion = _create_completion()

    with pytest.raises(ValueError, match="Unsupported message content type 'audio'"):
        completion._parse_chat_message_content(
            role="user",
            content=[{"type": "audio", "audio": "..."}],
        )
