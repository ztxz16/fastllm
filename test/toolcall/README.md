# Toolcall Parser Golden Tests

These are deterministic golden tests for toolcall parsers. They exercise parser
inputs and OpenAI-compatible tool call shapes directly from the source tree.

They do not:

- start a server
- download a model
- require GPU access
- require new test dependencies

The current focus is the DeepSeek V4 DSML parser. These tests verify parser and
API formatting behavior; they do not verify whether a real model chooses the
right tool.

Run from the repository root:

```bash
python3 test/toolcall/test_parser_golden.py
```

Or run through unittest discovery:

```bash
python3 -m unittest discover -s test/toolcall -p 'test*.py'
```

The tests import the existing OpenAI server protocol/parser modules. If imports
fail because server-side Python dependencies such as `shortuuid` are missing,
install the existing server requirements:

```bash
pip install -r requirements-server.txt
```

Known future coverage gaps:

- `parameter name="arguments"` wrapper unwrapping is not covered because the
  current DeepSeek V4 parser treats it as a normal parameter.
- DeepSeek V4 streaming currently emits complete arguments after a full
  `invoke` closes; incremental argument streaming is not required by this suite.
- Streaming behavior for a complete `invoke` under the wrong outer block name is
  not fixed as a golden case yet.
- These tests do not cover server integration, chat template rendering, or
  model-side tool selection.
