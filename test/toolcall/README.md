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

## Software-Development Tool Matrix

The deterministic suite now includes coding-agent style tools in addition to
weather/time examples:

- `read_file(path, start_line, end_line)`
- `search_code(query, path, file_pattern)`
- `run_tests(command, cwd, timeout_seconds)`
- `apply_patch(path, patch)`
- `git_status()`
- `git_diff(path)`
- `list_files(path, recursive)`

These tests verify parser output, OpenAI response shape, streaming
`delta.tool_calls`, and native tool-name constraints with realistic argument
types: arrays, booleans, numbers, zero-argument calls, optional arguments, and
long patch strings containing quotes, braces, and newlines. The tests do not
read files, apply patches, run shell commands, or mutate the repository.

## Manual Live Smoke Tests

The live runner is a manual diagnostic tool for already-running
OpenAI-compatible servers. It does not start a server, download a model, or use
GPU commands. It is not part of unittest discovery and is not part of the
deterministic parser golden tests.

List manual cases without contacting a server:

```bash
python3 test/toolcall/run_live_toolcall_tests.py --list-cases
```

Run one live case against a server that you started separately:

```bash
python3 test/toolcall/run_live_toolcall_tests.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model DeepSeek-V4-Flash \
  --case-id live_openai_baseline_get_weather \
  --verbose \
  --dump-dir /tmp/fastllm-toolcall-live
```

Run the manual matrix in report-only mode to measure model/tool adherence
without making failures fail the shell command:

```bash
python3 test/toolcall/run_live_toolcall_tests.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model DeepSeek-V4-Flash \
  --repeat 3 \
  --report-only \
  --dump-dir /tmp/fastllm-toolcall-live
```

The matrix is meant to distinguish parser/server bugs, protocol bugs,
model-adherence failures, and generation-constraint effectiveness. It covers
baseline `get_weather`, short-name `weather`, required and named tool choice,
parallel calls, tool-result roundtrip, no-tool plain text, and strict-schema
missing-required diagnostics. The runner reports stable `error_code` values such
as `invalid_tool_name`, `no_tool_call`, `invalid_arguments_json`,
`missing_required_argument`, `stream_missing_done`, and `http_error`.

Manual software-development smoke cases are also available. They ask the model
to call coding-agent style tools but still do not execute those tools:
For roundtrip cases, mocked structured tool results are sent as JSON strings in
the tool-role message content for OpenAI-compatible server compatibility.

```bash
python3 test/toolcall/run_live_toolcall_tests.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model DeepSeek-V4-Flash \
  --case-id live_openai_dev_search_code \
  --verbose \
  --dump-dir /tmp/fastllm-toolcall-live-dev
```

```bash
python3 test/toolcall/run_live_toolcall_tests.py \
  --base-url http://127.0.0.1:8080/v1 \
  --model DeepSeek-V4-Flash \
  --case-id live_openai_dev_apply_patch_dry_run \
  --verbose \
  --dump-dir /tmp/fastllm-toolcall-live-dev
```

Known future coverage gaps:

- `parameter name="arguments"` wrapper unwrapping is not covered because the
  current DeepSeek V4 parser treats it as a normal parameter.
- DeepSeek V4 streaming currently emits complete arguments after a full
  `invoke` closes; incremental argument streaming is not required by this suite.
- Streaming behavior for a complete `invoke` under the wrong outer block name is
  not fixed as a golden case yet.
- Deterministic tests do not cover real model quality. Use the manual live
  runner for model-side tool selection and constraint impact diagnostics.
