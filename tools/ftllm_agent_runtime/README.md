# ftllm-agent-runtime

Linux x86-64 companion wheel for FastLLM WebUI. It bundles the official Pi
standalone executable and exposes a small Python API over Pi's JSONL RPC mode.
End users do not need Node.js, npm, or Bun.

Runtime requirements are CPython 3.9+, Linux x86-64, and glibc 2.17 or newer.
The wheel has no additional Python package dependencies. The pinned 0.84.4
prototype is approximately 37 MiB to download and 102 MiB after installation.

The bridge keeps uploaded-file analysis isolated and read-only. It exposes
snapshot tools plus WebUI-mediated public-web tools:

- `runtime_info`
- `list_project_files`
- `read_project_file`
- `search_project_files`
- `web_search`
- `read_web_page`

Uploaded source is copied into an isolated temporary directory before Pi
starts. Web requests are delegated to FastLLM's SSRF-protected `WebAgent`
through an authenticated, request-scoped localhost bridge.

Callers can explicitly pass `working_directory` to create a coding-agent run
over a real directory. That opt-in mode starts Pi in the selected directory,
loads its `AGENTS.md`/`CLAUDE.md` context, and enables `read`, `bash`, `edit`,
`write`, `grep`, `find`, and `ls`. Those tools can modify files and execute
commands with the permissions of the hosting process, so applications should
validate and clearly display the selected directory before starting a run.
FastLLM WebUI limits selection to `--agent-workspace-root` (the current user's
home directory by default) and only enables this mode on a loopback listener.
Using it with a non-loopback `--host` additionally requires the explicit
`--allow-remote-workspace-agent` flag.

## Build the Linux wheel

```bash
python scripts/fetch_pi.py
python -m build --wheel
```

For an offline/repeatable build, download the pinned release archive first:

```bash
python scripts/fetch_pi.py --archive /path/to/pi-linux-x64.tar.gz
python -m build --wheel --no-isolation
```

The fetch script verifies the upstream release archive and license using
hard-coded SHA-256 digests before copying anything into the package tree.
Use `--cache-dir /path/to/cache` to retain the verified archive, and add
`--offline` to require local files only. The repository's `make_portable.sh`
and `desktop/package.sh` build and include this runtime automatically.

## Install and enable in WebUI

```bash
python -m pip install \
  dist/ftllm_agent_runtime-0.3.2-py3-none-linux_x86_64.whl

ftllm webui /path/to/model
```

Pi is the default agent runtime for code tasks and Web Agent searches. Use
`--agent-runtime builtin` to force the original single-model-call paths, or
`--agent-runtime auto` to prefer Pi while allowing a missing package fallback.
Library callers can pass a `threading.Event` as `cancel_event` to
`PiAgentRuntime.stream()`; setting it terminates the request-scoped Pi process.
Tool activity is normalized into `tool_start`, `tool_update`, and `tool_end`
events with stable call IDs, bounded arguments, output text, and error state.
The optional `images` argument accepts image paths and MIME types and forwards
up to six images through Pi's RPC prompt format.

## Smoke test

```bash
ftllm-agent-runtime info
ftllm-agent-runtime probe \
  --api-base http://127.0.0.1:8080/v1 \
  --model your-model \
  --file README.md \
  "Read the project file and summarize its first heading."
```

To smoke-test the writable coding-agent mode in a disposable project:

```bash
ftllm-agent-runtime probe \
  --api-base http://127.0.0.1:8080/v1 \
  --model your-model \
  --directory /path/to/project \
  "Inspect the project and run its tests."
```

## Upstream

Pi Agent is distributed under the MIT License. The pinned upstream license is
committed at `src/ftllm_agent_runtime/licenses/PI_LICENSE` and included in
built wheels.
