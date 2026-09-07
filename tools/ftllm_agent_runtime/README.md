# ftllm-agent-runtime

Linux x86-64 companion wheel for FastLLM WebUI. It bundles the official Pi
standalone executable, ripgrep, and fd, and exposes a small Python API over Pi's JSONL RPC mode.
End users do not need Node.js, npm, or Bun.

Runtime requirements are CPython 3.9+, Linux x86-64, glibc 2.17 or newer, and
a CPU with SSE4.2 (the bundled Pi uses Bun's x64 baseline build).
Alpine/musl, ARM64, Windows, and macOS need separate runtime builds.
The wheel has no additional Python package dependencies. Runtime 0.3.3 bundles
Pi 0.84.4, ripgrep 14.1.1, and fd 10.2.0, including their license files.

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
python scripts/fetch_tools.py
python -m build --wheel --outdir build/wheels
```

For an offline/repeatable build, populate both archive caches once, then build
without network access:

```bash
python scripts/fetch_pi.py --cache-dir /path/to/cache/pi --offline
python scripts/fetch_tools.py --cache-dir /path/to/cache/agent-tools --offline
python -m build --wheel --no-isolation --outdir build/wheels
```

The fetch scripts verify upstream release archives using hard-coded SHA-256
digests and copy the corresponding licenses into the package tree.
Use `--cache-dir /path/to/cache` to retain the verified archive, and add
`--offline` to require local files only. The repository's `make_portable.sh`
and `desktop/package.sh` build and include this runtime automatically.
Wheel builds reject missing binaries, extensions, themes, and licenses.

Before publishing, install `build`, `auditwheel`, `patchelf`, and `twine` in
the build environment, then audit and label the wheel:

```bash
python -m auditwheel repair --plat manylinux_2_17_x86_64 \
  --wheel-dir dist build/wheels/ftllm_agent_runtime-0.3.3-py3-none-linux_x86_64.whl
python -m auditwheel show dist/*-manylinux*.whl
python -m twine check --strict dist/*-manylinux*.whl
```

Upload only the audited `manylinux` wheel, using Twine's token prompt:

```bash
python -m twine upload --username __token__ dist/*-manylinux*.whl
```

The distribution installs native executables into `platlib`, while keeping
the Python bridge's `py3-none` tag because it has no CPython extension ABI.
The manylinux audit checks shared-library compatibility; it does not replace
runtime tests on supported distributions and CPUs. The runtime has been
smoke-tested on Ubuntu 22.04 x86-64, including Pi's `read`, `find`, and `grep`.

## Install and enable in WebUI

Install the companion wheel into the same Python environment as FastLLM:

```bash
python -m pip install ftllm-agent-runtime==0.3.3

ftllm webui /path/to/model
```

For a local build, pass the audited wheel's path to `pip install` instead.
Restart an already-running FastLLM process after installing the package.

FastLLM builds that include **Launcher → Studio → Install Agent dependencies**
install this wheel with the Launcher's Python and pip. The installer uses pip's
configured index/mirror and cache, and installs into the current user's
`${XDG_DATA_HOME:-~/.local/share}/ftllm/agent-runtime/` directory with `--target`.
It verifies Pi and both search tools before activating the installation, then
enables the current Studio without restarting the model. Failed installations
leave any previous managed runtime intact and can be retried from the UI.

The companion wheel includes `rg` and `fd` beside Pi. The bridge prepends this
directory to the child process's `PATH`, so directory searches work even when
neither command is installed system-wide. Pi runs in offline mode and does not
download additional tools during use.

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
Search-tool license files and pinned version/source information are included
under `ftllm_agent_runtime/licenses/agent-tools/`.
