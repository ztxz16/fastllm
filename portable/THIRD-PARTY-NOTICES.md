# Third-party notices

This bundle contains CPython and third-party Python/native distributions. Their
license and notice files remain in the corresponding runtime and `*.dist-info`
directories.

In particular, the bundled `nvidia-cuda-runtime-cu12`, `nvidia-cublas-cu12`, and
`nvidia-nccl-cu12` distributions are NVIDIA software and remain subject to the
license terms shipped with those distributions. Building or redistributing this
bundle does not replace or alter those terms.

The redistributable CPython runtime is sourced from the
`astral-sh/python-build-standalone` project. Its CPython and component license
files are retained under `runtime/`.

Pi Agent is bundled from its pinned official standalone release under the MIT
License. Its license remains at
`runtime/lib/python3.11/site-packages/ftllm_agent_runtime/licenses/PI_LICENSE`.
The FastLLM runtime bridge and project extension are distributed under the
Apache License 2.0; their license is retained in the companion wheel metadata.

The standalone ripgrep and fd search tools are included for Pi's offline
directory tools. Pinned versions, upstream URLs, and archive hashes are recorded
in `runtime/share/ftllm-agent-tools/manifest.json`; the corresponding licenses
are retained in that directory under `rg/` and `fd/`.
