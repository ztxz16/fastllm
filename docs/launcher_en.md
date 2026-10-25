# Browser launcher and Studio

[中文](launcher.md) · [Documentation index](README_EN.md)

After installation, run `ftllm` to open the local launcher, or use:

```sh
ftllm launch
```

Add a profile → select a model → choose Automatic configuration → save and start. The launcher provides model downloads, command previews, logs, and Studio.

## Startup and access

`ftllm` (or `ftllm launch`) listens only on `127.0.0.1:8000` by default and opens the browser after the service is ready; use `ftllm launch --no-browser` to disable automatic opening. The page can download models from ModelScope, save and preview launch profiles, and run either `ftllm server` or the chat-oriented `ftllm webui`. For a new profile, select a local model and Automatic configuration; `--auto` plans inference settings when the service starts. Expand Resource limits or Model parameters to override individual options. Leave Chunk and Maximum concurrency blank to restore automatic planning. See the [`--auto` tuning guide](auto_en.md#launcher-automatic-mode). Custom configuration remains available. Its interface supports Simplified Chinese and English, preferring the last selection and otherwise following the browser language; terminal output from `ftllm launch` always uses English. To access it from the LAN, run `ftllm launch --host 0.0.0.0`; the terminal and Launcher page then list loopback, LAN, and directly assigned public-interface addresses when available. Public access also requires the host firewall and cloud security group to allow the port, plus port forwarding when behind NAT; Launcher does not discover NAT public mappings. Non-loopback access uses unencrypted HTTP and should only be enabled on a trusted network. It shares profiles with the terminal wizard and stops its managed download and model processes when the launcher exits. Run `ftllm launch --help` for other options.

## Launch settings and model downloads

**Model location** comes first, followed by Basic settings. The two modes are Automatic configuration and Custom configuration. Three sections below are collapsed by default:

| Section | Settings and launch options |
| --- | --- |
| Resource limits | GPU memory ratio → `--gpu_mem_ratio` (e.g. `0.9`); host memory limit → `--force_ddr_limit` (e.g. `120g`) |
| Vision module (optional) | Choose a matching GGUF vision module file → `--mmproj`; leave blank to skip the external module |
| Model parameters | Chunk → `--chunked_prefill_size`; Maximum concurrency → `--max_batch`; Custom arguments and Custom environment variables |

In automatic mode, leave Chunk, Maximum concurrency, and Resource limits blank to follow automatic planning. Custom arguments are appended to the command. Environment variables use `KEY=VALUE`, separated by spaces or newlines; both custom fields apply in automatic and custom modes. Check the command preview after saving, then restart the model service to apply changes. See the [`--auto` guide](auto_en.md) for parameter effects and budget semantics.

Download clusters are collapsed by default, with **DeepSeek · V4.1 Flash** first. Expand it to select the original model or one of these Fastllm quantizations:

- `huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-3.0bit`
- `huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-2.5bit`
- `huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-2.0bit`

Selecting a card fills in the corresponding ModelScope ID. These three variants download the full repository by default, with Download scope left blank. Confirm the destination and click Download; selecting a card alone does not start a download.

## Appearance and agents

Once the API server is running, **API information** in the top bar shows and copies its Base URL, model name, and API key. The key is hidden until revealed. For remote access, replace a loopback address with the server address or forward the API port.

Launcher’s **Appearance** selector offers Light and Dark modes. It defaults to Light and remembers your selection. Studio switches with Launcher while preserving the current conversation and draft.

Navigation has two groups: **Model management** and **agent**. The agent group contains Studio, **DeepSeek Harness**, **OpenCode**, **Codex**, and **Claude Code**. Use **Manage** beside the group or on an agent page to install, upgrade, or remove the last four tools through the existing plugin system. Installation works without a running model; removal keeps sessions and workspaces. Harness embeds its native interface and connects it to the active model API. **Install and open Harness** is also available, with installation progress shown on the page. No existing Node/npm installation is required. See the [Harness integration guide (Chinese)](launcher-harness.md) for storage locations and deployment limitations.

The same navigation also provides the native **OpenCode** web app and **Codex / Claude Code** conversation pages, connected to your local model with tool interactions and separate sessions. These tools are not installed by default: click the installation button. OpenCode and Codex can reuse an existing command; Claude Code uses a private Agent SDK runtime with a Harness-like interface. See the [native agent integration guide (Chinese)](launcher-agents.md) for installation, permissions, and deployment details.

The “自定义界面” (Customize interface) entry in Launcher's main navigation opens a dedicated page for conversational changes to pages, Studio features, status widgets, and global skins. It manages separate conversations with their own requests, change summaries, and drafts, saved in the current browser across reloads. An interactive preview of the whole application supports follow-up requests, file editing, cancellation, and explicit application of changes. Customizations live in `~/.fastllm/plugins` and support hot updates, disabling, deletion, and rollback; editing cannot write core application files. See the [customization guide (Chinese)](launcher-plugins.md) for usage and the plugin format.

Chat, Agent replies, and reasoning support Markdown tables, headings, nested lists, task lists, blockquotes, and links, including saved conversations. Click **Preview** in an HTML code block to open an interactive page with HTML, CSS, and inline JavaScript. Previews run in an isolated environment, load only embedded resources, and clear their temporary page storage when closed. Closing the preview returns to your conversation. The Markdown parser is bundled locally and needs no network download.

## Studio and WebUI

Once the API Server is ready, click **Open Studio** to use chat, saved conversations, Markdown, attachments, reasoning, and agents directly in Launcher's content area. Model management navigation stays visible, so you can visit launch profiles, downloads, logs, and hardware, then return to the same conversation. Launcher and standalone `ftllm webui` share the chat component and backend; the component adapts its colors, sizing, and language to Launcher. It connects to the active model with the configured API key, without another WebUI process or port. Conversations use WebUI’s existing local storage and survive page refreshes. Stopping or switching models cancels active WebUI tasks and disposes of the old component.

The Studio reads reasoning options from the active model's `/v1/models` metadata and sends the selected value unchanged. DeepSeek V4.1, for example, offers `none / low / high / max`. Servers without this metadata show **Model default**. Saved legacy choices are migrated for the corresponding model.

WebUI does not load a model in its own process, so start an OpenAI-compatible API server first. Its optional `model` positional argument is only a model-name hint; when omitted, the WebUI discovers the model from `/v1/models`.

## Agent runtime and workspaces

Code analysis and web search use the Pi agent runtime by default. On Linux
x86-64, open `ftllm launch` → **Studio** → **Install Agent dependencies** to install
`ftllm-agent-runtime==0.3.3` with pip. The wheel includes Pi and the `rg`/`fd`
search tools and downloads about 43 MB, using pip's configured mirror, proxy, and cache.
Installation needs no administrator privileges, Node.js, npm, or Bun.
Packages are stored in `${XDG_DATA_HOME:-~/.local/share}/ftllm/agent-runtime/`
without modifying system Python packages. The UI shows installation status and errors with a retry button.
The current Studio enables Pi when installation completes; later launches discover it automatically.
Portable bundles with a complete runtime need no additional installation. You can also run
`python -m pip install ftllm-agent-runtime==0.3.3` in FastLLM's Python environment
and restart FastLLM, or build
the companion wheel as described in [`tools/ftllm_agent_runtime/`](../tools/ftllm_agent_runtime/).
Use `--agent-runtime builtin`
to select the original single-call paths when the companion wheel is absent.

Launcher automatically uses an installed Pi runtime. **New Agent** lets you select a project directory; use `ftllm launch --agent-workspace-root /path/to/projects` to set the selectable root (your home directory by default). Directory agents are enabled by default on both local and remote Launcher listeners, for example `ftllm launch --host 0.0.0.0 --agent-workspace-root /path/to/projects`. Use `--disable-workspace-agent` to disable directory browsing, creating directory agents, and executing saved directory-agent tasks; ordinary chat remains available. Directory agents can modify files and execute commands, so restrict access to trusted users. The interface explains when the runtime is unavailable or directory agents have been disabled.
