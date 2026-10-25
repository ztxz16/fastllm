# 浏览器启动器与工作室

[English](launcher_en.md) · [文档索引](README.md)

安装后运行 `ftllm`，浏览器会打开本地启动器。也可以使用：

```sh
ftllm launch
```

新增启动项 → 选择模型 → 自动配置 → 保存并启动。常用操作是下载模型、预览命令、查看日志和打开工作室。

## 启动与访问

`ftllm`（或 `ftllm launch`）默认仅监听 `127.0.0.1:8000`，并在服务就绪后自动打开浏览器；使用 `ftllm launch --no-browser` 可以关闭自动打开。页面可以从 ModelScope 下载模型、保存启动配置、预览命令，并选择托管 `ftllm server` 或聊天 `ftllm webui`。新增启动项选择本地模型和“自动配置”后，启动时由 `--auto` 规划推理参数。需要调整时展开“资源限制”或“模型参数”；Chunk 和最大并发留空即可恢复自动规划。参数说明见 [`--auto` 调节指南](auto.md#启动器自动模式)。也可切换到自定义配置。界面支持简体中文和英文，会优先使用上次选择的语言，否则跟随浏览器语言；`ftllm launch` 的终端日志固定使用英文。需要从局域网访问时使用 `ftllm launch --host 0.0.0.0`；终端和 Launcher 页面随后会列出本机、局域网以及网卡上直接配置的公网访问地址（若有）。公网访问还需要放行主机防火墙及云安全组，经过 NAT 时还需配置端口映射；Launcher 不会自动探测 NAT 的公网地址。非本机监听使用未加密 HTTP，请仅在可信网络中使用。它与终端向导共用配置文件；关闭 Launcher 时，由它托管的下载和模型进程也会停止。使用 `ftllm launch --help` 查看其他选项。

## 启动项与模型下载

启动项顶部是**文件选择**，基础配置紧随其后。只保留“自动配置”和“自定义配置”两种模式。下面三个栏目默认折叠：

| 栏目 | 设置与启动参数 |
| --- | --- |
| 资源限制 | 显存比例 → `--gpu_mem_ratio`（例如 `0.9`）；内存上限 → `--force_ddr_limit`（例如 `120g`） |
| 视觉模块（可选） | 选择匹配的 GGUF 视觉模块文件 → `--mmproj`；留空不加载外部视觉模块 |
| 模型参数 | Chunk → `--chunked_prefill_size`；最大并发 → `--max_batch`；自定义参数与自定义环境变量 |

自动模式下，Chunk、最大并发和资源限制留空即沿用自动规划。自定义参数附加到命令末尾；环境变量使用 `KEY=VALUE`，以空格或换行分隔，两项在自动和自定义模式下都生效。保存后查看命令预览，重启模型服务应用修改；参数影响与预算含义见 [`--auto` 调节指南](auto.md)。

下载页的模型簇默认折叠，**DeepSeek · V4.1 Flash** 排在最前。展开后可以选择原版或以下 Fastllm 量化版本：

- `huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-3.0bit`
- `huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-2.5bit`
- `huangyuyang/DeepSeek-V4.1-Flash-1017-Fastllm-2.0bit`

点击版本卡片会填入对应的 ModelScope ID；这三个版本默认下载完整仓库，下载范围留空。确认保存目录后点击下载；选择卡片本身不会开始下载。

## 界面与 Agent

Launcher 的「界面主题」提供浅色模式和黑夜模式，默认浅色并记住手动选择。工作室会同步切换主题，保留当前会话和输入草稿。

左侧导航分为「模型管理」和「agent」两组。agent 包含工作室、**DeepSeek Harness**、**OpenCode**、**Codex**、**Claude Code**；后四者可通过分组旁或页面中的“管理”安装、升级和删除独立运行环境，复用现有插件管理。安装无需启动模型，删除保留会话和工作目录。Harness 嵌入原生页面并连接当前模型 API，也可点击“安装并打开 Harness”。无需预装 Node/npm，会话保存位置和部署限制见 [Harness 接入说明](launcher-harness.md)。

同一位置还提供 **OpenCode** 原生网页及 **Codex / Claude Code** 会话页面，支持连接本地模型、工具交互与独立会话。外部工具均不默认安装，需点击各自的安装按钮；OpenCode / Codex 可复用已有命令，Claude Code 使用独立 Agent SDK 运行环境，界面风格接近 Harness。安装、权限和部署说明见 [原生 agent 接入说明](launcher-agents.md)。

Launcher 主导航中的「自定义界面」支持在模型 API 启动后，用自然语言对话定制页面、工作室功能、主界面状态栏和全局皮肤。独立子页面提供会话管理，每个会话分别保留需求、修改摘要与草稿，刷新浏览器后也可恢复；右侧显示整个应用的交互预览，支持连续修改、编辑文件、取消生成和确认应用。自定义内容保存在 `~/.fastllm/plugins`，支持热更新、停用、删除和恢复上一版；核心代码不参与编辑。使用说明与插件格式见 [自定义界面文档](launcher-plugins.md)。

对话、Agent 回复和思考过程支持 Markdown 表格、标题、嵌套列表、任务列表、引用及链接，历史会话也会按相同格式显示。HTML 代码块右上角的「预览」可打开交互页面，支持 HTML、CSS 和内联 JavaScript。预览在隔离环境中运行，仅加载内嵌资源，页面存储在关闭后清空；关闭预览可继续原来的对话。Markdown 解析资源随安装包提供，无需联网下载。

## 工作室与 WebUI

API Server 就绪后，点击「打开工作室」即可在 Launcher 内容区直接使用聊天、历史会话、Markdown、附件、思考过程和智能体功能。模型管理导航始终保留，可随时切换到启动、下载、日志和硬件页面，返回「工作室」后继续当前会话。Launcher 与独立的 `ftllm webui` 共用聊天组件和后端，界面配色、尺寸及语言会适配 Launcher。组件自动连接当前模型并使用启动配置中的 API Key，无需另开 WebUI 服务或端口。会话沿用 WebUI 的本地存储，刷新页面后仍然保留；停止或切换模型时会取消正在运行的 WebUI 任务并清理旧组件。

工作室的思考强度由当前模型的 `/v1/models` 能力信息决定，选择后按原值发送。例如 DeepSeek V4.1 提供 `none / low / high / max`。未提供选项的服务显示“模型默认”；旧会话的思考设置会按对应模型迁移。

WebUI 不会在自身进程内加载模型，请先启动 OpenAI 兼容 API Server。WebUI 的可选 `model` 位置参数只用于推导 API 模型名；省略时会从 `/v1/models` 自动发现。

## Agent 运行时与工作目录

代码分析和联网搜索默认使用 Pi 智能体运行时。Linux x86-64 用户可打开
`ftllm launch` →「工作室」→「安装 Agent 依赖」，通过 pip 安装
`ftllm-agent-runtime==0.3.3`，其中已包含 Pi 及 `rg`、`fd` 搜索工具。
安装需要联网下载约 43 MB，沿用当前 pip 的镜像、代理和缓存配置；不需要管理员权限、Node.js、npm 或 Bun。
安装包保存在 `${XDG_DATA_HOME:-~/.local/share}/ftllm/agent-runtime/`，不修改系统 Python 包。
安装期间显示状态，失败后可查看错误并重试；完成后当前工作室即可使用，后续启动也会自动识别。
已有完整运行时的绿色包无需重复安装。也可在 ftllm 所在的 Python 环境中执行
`python -m pip install ftllm-agent-runtime==0.3.3` 后重启 ftllm，或按
[`tools/ftllm_agent_runtime/`](../tools/ftllm_agent_runtime/) 中的说明构建配套 wheel。尚未安装时可通过
`--agent-runtime builtin` 使用原有单轮链路。

Launcher 会自动使用已安装的 Pi 运行时；「新建 Agent」可选择工作目录。通过 `ftllm launch --agent-workspace-root /path/to/projects` 指定可选目录的根路径，默认为用户主目录。Launcher 的目录 Agent 默认启用，本机和远程监听均可使用，例如 `ftllm launch --host 0.0.0.0 --agent-workspace-root /path/to/projects`。使用 `--disable-workspace-agent` 可关闭目录 Agent，同时禁止目录浏览、新建目录 Agent 及继续执行已保存的目录 Agent 任务；普通对话仍可使用。目录 Agent 可修改文件和执行命令，请仅对可信用户开放。运行时缺失或目录 Agent 被关闭时，界面会显示原因。
