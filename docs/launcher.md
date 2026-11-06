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

## 并行服务、GPU 选择与视频生成

启动项的「模型类型 / 后端」支持 LLM 和 **MiniMax H3 / Pika H3 视频**。
多个启动项可以同时运行，各自使用不同端口。顶部「当前服务」选择器决定 API 信息、
停止按钮和外部 Agent 连接哪个服务；启动项卡片也可以直接管理或停止对应服务。
停止一个模型不会停止其他模型，退出 Launcher 则停止全部托管进程。
对话历史由多个 LLM 共用，同一会话同时只允许一个生成任务；可以从另一模型界面停止它。
Harness、OpenCode、Codex、Claude 共用各自的安装和工作目录，每种外部 Agent 保持一个活动实例，
连接另一模型时会停止旧实例。工作区 Pi Agent 的任务按会话独立运行。

每个启动项都可以勾选「可用 GPU」，或填写逗号分隔的卡号 / GPU UUID。
编号对应 Launcher 当前可见的 GPU；子进程使用独立的 `CUDA_VISIBLE_DEVICES`，
内部重新编号为 `0,1,…`。例如 LLM 选择 `0`、视频选择 `1`，两者分别占用不同卡；
视频选择 `1,2` 时会在这两张卡上使用张量并行。自定义 LLM 的 CUDA / TP 参数应填写
**限制后的逻辑编号**。LLM 自动配置也只考虑所选 GPU。已有父进程 GPU 限制会继续生效，
无法通过该配置扩大可见范围。留空表示继承可见范围；视频此时默认用逻辑 GPU 0。
这是设备可见性限制，不是 GPU 独占预约。

视频启动流程：

1. 按 [H3 构建说明](../tools/minimax_h3/README.md#构建) 编译 `minimax_h3_server`。
   在启动项中填写该可执行文件的绝对路径，也可将其加入 `PATH`，或设置
   `FTLLM_H3_SERVER=/path/to/minimax_h3_server`。源码 `build` / `build-cuda` 目录也会自动查找。
2. 选择视频后端，填写 H3 或 Pika 模型目录、监听地址、端口、API Key 和可用 GPU。
   Pika 自动识别，无需 `--pika`；基座不在相邻 `MiniMax-H3` 目录或 `base_model` 链接下时，
   填写「MiniMax H3 基座目录」。Pika 仍需完整基座。视频输出目录默认是
   `~/.local/share/fastllm/videos/<端口>`。
   显存不足时填写「每卡显存预算」（GiB），例如 `4`；留空或 `0` 保留默认策略。
   启动时实际预留显存；预算充足时保持驻留，较小时逐层加载。双卡时该预算分别作用于每张卡。
   「主机内存预算」由整个服务共享，至少为 `2` GiB；`2` 从磁盘分块读权重，超出的部分用于缓存。
   留空或 `0` 保留原有缓存。预算不是操作系统硬隔离，需另外为系统与媒体处理留出内存。
   默认 Pika 832×480／124 帧文生视频已验证 4 GiB 预算；更大输入需要提高预算。
3. 启动并等待就绪。顶部「API 信息」可复制模型名、认证信息，以及
   `POST /v1/videos`、查询、取消和下载接口及请求示例。
4. 打开工作室的「视频生成」，选择视频服务、描述、分辨率、时长、可选种子及首尾帧。
   Pika 固定 6 步，H3 默认 50 步。任务支持进度、取消、预览和 MP4 下载；刷新页面会重新读取
   服务中的任务列表。参考图片上传总大小限制为 31 MiB，工作室单个结果下载限制为 256 MiB。

工作室的「对话 / 工作区」有独立的对话模型选择器，切换顶部服务或查看视频不会切断聊天。
同时启动 LLM 和视频模型，安装 Pi 运行时并新建一个工作目录 Agent 后，可以直接说：
“生成一段小猫走过花园的 5 秒视频，保存到 videos 目录。”
Launcher 会为本轮工作区任务提供实时生成服务目录和调用命令；Pi 用现有 `bash` 工具
创建任务、轮询、下载到工作区，结果也会作为会话视频附件显示。无需升级现有 Pi 0.3.3。
该联动针对**工作目录 Agent**，普通聊天不会执行本地命令。停止 Agent 会取消本轮创建的未完成任务。
视频较长时，Agent 达到自己的时间或轮次限制后可以继续任务，视频服务的任务仍可在 Demo 中查看。

### 增加其他生成后端

`service_backends.py` 注册后端 ID、能力列表、启动命令、校验、就绪探测和 API 契约；
`launcher_services.py` 统一管理进程、状态、服务目录及认证代理；`media_agent.py` 提供
单轮有效的本地生成桥接。桥接只访问已注册服务的允许接口，不向 Agent 暴露模型 API Key
或 Launcher 控制令牌。新增图片、音频或世界模型时，可增加后端与对应生成协议适配、
工作室组件，复用进程 / GPU / 服务目录机制。当前实际注册的生成能力为 `video.generate`。

## 界面与 Agent

API 服务启动后，顶部「API 信息」可查看并复制当前服务的 Base URL、模型名称和 API Key。Key 默认隐藏，可点击显示。远程访问时请将回环地址替换为服务器地址，或转发对应的 API 端口。

Launcher 的「界面主题」提供浅色模式和黑夜模式，默认浅色并记住手动选择。工作室会同步切换主题，保留当前会话和输入草稿。

左侧导航分为「模型管理」和「agent」两组。agent 包含工作室、**DeepSeek Harness**、**OpenCode**、**Codex**、**Claude Code**；后四者可通过分组旁或页面中的“管理”安装、升级和删除独立运行环境，复用现有插件管理。安装无需启动模型，删除保留会话和工作目录。Harness 嵌入原生页面并连接当前模型 API，也可点击“安装并打开 Harness”。无需预装 Node/npm，会话保存位置和部署限制见 [Harness 接入说明](launcher-harness.md)。

同一位置还提供 **OpenCode** 原生网页及 **Codex / Claude Code** 会话页面，支持连接本地模型、工具交互与独立会话。外部工具均不默认安装，需点击各自的安装按钮；OpenCode / Codex 可复用已有命令，Claude Code 使用独立 Agent SDK 运行环境，界面风格接近 Harness。安装、权限和部署说明见 [原生 agent 接入说明](launcher-agents.md)。

Launcher 主导航中的「自定义界面」支持在模型 API 启动后，用自然语言对话定制页面、工作室功能、主界面状态栏和全局皮肤。独立子页面提供会话管理，每个会话分别保留需求、修改摘要与草稿，刷新浏览器后也可恢复；右侧显示整个应用的交互预览，支持连续修改、编辑文件、取消生成和确认应用。自定义内容保存在 `~/.fastllm/plugins`，支持热更新、停用、删除和恢复上一版；核心代码不参与编辑。使用说明与插件格式见 [自定义界面文档](launcher-plugins.md)。

对话、Agent 回复和思考过程支持 Markdown 表格、标题、嵌套列表、任务列表、引用及链接，历史会话也会按相同格式显示。HTML 代码块右上角的「预览」可打开交互页面，支持 HTML、CSS 和内联 JavaScript。预览在隔离环境中运行，仅加载内嵌资源，页面存储在关闭后清空；关闭预览可继续原来的对话。Markdown 解析资源随安装包提供，无需联网下载。

## 工作室与 WebUI

API Server 就绪后，点击「打开工作室」即可在 Launcher 内容区直接使用聊天、历史会话、Markdown、附件、思考过程和智能体功能。模型管理导航始终保留，可随时切换到启动、下载、日志和硬件页面，返回「工作室」后继续当前会话。Launcher 与独立的 `ftllm webui` 共用聊天组件和后端，界面配色、尺寸及语言会适配 Launcher。组件连接工作室选择的对话模型并使用启动配置中的 API Key，无需另开 WebUI 服务或端口。会话沿用 WebUI 的本地存储，刷新页面后仍然保留；停止对应模型或切换工作室的对话模型时会取消该 WebUI 的任务并清理组件；顶部服务选择与视频页签切换不影响已有聊天。

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
