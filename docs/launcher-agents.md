# Launcher 中使用 OpenCode、Codex 和 Claude Code

`ftllm launch` 的导航分为“模型管理”和“agent”两组。前一组包含启动服务、下载模型、运行日志、硬件信息；后一组包含工作室、DeepSeek Harness、OpenCode、Codex、Claude Code。四个原生 agent 使用 Launcher 当前启动的 FastLLM 模型，分别保存会话。

**pip 包只包含适配代码和页面。进入页面、刷新、启用插件都不会安装外部工具。** 点击 agent 分组旁或对应页面中的“管理”，可在模型服务未启动时手动安装。也可以启动模型后点击 **安装并打开 OpenCode / Codex**。安装进度、取消和失败重试与 Harness 一致。已安装时，打开栏目可直接启动工具。

## 使用

1. 运行 `ftllm launch`，启动一个支持工具调用的模型 API。
2. 点击 OpenCode、Codex 或 Claude Code；首次使用时手动点击安装按钮。
3. OpenCode 使用原生网页：点击 **Add project**，选择运行 Launcher 的机器上的目录，再创建会话。
4. Codex 左侧按工作目录分组显示会话，右侧显示当前会话和它实际绑定的目录。点击“新建会话”，在“会话工作目录”旁点击文件夹按钮，浏览运行 Launcher 的机器并选择目录，也可直接输入路径；发送第一条消息时创建会话并绑定该目录。点击目录旁的 `+` 可直接在该目录下准备新会话。已有会话使用各自保存的目录，切换会话不会改变目录。
5. Claude Code 使用接近 Harness 的简洁会话侧栏和聊天区，跟随 Launcher 的浅色 / 黑夜模式。目录选择、按项目分组、独立草稿、Markdown、回车发送、重命名和归档复用同一套会话界面。底部可选择当前模型声明的思考档位。

Claude Code 基于官方 Agent SDK 运行，通过 FastLLM 的 `/v1/messages` 使用当前模型，无需登录 Anthropic。工具和命令由真正的 Claude Code 执行；保留默认权限检查，需要批准的操作会显示“允许一次 / 拒绝 / 取消本轮”，工具提出的问题也可在页面回答。这里的权限确认不等于操作系统沙箱。停止、刷新和重新打开后可以恢复已保存的会话；新建会话会绑定所选目录，不会修改另一会话的目录。

Claude Code 的安装、升级 / 重装、删除、启用 / 停用与其他 agent 使用相同管理面板。它需要独立 SDK 运行环境，即使 PATH 中已有 `claude` 命令也不会复用该命令或用户的 `~/.claude`。删除只移除 `~/.fastllm/claude/runtime/`，保留 `home/` 中的原生会话、界面历史以及工作目录。

Codex 页面支持会话搜索、切换、重命名、归档、逐项消息和工具输出、文件修改详情、审批、回答工具问题和取消生成。用户消息靠右，Codex 回复靠左；`Enter` 发送消息，`Shift+Enter` 换行，也支持 `Ctrl+Enter` / `Cmd+Enter`。中文输入法选词时的回车不会发送。每个会话的输入草稿及各目录下的新会话草稿保存在当前浏览器，完整会话由 Codex 保存；切换栏目保留页面，刷新后可以恢复会话和待审批请求。

消息、计划和思考内容复用工作室的 Markdown 渲染器，支持流式更新、标题、表格、列表、链接、代码块和代码复制；刷新后历史消息也会重新渲染。命令输出和文件差异保留纯文本格式，消息中的 HTML 不会作为页面代码执行。

Codex 输入框下方提供“思考档位”，按当前模型和会话记住选择；新会话从模型默认档位开始。OpenCode 使用输入框中的 **Choose model variant** 原生菜单，Harness 使用原生模型设置中的思考档位。启动 agent 时会读取当前服务 `/v1/models` 的能力声明，模型使用别名也可识别。例如 Qwen3.5 支持 `low / medium / xhigh`，Kimi K3 支持 `low / high / max`；不会向这些模型发送不支持的通用档位。没有声明可调档位的模型继续使用服务默认设置，Codex 会显示禁用的“模型默认”及原因。更新适配代码后需重启 Launcher 并重新打开 agent；思考参数的服务端处理更新需同时重启模型服务。

两者都能运行命令和修改文件。Codex 使用 `workspace-write` 和 `on-request`，需要额外权限时在页面请求批准；OpenCode 使用原生权限交互。工作目录用于组织任务，不代表操作系统级隔离。它们与原来只允许前端扩展的“自定义界面”权限不同；用户自建界面仍不能声明原生执行能力。

agent 管理面板复用插件注册表和管理接口，也可从 **自定义界面 → 已有自定义项与恢复 → 管理** 打开。工作室不提供外部运行环境管理。

- **安装**：下载安装独立运行环境，无需启动模型；已有系统命令时可选择安装独立版本。
- **升级 / 重装**：安装面板标明的当前支持版本；已是该版本时重新安装。先停止对应 agent，在临时目录完成安装和验证后替换运行环境，失败时保留旧版本。
- **删除**：停止该 agent 并仅移除其 `runtime/`，保留会话、配置和工作目录。不会删除内置页面或系统安装，之后可重新安装；若 PATH 中有可用命令，会继续显示为使用系统安装。
- **启用 / 停用**：复用插件开关。停用会停止对应进程或取消安装，保留运行环境、会话、文件和模型服务。

停止模型、切换模型或退出 Launcher 时，会同时停止对应的 agent 或取消安装、升级。

## 安装与数据

| 工具 | 点击安装时使用的版本 | 默认数据目录 | 模型接口 |
| --- | --- | --- | --- |
| OpenCode | `opencode-ai@1.18.26` | `~/.fastllm/opencode/` | `/v1/chat/completions` |
| Codex | `@openai/codex@0.153.4` | `~/.fastllm/codex/` | `/v1/responses` |
| Claude Code | `@anthropic-ai/claude-agent-sdk@0.3.266`，内含 Claude Code `2.1.266` | `~/.fastllm/claude/` | `/v1/messages` |

每个目录中的 `runtime/` 保存独立 Node/npm 和工具；`workspace/` 是默认工作目录。OpenCode 的配置、缓存、会话分别位于 `config/`、`cache/`、`data/`、`state/`；Codex 的配置和会话位于 `home/`。不覆盖用户原有的全局 OpenCode / Codex 配置和登录信息。

Claude Code 的 `home/launcher-sessions/` 保存页面使用的会话目录、标题、归档状态和消息历史；原生 SDK 同时在独立 `home/` 内保存用于续聊的会话。密钥只通过子进程环境传递。模型、工作目录和权限策略由适配器设置，不加载全局或项目 settings 来替换这些选项。

思考参数通过 Anthropic `output_config.effort` 传给 FastLLM，并使用与 Chat Completions 一致的模型专用规则；未声明档位时保留服务默认行为，避免 Claude Code 自动补上模型不支持的 `high`。升级这部分适配后，需要同时重启 Launcher 和模型 API 服务。

安装器复用 Harness 的 Node 24.20.0 下载与 SHA-256 校验，支持 Linux、macOS、Windows 的 x64/ARM64 安装包选择；当前完整验证的平台是 Linux x64。npm 沿用用户的包源、代理和证书配置。安装在临时目录完成并验证后才切换到正式目录，失败保留原有文件；同一工具的安装使用文件锁互斥。

Linux x64 实测独立运行环境的磁盘占用约为 OpenCode **491 MiB**、Codex **459 MiB**（按 `du` 统计，硬链接不重复计算），不含 npm 缓存和会话数据。Launcher 的网页/终端代理使用轻量 Python 依赖 `httpx` 和 `websockets`，由 ftllm 的服务依赖提供，不属于外部 Agent 运行时。

PATH 中已有的 `opencode` 或 `codex` 可以直接复用；上表版本经过验证，其他版本的协议或网页可能不同。适配器禁止 OpenCode 自动升级。FastLLM 地址、模型名称、上下文容量自动配置；模型密钥通过子进程环境传入，不写入生成的配置正文。Codex 不需要登录 OpenAI 账号来使用此本地提供方。

适配器不压缩、裁剪或重新拼接会话历史。OpenCode 的自动压缩与裁剪已关闭；Codex 配置了高于模型容量的自动压缩阈值。模型或运行时报告的上下文超限会显示在页面。原生工具自身对单次命令输出的长度限制仍遵循工具的实现。

## 嵌入和部署

Claude Code 与 Codex 的会话界面通过 Launcher 已有的认证接口访问 stdio 桥接，不需要开放额外网页端口。Claude Code 在 Linux x64 验证了真实 SDK 的流式回复、批准后写入文件、原生会话续聊与思考档位传递；其他平台尚未做端到端验证。参考官方 [Agent SDK](https://code.claude.com/docs/en/agent-sdk/typescript)、[网关配置](https://code.claude.com/docs/en/llm-gateway)与[环境变量](https://code.claude.com/docs/zh-CN/env-vars)。

OpenCode 使用原生静态资源和 API。内部服务只监听回环地址，外部入口通过独立端口代理 HTTP、SSE 和终端 WebSocket。入口使用独立 token/HttpOnly cookie，服务密码只由后端注入。页面与 Launcher DOM 位于不同 origin。

内嵌 OpenCode 的浅色/深色模式跟随 Launcher，切换主题不会刷新会话或丢失输入。内置的外观适配缩减原生窗口的重复边框和留白，并调整项目栏在嵌入区域内的布局；聊天、项目选择、文件差异、审批和终端继续由 OpenCode 原生界面处理。适配样式以表中的已验证版本为依据，不改写安装包。

通过 localhost、局域网 IP 或主机名访问同一个 Launcher 时，OpenCode 会为当前浏览器提供对应地址的嵌入入口，复用已有进程，并保留其他已打开页面的连接。

与当前 Harness 集成相同，OpenCode 嵌入要求 HTTP、IPv4 或主机名，并且浏览器能访问分配的额外端口；HTTPS、IPv6 和仅开放 Launcher 单端口的反向代理尚未适配。网页是否可离线使用取决于所复用的 OpenCode 安装包是否带有原生网页资源，不能将任意 PATH 版本都视为离线包。

Codex 通过 app-server 的 stdio JSON-RPC 连接到 Launcher 后端。网页沿用 Launcher 的鉴权和端口，不另外公开 app-server。后端只提供此页面需要的会话、回合和审批操作，不提供任意 JSON-RPC 转发或用户插件后端注册。尚未适配的原生请求会在页面提示，可取消对应回合。

接口依据：[OpenCode Web](https://opencode.ai/docs/web/)、[OpenCode 自定义提供方](https://opencode.ai/docs/providers/)、[Codex app-server](https://learn.chatgpt.com/docs/app-server)、[Codex 配置](https://learn.chatgpt.com/docs/config-file/config-reference)。
