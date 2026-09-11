# Launcher 中使用 DeepSeek Harness

Launcher 主导航的“工作室”下面提供 **DeepSeek Harness** 入口。启动模型 API 后点击入口，已安装的 Harness 会启动独立进程，并在内容区嵌入原生页面；尚未安装时显示安装按钮。切换到其他栏目再返回时保留页面状态。

Harness 作为内置页面插件，复用已有的插件注册表、启停状态和自定义界面管理页。验证版本为 `@deepseek-ai/dsh@0.1.5-alpha.1`。

## 插件管理

在左侧 **agent → 管理**、Harness 页面上的“管理”，或 **自定义界面 → 已有自定义项与恢复 → DeepSeek Harness → 管理** 中打开统一管理面板：

- 查看未安装、已安装、安装中或运行中等状态。
- 点击“安装”即可安装独立运行环境，无需先启动模型。
- 点击“升级 / 重装”安装面板标明的当前支持版本，已是该版本时重新安装。升级前停止 Harness，失败保留旧运行环境。
- 点击“删除”仅移除独立运行环境，保留会话和工作目录，不删除系统安装；删除后可重新安装。
- 点击“停用”关闭 Harness 或取消当前安装，页面显示停用提示；模型服务、已安装环境和会话数据保留。
- 点击“启用”恢复页面，开关会持久保存；启用本身不下载依赖，也不启动 Harness。

页面、脚本、样式和 `plugin.json` 位于发布包的 `ui_plugins/harness/`，遵循内置插件只读规则。安装和进程生命周期由应用注册的后端处理；用户创建的自定义界面仍使用原有沙箱权限，不能声明原生插件或执行安装命令。

## 安装与启动

```bash
ftllm launch
```

1. 在 Launcher 中启动模型 API。
2. 点击左侧 **DeepSeek Harness**。尚未安装时，手动点击 **安装并打开 Harness** 下载独立运行环境，或提前在管理面板中安装；仅进入、刷新或切换页面不会安装。页面显示下载量、依赖安装和验证进度，“安装并打开”完成后自动打开。
3. 首次进入按 Harness 提示选择工作区，即可聊天和使用工具。

不需要预装 Node/npm，也不会修改系统 Node 或全局 npm 包。运行环境保存在 `~/.fastllm/deepseek-harness/runtime`，后续直接复用。pip 安装 ftllm 时只安装集成代码，不下载 Harness 本体。

一键安装为 Linux、macOS、Windows 的 x64/ARM64 提供对应 Node 24.20.0 运行环境，从 Node 官方站点下载并检查 SHA-256，再通过 npm 安装固定版本的 Harness。当前实际验证了 Linux x64，运行环境约占 **440 MiB**（不含 npm 缓存和会话数据）；其他平台大小可能不同。首次安装需要能访问 Node 下载站点及 npm 包源，npm 沿用用户的源、代理和证书配置。

安装期间可以切换栏目，返回后继续显示进度；也可以点击“取消安装”。失败时显示错误，点击“重试”即可重新安装。下载和安装先在临时目录完成，验证成功后才启用；多个 Launcher 不会同时向同一目录安装。

已经安装了可用 `dsh` 的用户可直接复用 PATH 中的命令。此前手动安装到 `runtime/node_modules/node` 的运行环境也继续支持。

Launcher 自动配置 FastLLM 提供方的 API 地址、模型名称、密钥和已识别的上下文容量。密钥只通过子进程环境传递。工作区选择器直接显示在嵌入页中。

Launcher 设置 `compat.supportsDeveloperRole: false`，让 Harness 使用 `system` 发送系统提示词，避免 Qwen 模板拒绝 `developer` 角色。直接连接 API 时，Qwen3.5/3.8 服务也会将 `developer` 指令与显式 `system` 指令按原顺序合并为开头的系统消息，保留用户消息和工具调用记录。

模型输出预算通过 `compat.maxTokensField: max_tokens` 发送，兼容只支持旧字段的 FastLLM 服务。新版服务也接受 `max_completion_tokens`；两者同时提供时，以非空的 `max_completion_tokens` 为准。

## 流式连接与诊断

Harness 的 `llm-pi-ai.providers.fastllm.streamIdleTimeoutMs` 控制等待模型有效输出的空闲时限，默认 300000 毫秒。SSE 注释心跳只维持传输连接，不会重置这个时限。若报错包含 `pi-ai stream idle timeout`，应结合排队和 prefill 耗时调整该配置。模型以 `finish_reason: length` 结束则表示本次输出预算耗尽，与网络断开不同。

Harness 退出时，Launcher 将最近的日志尾部脱敏后保存到 `~/.fastllm/deepseek-harness/logs/latest.log`，上一份保存在 `previous.log`。每份读取最多 64 KiB，去掉边界处的不完整行；已知模型密钥、URL 中的令牌和 Authorization 值会隐藏。日志文件在 POSIX 上以 0600 权限创建。原始临时日志仍随启动目录清理。

## 实验性恢复扩展（默认关闭）

Launcher 默认不加载恢复扩展。需要试用时，在启动 Launcher 前显式设置环境变量（仅值 `1` 开启）：

```bash
FTLLM_HARNESS_EXPERIMENTAL_RECOVERY=1 ftllm launch
```

取消该环境变量并重启 Launcher 和 Harness 后恢复默认行为。角色兼容、输出预算、SSE 心跳和诊断日志独立于此开关。

**已知限制：** 文本续写可能重复截断处的内容；headless 标准输出只包含最后一段回答，前面的片段需从会话记录读取；恢复额度耗尽时，最后生成的一段文本可能只保存在失败尝试中，没有进入正式回答。退出码 0 不保证续写后的完整文本无重复。这些问题仍待修复。

启用后，扩展使用同一会话中的实际结果继续任务：

- 文本达到输出上限后，保留前文并排队一个续写回合。
- 只有思考、没有答案或工具调用时，要求模型减少思考并给出可用输出。
- 工具调用被服务端拒绝或截断时，将恢复提示写入会话，再要求模型重建合法调用；被拒绝的调用不会执行，之前成功的工具结果继续保留。

恢复提示带有插件来源并记录在会话中。每次用户输入默认最多自动恢复 2 次，续写和上述重试共用这个额度；达到上限后返回明确错误。用户取消后不发起恢复。摘要压缩、上下文超限、网络重试仍遵循 Harness 本身的策略。

上述错误恢复期间若有待处理输入，扩展会保留队列、结束当前失败轮次，并交由 Harness 在新轮次正常读取输入。即时指令保持原有顺序，排队的后续任务仍各自进入独立轮次；即使恢复额度已耗尽，新输入也会优先处理。会话记录会将交接前的轮次标为扩展主动中止，随后记录新输入的执行结果。

启用时，Launcher 生成的 patch 会插入 `fastllm-harness-recovery`，其 `config.maxRecoveries` 为 2。独立使用 Harness 时，可在 patch 的 `insert` 中加载已安装 ftllm 包目录内的 `harness_recovery.mjs`，配置 `provider: fastllm`、`maxRecoveries: 2`。`maxRecoveries` 可设为 0–10；0 只禁止自动重试和续写，仍会进行错误分类及待处理输入交接，不等于卸载扩展。扩展使用 Harness 的公开插件钩子，不修改 npm 依赖。

## 数据和生命周期

- 默认配置和会话保存在 `~/.fastllm/deepseek-harness/home`，默认工作目录为 `~/.fastllm/deepseek-harness/workspace`，也可以在 Harness 中选择其他工作区。
- “停止 Harness”仅关闭 Harness；停止模型、切换模型或退出 Launcher 时，也会关闭对应的 Harness 进程或取消尚未完成的安装。已安装环境、会话和文件保留。
- 启动失败时显示错误，可以点击“重试”。刷新 Launcher 后可以重新连接仍在运行的 Harness，或继续查看安装进度。
- 原有工作室与 Harness 分别保存会话。

当前嵌入使用同一主机的独立 HTTP 代理端口，页面与 Launcher DOM 保持隔离。Harness 内部服务只监听回环地址，代理在后端完成原生 token/cookie 认证；浏览器使用代理独立的 token/HttpOnly cookie，原生凭据不发送给浏览器。

通过 localhost、局域网 IP 或主机名访问同一个 Launcher 时，每个浏览器获得对应地址的入口，复用已有 Harness 进程。Harness 重启导致入口地址变化时，已打开页面会自动重连；普通栏目切换和状态轮询仍保留当前页面。

支持 IPv4 地址或主机名；HTTPS、IPv6 和仅开放 Launcher 一个端口的反向代理部署尚未适配。远程访问需要同时能访问代理分配的端口。
