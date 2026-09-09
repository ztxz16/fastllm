# FastLLM Launcher 桌面绿色包

此目录把现有 `ftllm launch` 网页界面嵌入 Electron，并复用仓库根目录的
`make_portable.sh` 生成完整 FastLLM 运行时。最终用户不需要安装 Electron、Node.js、
Python、CUDA Toolkit、cuBLAS、NCCL 或 Python 包。
包内同时包含 Pi Agent 和 FastLLM 桥接扩展，启动模型后可直接使用“工作室”中的目录 Agent。
目录搜索所需的 ripgrep、fd 也会随包提供，不依赖目标机器预先安装这些工具。

## Linux 使用入口

解压后双击根目录应用即可使用，无需先运行初始化命令：

```text
Fastllm-Launcher     桌面应用；FastLLM 速度 F + 显示器与启动箭头 + APP
ftllm-launch-webui   网页服务；FastLLM 速度 F + 浏览器与网络地球 + WEB
ftllm               命令行入口；无自定义图标，双击打开已配置环境的终端
launch.sh           兼容网页启动脚本
README.html         离线使用文档；双击用浏览器阅读
support/            运行时、依赖、图标和辅助工具
```

三个应用通过 `entrypoint.c` 编译为原生 ELF，执行权限随 tar 归档保留，
文件管理器可以直接启动。通过 `/proc/self/exe` 定位包目录，支持整体移动、
中文和空格路径，以及包外的符号链接。`support/entrypoint.sh` 负责启动实际程序；
在图形会话中无参数、无终端启动 `ftllm` 或 `ftllm-launch-webui` 时，自动打开系统终端。
带参数的命令行调用与无桌面运行保持原有语义。

不使用隐藏目录或 `.hidden` 列表。Electron/Python 运行时、动态库、辅助工具、
许可证和构建清单都位于 `support/`，Python 位于 `support/runtime/`。

首次在桌面会话中启动应用时，在后台注册 `support/icons/` 中的两枚 SVG 图标，
并通过 GIO 设置文件图标。启动不会等待图标初始化，不需要 sudo，也不更改包内文件内容。
只有需要手动修复图标时才使用 `./ftllm --setup-desktop`。Electron 窗口继续使用 PNG 图标。
辅助 `.desktop` 文件放在 `support/desktop/`，遵循
[Desktop Entry 规范](https://specifications.freedesktop.org/desktop-entry/latest/)，不作为首次启动入口。

只写 `.desktop` 的 `Icon` 字段不足以让所有文件管理器显示图标。例如 Nautilus 将
普通目录中的桌面文件按文档展示，它的自定义文件图标使用独立的
[元数据](https://github.com/GNOME/nautilus/blob/gnome-42/src/nautilus-file.c)。
这些用户元数据不能随 tar 包迁移，因此首次启动前可能显示系统通用应用图标。
原生入口可以直接运行；启动后自动配置并通知文件管理器刷新图标。
`ftllm` 保留系统默认图标；初始化时会清除旧版为它设置的自定义图标。
HTML 文档的样式和插图全部随包提供，断网也能正常阅读。
两枚图标沿用 `tools/fastllm_pytools/fastllm_icon.svg` 的 F 和速度线，强调 FastLLM 的快速推理特点。
文档中的“详细参数手册”链接到包内 `support/docs/parameters.html`，完全离线可读。
`build_parameter_docs.py` 使用包内 Python 和已安装的 CLI 解析器生成参数名、别名、默认值与取值类型，
再结合中文说明、逐项写法与完整启动示例，覆盖模型、设备、量化、缓存、MoE、投机解码、服务和工具参数。
页面模板为 `BUNDLE-PARAMETERS.html.in`；新增 CLI 选项也会出现在手册中。

无桌面服务器可直接使用 `./ftllm server ...`、`./launch.sh --no-browser`。
`source ./support/env.sh` 后可直接输入 `ftllm`、`ftllm-check` 等命令；
双击 `ftllm` 打开的终端会自动启用这个环境。完整使用示例见成品根目录 `README.html`。

## Windows 构建

同时发布 wheel、Electron 绿色包、校验和与验证报告时，使用仓库根目录的
[`make_release.ps1`](../make_release.ps1)，详见 [Windows 发布流程](../docs/windows-release.md)。

Windows 桌面包使用独立的 PowerShell 入口，最终双击 `FastLLM-Launcher.exe`
打开 Electron 窗口，不打开系统浏览器：

主目录与 Linux 版一样集中保留用户入口：

```text
FastLLM-Launcher.exe   Electron 桌面应用
ftllm-launch-webui.exe 浏览器启动器
ftllm.exe             命令行；双击打开已配置环境的 PowerShell
README.html           中文离线说明
support/              Electron、Python、DLL、许可证、构建清单和默认数据
```

三个入口均为静态链接 CRT 的原生 EXE，按自身路径定位 `support/`，支持整体移动、
中文和空格路径。已有 PowerShell 可执行 `. .\support\env.ps1` 启用包内命令。

```powershell
# 从当前源码编译并生成完整 Electron ZIP（默认多 CUDA 架构）
powershell -NoProfile -ExecutionPolicy Bypass -File .\desktop\package.ps1

# SM86 / RTX 30 系列；复用增量构建目录和下载缓存
powershell -NoProfile -ExecutionPolicy Bypass -File .\desktop\package.ps1 `
  -CudaArch 86 -BuildDirectory build-fastllm-benchmark-sm86 -RequireCuda

# 使用已经编译好的 Windows wheel；CudaArch 应与 wheel 一致
powershell -NoProfile -ExecutionPolicy Bypass -File .\desktop\package.ps1 `
  -Wheel .\build-fastllm-benchmark-sm86\cuda\tools\dist\ftllm-0.1.8.2-py3-none-win_amd64.whl `
  -CudaArch 86 -Offline
```

默认输出 `build-desktop-dist/FastLLM/`、带版本和架构的 ZIP 以及 `.zip.sha256`。
已有目录不会覆盖；再次构建请指定新的 `-OutputDirectory`。完整解压目录后运行 EXE，
不能只复制单个 EXE。目标机无需安装 Python、Node/npm、Electron、CUDA Toolkit、
VC++ Redistributable 或其他包；只需要 Windows 10/11 x64 和支持 AVX2/FMA/F16C
的 CPU，GPU 推理仍需要匹配的 NVIDIA 驱动。模型权重不随软件打包。

构建机需要 Visual Studio 2022 C++/Windows SDK/CMake，以及 GPU 编译所需 CUDA 12.9；
Python 和 Electron 由脚本下载并验证 SHA256，无需预装 Node/npm。运行时版本与校验值
统一维护在 `portable/windows/runtime-lock.json`。`-CpuOnly` 可构建
纯 CPU 包，`-Offline` 只使用缓存，`-CacheDirectory` 可修改缓存位置。

默认验证实际 Electron BrowserWindow、隔离后的 PATH、认证 API、工作室前端模块、
中文空格路径迁移，以及发送原生窗口关闭消息后的进程退出。`-SmokeModel 本地模型路径`
会进一步在 GPU 上加载模型、打开内嵌工作室、执行聊天请求，并验证关闭窗口时模型也退出。
需要双卡的模型同时使用 `-SmokeTp 2`；默认验证单卡。
测试期间窗口隐藏，截图和结果保留在输出目录的 `electron-test/`，不混入发布 ZIP。
完整成品包括 Electron 的所有 PE 文件都会执行 DLL 依赖审计，运行库随包携带。
最终归档验证同时写入输出目录的 `archive-verification.json`。

Windows 内部服务只绑定回环地址，由 Electron 自动启动和停止；关闭窗口会先请求
正常清理，再以 Windows 进程树终止作兜底。配置和日志默认位于 `support/data/`。

## Linux 构建

先生成当前源码对应的 wheel，再打桌面包：

```bash
./make_whl.sh
./desktop/package.sh --force
```

也可以显式指定已有 wheel，或让脚本先调用 `make_whl.sh`：

```bash
./desktop/package.sh --wheel /path/to/ftllm.whl
./desktop/package.sh --build-wheel
```

默认输出到 `portable-dist/`，产物是 `.tar.gz` 和对应的 `.sha256`。常用选项：

```text
--keep-dir       同时保留未压缩目录
--offline        只使用已有的 Python/Electron/Pi/搜索工具下载缓存和 wheelhouse
--skip-tests     跳过运行时冒烟测试
--format tar.zst 改用 zstd 压缩
```

构建机要求 Linux x86_64、C 编译器（默认 `cc`，可通过 `CC` 指定）、Python 3、
curl 或 wget、unzip、readelf、ldd 和常规 GNU 归档工具。建议在项目支持的最低发行版
Ubuntu 22.04 上构建，因为脚本会把 Electron
依赖的非 glibc 系统动态库一并收集，并拒绝高于 GLIBC 2.35 的 ELF。Electron 固定到
经过 SHA256 校验的官方预编译版本；升级时必须同时更新版本与校验值。

Pi 使用 `tools/ftllm_agent_runtime/scripts/fetch_pi.py` 固定并校验的版本。
打包时从当前仓库构建 Agent wheel，构建工具在临时环境中安装；下载缓存位于
`build-portable-cache/`。完整在线构建一次后可使用 `--offline` 复用缓存。
默认测试包括包内 Pi 的实际启动、临时项目文件读取及本地模拟模型的工具调用往返。
动态库收集同时包含 NSS 运行时加载的模块，以保证 Electron 能初始化证书和网络功能。

## 设计边界

- 包是可移动目录，不写系统安装路径；Electron 默认把数据放在包内 `support/data/`。
  浏览器 Launcher 和命令行沿用当前用户的配置目录。复制或迁移时应保留整个目录。
- Launcher 只监听 `127.0.0.1`，Electron 自动附加控制令牌并阻止主窗口跳转到外部页面。
- NVIDIA 驱动库和 glibc 不会打包。驱动必须与内核匹配，而 glibc 必须与宿主系统动态
  加载器保持一致；构建时会扫描整个成品并拒绝混入 `libcuda.so` 或其他 NVIDIA 驱动库。
- 最终用户需要满足基线的 Linux/glibc；仅 Electron 桌面窗口需要 X11/Wayland。
  `ftllm server`、`ftllm launch --no-browser` 和终端向导均可在无桌面的机器上使用。
