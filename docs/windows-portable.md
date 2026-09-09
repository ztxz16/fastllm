# Windows 绿色包

一键生成 wheel、Electron 应用和发布验证报告，参见 [Windows 发布流程](windows-release.md)。

需要 **Electron 桌面应用** 时，请使用 [桌面打包脚本与说明](../desktop/README.md#windows-构建)：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\desktop\package.ps1
```

产物入口是 `FastLLM-Launcher.exe`，内置 Electron，不使用系统浏览器。
桌面包主目录还提供网页与命令行入口、`README.html`，运行时和辅助文件集中在
`support/`，与 Linux 桌面包的布局一致。
以下 `make_portable.ps1` 是底层 CLI/浏览器运行时包，桌面脚本会复用它的运行环境。

`make_portable.ps1` 构建 `ftllm launch` 的 Windows x64 ZIP。完整解压后双击
`Launch.cmd` 或 `ftllm.exe`，即可打开浏览器部署页面；也可以运行
`ftllm.exe launch`、`ftllm.exe server ...` 等命令。

## 构建与输出

构建机需要 Visual Studio 2022 C++ 工具链、Windows SDK、CMake，以及编译 GPU
后端时使用的 CUDA Toolkit 12.9。脚本自动下载有 SHA256 校验的 Python 3.11.15、
Pi 0.84.4、ripgrep 和 fd；不要求构建机预装 Python。首次构建需联网。

```powershell
# 编译 CPU + CUDA，打包并验证；默认使用多架构列表
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_portable.ps1

# RTX 30 系列 / SM86 原生代码和 PTX，缩短编译时间
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_portable.ps1 -CudaArch 86 -RequireCuda

# CPU-only 不需要 CUDA Toolkit；请使用独立构建/输出目录
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_portable.ps1 -CpuOnly `
  -BuildDirectory build-portable-cpu -OutputDirectory build-portable-cpu-dist

# 复用已有 Windows wheel
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_portable.ps1 `
  -Wheel .\build-fastllm-windows\cuda\tools\dist\ftllm-0.1.8.2-py3-none-win_amd64.whl `
  -CudaArch 86
```

默认输出到 `build-portable-dist`：版本/架构命名的 ZIP、对应 `.zip.sha256`，
以及已解压的 `ftllm` 目录。目录已存在时拒绝覆盖，使用新的 `-OutputDirectory`
即可保留多个构建。解压目录名保持简短，避免 Windows Explorer 的路径长度限制；
建议解压到 `D:\apps\ftllm` 等短路径。

`-Wheel` 模式不会重编译原生库，`-CudaArch` 必须与输入 wheel 的编译参数一致。
裸架构值（例如 `86`）包含该架构的原生代码和 PTX；`86-real` 只包含原生代码。
其他显卡是否能使用该 wheel，取决于实际编译架构和驱动对 PTX 的支持。

`-Offline` 使用 `build-portable-cache` 中已缓存的归档和 wheelhouse；缺失缓存时
报错，不临时安装系统依赖。`-CacheDirectory` 可切换缓存目录，`-Constraints`
可使用另一组 Python 依赖版本约束。`-SkipTests` 只用于调试构建流程，发布应保留验证。

## 运行环境与包内内容

目标电脑需要 Windows 10/11 x64 和支持 AVX2/FMA/F16C 的 CPU。包内包含 Python、
所有声明的 Python 依赖、CUDA runtime、cuBLAS、VC++ runtime、CPU fallback、
Pi Agent 和其搜索工具，不要求安装 Python、Node.js、Git Bash、CUDA Toolkit 或
VC++ Redistributable。Windows Agent 使用系统 PowerShell 执行命令。

GPU 模式仍需要系统 NVIDIA 显卡驱动；内核驱动无法通过解压程序目录安装。
CPU 模式无需显卡驱动。模型权重不包含在包中，本地模型可离线使用；下载模型及
联网搜索需要网络。Windows 系统浏览器用于显示 `ftllm launch` 网页界面。

`ftllm.exe` 是静态链接 C runtime 的原生启动器，从自身目录定位 Python并转发参数
及退出码。Python `_pth` 配置隔离系统 Python、用户 site-packages、注册表和
`PYTHONPATH`；子进程使用同一运行时。DLL 搜索包含包内 CRT/CUDA 库目录。

默认配置、模型下载缓存位于包内 `data`，聊天记录沿用用户目录 `.fastllm`。
`python.cmd` 提供包内 Python，`ftllm-check.cmd --require-cuda` 可验证 GPU 环境。

## 验证与追溯

构建会运行 `pip check`、依赖导入、CPU fallback 装载、Pi 版本检查、所有 PE 文件
的直接/延迟导入审计，以及 `ftllm.exe launch` 的真实 HTTP 启动、页面/API、硬件
检测和退出测试。测试前将目录移动到含中文和空格的路径，并将 `PATH` 限制为
Windows System32，移除外部 Python/CUDA 环境变量。`-RequireCuda` 额外要求装载
CUDA 后端并枚举到真实 GPU。

`BUILD-INFO.json` 记录源码版本、工作区是否有修改、wheel SHA256、编译架构和运行时
来源；`requirements-lock.txt` 记录最终依赖版本；`DLL-DEPENDENCIES.txt` 记录 DLL
依赖闭包；`MANIFEST.sha256` 记录包内每个文件的 SHA256。
