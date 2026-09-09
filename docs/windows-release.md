# Windows wheel 与 Electron 发布流程

适配范围、代码整理和开发分支建议见 [Windows 适配整理](windows-adaptation.md)。

仓库根目录的 `make_release.ps1` 将当前工作区构建为 Windows wheel 和 Electron
绿色包，并统一输出校验和、构建日志及验证报告。

## 一键发布

构建机需要 Windows x64、Visual Studio 2022 C++/Windows SDK 和 CMake。CUDA 发布
还需要 CUDA Toolkit（已验证 12.9），包括用于核验实际架构的 `cuobjdump.exe`。
脚本自动准备 Python、Electron 和打包依赖，不要求预装 Python 或 Node/npm。

在仓库根目录运行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_release.ps1

# 指定输出目录；验证本机 CUDA，并通过 Electron 加载本地模型完成一次对话
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_release.ps1 `
  -OutputDirectory build-release-local `
  -RequireCuda -SmokeModel "D:\models\Qwen3-4B-FP8"

# 纯 CPU 发布，不需要 CUDA Toolkit
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_release.ps1 `
  -CpuOnly -BuildDirectory build-windows-cpu -OutputDirectory build-release-cpu
```

默认架构直接读取 `make_whl.sh` 中的 `CUDA_ARCH_LIST`。`-CudaArch "86"` 等参数可以
覆盖默认列表。版本从 wheel 元数据读取；`-Version "0.1.8.2"` 只要求版本匹配，
不会修改源代码里的版本号。

没有本机 GPU 时可以构建和核验 CUDA 包。`-RequireCuda` 要求实际加载 CUDA 后端并
枚举到 GPU；`-SmokeModel` 额外进行 GPU 模型推理。报告中的 `gpu_inference_tested`
只有在模型实测成功时才为 `true`。`-CpuOnly` 不接受这两个 GPU 测试参数。
需要双卡的模型可同时指定 `-SmokeTp 2`，使用 GPU 0、1 完成桌面聊天和退出验证。

## 复用 wheel、缓存和编译结果

```powershell
# 复用已有 wheel；CudaArch 必须与其实际编译参数一致
powershell -NoProfile -ExecutionPolicy Bypass -File .\make_release.ps1 `
  -Wheel "D:\wheels\ftllm-0.1.8.2-py3-none-win_amd64.whl" `
  -CudaArch "86" -OutputDirectory build-release-from-wheel -Offline

# 继续已有多架构编译；保持上次的构建目录及 CMake 参数
.\make_release.ps1 -BuildDirectory build-release-0.1.8.2/native `
  -OutputDirectory build-release-next -Offline `
  -CMakeArgs "-DCMAKE_CUDA_FLAGS=--split-compile=2"
```

发布入口总是增量编译。中断后重新运行时，保留相同的 `-BuildDirectory`、
`-CudaArch` 和 `-CMakeArgs`，即可复用已完成的目标文件。再次发布使用新的
`-OutputDirectory`；已存在的应用目录或完整发布结果会被拒绝覆盖。

`-Offline` 使用已有的 `build-portable-cache` 下载归档与 wheelhouse，缓存缺失或
SHA256 不匹配时直接报错。`-CacheDirectory`、`-Constraints` 可指定缓存及 Python
依赖约束；`-CMakePath`、`-CuobjdumpPath` 可指定工具路径；`-Jobs` 控制构建并行数。
即使使用现成的 CUDA wheel，发布验证仍需 `cuobjdump`。

## 输出与验证

默认输出目录为 `build-windows-release`：

```text
FastLLM/                         三个原生入口、README.html 和 support/，解压即可运行
ftllm-<版本>-py3-none-win_amd64.whl
FastLLM-Launcher-<版本>-windows-x64-<架构>-<提交>.zip
*.sha256 / SHA256SUMS.txt         最终 wheel 与 ZIP 的 SHA256
README.md                        成品使用说明
wheel-verification.json          wheel 内容、版本、CPU/CUDA 库及实际架构
cuda-elf.txt / cuda-ptx.txt       CUDA 二进制架构清单（CUDA 发布）
archive-verification.json        ZIP 清单、逐文件哈希及构建信息
release-verification.json        发布结果汇总与 GPU 实测状态
release-*.log                    完整发布日志
electron-test/                   实际窗口测试、截图及退出清理结果
```

验证覆盖 wheel 元数据与入口、DLL 架构、隔离 Python、Python 依赖、原生 DLL
依赖闭包、CPU 回退、中文空格路径迁移、Electron 启动和关闭，以及最终归档的
全部文件哈希。从源码构建时还比较 wheel 资源与当前源码、原生 DLL 与编译结果。
`-Wheel` 模式只验证输入 wheel 本身，`source_assets_checked` 和
`native_build_checked` 会明确记录为 `false`。

完整解压 ZIP 后双击 `FastLLM/FastLLM-Launcher.exe`。绿色包包含用户态运行环境，
GPU 模式仍需系统 NVIDIA 驱动，模型权重另行提供。

主目录只保留 `FastLLM-Launcher.exe`、`ftllm-launch-webui.exe`、`ftllm.exe`、
`README.html` 和 `support/`。Electron、Python、动态库、许可证和逐文件清单集中放在
`support/`；构建信息为 `support/BUILD-INFO.json`，清单为 `support/MANIFEST.sha256`。
外部发布日志、截图和验证报告保留在输出目录中，不放进绿色包主目录。

## 脚本职责与维护

| 入口或目录 | 用途 |
| --- | --- |
| `make_release.ps1` | wheel、Electron、验证报告和校验和的一键发布 |
| `make_whl.ps1` | 单独构建 wheel；支持 `-Incremental`、CPU、CUDA 和 NCCL 配置 |
| `make_portable.ps1` | 构建 CLI/浏览器运行时，供桌面包复用 |
| `desktop/package.ps1` | 单独构建 Electron 包；支持复用 wheel |
| `tools/windows/common.ps1` | 路径、命令执行、CMake 查找、下载与 Python 引导环境 |
| `portable/windows/runtime-lock.json` | Windows Python、Electron、Pi、rg、fd 的版本、来源和 SHA256 |
| `tools/windows/verify_wheel.py` | 可独立使用的 wheel 与 CUDA 架构验证 |
| `tools/windows/collect_release.py` | 核对验证结果并收集最终发布产物 |

升级运行时时统一更新 `runtime-lock.json` 的版本、下载地址和 SHA256，并执行完整
发布验证。Pi 的版本需同时与 `tools/ftllm_agent_runtime/scripts/fetch_pi.py` 匹配。
当前 Python 环境使用 CPython 3.11；更换 Python 大小版本时还需更新 wheelhouse
标签、`python311._pth` 等与 ABI 有关的配置。

单独验证 wheel 的示例：

```powershell
.\build-portable-cache\bootstrap\python\python.exe -I -B -X utf8 `
  .\tools\windows\verify_wheel.py "D:\wheels\ftllm-0.1.8.2-py3-none-win_amd64.whl" `
  --cuda-arch "86" --report build-wheel-check/result.json
```

脚本回归测试：

```powershell
.\build-portable-cache\bootstrap\python\python.exe -I -B -m unittest discover -s tools/windows/tests -v
```
