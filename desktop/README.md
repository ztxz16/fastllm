# FastLLM Launcher 桌面绿色包

此目录把现有 `ftllm launch` 网页界面嵌入 Electron，并复用仓库根目录的
`make_portable.sh` 生成完整 FastLLM 运行时。最终用户不需要安装 Electron、Node.js、
Python、CUDA Toolkit、cuBLAS、NCCL 或 Python 包。
包内同时包含 Pi Agent 和 FastLLM 桥接扩展，启动模型后可直接使用“工作室”中的目录 Agent。
目录搜索所需的 ripgrep、fd 也会随包提供，不依赖目标机器预先安装这些工具。

## 使用入口

解压目录直接提供 `./ftllm server ...`、`./ftllm launch ...` 等命令，以及独立的
`./launch.sh` 网页启动脚本。无桌面服务器使用 `./launch.sh --no-browser`；有桌面时
也可以运行 `./FastLLM-Launcher` 打开 Electron 窗口。`source ./env.sh` 后可直接输入
`ftllm` 命令。成品根目录的 `README.md` 包含模型服务、远程网页访问和环境检查示例。

桌面资源与命令行运行时共用包根目录，Python 位于 `runtime/`，不再嵌套 `ftllm/` 子目录。

## 构建

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

构建机要求 Linux x86_64、Python 3、curl 或 wget、unzip、readelf、ldd 和常规 GNU
归档工具。建议在项目支持的最低发行版 Ubuntu 22.04 上构建，因为脚本会把 Electron
依赖的非 glibc 系统动态库一并收集，并拒绝高于 GLIBC 2.35 的 ELF。Electron 固定到
经过 SHA256 校验的官方预编译版本；升级时必须同时更新版本与校验值。

Pi 使用 `tools/ftllm_agent_runtime/scripts/fetch_pi.py` 固定并校验的版本。
打包时从当前仓库构建 Agent wheel，构建工具在临时环境中安装；下载缓存位于
`build-portable-cache/`。完整在线构建一次后可使用 `--offline` 复用缓存。
默认测试包括包内 Pi 的实际启动、临时项目文件读取及本地模拟模型的工具调用往返。
动态库收集同时包含 NSS 运行时加载的模块，以保证 Electron 能初始化证书和网络功能。

## 设计边界

- 包是可移动目录，不写系统安装路径；默认把数据放在包内 `data/`。
- Launcher 只监听 `127.0.0.1`，Electron 自动附加控制令牌并阻止主窗口跳转到外部页面。
- NVIDIA 驱动库和 glibc 不会打包。驱动必须与内核匹配，而 glibc 必须与宿主系统动态
  加载器保持一致；构建时会扫描整个成品并拒绝混入 `libcuda.so` 或其他 NVIDIA 驱动库。
- 最终用户需要满足基线的 Linux/glibc；仅 Electron 桌面窗口需要 X11/Wayland。
  `ftllm server`、`ftllm launch --no-browser` 和终端向导均可在无桌面的机器上使用。
