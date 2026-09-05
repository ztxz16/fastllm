# FastLLM Launcher 桌面绿色包

此目录把现有 `ftllm launch` 网页界面嵌入 Electron，并复用仓库根目录的
`make_portable.sh` 生成完整 FastLLM 运行时。最终用户不需要安装 Electron、Node.js、
Python、CUDA Toolkit、cuBLAS、NCCL 或 Python 包。
包内同时包含 Pi Agent 和 FastLLM 桥接扩展，启动模型后可直接使用“工作室”中的目录 Agent。
目录搜索所需的 ripgrep、fd 也会随包提供，不依赖目标机器预先安装这些工具。

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

## 设计边界

- 包是可移动目录，不写系统安装路径；默认把数据放在包内 `data/`。
- Launcher 只监听 `127.0.0.1`，Electron 自动附加控制令牌并阻止主窗口跳转到外部页面。
- NVIDIA 驱动库和 glibc 不会打包。驱动必须与内核匹配，而 glibc 必须与宿主系统动态
  加载器保持一致；构建时会扫描整个成品并拒绝混入 `libcuda.so` 或其他 NVIDIA 驱动库。
- 最终用户仍需要 Linux 图形桌面（X11/Wayland）和满足基线的 glibc。这些是操作系统
  能力，不是需要额外安装的 FastLLM 依赖。
