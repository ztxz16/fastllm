FastLLM Windows x64 绿色包

完整解压 ZIP 后双击 Launch.cmd（或 ftllm.exe），浏览器会打开部署页面。
命令行：ftllm.exe launch；其他命令：ftllm.exe --help。
自检：ftllm-check.cmd；强制验证 CUDA：ftllm-check.cmd --require-cuda。
自带 Python：python.cmd；不需要安装 Python、pip、Node.js、CUDA Toolkit、VC++ 运行库。
关闭启动窗口或按 Ctrl+C 停止启动器。部署模型需要准备本地权重，或联网下载模型。

系统要求：64 位 Windows 10/11，支持 AVX2/FMA/F16C 的 CPU。
GPU 模式还需受支持的 NVIDIA GPU 和系统显卡驱动；驱动由 Windows/NVIDIA 安装，
无法作为绿色程序随文件夹加载。无显卡驱动时可使用 CPU 后端。
编译的 CUDA 架构、源代码版本、依赖列表见 BUILD-INFO.json / requirements-lock.txt。
BUILD-INFO.json 标记的架构适用范围以实际编译为准，不能把单架构包用于所有显卡。

Python、CUDA 用户态运行库、Pi Agent、模型下载、WebUI/API、文档处理依赖均在包内。
本地模型启动无需联网；下载模型和联网搜索功能需要网络。模型权重不包含在本包中。
启动器配置及模型下载缓存默认写入 data；聊天记录默认保存在用户 .fastllm 目录。
包内 Python 与系统 Python 隔离。不要移动或单独复制 runtime 中的文件。
第三方软件许可证保留在 runtime、各包 dist-info/licenses 和 THIRD-PARTY 目录。
