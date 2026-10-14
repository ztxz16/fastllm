# Linux x86_64 wheel 构建

发布环境采用 Ubuntu 20.04 / glibc 2.31、GCC 11、CUDA 12.9 和 Binutils 2.38，
目标是兼容 Ubuntu 20.04/22.04。运行 wheel 需要 Python >= 3.10；构建 Python 可用 3.8。

在仓库根目录执行：

```bash
docker build -f whl_docker/Dockerfile --build-arg BUILD_JOBS=32 -t fastllm-wheel:cu129-u20 .
docker run --rm -v "$PWD:/src" -w /src fastllm-wheel:cu129-u20 \
  bash make_whl.sh --cpu-jobs 32 --cuda-jobs 12 --nvcc-threads 2
```

编译不需要 GPU。默认产物仍在 `build-fastllm/tools/dist`，兼容现有桌面版和 portable
打包脚本。CPU/CUDA 中间文件分别在 `build-wheel/cpu` 和 `build-wheel/cuda`，
日志为 `build-wheel/build.log`。可用 `--output-dir` 指定产物目录，
用新的 `--build-dir` 做干净构建；默认复用已有构建目录。

大内存、多核机器可使用 `--cpu-jobs 96 --cuda-jobs 24 --nvcc-threads 4 --split-compile 2`。
NVCC 内部线程会叠加在 CUDA 并行任务上，普通机器应降低线程数。
`--nightly` 生成 nightly 包；完整选项见 `bash make_whl.sh --help`。

没有 Docker 权限时，可在具备 `CAP_SYS_CHROOT` 的机器上准备 Ubuntu 20.04 rootfs，
安装相同工具链并在其中执行脚本。编译、链接和 libnuma/NCCL 都必须使用兼容的
旧系统依赖；仅更换 GCC 不会降低宿主机 glibc 要求。

- 默认 `--max-glibc 2.31`，提前拒绝更新的构建环境，打包时再次审计 ELF 导入符号。
- 使用 `linux_x86_64` 标签；glibc 检查不等同于 manylinux 认证。
- 默认保留全架构和 PTX，以 CUDA `size` 压缩并 strip 暂存的动态库。
  该压缩格式要求 CUDA 12.4 发布时期或更新的驱动，特定 GPU/PTX 可能要求更高版本。
  `--compression legacy` 可使用原压缩方式；普通 CMake 构建仍默认 legacy。
- CPU/CUDA 库保留 `$ORIGIN` 搜索路径并附带 libnuma，CUDA 库由 pip 依赖和驱动提供。
- 每次打包读取当前 Python 源码，避免增量构建遗漏 Python 修改。
  wheel 同目录的 JSON 记录 ELF 依赖、glibc 要求、大小和 SHA256，阶段耗时显示在脚本输出中。

在 Python 3.10/3.11 虚拟环境中安装：

```bash
python -m pip install -c tools/wheel/runtime-cu129.txt build-fastllm/tools/dist/ftllm-0.1.8.2-py3-none-linux_x86_64.whl
python -c 'from ftllm import llm; print(llm.get_cpu_threads())'
```

发布前验证目标系统的依赖安装、CPU/CUDA 加载、GPU 运算和 CUDA Graph。
包含某个架构的二进制不代表已经在对应物理 GPU 上验证。
