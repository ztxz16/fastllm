# Vast CUDA 12.6 环境镜像

将已经通过 GPU 权重、Dense 层和 MoE EP 单层验收的环境导出，在 Windows
Docker Desktop 的 Linux 引擎上构建。导出和构建均不编译 FastLLM。

## 范围

- 基础镜像固定为 `vastai/base-image:cuda-13.0.3-auto`，Ubuntu 24.04 x86_64。
- 从原实例锁定 CUDA 12.6、NCCL `2.28.3-1+cuda12.9`、系统 Python 和已有构建工具。
- 使用原实例的 `/venv/main` Conda/venv 归档，保留绝对安装位置；不重新下载 Python 包。
- 构建时检查 Python 版本、ABI、基础解释器路径和全部包版本。
- editable 包和旧式 egg-link 指向环境外的代码，导出时拒绝，需单独处理。
- 代码、模型、验收报告和其他工作目录单独迁移；本工具不复制整个容器根目录。
- 此镜像保留原基础镜像的 CUDA 13 组件和驱动要求，默认选择 CUDA 12.6。
  不能将新实例的宿主机驱动要求仅按 CUDA 12.6 判断。
- venv 一般不可随意迁移。本工具只针对同一基础镜像、同一绝对路径；环境检查
  不能替代在新 GPU 实例上重跑功能验收。

## 原 Vast 服务器：导出

保持当前环境稳定，导出期间不要同时修改 Python 环境或安装系统包。

```bash
cd /home/project/vllm_fastllm
git fetch origin
git switch sk
git pull --ff-only origin sk
unset PYTHONHOME PYTHONPATH LD_PRELOAD
export PYTHONNOUSERSITE=1
/venv/main/bin/python3 tools/vast_image/export_env.py export
```

输出目录形如 `/home/models/fastllm-cu126-image.XXXXXXXX`。中断后用
`export --output 原目录` 重试；包版本变化时必须生成新目录。
源中已无法获取某个锁定版本时会报错，不替换成其他版本。仅在缺少软件源索引时
先执行 `apt-get update`，再重试导出。

APT 只用于解析地址。安装包优先从 `/tmp/nccl-cu12-download` 和
`/var/cache/apt/archives` 校验后复用，缺少的文件通过 aria2 下载。
Python 环境归档可能较大，导出目录需有足够剩余空间。

## Windows：传输和构建

使用已连接 Vast 的 MobaXterm 左侧 SFTP 面板，将整个导出目录传到
`C:\fastllm-cu126-image`。这是通过已有 SSH 身份认证的文件传输；无需开放 HTTP
端口。不要在 Windows 解压目录内的 `python-env.tar.gz`，Linux 权限和软链接
由 Docker 构建时恢复。

在 Windows PowerShell 中执行：

```powershell
Set-Location C:\fastllm-cu126-image
docker build --platform linux/amd64 --progress=plain -t moguizhiz1313/fastllm-cu126:cu126-nccl2283 .
if ($LASTEXITCODE -ne 0) { throw "镜像构建失败，请保留日志。" }
docker run --rm --entrypoint /venv/main/bin/python3 moguizhiz1313/fastllm-cu126:cu126-nccl2283 /opt/fastllm-image/export_env.py verify --context /opt/fastllm-image
if ($LASTEXITCODE -ne 0) { throw "镜像内环境版本检查失败。" }
```

镜像层由 Docker 按 Registry 协议拉取；普通 `.deb` 下载仍由 aria2 完成。
构建会按基础镜像的实际状态补齐依赖，随后用 `dpkg` 从本地文件安装，避免 APT
重新下载已经取得的包。全部锁定包设为 hold，普通升级不会改变它们。

## Windows：登录并推送

先在 Docker Hub 创建私有仓库 `moguizhiz1313/fastllm-cu126`。
登录提示 Password 时输入 Docker Hub 的 Read/Write Token，无需把 Token 写进文件。

```powershell
docker login --username moguizhiz1313
if ($LASTEXITCODE -ne 0) { throw "Docker Hub 登录失败。" }
docker push moguizhiz1313/fastllm-cu126:cu126-nccl2283
if ($LASTEXITCODE -ne 0) { throw "镜像推送失败。" }
```

确认推送成功后，Vast 模板的 Image Path:Tag 填
`moguizhiz1313/fastllm-cu126:cu126-nccl2283`，Launch Mode 选择 Jupyter + SSH，
配置私有仓库拉取凭据。新实例迁移代码和模型后重新验收，通过后再处理旧实例。
