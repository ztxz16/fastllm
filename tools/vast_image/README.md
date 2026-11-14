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
导出阶段预下载 CUDA/NCCL 包；Python 和构建工具的系统包只记录版本锁，
由同一基础镜像复用，构建时仅补齐有差异的包，避免无故重新下载旧版系统包。
Python 环境归档可能较大，导出目录需有足够剩余空间。

## 传输：Hugging Face Bucket + aria2

已经通过 Vast Cloud Copy 将导出目录备份到 Hugging Face Bucket 时，可在 Windows
直接下载。上传的是镜像构建目录；完成下方 Docker 构建及推送后才有可供 Vast 拉取的镜像。

Windows 先准备 `aria2c`（尚未安装时用 `winget install --id aria2.aria2 --exact --source winget`
引导安装并重新打开 PowerShell）。取得本仓库 `tools/vast_image/download_hf.ps1` 后执行：

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File .\download_hf.ps1 -Bucket devilson/fastllm-backup -Prefix fastllm-cu126-image -Destination C:\fastllm-cu126-image
if ($LASTEXITCODE -ne 0) { throw "备份下载或校验失败。" }
```

提示时输入有该桶读取权限的 Hugging Face Token（`hf_` 开头，输入不回显）。不是
Vast API Key、Docker Hub Token 或 S3 Access Key。Token 只用于向 Hugging Face
请求下载地址，不写入脚本、命令行或磁盘，也不发送到 CDN。aria2 接收短期签名地址，
分段下载并校验 `SHA256SUMS` 中的每个文件；完整缓存直接复用，中断后重跑即可。
只有在全部文件通过校验后，才继续下方 Docker 构建。无需解压 `python-env.tar.gz`。

脚本要求桶内目录直接包含导出时生成的 `SHA256SUMS`、`Dockerfile`、`manifest.json`
等文件及 `debs/`。如果备份时多嵌套了一层目录，须相应调整 `-Prefix`。HF 的 Xet
内容哈希不能当作文件 SHA256；本流程始终使用导出时生成的 SHA256 清单。

## 传输：Direct SSH + aria2

大文件优先通过 Vast 的 Direct SSH 建立通道，再由 Windows aria2 下载。
代理 SSH 可能限制速度；aria2 的多个 HTTP 请求仍共用一条 SSH 连接，不能保证
比已有直连 SFTP 更快。下载工具只依赖 Python 标准库，仅监听回环地址，支持
HTTP Range 和 SHA256 校验，不需要新增公网端口，也不提供目录浏览。

原 Vast 服务器执行（替换第一行的导出目录）：

```bash
(
set -euo pipefail
image_export=/home/models/fastllm-cu126-image.XXXXXXXX
image_archive="${image_export}.tar"
# 导出目录保持不变；已打包完成则复用，避免中断下载后重新打包。
if [ ! -s "$image_archive" ]; then
  tar -cf "${image_archive}.part" -C "$image_export" .
  mv "${image_archive}.part" "$image_archive"
fi
cd /home/project/vllm_fastllm
python3 tools/vast_image/serve_archive.py "$image_archive"
)
```

看到“下载服务已启动”后，保留这个终端。归档会占用约一份导出目录的额外空间。
如果导出内容发生变化，需另建归档和下载文件名，不能沿用旧的断点文件。

Windows 第一个 PowerShell 窗口执行，用实例的 Direct SSH IP 和端口替换占位值：

```powershell
ssh -p <直连端口> -N -T -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 -o ServerAliveCountMax=3 -L 127.0.0.1:18765:127.0.0.1:18765 root@<公网IP>
```

如果 MobaXterm 使用指定私钥，在 `ssh` 后加 `-i "私钥完整路径"`；OpenSSH
需要它支持的私钥格式，不能直接使用 PuTTY 的 `.ppk` 文件。连接后没有输出是
正常的，保留窗口。首次连接需核对服务器主机密钥。此处复用已有 SSH 身份认证。

Windows 第二个 PowerShell 窗口执行：

```powershell
$ErrorActionPreference = "Stop"
$imageDownload = Join-Path $env:USERPROFILE "Downloads\fastllm-image-transfer"
New-Item -ItemType Directory -Force $imageDownload | Out-Null
$imageMeta = Invoke-RestMethod "http://127.0.0.1:18765/metadata"
aria2c -c -x 16 -s 16 -k 1M --file-allocation=none --auto-file-renaming=false --no-proxy=127.0.0.1 --check-integrity=true "--checksum=sha-256=$($imageMeta.sha256)" -d "$imageDownload" -o fastllm-cu126-image.tar "http://127.0.0.1:18765/archive.tar"
if ($LASTEXITCODE -ne 0) { throw "下载未完成；保留归档和 .aria2 文件，修复连接后重跑。" }
New-Item -ItemType Directory -Force C:\fastllm-cu126-image | Out-Null
tar -xf (Join-Path $imageDownload "fastllm-cu126-image.tar") -C C:\fastllm-cu126-image
if ($LASTEXITCODE -ne 0) { throw "归档解包失败。" }
```

Windows 尚无 aria2 时，先用 `winget install --id aria2.aria2 --exact --source winget`
引导安装，再重新打开 PowerShell。仅此次引导安装使用其他下载方式。
完成后可在服务器下载服务和 SSH 通道窗口按 Ctrl+C 停止。只解包外层 `.tar`；
目录内的 `python-env.tar.gz` 保留原样，Linux 权限和软链接由 Docker 构建恢复。

## Windows：构建

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
