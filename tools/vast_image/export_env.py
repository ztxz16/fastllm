#!/usr/bin/env python3
"""导出已验收的 Vast CUDA 12.6 环境，并在同基础镜像内恢复；不编译项目。"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys
import sysconfig
import tempfile
from urllib.parse import unquote, urlsplit

BASE_IMAGE = "vastai/base-image:cuda-13.0.3-auto"
VENV = Path("/venv/main")
NCCL_VERSION = "2.28.3-1+cuda12.9"
REQUIRED = {"cuda-nvcc-12-6", "cuda-cudart-dev-12-6", "libcublas-dev-12-6",
            "libnccl2", "libnccl-dev"}
EXTRA = {"aria2", "build-essential", "cmake", "ninja-build", "make", "gcc", "g++",
         "git", "libnuma-dev", "pkg-config", "ca-certificates"}
CUDA_PACKAGE = re.compile(r"(?:cuda-.+-12-6|cuda-toolkit(?:-12(?:-6)?)?-config-common|libcublas(?:-dev)?-12-6|libnccl(?:2|-dev))")


def run(command, **kwargs):
    """/** 执行无 shell 展开的命令；子进程失败时停止导出或恢复。 */"""
    return subprocess.run(command, check=True, text=True,
                          env={**os.environ, "LC_ALL": "C"}, **kwargs)


def capture(command):
    """/** 取得只读命令输出，保留失败状态而不将失败结果当成有效清单。 */"""
    return run(command, stdout=subprocess.PIPE).stdout


def sha256(path):
    """/** 分块计算文件 SHA256，避免大安装包或虚拟环境归档占满内存。 */"""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_uris(output):
    """/** 解析 APT 的下载计划；只接受安全文件名、HTTP URL 和 SHA256。 */"""
    records = {}
    for line in output.splitlines():
        if not line.startswith("'"):
            continue
        parts = shlex.split(line)
        if len(parts) != 4:
            raise ValueError("APT 下载计划字段不完整")
        url, name, size, checksum = parts
        if (urlsplit(url).scheme not in ("http", "https")
                or not re.fullmatch(r"[A-Za-z0-9_.+%~-]+\.deb", name)
                or not size.isdecimal()
                or not re.fullmatch(r"SHA256:[0-9a-fA-F]{64}", checksum)):
            raise ValueError("APT 下载计划包含不支持的 URL、文件名或校验算法")
        record = dict(url=url, name=name, size=int(size), sha256=checksum[7:].lower())
        if name in records and records[name] != record:
            raise ValueError("APT 下载计划存在冲突文件名")
        records[name] = record
    return list(records.values())


def matches(path, record):
    """/** 先比较大小，再校验内容；只有完整匹配的缓存文件才能复用。 */"""
    return (path.is_file() and path.stat().st_size == record["size"]
            and sha256(path) == record["sha256"])


def fetch_debs(specs, destination, caches, resolve=False):
    """/** 解析并取得指定版本的 Debian 安装包。
     * APT 仅生成下载计划；缓存必须通过 SHA256 校验，缺少的文件由 aria2
     * 断点续传。本函数不安装软件，任何解析、下载或校验失败均抛出异常。
     * @param specs 精确的 package=version 列表。
     * @param destination 保存安装包及断点信息的独立目录。
     * @param caches 可复用的只读缓存目录列表。
     * @param resolve true 时根据构建容器状态补齐依赖，false 仅导出指定包。
     * @return 已校验文件的 URL、名称、大小和 SHA256 记录。
     */"""
    destination.mkdir(parents=True, exist_ok=True)
    command = ["apt-get", "-o", "Acquire::ForceHash=sha256", "--print-uris"]
    if resolve:
        # 独立空缓存使 APT 列出完整计划；不能将全局缓存中的无关包一起安装。
        planning = destination / "apt-plan"
        planning.mkdir()
        command += ["-o", "Dir::Cache::archives=" + str(planning),
                    "--yes", "--download-only", "--no-install-recommends",
                    "--no-remove", "--allow-downgrades", "--allow-change-held-packages", "install"]
    else:
        command += ["download"]
    records = parse_uris(capture(command + specs))
    if not records and not resolve:
        raise RuntimeError("APT 未返回安装包地址；请更新软件源索引后重试")

    pending = []
    for record in records:
        target = destination / record["name"]
        if matches(target, record):
            continue
        aliases = {record["name"], unquote(urlsplit(record["url"]).path.rsplit("/", 1)[-1])}
        source = next((cache / alias for cache in caches for alias in aliases
                       if Path(alias).name == alias and matches(cache / alias, record)), None)
        if source is not None:
            shutil.copyfile(source, target)
            print("复用已校验安装包：" + target.name, flush=True)
        else:
            pending.append(record)

    if pending:
        if not shutil.which("aria2c"):
            raise RuntimeError("需要 aria2c 下载缺少的安装包，请先安装 aria2")
        listing = destination / "aria2-input.txt"
        listing.write_text("".join(
            f'{row["url"]}\n  out={row["name"]}\n  checksum=sha-256={row["sha256"]}\n'
            for row in pending), encoding="utf-8")
        run(["aria2c", "--no-conf=true", "-c", "-x", "16", "-s", "16", "-j", "2", "-k", "1M",
             "--file-allocation=none", "--check-integrity=true", "--auto-file-renaming=false",
             "--allow-overwrite=true", "--dir=" + str(destination), "--input-file=" + str(listing)])
        listing.unlink()
    for record in records:
        if not matches(destination / record["name"], record):
            raise RuntimeError("安装包校验失败：" + record["name"])
    return records


def python_state():
    """/** 记录解释器 ABI 和安装包版本，不导出环境变量或软件源凭据。 */"""
    packages = sorted((dist.metadata["Name"], dist.version)
                      for dist in importlib.metadata.distributions())
    return dict(version=list(sys.version_info[:3]), soabi=sysconfig.get_config_var("SOABI"),
                prefix=sys.prefix, base_executable=os.path.realpath(getattr(sys, "_base_executable", sys.executable)),
                packages=packages)


def selected_packages():
    """/** 锁定已安装的 CUDA 12.6、NCCL、系统 Python 和现有构建工具版本。 */"""
    output = capture(["dpkg-query", "-W", "-f=${Package}\t${Version}\t${Architecture}\t${Status}\n"])
    installed = {}
    for line in output.splitlines():
        name, version, architecture, status = line.split("\t", 3)
        if status.endswith(" ok installed") and architecture in ("amd64", "all"):
            installed[name] = version
    if not REQUIRED <= installed.keys():
        raise RuntimeError("尚未完整安装 CUDA 12.6 / NCCL 开发包")
    if any(installed[name] != NCCL_VERSION for name in ("libnccl2", "libnccl-dev")):
        raise RuntimeError("NCCL 尚未切换到已验收的 " + NCCL_VERSION)
    selected = {name: version for name, version in installed.items()
                if name in EXTRA or CUDA_PACKAGE.fullmatch(name)
                or re.fullmatch(r"(?:lib)?python3(?:\.12)?(?:-.*)?", name)}
    return selected, output


def export(args):
    """/** 在原 Vast 实例导出通过验收的 CUDA、NCCL 和 Python 环境。
     * 检查系统与环境边界、锁定版本、取得安装包，再打包原位置的 Conda/venv。
     * 不修改已安装软件或编译代码。输出独立构建目录；中断后保留已下载包，
     * 版本变化、外部 editable 依赖或归档期间状态变化时拒绝继续。
     * @param args 命令行参数，包含可选输出目录和安装包缓存目录。
     * @return 无返回值；成功时输出目录路径，失败通过异常交给主入口返回非零状态。
     */"""
    release = platform.freedesktop_os_release()
    if platform.machine() != "x86_64" or release.get("ID") != "ubuntu" or release.get("VERSION_ID") != "24.04":
        raise RuntimeError("此配方仅适用于 Ubuntu 24.04 x86_64 的已验收实例")
    is_environment = sys.prefix != sys.base_prefix or (VENV / "conda-meta").is_dir()
    if Path(sys.prefix) != VENV or not is_environment or VENV.is_symlink():
        raise RuntimeError("请用 /venv/main/bin/python3 执行；Conda/venv 环境必须实际位于 /venv/main")
    for dist in importlib.metadata.distributions():
        direct = json.loads(dist.read_text("direct_url.json") or "{}")
        if direct.get("dir_info", {}).get("editable"):
            raise RuntimeError("需先单独处理 editable 安装包：" + dist.metadata["Name"])
    if list(VENV.glob("lib/python*/site-packages/*.egg-link")):
        raise RuntimeError("虚拟环境含外部 egg-link，请先单独处理这些依赖")
    packages, system_listing = selected_packages()
    specs = [name + "=" + version for name, version in sorted(packages.items())]
    state = python_state()
    context = args.output.resolve() if args.output else Path(tempfile.mkdtemp(
        prefix="fastllm-cu126-image.", dir="/home/models"))
    context.mkdir(parents=True, exist_ok=True)
    lock = context / "packages.lock"
    locked = "\n".join(specs) + "\n"
    if any(context.iterdir()) and (not lock.is_file() or lock.read_text() != locked):
        raise RuntimeError("输出目录已有其他内容或版本已变化，请使用新目录")
    lock.write_text(locked)
    (context / "system-packages.txt").write_text(system_listing)
    print("导出目录（中断后可用 --output 此路径重试）：" + str(context), flush=True)

    # 相同基础镜像中的 Python/构建工具可直接复用，避免强制下载已下架的旧版本。
    # 这些版本仍在锁中，构建时只在确有差异或缺失时解析并补齐。
    deb_specs = [name + "=" + version for name, version in sorted(packages.items())
                 if CUDA_PACKAGE.fullmatch(name)]
    records = fetch_debs(deb_specs, context / "debs", args.cache_dir)
    archive = context / "python-env.tar.gz"
    print("打包 /venv/main（保留 Linux 权限、软链接和绝对安装位置）", flush=True)
    run(["tar", "-C", "/", "-czf", str(archive) + ".partial", "venv/main"])
    if state != python_state() or packages != selected_packages()[0]:
        raise RuntimeError("导出期间包版本发生变化，请停止安装操作后重新导出")
    Path(str(archive) + ".partial").replace(archive)
    source = Path(__file__).resolve().parent
    for name in ("Dockerfile", "export_env.py", "README.md"):
        shutil.copyfile(source / name, context / name)
    repository = source.parent.parent
    commit = capture(["git", "-C", str(repository), "rev-parse", "HEAD"]).strip()
    manifest = dict(base_image=BASE_IMAGE, git_commit=commit, python=state,
                    packages=packages, debs=records)
    (context / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    names = ["Dockerfile", "export_env.py", "README.md", "manifest.json", "packages.lock",
             "system-packages.txt", "python-env.tar.gz", ".dockerignore"]
    (context / ".dockerignore").write_text("**\n" + "".join("!" + name + "\n" for name in names)
                                          + "!SHA256SUMS\n!debs/\n!debs/*.deb\n")
    files = [context / name for name in names] + [context / "debs" / row["name"] for row in records]
    (context / "SHA256SUMS").write_text("".join(
        sha256(path) + "  " + path.relative_to(context).as_posix() + "\n" for path in sorted(files)))
    print("导出成功。将整个目录传到 Windows 后按 README 构建；尚未创建或推送镜像。", flush=True)


def restore(context):
    """/** 在 Docker 构建容器中恢复锁定的系统包和 Python 环境。
     * 校验导出文件后，解析本容器缺少的依赖，通过本地 dpkg 安装并锁定版本。
     * 然后替换构建容器自带的 /venv/main，并用恢复的解释器验证版本与 ABI。
     * 必须设置专用构建标志；失败立即抛出异常，不放宽依赖或版本约束。
     * @param context export 生成并只读挂载到构建容器的目录。
     * @return 无返回值；成功表示环境恢复及版本检查完成，不代表 GPU 验收通过。
     */"""
    if os.environ.get("FASTLLM_IMAGE_BUILD") != "1":
        raise RuntimeError("restore 只能由配套 Dockerfile 在构建容器内调用")
    run(["sha256sum", "-c", "SHA256SUMS"], cwd=context)
    manifest = json.loads((context / "manifest.json").read_text())
    specs = (context / "packages.lock").read_text().splitlines()
    expected = [name + "=" + version for name, version in sorted(manifest["packages"].items())]
    if specs != expected or manifest["base_image"] != BASE_IMAGE:
        raise RuntimeError("版本锁与环境清单不一致")

    with tempfile.TemporaryDirectory(prefix="fastllm-local-debs.") as temporary:
        destination = Path(temporary)
        fetch_debs(specs, destination, [context / "debs", Path("/var/cache/apt/archives")], resolve=True)
        # 独立校验导出的 CUDA/NCCL 包，保证导出的每个包都进入安装集合。
        for record in manifest["debs"]:
            source = context / "debs" / record["name"]
            if not matches(source, record):
                raise RuntimeError("导出安装包损坏：" + record["name"])
            target = destination / record["name"]
            if not target.exists():
                shutil.copyfile(source, target)
        run(["dpkg", "--unpack"] + [str(path) for path in sorted(destination.glob("*.deb"))])
        run(["dpkg", "--configure", "--pending"])
    run(["apt-get", "check"])
    run(["ldconfig"])
    run(["apt-mark", "hold"] + sorted(manifest["packages"]))

    # 同一基础镜像、同一路径恢复；先移除镜像自带 venv，避免遗留不同版本文件。
    if VENV.is_symlink():
        VENV.unlink()
    elif VENV.exists():
        shutil.rmtree(VENV)
    run(["tar", "-C", "/", "-xzf", str(context / "python-env.tar.gz")])
    run([str(VENV / "bin/python3"), str(context / "export_env.py"), "verify", "--context", str(context)])


def verify(context):
    """/** 检查当前 Python ABI、安装包及系统包是否符合导出清单。
     * 不启动 GPU 或编译器，也不加载第三方 Python 模块。版本一致性检查
     * 不能代替新实例上的 GPU 功能验收；发现差异立即报错。
     * @param context 包含 manifest.json 的导出目录或镜像内元数据目录。
     * @return 无返回值；不一致通过异常报告，成功时打印明确的检查范围。
     */"""
    manifest = json.loads((context / "manifest.json").read_text())
    current = json.loads(json.dumps(python_state()))
    if current != manifest["python"]:
        fields = [key for key in current if current[key] != manifest["python"].get(key)]
        raise RuntimeError("Python 环境与导出时不一致：" + ", ".join(fields))
    installed, _ = selected_packages()
    if any(installed.get(name) != version for name, version in manifest["packages"].items()):
        raise RuntimeError("恢复后的系统包版本不符合锁定清单")
    print("环境版本检查通过；GPU 功能仍需在新 Vast 实例验收。", flush=True)


def main():
    """/** 分离原实例导出与镜像内恢复入口；发生错误返回非零状态。 */"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("export", "restore", "verify"))
    parser.add_argument("--output", type=Path, help="导出目录；省略时在 /home/models 下新建")
    parser.add_argument("--context", type=Path, help="restore/verify 使用的构建目录")
    parser.add_argument("--cache-dir", type=Path, action="append",
                        default=[Path("/tmp/nccl-cu12-download"), Path("/var/cache/apt/archives")])
    args = parser.parse_args()
    if args.action != "export" and args.context is None:
        parser.error("restore/verify 需要 --context")
    if args.action == "export":
        export(args)
    elif args.action == "restore":
        restore(args.context.resolve())
    else:
        verify(args.context.resolve())


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        sys.exit("环境迁移失败：" + str(error))
