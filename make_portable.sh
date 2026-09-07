#!/usr/bin/env bash

set -Eeuo pipefail

# Keep build paths out of bytecode and leave the relocatable runtime immutable.
export PYTHONDONTWRITEBYTECODE=1

ROOT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
PORTABLE_ASSETS_DIR="${ROOT_DIR}/portable"
DEFAULT_DIST_DIR="${ROOT_DIR}/build-fastllm/tools/dist"

# Keep this interpreter release fixed so the same source tree produces the same
# Python runtime. The GNU build has a GLIBC 2.17 baseline and is relocatable.
PYTHON_VERSION="3.11.15"
PYTHON_BUILD="20260718"
PYTHON_ARCHIVE="cpython-${PYTHON_VERSION}+${PYTHON_BUILD}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/${PYTHON_BUILD}/cpython-${PYTHON_VERSION}%2B${PYTHON_BUILD}-x86_64-unknown-linux-gnu-install_only_stripped.tar.gz"
PYTHON_SHA256="23ccae6f1ff73e8aa8378436f869da003b8eb7d6c95f2bc706f494115ba1447d"

TARGET_GLIBC="2.35"
MIN_NVIDIA_DRIVER="525.60.13"
RECOMMENDED_NVIDIA_DRIVER="575.57.08"

wheel_path=""
output_dir="${DEFAULT_DIST_DIR}"
output_dir_explicit=0
constraints_path="${PORTABLE_ASSETS_DIR}/constraints.txt"
use_constraints=1
archive_format="tar.gz"
keep_dir=0
directory_only=0
force=0
run_tests=1
offline=0

usage() {
    cat <<'EOF'
用法：
  ./make_portable.sh [wheel]
  ./make_portable.sh --wheel PATH [选项]

把 ftllm wheel、Python 3.11、CUDA 12 runtime、cuBLAS、NCCL 和全部 Python
依赖以及 Pi Agent、ripgrep、fd 打成解压即用的 x86_64 Linux 绿色包。

选项：
  --wheel PATH           输入 wheel；默认选择 build-fastllm/tools/dist 中最新的 ftllm wheel
  --output-dir DIR       输出目录；默认 build-fastllm/tools/dist；不可写时回退到 portable-dist
  --constraints PATH     指定 pip constraints 文件
  --no-constraints       不使用仓库内的固定依赖版本
  --format tar.gz|tar.zst
                         压缩格式；默认 tar.gz（目标机无需安装 zstd）
  --keep-dir             除压缩包外，同时保留未压缩目录
  --directory-only       只生成未压缩目录，便于被其他打包工具复用
  --skip-tests           跳过启动器、Pi Agent、原生库和 CUDA 冒烟测试
  --offline              不访问网络，仅使用已缓存的 Python、wheelhouse、Pi 和搜索工具
  --force                覆盖同名输出
  -h, --help             显示帮助

环境变量：
  FTLLM_PORTABLE_CACHE_DIR  下载缓存目录
  SOURCE_DATE_EPOCH         固定构建时间戳，以获得可复现的归档元数据
EOF
}

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

log() {
    printf '[ftllm-portable] %s\n' "$*"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "缺少构建命令：$1"
}

absolute_path() {
    local input_path="$1"
    local input_dir
    local input_name
    input_dir="$(dirname -- "$input_path")"
    input_name="$(basename -- "$input_path")"
    printf '%s/%s\n' "$(CDPATH= cd -- "$input_dir" && pwd -P)" "$input_name"
}

select_default_wheel() {
    local candidate
    local selected=""
    shopt -s nullglob
    for candidate in "${DEFAULT_DIST_DIR}"/ftllm-*.whl; do
        if [[ -z "$selected" || "$candidate" -nt "$selected" ]]; then
            selected="$candidate"
        fi
    done
    shopt -u nullglob
    [[ -n "$selected" ]] || die "${DEFAULT_DIST_DIR} 中没有 ftllm wheel，请先运行 make_whl.sh"
    printf '%s\n' "$selected"
}

download_python() {
    local destination="$1"
    local partial="${destination}.part"

    if [[ -f "$destination" ]]; then
        if printf '%s  %s\n' "$PYTHON_SHA256" "$destination" | sha256sum --check --status; then
            log "复用 Python 缓存：$(basename -- "$destination")"
            return
        fi
        log "Python 缓存校验失败，重新下载"
        rm -f -- "$destination"
    fi

    ((offline == 0)) \
        || die "--offline 模式缺少有效的 Python 缓存：$destination"

    rm -f -- "$partial"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --retry-delay 2 --output "$partial" "$PYTHON_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="$partial" "$PYTHON_URL"
    else
        die "需要 curl 或 wget 下载可重定位 Python"
    fi
    printf '%s  %s\n' "$PYTHON_SHA256" "$partial" | sha256sum --check --status \
        || die "Python 归档 SHA256 校验失败"
    mv -- "$partial" "$destination"
}

version_is_greater() {
    local left="$1"
    local right="$2"
    local greatest
    greatest="$(printf '%s\n%s\n' "$left" "$right" | LC_ALL=C sort -V | tail -n 1)"
    [[ "$greatest" == "$left" && "$left" != "$right" ]]
}

patch_python_shebangs() {
    local runtime_dir="$1"

    "${runtime_dir}/bin/python3" - "${runtime_dir}/bin" <<'PY'
import sys
from pathlib import Path

bin_dir = Path(sys.argv[1])
relative_exec = b'''\'\'\'exec' "$(dirname -- "$(realpath -- "$0")")/python3" "$0" "$@"\n'''
polyglot_header = b"#!/bin/sh\n" + relative_exec + b"' '''\n"

for path in bin_dir.iterdir():
    if not path.is_file():
        continue
    try:
        content = path.read_bytes()
    except OSError:
        continue
    lines = content.splitlines(keepends=True)
    if not lines:
        continue

    # A short pip build path produces a normal Python shebang. Convert it to a
    # shell/Python polyglot so the sibling interpreter is found after moving.
    if lines[0].startswith(b"#!") and b"python" in lines[0].lower():
        path.write_bytes(polyglot_header + b"".join(lines[1:]))
        continue

    # A long pip build path already produces a polyglot launcher, but embeds
    # the absolute temporary interpreter path on its second line.
    if (
        lines[0].rstrip(b"\r\n") == b"#!/bin/sh"
        and len(lines) >= 3
        and lines[1].startswith(b"'''exec'")
        and b"python" in lines[1].lower()
    ):
        lines[1] = relative_exec
        path.write_bytes(b"".join(lines))
PY
}

audit_elf_glibc() {
    local bundle_dir="$1"
    local raw_report="$2"
    local final_report="${bundle_dir}/ELF-GLIBC-REQUIREMENTS.txt"
    local elf_file
    local required
    local max_required="0"

    : > "$raw_report"
    while IFS= read -r -d '' elf_file; do
        if ! readelf -h "$elf_file" >/dev/null 2>&1; then
            continue
        fi
        required="$(
            readelf --version-info "$elf_file" 2>/dev/null \
                | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
                | LC_ALL=C sort -Vu \
                | tail -n 1 \
                || true
        )"
        if [[ -n "$required" ]]; then
            printf '%s  %s\n' "$required" "${elf_file#"${bundle_dir}/"}" >> "$raw_report"
        fi
    done < <(find "$bundle_dir" -type f -print0)

    if [[ -s "$raw_report" ]]; then
        max_required="$(cut -d' ' -f1 "$raw_report" | sed 's/^GLIBC_//' | LC_ALL=C sort -Vu | tail -n 1)"
    fi

    {
        printf 'Target maximum: GLIBC_%s (Ubuntu 22.04 baseline)\n' "$TARGET_GLIBC"
        printf 'Bundle maximum: GLIBC_%s\n\n' "$max_required"
        LC_ALL=C sort -V "$raw_report"
    } > "$final_report"

    if version_is_greater "$max_required" "$TARGET_GLIBC"; then
        grep "^GLIBC_${max_required} " "$raw_report" >&2 || true
        die "包内 ELF 需要 GLIBC_${max_required}，高于 Ubuntu 22.04 的 GLIBC_${TARGET_GLIBC}"
    fi
    log "ELF 兼容审计通过：最高 GLIBC_${max_required}（上限 GLIBC_${TARGET_GLIBC}）"
}

write_build_info() {
    local python_bin="$1"
    local destination="$2"
    local package_name="$3"
    local package_version="$4"
    local wheel_name="$5"
    local wheel_sha="$6"
    local constraints_sha="$7"
    local build_time="$8"

    "$python_bin" - "$destination" "$package_name" "$package_version" \
        "$wheel_name" "$wheel_sha" "$constraints_sha" "$build_time" \
        "$PYTHON_VERSION" "$PYTHON_BUILD" "$PYTHON_SHA256" "$TARGET_GLIBC" \
        "$MIN_NVIDIA_DRIVER" "$RECOMMENDED_NVIDIA_DRIVER" "$agent_wheel_path" <<'PY'
import json
import importlib.metadata
import platform
import sys
from pathlib import Path
import hashlib
from ftllm_agent_runtime.runtime import PI_VERSION

(
    destination,
    package_name,
    package_version,
    wheel_name,
    wheel_sha256,
    constraints_sha256,
    build_time,
    python_version,
    python_build,
    python_sha256,
    target_glibc,
    min_driver,
    recommended_driver,
    agent_wheel,
) = sys.argv[1:]

data = {
    "format_version": 1,
    "package": {"name": package_name, "version": package_version},
    "wheel": {"filename": wheel_name, "sha256": wheel_sha256},
    "agent_runtime": {
        "name": "ftllm-agent-runtime",
        "version": importlib.metadata.version("ftllm-agent-runtime"),
        "pi_version": PI_VERSION,
        "wheel": {
            "filename": Path(agent_wheel).name,
            "sha256": hashlib.sha256(Path(agent_wheel).read_bytes()).hexdigest(),
        },
    },
    "python": {
        "version": python_version,
        "standalone_build": python_build,
        "archive_sha256": python_sha256,
    },
    "platform": {
        "machine": platform.machine(),
        "target_glibc_max": target_glibc,
        "ubuntu_targets": ["22.04", "24.04", "26.04"],
    },
    "cuda": {
        "major": 12,
        "minimum_compute_capability": "6.0",
        "embedded_native_targets": ["60", "70", "75", "80", "86", "89", "90", "100", "120"],
        "minimum_linux_driver": min_driver,
        "recommended_linux_driver": recommended_driver,
    },
    "constraints_sha256": constraints_sha256 or None,
    "built_at_utc": build_time,
}

with open(destination, "w", encoding="utf-8") as output:
    json.dump(data, output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")
PY
}

create_manifest() {
    local bundle_dir="$1"
    local manifest="${bundle_dir}/MANIFEST.sha256"
    local relative_file

    (
        cd "$bundle_dir"
        : > MANIFEST.sha256
        while IFS= read -r -d '' relative_file; do
            sha256sum "$relative_file" >> MANIFEST.sha256
        done < <(
            find . -type f ! -name MANIFEST.sha256 -print0 | LC_ALL=C sort -z
        )
    )
}

create_archive() {
    local bundle_dir="$1"
    local archive_path="$2"
    local epoch="$3"
    local parent_dir
    local bundle_name
    parent_dir="$(dirname -- "$bundle_dir")"
    bundle_name="$(basename -- "$bundle_dir")"

    case "$archive_format" in
        tar.gz)
            if command -v pigz >/dev/null 2>&1; then
                tar --sort=name --mtime="@${epoch}" --owner=0 --group=0 --numeric-owner \
                    -C "$parent_dir" -cf - "$bundle_name" | pigz -6n > "$archive_path"
            else
                tar --sort=name --mtime="@${epoch}" --owner=0 --group=0 --numeric-owner \
                    -C "$parent_dir" -cf - "$bundle_name" | gzip -6n > "$archive_path"
            fi
            ;;
        tar.zst)
            require_command zstd
            tar --sort=name --mtime="@${epoch}" --owner=0 --group=0 --numeric-owner \
                -C "$parent_dir" -cf - "$bundle_name" | zstd -T0 -10 -q -o "$archive_path"
            ;;
        *)
            die "未知压缩格式：$archive_format"
            ;;
    esac
}

while (($#)); do
    case "$1" in
        --wheel)
            (($# >= 2)) || die "--wheel 缺少参数"
            wheel_path="$2"
            shift 2
            ;;
        --output-dir)
            (($# >= 2)) || die "--output-dir 缺少参数"
            output_dir="$2"
            output_dir_explicit=1
            shift 2
            ;;
        --constraints)
            (($# >= 2)) || die "--constraints 缺少参数"
            constraints_path="$2"
            use_constraints=1
            shift 2
            ;;
        --no-constraints)
            use_constraints=0
            shift
            ;;
        --format)
            (($# >= 2)) || die "--format 缺少参数"
            archive_format="$2"
            shift 2
            ;;
        --keep-dir)
            keep_dir=1
            shift
            ;;
        --directory-only)
            directory_only=1
            keep_dir=1
            shift
            ;;
        --skip-tests)
            run_tests=0
            shift
            ;;
        --offline)
            offline=1
            shift
            ;;
        --force)
            force=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        --)
            shift
            break
            ;;
        -*)
            die "未知选项：$1"
            ;;
        *)
            [[ -z "$wheel_path" ]] || die "只能指定一个 wheel"
            wheel_path="$1"
            shift
            ;;
    esac
done

(($# == 0)) || die "多余参数：$*"
[[ "$(uname -s)" == "Linux" ]] || die "绿色包构建目前只支持 Linux"
[[ "$(uname -m)" == "x86_64" ]] || die "绿色包构建目前只支持 x86_64"
[[ "$archive_format" == "tar.gz" || "$archive_format" == "tar.zst" ]] \
    || die "--format 只能是 tar.gz 或 tar.zst"

require_command tar
require_command sha256sum
require_command readelf
require_command sed
require_command grep
require_command sort
require_command find
require_command install
require_command date
require_command cut
require_command du
if ((! directory_only)); then
    if [[ "$archive_format" == "tar.gz" ]]; then
        require_command gzip
    else
        require_command zstd
    fi
fi

if [[ -z "$wheel_path" ]]; then
    wheel_path="$(select_default_wheel)"
fi
[[ -f "$wheel_path" ]] || die "wheel 不存在：$wheel_path"
wheel_path="$(absolute_path "$wheel_path")"

if (( ! output_dir_explicit )); then
    if ! mkdir -p "$output_dir" 2>/dev/null || [[ ! -w "$output_dir" ]]; then
        output_dir="${ROOT_DIR}/portable-dist"
        log "默认 dist 目录不可写，输出改为：${output_dir}"
    fi
fi
mkdir -p "$output_dir" 2>/dev/null || die "无法创建输出目录：$output_dir"
[[ -w "$output_dir" ]] || die "输出目录不可写：$output_dir"
output_dir="$(CDPATH= cd -- "$output_dir" && pwd -P)"

if ((use_constraints)); then
    [[ -f "$constraints_path" ]] || die "constraints 不存在：$constraints_path"
    constraints_path="$(absolute_path "$constraints_path")"
fi

cache_dir="${FTLLM_PORTABLE_CACHE_DIR:-${ROOT_DIR}/build-portable-cache}"
mkdir -p "$cache_dir"
cache_dir="$(CDPATH= cd -- "$cache_dir" && pwd -P)"
python_archive_path="${cache_dir}/${PYTHON_ARCHIVE}"
download_python "$python_archive_path"

build_root="$(mktemp -d "${output_dir}/.ftllm-portable-build.XXXXXXXX")"
cleanup() {
    if [[ -n "${build_root:-}" && -d "$build_root" && "$build_root" == "${output_dir}/.ftllm-portable-build."* ]]; then
        rm -r -- "$build_root"
    fi
}
trap cleanup EXIT INT TERM

runtime_staging="${build_root}/runtime"
mkdir -p "$runtime_staging"
tar -xzf "$python_archive_path" --strip-components=1 -C "$runtime_staging"
python_bin="${runtime_staging}/bin/python3"
[[ -x "$python_bin" ]] || die "Python 归档结构异常：缺少 bin/python3"

IFS=$'\t' read -r package_name package_version wheel_tag < <(
    "$python_bin" - "$wheel_path" <<'PY'
import email
import sys
import zipfile

wheel_path = sys.argv[1]
with zipfile.ZipFile(wheel_path) as wheel:
    metadata_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
    wheel_name = next(name for name in wheel.namelist() if name.endswith(".dist-info/WHEEL"))
    metadata = email.message_from_bytes(wheel.read(metadata_name))
    wheel_metadata = email.message_from_bytes(wheel.read(wheel_name))

print(
    metadata.get("Name", ""),
    metadata.get("Version", ""),
    wheel_metadata.get("Tag", ""),
    sep="\t",
)
PY
)
[[ -n "$package_name" && -n "$package_version" ]] || die "无法读取 wheel 元数据"
case "${package_name,,}" in
    ftllm|ftllm-nightly) ;;
    *) die "输入不是 ftllm wheel：${package_name}" ;;
esac
[[ "$package_version" =~ ^[A-Za-z0-9][A-Za-z0-9._+!-]*$ ]] \
    || die "wheel 版本字符串不安全：${package_version}"
[[ "$wheel_tag" == *manylinux*"x86_64"* ]] || die "wheel 不是 manylinux x86_64：${wheel_tag}"

package_slug="${package_name,,}"
package_slug="${package_slug//_/-}"
bundle_name="${package_slug}-${package_version}-linux-x86_64-portable-cu12"
archive_suffix="$archive_format"
archive_path="${output_dir}/${bundle_name}.${archive_suffix}"
checksum_path="${archive_path}.sha256"
final_dir="${output_dir}/${bundle_name}"

if ((! force)); then
    if ((! directory_only)); then
        for existing_path in "$archive_path" "$checksum_path"; do
            [[ ! -e "$existing_path" ]] \
                || die "输出已存在：$existing_path（使用 --force 覆盖）"
        done
    fi
    if ((keep_dir)) && [[ -e "$final_dir" ]]; then
        die "输出目录已存在：$final_dir（使用 --force 覆盖）"
    fi
fi

bundle_dir="${build_root}/${bundle_name}"
mkdir -p "$bundle_dir"
mv -- "$runtime_staging" "${bundle_dir}/runtime"
runtime_dir="${bundle_dir}/runtime"
python_bin="${runtime_dir}/bin/python3"

wheelhouse_dir="${cache_dir}/wheelhouse-cp311-linux-x86_64"
mkdir -p "$wheelhouse_dir"
pip_constraint_args=()
if ((use_constraints)); then
    pip_constraint_args+=(--constraint "$constraints_path")
fi

# Resolve for the oldest supported host even when this script runs on Ubuntu
# 24.04/26.04. Multiple platform tags let pip choose older compatible wheels
# (for example manylinux_2_28) while refusing binaries newer than GLIBC 2.35.
download_target_args=()
for glibc_minor in {35..17}; do
    download_target_args+=(--platform "manylinux_2_${glibc_minor}_x86_64")
done
download_target_args+=(
    --platform manylinux2014_x86_64
    --platform manylinux2010_x86_64
    --platform manylinux1_x86_64
    --python-version 3.11
    --implementation cp
    --abi cp311
    --abi abi3
    --abi none
)

if ((! offline)); then
    log "下载缺失依赖到可复用 wheelhouse"
    PIP_CACHE_DIR="${cache_dir}/pip" "$python_bin" -m pip download \
        --disable-pip-version-check \
        --no-input \
        --only-binary=:all: \
        --dest "$wheelhouse_dir" \
        "${download_target_args[@]}" \
        "${pip_constraint_args[@]}" \
        "$wheel_path"
fi

log "从本地 wheelhouse 安装 ${package_name} ${package_version} 及全部依赖"
pip_install_args=(
    install
    --disable-pip-version-check
    --no-input
    --no-compile
    --no-index
    --find-links "$wheelhouse_dir"
)
pip_install_args+=("${pip_constraint_args[@]}")
"$python_bin" -m pip "${pip_install_args[@]}" "$wheel_path"

# Build the companion wheel from this checkout, in a disposable environment.
# Do not reuse a version-only wheel cache: bridge/extension code may have changed.
log "构建内置 Pi Agent 运行时"
agent_source="${build_root}/agent-source"
"$python_bin" - "${ROOT_DIR}/tools/ftllm_agent_runtime" "$agent_source" <<'PY'
import shutil
import sys

shutil.copytree(sys.argv[1], sys.argv[2], ignore=shutil.ignore_patterns(
    "build", "dist", "*.egg-info", "__pycache__", "*.pyc", "bin",
))
PY
agent_fetch_args=(--cache-dir "${cache_dir}/pi")
((offline == 0)) || agent_fetch_args+=(--offline)
"$python_bin" "${agent_source}/scripts/fetch_pi.py" "${agent_fetch_args[@]}"
agent_build_requirements=(setuptools==80.9.0 wheel==0.45.1)
if ((! offline)); then
    "$python_bin" -m pip download --disable-pip-version-check --no-input \
        --only-binary=:all: --dest "$wheelhouse_dir" "${agent_build_requirements[@]}"
fi
agent_build_env="${build_root}/agent-build-env"
"$python_bin" -m venv --without-pip "$agent_build_env"
"$python_bin" -m pip --python "$agent_build_env" install \
    --disable-pip-version-check --no-input --no-index --find-links "$wheelhouse_dir" \
    "${agent_build_requirements[@]}"
"$python_bin" -m pip --python "$agent_build_env" wheel \
    --disable-pip-version-check --no-input --no-index --no-deps --no-build-isolation \
    --wheel-dir "${build_root}/agent-wheel" "$agent_source"
agent_wheels=("${build_root}"/agent-wheel/ftllm_agent_runtime-*.whl)
[[ ${#agent_wheels[@]} == 1 && -f "${agent_wheels[0]}" ]] \
    || die "没有生成唯一的 ftllm-agent-runtime wheel"
agent_wheel_path="${agent_wheels[0]}"
"$python_bin" -m pip install --disable-pip-version-check --no-input \
    --no-index --no-deps --no-compile "$agent_wheel_path"
agent_tools_args=(--runtime-dir "$runtime_dir" --cache-dir "${cache_dir}/agent-tools")
((offline == 0)) || agent_tools_args+=(--offline)
"$python_bin" "${PORTABLE_ASSETS_DIR}/fetch_agent_tools.py" "${agent_tools_args[@]}"

mkdir -p "${bundle_dir}/libexec"
install -m 0755 "${PORTABLE_ASSETS_DIR}/launcher.sh" "${bundle_dir}/ftllm"
install -m 0755 "${PORTABLE_ASSETS_DIR}/launch.sh" "${bundle_dir}/launch.sh"
install -m 0755 "${PORTABLE_ASSETS_DIR}/launcher.sh" "${bundle_dir}/ftllm-agent-runtime"
install -m 0755 "${PORTABLE_ASSETS_DIR}/launcher.sh" "${bundle_dir}/python"
install -m 0755 "${PORTABLE_ASSETS_DIR}/launcher.sh" "${bundle_dir}/pip"
install -m 0755 "${PORTABLE_ASSETS_DIR}/launcher.sh" "${bundle_dir}/ftllm-check"
install -m 0644 "${PORTABLE_ASSETS_DIR}/activate.sh" "${bundle_dir}/libexec/activate.sh"
install -m 0755 "${PORTABLE_ASSETS_DIR}/check.py" "${bundle_dir}/libexec/check.py"
install -m 0644 "${PORTABLE_ASSETS_DIR}/env.sh" "${bundle_dir}/env.sh"
install -m 0644 "${PORTABLE_ASSETS_DIR}/README.md.in" "${bundle_dir}/README.md"
install -m 0644 "${PORTABLE_ASSETS_DIR}/THIRD-PARTY-NOTICES.md" "${bundle_dir}/THIRD-PARTY-NOTICES.md"

sed -i \
    -e "s/@PACKAGE_NAME@/${package_name}/g" \
    -e "s/@PACKAGE_VERSION@/${package_version}/g" \
    -e "s/@PYTHON_VERSION@/${PYTHON_VERSION}/g" \
    -e "s/@MIN_NVIDIA_DRIVER@/${MIN_NVIDIA_DRIVER}/g" \
    -e "s/@RECOMMENDED_NVIDIA_DRIVER@/${RECOMMENDED_NVIDIA_DRIVER}/g" \
    "${bundle_dir}/README.md"

# Console scripts made by pip contain the temporary build path in their shebang
# or polyglot shell header. Point every one at its sibling bundled interpreter.
patch_python_shebangs "$runtime_dir"

"$python_bin" -m pip --disable-pip-version-check list --format=freeze \
    | LC_ALL=C sort -f > "${bundle_dir}/requirements.lock.txt"
if ((use_constraints)); then
    install -m 0644 "$constraints_path" "${bundle_dir}/build-constraints.txt"
    constraints_sha="$(sha256sum "$constraints_path" | cut -d' ' -f1)"
else
    constraints_sha=""
fi

wheel_sha="$(sha256sum "$wheel_path" | cut -d' ' -f1)"
build_epoch="${SOURCE_DATE_EPOCH:-$(date +%s)}"
[[ "$build_epoch" =~ ^[0-9]+$ ]] || die "SOURCE_DATE_EPOCH 必须是非负整数"
build_time="$(date -u -d "@${build_epoch}" '+%Y-%m-%dT%H:%M:%SZ')"
write_build_info "$python_bin" "${bundle_dir}/BUILD-INFO.json" \
    "$package_name" "$package_version" "$(basename -- "$wheel_path")" "$wheel_sha" \
    "$constraints_sha" "$build_time"

audit_elf_glibc "$bundle_dir" "${build_root}/elf-glibc.raw"

# Moving the finished tree to a path containing a space catches absolute shebang
# and relocation bugs without duplicating the roughly 2 GB environment.
relocation_parent="${build_root}/relocation test"
mkdir -p "$relocation_parent"
mv -- "$bundle_dir" "${relocation_parent}/${bundle_name}"
bundle_dir="${relocation_parent}/${bundle_name}"
runtime_dir="${bundle_dir}/runtime"

if ((run_tests)); then
    log "运行可重定位启动器和原生库冒烟测试"
    "${bundle_dir}/ftllm" --version
    "${bundle_dir}/ftllm" --help >/dev/null
    "${bundle_dir}/ftllm" server --help >/dev/null
    "${bundle_dir}/ftllm" tui --help >/dev/null
    "${bundle_dir}/launch.sh" --help >/dev/null
    "${bundle_dir}/python" "${PORTABLE_ASSETS_DIR}/smoke_launcher.py" "$bundle_dir"
    "${bundle_dir}/python" "${PORTABLE_ASSETS_DIR}/smoke_launcher.py" "$bundle_dir" --entrypoint launch.sh
    "${bundle_dir}/ftllm-agent-runtime" info
    "${bundle_dir}/python" "${PORTABLE_ASSETS_DIR}/smoke_agent.py"
    embedded_build_paths="$(grep -RIlF "$build_root" "${bundle_dir}/runtime/bin" || true)"
    [[ -z "$embedded_build_paths" ]] \
        || die "console script 仍包含临时构建路径：${embedded_build_paths}"
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
        "${bundle_dir}/ftllm-check" --require-cuda
    else
        "${bundle_dir}/ftllm-check"
    fi
fi

# Do this after all Python-based checks so no temporary build paths or redundant
# bytecode files enter the portable package.
find "$runtime_dir" -type f \( -name '*.pyc' -o -name '*.pyo' \) -delete
find "$runtime_dir" -depth -type d -name __pycache__ -empty -delete

create_manifest "$bundle_dir"

if ((! directory_only)); then
    log "创建归档：$(basename -- "$archive_path")"
    archive_staging="${build_root}/${bundle_name}.${archive_suffix}.new"
    checksum_staging="${build_root}/${bundle_name}.${archive_suffix}.sha256.new"
    create_archive "$bundle_dir" "$archive_staging" "$build_epoch"
    archive_sha="$(sha256sum "$archive_staging" | cut -d' ' -f1)"
    printf '%s  %s\n' "$archive_sha" "$(basename -- "$archive_path")" > "$checksum_staging"
    mv -f -- "$archive_staging" "$archive_path"
    mv -f -- "$checksum_staging" "$checksum_path"
fi

if ((keep_dir)); then
    if [[ -e "$final_dir" ]]; then
        ((force)) || die "输出目录在构建期间出现：$final_dir"
        [[ "$final_dir" == "${output_dir}/${bundle_name}" ]] \
            || die "拒绝删除非预期目录：$final_dir"
        rm -r -- "$final_dir"
    fi
    mv -- "$bundle_dir" "$final_dir"
fi

if ((! directory_only)); then
    archive_size="$(du -h "$archive_path" | cut -f1)"
    log "完成：${archive_path}（${archive_size}）"
    log "校验：${checksum_path}"
fi
if ((keep_dir)); then
    log "目录：${final_dir}"
fi
