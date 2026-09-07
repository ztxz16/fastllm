#!/usr/bin/env bash

set -Eeuo pipefail

export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
ROOT_DIR="$(CDPATH= cd -- "${SCRIPT_DIR}/.." && pwd -P)"
DEFAULT_WHEEL_DIR="${ROOT_DIR}/build-fastllm/tools/dist"
DEFAULT_OUTPUT_DIR="${ROOT_DIR}/portable-dist"

ELECTRON_VERSION="44.1.1"
ELECTRON_ARCHIVE="electron-v${ELECTRON_VERSION}-linux-x64.zip"
ELECTRON_URL="https://github.com/electron/electron/releases/download/v${ELECTRON_VERSION}/${ELECTRON_ARCHIVE}"
ELECTRON_SHA256="043327f5bf2c492f744a806544d1aabd0dbec8674f10d3043ef0c455291b3a33"
TARGET_GLIBC="2.35"

wheel_path=""
output_dir="${DEFAULT_OUTPUT_DIR}"
archive_format="tar.gz"
build_wheel=0
keep_dir=0
force=0
offline=0
run_tests=1

usage() {
    cat <<'EOF'
用法：
  ./desktop/package.sh [选项]

把 Electron、ftllm/Pi Agent 绿色运行时和桌面所需动态库组装为 Linux x86_64 绿色包。

选项：
  --wheel PATH       使用指定的 ftllm wheel；默认选择 build-fastllm/tools/dist 中最新文件
  --build-wheel      打包前先运行 make_whl.sh；不能与 --wheel 同时使用
  --output-dir DIR   输出目录；默认 portable-dist
  --format FORMAT    tar.gz 或 tar.zst；默认 tar.gz
  --keep-dir         同时保留未压缩目录
  --offline          不访问网络，只使用已有 Python/Electron/Pi 缓存及 wheelhouse
  --skip-tests       跳过运行时冒烟测试
  --force            覆盖同名产物
  -h, --help         显示帮助

环境变量：
  FTLLM_DESKTOP_CACHE_DIR  Electron 下载缓存目录
  FTLLM_PORTABLE_CACHE_DIR Python 与 wheelhouse 缓存目录
  FTLLM_PACKAGE_JOBS       pigz 压缩线程数；默认 4
  SOURCE_DATE_EPOCH        固定归档时间戳
EOF
}

die() {
    printf '错误：%s\n' "$*" >&2
    exit 1
}

log() {
    printf '[ftllm-desktop] %s\n' "$*"
}

require_command() {
    command -v "$1" >/dev/null 2>&1 || die "缺少构建命令：$1"
}

absolute_path() {
    local value="$1"
    local directory
    directory="$(dirname -- "$value")"
    printf '%s/%s\n' "$(CDPATH= cd -- "$directory" && pwd -P)" "$(basename -- "$value")"
}

select_default_wheel() {
    local candidate
    local selected=""
    shopt -s nullglob
    for candidate in "${DEFAULT_WHEEL_DIR}"/ftllm-*.whl; do
        if [[ -z "$selected" || "$candidate" -nt "$selected" ]]; then
            selected="$candidate"
        fi
    done
    shopt -u nullglob
    [[ -n "$selected" ]] || die "没有可用 wheel；请先运行 make_whl.sh 或使用 --build-wheel"
    printf '%s\n' "$selected"
}

download_electron() {
    local destination="$1"
    local partial="${destination}.part"
    if [[ -f "$destination" ]] \
        && printf '%s  %s\n' "$ELECTRON_SHA256" "$destination" \
            | sha256sum --check --status; then
        log "复用 Electron 缓存：${ELECTRON_ARCHIVE}"
        return
    fi
    rm -f -- "$destination" "$partial"
    ((offline == 0)) || die "--offline 模式缺少有效 Electron 缓存：$destination"
    log "下载 Electron ${ELECTRON_VERSION}"
    if command -v curl >/dev/null 2>&1; then
        curl --fail --location --retry 3 --retry-delay 2 --output "$partial" "$ELECTRON_URL"
    elif command -v wget >/dev/null 2>&1; then
        wget --tries=3 --output-document="$partial" "$ELECTRON_URL"
    else
        die "需要 curl 或 wget 下载 Electron"
    fi
    printf '%s  %s\n' "$ELECTRON_SHA256" "$partial" | sha256sum --check --status \
        || die "Electron 归档 SHA256 校验失败"
    mv -- "$partial" "$destination"
}

version_is_greater() {
    local left="$1"
    local right="$2"
    local greatest
    greatest="$(printf '%s\n%s\n' "$left" "$right" | LC_ALL=C sort -V | tail -n 1)"
    [[ "$greatest" == "$left" && "$left" != "$right" ]]
}

read_wheel_metadata() {
    python3 - "$1" <<'PY'
import email
import sys
import zipfile

with zipfile.ZipFile(sys.argv[1]) as wheel:
    metadata_path = next(name for name in wheel.namelist() if name.endswith(".dist-info/METADATA"))
    metadata = email.message_from_bytes(wheel.read(metadata_path))
print(metadata.get("Name", ""), metadata.get("Version", ""), sep="\t")
PY
}

audit_glibc() {
    local bundle_dir="$1"
    local raw_report="$2"
    local final_report="${bundle_dir}/ELF-GLIBC-REQUIREMENTS.txt"
    local elf_file
    local required
    local maximum="0"
    : > "$raw_report"
    while IFS= read -r -d '' elf_file; do
        readelf -h "$elf_file" >/dev/null 2>&1 || continue
        required="$(
            readelf --version-info "$elf_file" 2>/dev/null \
                | grep -oE 'GLIBC_[0-9]+\.[0-9]+' \
                | LC_ALL=C sort -Vu \
                | tail -n 1 \
                || true
        )"
        [[ -n "$required" ]] && printf '%s  %s\n' "$required" "${elf_file#"${bundle_dir}/"}" >> "$raw_report"
    done < <(find "$bundle_dir" -type f -print0)
    if [[ -s "$raw_report" ]]; then
        maximum="$(cut -d' ' -f1 "$raw_report" | sed 's/^GLIBC_//' | LC_ALL=C sort -Vu | tail -n 1)"
    fi
    {
        printf 'Target maximum: GLIBC_%s (Ubuntu 22.04 baseline)\n' "$TARGET_GLIBC"
        printf 'Bundle maximum: GLIBC_%s\n\n' "$maximum"
        LC_ALL=C sort -V "$raw_report"
    } > "$final_report"
    if version_is_greater "$maximum" "$TARGET_GLIBC"; then
        die "包内 ELF 需要 GLIBC_${maximum}，高于目标 GLIBC_${TARGET_GLIBC}"
    fi
    log "ELF 兼容审计通过：最高 GLIBC_${maximum}"
}

audit_driver_boundary() {
    local bundle_dir="$1"
    local bundled_drivers
    bundled_drivers="$(
        find "$bundle_dir" \( -type f -o -type l \) \
            \( -name 'libcuda.so*' -o -name 'libcudadebugger.so*' \
               -o -name 'libnvidia-*' -o -name 'libGLX_nvidia.so*' \
               -o -name 'libEGL_nvidia.so*' -o -name 'libnvcuvid.so*' \
               -o -name 'libnvoptix.so*' -o -name 'libvdpau_nvidia.so*' \) \
            -print
    )"
    if [[ -n "$bundled_drivers" ]]; then
        printf '%s\n' "$bundled_drivers" >&2
        die "检测到不应打包的 NVIDIA 驱动库"
    fi
    log "驱动边界审计通过：libcuda.so.1 由宿主机 NVIDIA 驱动提供"
}

create_manifest() {
    local bundle_dir="$1"
    (
        cd "$bundle_dir"
        : > MANIFEST.sha256
        while IFS= read -r -d '' file; do
            sha256sum "$file" >> MANIFEST.sha256
        done < <(find . -type f ! -name MANIFEST.sha256 -print0 | LC_ALL=C sort -z)
    )
}

create_archive() {
    local bundle_dir="$1"
    local destination="$2"
    local epoch="$3"
    local parent name
    local package_jobs="${FTLLM_PACKAGE_JOBS:-4}"
    [[ "$package_jobs" =~ ^[1-9][0-9]*$ ]] \
        || die "FTLLM_PACKAGE_JOBS 必须是正整数"
    parent="$(dirname -- "$bundle_dir")"
    name="$(basename -- "$bundle_dir")"
    case "$archive_format" in
        tar.gz)
            if command -v pigz >/dev/null 2>&1; then
                tar --sort=name --mtime="@${epoch}" --owner=0 --group=0 --numeric-owner \
                    -C "$parent" -cf - "$name" | pigz -p "$package_jobs" -6n > "$destination"
            else
                tar --sort=name --mtime="@${epoch}" --owner=0 --group=0 --numeric-owner \
                    -C "$parent" -cf - "$name" | gzip -6n > "$destination"
            fi
            ;;
        tar.zst)
            tar --sort=name --mtime="@${epoch}" --owner=0 --group=0 --numeric-owner \
                -C "$parent" -cf - "$name" | zstd -T"$package_jobs" -10 -q -o "$destination"
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
        --build-wheel)
            build_wheel=1
            shift
            ;;
        --output-dir)
            (($# >= 2)) || die "--output-dir 缺少参数"
            output_dir="$2"
            shift 2
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
        --offline)
            offline=1
            shift
            ;;
        --skip-tests)
            run_tests=0
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
        *)
            die "未知选项：$1"
            ;;
    esac
done

[[ "$(uname -s)" == "Linux" && "$(uname -m)" == "x86_64" ]] \
    || die "桌面绿色包目前只支持 Linux x86_64"
[[ "$archive_format" == "tar.gz" || "$archive_format" == "tar.zst" ]] \
    || die "--format 只能是 tar.gz 或 tar.zst"
if ((build_wheel)) && [[ -n "$wheel_path" ]]; then
    die "--build-wheel 不能与 --wheel 同时使用"
fi

for command in python3 unzip sha256sum readelf ldd ldconfig find sort tar sed cut install date du grep; do
    require_command "$command"
done
if [[ "$archive_format" == "tar.gz" ]]; then
    require_command gzip
else
    require_command zstd
fi
[[ -x "${ROOT_DIR}/make_portable.sh" ]] || die "缺少 make_portable.sh"

if ((build_wheel)); then
    log "构建当前源码的 ftllm wheel"
    (cd "$ROOT_DIR" && ./make_whl.sh)
fi
if [[ -z "$wheel_path" ]]; then
    wheel_path="$(select_default_wheel)"
fi
[[ -f "$wheel_path" ]] || die "wheel 不存在：$wheel_path"
wheel_path="$(absolute_path "$wheel_path")"

IFS=$'\t' read -r package_name package_version < <(read_wheel_metadata "$wheel_path")
case "${package_name,,}" in
    ftllm|ftllm-nightly) ;;
    *) die "输入不是 ftllm wheel：${package_name}" ;;
esac
[[ "$package_version" =~ ^[A-Za-z0-9][A-Za-z0-9._+!-]*$ ]] || die "wheel 版本无效：$package_version"

mkdir -p "$output_dir" 2>/dev/null || die "无法创建输出目录：$output_dir"
[[ -w "$output_dir" ]] || die "输出目录不可写：$output_dir"
output_dir="$(CDPATH= cd -- "$output_dir" && pwd -P)"
cache_dir="${FTLLM_DESKTOP_CACHE_DIR:-${ROOT_DIR}/build-desktop-cache}"
mkdir -p "$cache_dir" 2>/dev/null || die "无法创建 Electron 缓存目录：$cache_dir"
[[ -w "$cache_dir" ]] || die "Electron 缓存目录不可写：$cache_dir"
cache_dir="$(CDPATH= cd -- "$cache_dir" && pwd -P)"
electron_archive_path="${cache_dir}/${ELECTRON_ARCHIVE}"
download_electron "$electron_archive_path"

bundle_name="FastLLM-Launcher-${package_version}-linux-x86_64-portable"
archive_path="${output_dir}/${bundle_name}.${archive_format}"
checksum_path="${archive_path}.sha256"
final_dir="${output_dir}/${bundle_name}"
if ((! force)); then
    [[ ! -e "$archive_path" ]] || die "输出已存在：$archive_path（使用 --force 覆盖）"
    [[ ! -e "$checksum_path" ]] || die "输出已存在：$checksum_path（使用 --force 覆盖）"
    ((! keep_dir)) || [[ ! -e "$final_dir" ]] || die "输出已存在：$final_dir（使用 --force 覆盖）"
fi

build_root="$(mktemp -d "${output_dir}/.ftllm-desktop-build.XXXXXXXX")"
cleanup() {
    if [[ -n "${build_root:-}" && -d "$build_root" \
        && "$build_root" == "${output_dir}/.ftllm-desktop-build."* ]]; then
        rm -r -- "$build_root"
    fi
}
trap cleanup EXIT INT TERM

runtime_output="${build_root}/portable-runtime"
mkdir -p "$runtime_output"
portable_args=(--wheel "$wheel_path" --output-dir "$runtime_output" --directory-only)
((run_tests)) || portable_args+=(--skip-tests)
((offline == 0)) || portable_args+=(--offline)
log "生成内置 ftllm 绿色运行时"
"${ROOT_DIR}/make_portable.sh" "${portable_args[@]}"
runtime_source="$(
    find "$runtime_output" -mindepth 1 -maxdepth 1 -type d \
        -name '*-portable-cu12' -print -quit
)"
[[ -n "$runtime_source" ]] || die "make_portable.sh 未生成运行时目录"

bundle_dir="${build_root}/${bundle_name}"
mv -- "$runtime_source" "$bundle_dir"
unzip -q "$electron_archive_path" -d "$bundle_dir"
[[ -x "${bundle_dir}/electron" ]] || die "Electron 归档结构异常"
mv -- "${bundle_dir}/electron" "${bundle_dir}/FastLLM-Launcher.bin"
rm -f -- "${bundle_dir}/resources/default_app.asar"
mkdir -p "${bundle_dir}/resources/app"
for app_file in package.json main.js runtime.js loading.html loading.js loading.css; do
    install -m 0644 "${SCRIPT_DIR}/app/${app_file}" "${bundle_dir}/resources/app/${app_file}"
done
install -m 0644 "${ROOT_DIR}/tools/fastllm_pytools/launcher_assets/launcher-icon.png" \
    "${bundle_dir}/resources/app/icon.png"
install -m 0755 "${SCRIPT_DIR}/launcher.sh" "${bundle_dir}/FastLLM-Launcher"
install -m 0644 "${SCRIPT_DIR}/BUNDLE-README.md.in" "${bundle_dir}/README.md"
mkdir -p "${bundle_dir}/licenses"
install -m 0644 "${ROOT_DIR}/LICENSE" "${bundle_dir}/licenses/FastLLM-LICENSE"

IFS=$'\t' read -r minimum_driver recommended_driver < <(
    python3 - "${bundle_dir}/BUILD-INFO.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as stream:
    cuda = json.load(stream)["cuda"]
print(cuda["minimum_linux_driver"], cuda["recommended_linux_driver"], sep="\t")
PY
)
sed -i \
    -e "s/@FTLLM_VERSION@/${package_version}/g" \
    -e "s/@MIN_NVIDIA_DRIVER@/${minimum_driver}/g" \
    -e "s/@RECOMMENDED_NVIDIA_DRIVER@/${recommended_driver}/g" \
    "${bundle_dir}/README.md"

mkdir -p "${bundle_dir}/third-party/system"
log "收集 Electron 的非 glibc 动态库闭包"
python3 "${SCRIPT_DIR}/collect_libraries.py" \
    --root "$bundle_dir" \
    --output "${bundle_dir}/lib" \
    --report "${bundle_dir}/ELECTRON-LIBRARIES.json" \
    --copyrights "${bundle_dir}/third-party/system" \
    --exclude "${bundle_dir}/runtime" \
    --optional libgtk-3.so.0 \
    --optional libgdk-3.so.0 \
    --optional libXss.so.1

wheel_sha="$(sha256sum "$wheel_path" | cut -d' ' -f1)"
build_epoch="${SOURCE_DATE_EPOCH:-$(date +%s)}"
[[ "$build_epoch" =~ ^[0-9]+$ ]] || die "SOURCE_DATE_EPOCH 必须是非负整数"
git_commit="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || true)"
git_dirty=""
if [[ -n "$git_commit" ]]; then
    if [[ -n "$(git -C "$ROOT_DIR" status --porcelain --untracked-files=normal 2>/dev/null)" ]]; then
        git_dirty=1
    else
        git_dirty=0
    fi
fi
python3 - "${bundle_dir}/DESKTOP-BUILD-INFO.json" "$package_name" "$package_version" \
    "$wheel_sha" "$ELECTRON_VERSION" "$ELECTRON_SHA256" "$git_commit" "$git_dirty" \
    "$build_epoch" <<'PY'
import datetime
import json
import sys

(
    destination,
    name,
    version,
    wheel_sha,
    electron,
    electron_sha,
    commit,
    dirty,
    epoch,
) = sys.argv[1:]
data = {
    "format_version": 1,
    "package": {"name": name, "version": version},
    "wheel_sha256": wheel_sha,
    "electron": {"version": electron, "archive_sha256": electron_sha},
    "source_commit": commit or None,
    "built_at_utc": datetime.datetime.fromtimestamp(int(epoch), datetime.timezone.utc).isoformat(),
    "platform": {"os": "linux", "architecture": "x86_64", "glibc_minimum": "2.35"},
    "data_directory": "data",
    "source_dirty": None if not dirty else dirty == "1",
}
with open(destination, "w", encoding="utf-8") as output:
    json.dump(data, output, ensure_ascii=False, indent=2, sort_keys=True)
    output.write("\n")
PY

audit_glibc "$bundle_dir" "${build_root}/desktop-elf.raw"
audit_driver_boundary "$bundle_dir"

if ((run_tests)); then
    log "运行桌面包冒烟测试"
    PYTHONPATH="$ROOT_DIR" python3 -m unittest discover \
        -s "${SCRIPT_DIR}/tests" -p 'test_*.py'
    ELECTRON_RUN_AS_NODE=1 "${bundle_dir}/FastLLM-Launcher" \
        --test "${SCRIPT_DIR}/tests/runtime.test.js"
    ELECTRON_RUN_AS_NODE=1 "${bundle_dir}/FastLLM-Launcher" \
        -p 'process.versions.electron' | grep -Fx "$ELECTRON_VERSION" >/dev/null
    "${bundle_dir}/ftllm" --version
    "${bundle_dir}/ftllm" server --help >/dev/null
    "${bundle_dir}/ftllm-check"
    "${bundle_dir}/python" "${ROOT_DIR}/portable/smoke_launcher.py" "$bundle_dir"
    "${bundle_dir}/python" "${ROOT_DIR}/portable/smoke_launcher.py" "$bundle_dir" --entrypoint launch.sh
    "${bundle_dir}/python" "${ROOT_DIR}/portable/smoke_agent.py" --check-studio
    if command -v xvfb-run >/dev/null 2>&1; then
        smoke_data="${build_root}/electron-smoke-data"
        smoke_log="${build_root}/electron-smoke.log"
        set +e
        FTLLM_LAUNCHER_DATA_DIR="$smoke_data" timeout --signal=TERM 15s \
            xvfb-run -a "${bundle_dir}/FastLLM-Launcher" --disable-gpu >"$smoke_log" 2>&1
        smoke_status=$?
        set -e
        if [[ "$smoke_status" != 0 && "$smoke_status" != 124 ]]; then
            tail -n 40 "$smoke_log" >&2
            die "Electron Launcher 冒烟测试失败，退出码 ${smoke_status}"
        fi
        grep -F "Starting bundled ftllm launch" "${smoke_data}/logs/desktop.log" >/dev/null \
            || die "Electron Launcher 未启动 ftllm launch"
    else
        log "未安装 xvfb-run，已跳过图形窗口冒烟测试"
    fi
fi

if find "${bundle_dir}/runtime" -type f \
    \( -name '*.pyc' -o -name '*.pyo' \) -print -quit | grep -q .; then
    die "内置 Python 运行时包含冗余字节码文件"
fi

create_manifest "$bundle_dir"
archive_staging="${build_root}/${bundle_name}.${archive_format}.new"
checksum_staging="${build_root}/${bundle_name}.${archive_format}.sha256.new"
log "创建绿色包归档：$(basename -- "$archive_path")"
create_archive "$bundle_dir" "$archive_staging" "$build_epoch"
archive_sha="$(sha256sum "$archive_staging" | cut -d' ' -f1)"
printf '%s  %s\n' "$archive_sha" "$(basename -- "$archive_path")" > "$checksum_staging"
mv -f -- "$archive_staging" "$archive_path"
mv -f -- "$checksum_staging" "$checksum_path"

if ((keep_dir)); then
    if [[ -e "$final_dir" ]]; then
        ((force)) || die "输出目录已存在：$final_dir"
        [[ "$final_dir" == "${output_dir}/${bundle_name}" ]] || die "拒绝删除非预期目录：$final_dir"
        rm -r -- "$final_dir"
    fi
    mv -- "$bundle_dir" "$final_dir"
fi

log "完成：${archive_path}（$(du -h "$archive_path" | cut -f1)）"
log "校验：${checksum_path}"
((keep_dir == 0)) || log "目录：${final_dir}"
