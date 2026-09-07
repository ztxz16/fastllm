#!/usr/bin/env python3
"""Register bundled icons and file-manager metadata for the current user."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


ENTRIES = (
    ("support/desktop/Fastllm-Launcher.desktop", "Fastllm-Launcher", "fastllm-portable-launcher"),
    ("support/desktop/ftllm-launch-webui.desktop", "ftllm-launch-webui", "fastllm-portable-browser"),
)


def xdg_directory(variable, default):
    value = Path(os.environ.get(variable, ""))
    return value if value.is_absolute() else Path.home() / default


def write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        stream.write(content)
    try:
        temporary.chmod(0o644)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run_optional(command):
    try:
        return subprocess.run(
            command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL, timeout=3,
        ).returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def setup(bundle, quiet=False):
    support = bundle / "support"
    theme = xdg_directory("XDG_DATA_HOME", ".local/share") / "icons/hicolor"
    icon_directory = theme / "scalable/apps"
    contents = {
        name: (support / "icons" / (name + ".svg")).read_bytes()
        for _, _, name in ENTRIES
    }
    cache = xdg_directory("XDG_CACHE_HOME", ".cache") / "fastllm/desktop-icons"
    stamp = cache / (hashlib.sha256(os.fsencode(bundle)).hexdigest() + ".json")
    state = json.dumps({
        "format_version": 4,
        "entries": ENTRIES,
        "icons": {name: hashlib.sha256(data).hexdigest() for name, data in contents.items()},
        "theme": str(theme),
        "session": os.environ.get("DBUS_SESSION_BUS_ADDRESS", ""),
    }, sort_keys=True).encode()
    icons_current = all(
        (icon_directory / (name + ".svg")).is_file()
        and (icon_directory / (name + ".svg")).read_bytes() == data
        for name, data in contents.items()
    )
    if quiet and icons_current and stamp.is_file() and stamp.read_bytes() == state:
        return True

    if not icons_current:
        for name, data in contents.items():
            write_file(icon_directory / (name + ".svg"), data)
        os.utime(theme, None)
        updater = shutil.which("gtk-update-icon-cache")
        if updater:
            run_optional([updater, "--force", "--ignore-theme-index", str(theme)])

    # Nautilus treats .desktop files in ordinary folders as documents. Its
    # custom-icon metadata is separate from the desktop entry's Icon key.
    # DING and other desktop launchers also use metadata::trusted.
    gio = shutil.which("gio")
    metadata_ready = bool(gio)
    for desktop, executable, name in ENTRIES:
        filenames = (desktop, executable, "launch.sh") if executable == "ftllm-launch-webui" else (desktop, executable)
        for filename in filenames:
            entry = bundle / filename
            if not entry.is_file():
                continue
            if not gio:
                continue
            icon_uri = (icon_directory / (name + ".svg")).as_uri()
            if not run_optional([gio, "set", "-t", "string", str(entry), "metadata::custom-icon", icon_uri]):
                metadata_ready = False
            else:
                # Some file managers cache metadata until a filesystem change.
                # Notify their file monitors without modifying file contents.
                try:
                    os.utime(entry, None, follow_symlinks=False)
                except OSError:
                    pass  # Read-only bundles can still use the registered icon.
            if filename == desktop:
                mode = entry.stat().st_mode
                if not mode & 0o111:
                    entry.chmod(mode | 0o111)
                if not run_optional([gio, "set", "-t", "string", str(entry), "metadata::trusted", "true"]):
                    metadata_ready = False

    # Older bundles gave the CLI a custom terminal icon. Remove that metadata
    # as well when updating an existing directory, leaving the system default.
    cli = bundle / "ftllm"
    if gio and cli.is_file():
        if run_optional([gio, "set", "-t", "unset", str(cli), "metadata::custom-icon"]):
            try:
                os.utime(cli, None, follow_symlinks=False)
            except OSError:
                pass
        else:
            metadata_ready = False

    if metadata_ready:
        write_file(stamp, state)
    if not quiet:
        print(f"已为当前用户安装 {len(contents)} 个 FastLLM 图标：{icon_directory}")
        if metadata_ready:
            print("已设置桌面应用和网页服务图标，ftllm 使用系统默认图标。")
        else:
            print("当前会话无法设置文件管理器图标；请在本机桌面终端运行 ./ftllm --setup-desktop。")
    return metadata_ready


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    try:
        setup(Path(__file__).resolve().parent.parent, quiet=args.quiet)
    except OSError as error:
        if not args.quiet:
            parser.exit(1, f"无法设置桌面图标：{error}\n")


if __name__ == "__main__":
    main()
