#!/usr/bin/env bash

set -euo pipefail

# Called by the native launcher, which resolves its location via /proc/self/exe.
ENTRYPOINT="$1"
shift
BUNDLE_ROOT=$(CDPATH= cd -- "$(dirname -- "$ENTRYPOINT")" && pwd -P)
SUPPORT_ROOT="${BUNDLE_ROOT}/support"
ENTRYPOINT_NAME="$(basename -- "$ENTRYPOINT")"
if [[ "$ENTRYPOINT_NAME" == ftllm && "${1:-}" == --setup-desktop ]]; then
    shift
    exec "${SUPPORT_ROOT}/runtime/bin/python3" -I -B "${SUPPORT_ROOT}/setup_desktop.py" "$@"
fi
if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" ]]; then
    "${SUPPORT_ROOT}/runtime/bin/python3" -I -B "${SUPPORT_ROOT}/setup_desktop.py" --quiet \
        >/dev/null 2>&1 &
fi

open_terminal() {
    local terminal
    for terminal in xdg-terminal-exec x-terminal-emulator gnome-terminal konsole xfce4-terminal mate-terminal xterm; do
        command -v "$terminal" >/dev/null 2>&1 || continue
        case "$terminal" in
            xdg-terminal-exec) exec "$terminal" "$@" ;;
            gnome-terminal) exec "$terminal" -- "$@" ;;
            xfce4-terminal|mate-terminal) exec "$terminal" -x "$@" ;;
            *) exec "$terminal" -e "$@" ;;
        esac
    done
    printf '没有找到终端程序，请在终端中运行此入口，或打开 Fastllm-Launcher。\n' >&2
    exit 1
}

case "$ENTRYPOINT_NAME" in
    Fastllm-Launcher)
        exec "${SUPPORT_ROOT}/FastLLM-Launcher" "$@"
        ;;
    ftllm-launch-webui)
        if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" && ! -t 0 && ! -t 1 && $# == 0 ]]; then
            open_terminal "${SUPPORT_ROOT}/launch.sh"
        fi
        exec "${SUPPORT_ROOT}/launch.sh" "$@"
        ;;
    ftllm)
        if [[ -n "${DISPLAY:-}${WAYLAND_DISPLAY:-}" && ! -t 0 && ! -t 1 && $# == 0 ]]; then
            open_terminal "${SUPPORT_ROOT}/terminal.sh"
        fi
        exec "${SUPPORT_ROOT}/ftllm" "$@"
        ;;
    *)
        printf '未知的 FastLLM 入口：%s\n' "$ENTRYPOINT_NAME" >&2
        exit 2
        ;;
esac
