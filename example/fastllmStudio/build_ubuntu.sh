#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
APP_NAME="FastllmStudio"
VERSION="${VERSION:-1.0.0}"
BUILD_DIR="${SCRIPT_DIR}/build"
DIST_DIR="${SCRIPT_DIR}/dist"
VENV_DIR="${BUILD_DIR}/venv"

# ---------- helpers ----------
info()  { echo -e "\033[1;32m[INFO]\033[0m  $*"; }
warn()  { echo -e "\033[1;33m[WARN]\033[0m  $*"; }
error() { echo -e "\033[1;31m[ERROR]\033[0m $*"; exit 1; }

# ---------- 1. system dependencies ----------
info "Checking system dependencies..."

MISSING_PKGS=()
for cmd in python3 pip3 pyside6-lrelease; do
    if ! command -v "$cmd" &>/dev/null; then
        MISSING_PKGS+=("$cmd")
    fi
done

if [ ${#MISSING_PKGS[@]} -ne 0 ]; then
    warn "Missing commands: ${MISSING_PKGS[*]}"
    info "Attempting to install system packages..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq python3 python3-pip python3-venv \
        libxkbcommon0 libgl1 libegl1 libfontconfig1 libdbus-1-3
fi

# ---------- 2. clean previous build ----------
info "Cleaning previous build artifacts..."
rm -rf "${BUILD_DIR}" "${DIST_DIR}"
mkdir -p "${BUILD_DIR}" "${DIST_DIR}"

# ---------- 3. virtual environment ----------
info "Creating virtual environment..."
python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

info "Installing Python dependencies..."
pip install --upgrade pip -q
pip install -r "${SCRIPT_DIR}/requirements.txt" -q
pip install pyinstaller -q

# ---------- 4. compile i18n .ts -> .qm ----------
info "Compiling translation files..."
I18N_DIR="${SCRIPT_DIR}/i18n"
for ts_file in "${I18N_DIR}"/*.ts; do
    [ -f "$ts_file" ] || continue
    qm_file="${ts_file%.ts}.qm"
    pyside6-lrelease "$ts_file" -qm "$qm_file"
    info "  $(basename "$ts_file") -> $(basename "$qm_file")"
done

# ---------- 5. PyInstaller bundle ----------
info "Building with PyInstaller..."

QML_DIR="${SCRIPT_DIR}/qml"

pyinstaller \
    --noconfirm \
    --clean \
    --name "${APP_NAME}" \
    --windowed \
    --add-data "${QML_DIR}:qml" \
    --add-data "${I18N_DIR}:i18n" \
    --hidden-import PySide6.QtQuick \
    --hidden-import PySide6.QtQml \
    --hidden-import PySide6.QtCore \
    --hidden-import PySide6.QtGui \
    --hidden-import PySide6.QtNetwork \
    --hidden-import openai \
    --hidden-import requests \
    --distpath "${DIST_DIR}" \
    --workpath "${BUILD_DIR}/pyinstaller" \
    --specpath "${BUILD_DIR}" \
    "${SCRIPT_DIR}/main.py"

# ---------- 6. verify output ----------
BINARY="${DIST_DIR}/${APP_NAME}/${APP_NAME}"
if [ ! -f "$BINARY" ]; then
    error "Build failed: binary not found at ${BINARY}"
fi

info "Build successful!"
info "  Output: ${DIST_DIR}/${APP_NAME}/"
info "  Binary: ${BINARY}"

# ---------- 7. optional: create .tar.gz ----------
info "Creating distributable archive..."
ARCHIVE="${DIST_DIR}/${APP_NAME}-${VERSION}-linux-x86_64.tar.gz"
tar -czf "${ARCHIVE}" -C "${DIST_DIR}" "${APP_NAME}"
info "  Archive: ${ARCHIVE}"

deactivate
info "Done."
