FastLLM Launcher - Windows x64 Electron portable application
==========================================================

Extract the ENTIRE ZIP to a writable folder, then double-click:

    FastLLM-Launcher.exe

This opens the Electron desktop application, not your web browser.
Do not copy the EXE out of this folder; its resources and ftllm runtime are required.
Closing the application also stops its managed model/download processes.

Bundled: Electron/Chromium/Node, isolated Python, FastLLM CPU fallback,
CUDA runtime and cuBLAS (CUDA build), MSVC/OpenMP runtime, Pi Agent,
ripgrep, fd, aria2 and all Python dependencies.
No pip/npm, Python, Node, CUDA Toolkit, VC redistributable installation,
administrator privileges or system PATH changes are required.

Requires Windows 10/11 x64. GPU acceleration requires a supported NVIDIA
GPU and an installed, CUDA-compatible NVIDIA driver (an OS device driver,
not redistributable inside a portable application). See BUILD-INFO.json
for the CUDA architectures included in this particular build.
Model weights are not included; choose existing local weights or download
them using the application. Local models can be used without internet.

Settings, logs, model caches and Electron profile default to data/ beside
the EXE. If the folder is read-only, data falls back to the user profile.
Logs: data/logs/desktop.log
Internal CLI: ftllm/ftllm.exe
Runtime diagnostics: ftllm/ftllm-check.cmd --smoke --audit

The internal authenticated service listens only on 127.0.0.1 and is managed
by Electron. No separate terminal or browser needs to be opened.
Unsigned local builds may trigger Windows SmartScreen.
BUILD-INFO.json records provenance; MANIFEST.sha256 covers package files.
Electron licenses: LICENSE and LICENSES.chromium.html
Other licenses: LICENSE-FastLLM and ftllm/THIRD-PARTY/
