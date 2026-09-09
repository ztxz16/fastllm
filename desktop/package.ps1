[CmdletBinding()]
param(
    [string]$Wheel = "",
    [switch]$CpuOnly,
    [string]$CudaArch = "",
    [string]$BuildDirectory = "build-fastllm-windows",
    [string]$OutputDirectory = "build-desktop-dist",
    [string]$CacheDirectory = "build-portable-cache",
    [string]$CMakePath = "",
    [string[]]$CMakeArgs = @(),
    [string]$Constraints = "portable/constraints.txt",
    [string]$SmokeModel = "",
    [int]$Jobs = 12,
    [switch]$Offline,
    [switch]$RequireCuda,
    [switch]$SkipTests
)

# No npm, Node.js, Python or Electron installation is needed on the build host.
# Native compilation still requires Visual Studio C++ and (unless CPU-only) CUDA.
Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT" -or -not [Environment]::Is64BitProcess) {
    throw "Run desktop/package.ps1 in 64-bit Windows PowerShell."
}
$RepoRoot = Split-Path -Parent $PSScriptRoot
. (Join-Path $RepoRoot "tools/windows/common.ps1")
if (-not $CudaArch) { $CudaArch = Get-FastllmCudaArch }
if ($Jobs -lt 1) { throw "Jobs must be at least 1." }
if ($CpuOnly -and ($RequireCuda -or $SmokeModel)) {
    throw "CpuOnly cannot be combined with RequireCuda or a GPU SmokeModel."
}
$Cache = Full-Path $CacheDirectory
$Output = Full-Path $OutputDirectory
if (Test-Path -LiteralPath (Join-Path $Output "FastLLM")) {
    throw "Output already exists. Choose another -OutputDirectory; existing packages are never overwritten."
}
New-Item -ItemType Directory -Force -Path $Cache, $Output | Out-Null
$Electron = Get-FastllmAsset "electron" $Cache -Offline:$Offline
# Short, unique staging paths avoid Windows MAX_PATH and retain failed builds.
$Stage = Join-Path $Output (".e-" + [Guid]::NewGuid().ToString("N").Substring(0, 8))
New-Item -ItemType Directory -Path $Stage | Out-Null
$runtimeArgs = @{
    Wheel = $Wheel; CpuOnly = $CpuOnly; CudaArch = $CudaArch
    BuildDirectory = $BuildDirectory; OutputDirectory = $Stage; CacheDirectory = $Cache
    CMakePath = $CMakePath; CMakeArgs = $CMakeArgs; Constraints = $Constraints
    Jobs = $Jobs; Offline = $Offline; RequireCuda = $RequireCuda
    SkipTests = $SkipTests; NoArchive = $true
}
& (Join-Path $RepoRoot "make_portable.ps1") @runtimeArgs
$Python = Join-Path $Cache "bootstrap/python/python.exe"
$builderArgs = @("-I", "-B", "-X", "utf8", (Join-Path $PSScriptRoot "windows/build.py"),
    "--runtime", (Join-Path $Stage "ftllm"), "--electron", $Electron, "--output", $Output,
    "--entrypoints", (Join-Path $Cache "launcher-build/Release"))
if ($SkipTests) { $builderArgs += "--skip-tests" }
if ($SmokeModel) { $builderArgs += @("--smoke-model", (Full-Path $SmokeModel)) }
Run $Python $builderArgs
