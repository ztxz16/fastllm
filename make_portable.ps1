[CmdletBinding()]
param(
    [string]$Wheel = "",
    [switch]$CpuOnly,
    [string]$CudaArch = "",
    [string]$BuildDirectory = "build-fastllm-windows",
    [string]$OutputDirectory = "build-portable-dist",
    [string]$CacheDirectory = "build-portable-cache",
    [string]$Constraints = "portable/constraints.txt",
    [string]$CMakePath = "",
    [string[]]$CMakeArgs = @(),
    [int]$Jobs = 12,
    [switch]$Offline,
    [switch]$RequireCuda,
    [switch]$NoArchive,
    [switch]$SkipTests
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT" -or -not [Environment]::Is64BitProcess) {
    throw "Run make_portable.ps1 in 64-bit Windows PowerShell."
}
$RepoRoot = $PSScriptRoot
. (Join-Path $RepoRoot "tools/windows/common.ps1")
if (-not $CudaArch) { $CudaArch = Get-FastllmCudaArch }
if ($Jobs -lt 1) { throw "Jobs must be at least 1." }
if ($CpuOnly -and $RequireCuda) { throw "CpuOnly cannot be combined with RequireCuda." }
$Cache = Full-Path $CacheDirectory
$Output = Full-Path $OutputDirectory
if (Test-Path -LiteralPath (Join-Path $Output "ftllm")) {
    throw "Output already exists. Choose another -OutputDirectory."
}
$CMakePath = Get-FastllmCMake $CMakePath
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$PythonExe = Initialize-FastllmPython $Cache -Offline:$Offline
foreach ($asset in @("pi", "rg", "fd")) {
    $null = Get-FastllmAsset $asset $Cache -Offline:$Offline
}

if (-not $Wheel) {
    $buildArgs = @{
        Incremental = $true; BuildDirectory = $BuildDirectory; CudaArch = $CudaArch
        Python = $PythonExe; CMakePath = $CMakePath; Jobs = $Jobs; CpuOnly = $CpuOnly
        UseNccl = "OFF"; CMakeArgs = $CMakeArgs
    }
    & (Join-Path $RepoRoot "make_whl.ps1") @buildArgs
    $backend = if ($CpuOnly) { "cpu" } else { "cuda" }
    $dist = Join-Path (Full-Path $BuildDirectory) "$backend/tools/dist"
    $Wheel = (Get-ChildItem -LiteralPath $dist -Filter '*-win_amd64.whl' |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
}
$Wheel = Full-Path $Wheel

$LauncherBuild = Join-Path $Cache "launcher-build"
Run $CMakePath @("-S", (Join-Path $RepoRoot "portable/windows"), "-B", $LauncherBuild, "-G", "Visual Studio 17 2022", "-A", "x64")
Run $CMakePath @("--build", $LauncherBuild, "--config", "Release", "--parallel", "$Jobs")

$builderArgs = @("-I", "-B", "-X", "utf8", (Join-Path $RepoRoot "portable/windows/build.py"),
    "--wheel", $Wheel, "--output", $Output, "--cache", $Cache,
    "--constraints", (Full-Path $Constraints), "--cuda-arch", $CudaArch,
    "--launcher", (Join-Path $LauncherBuild "Release/ftllm.exe"))
if ($Offline) { $builderArgs += "--offline" }
if ($RequireCuda) { $builderArgs += "--require-cuda" }
if ($SkipTests) { $builderArgs += "--skip-tests" }
if ($NoArchive) { $builderArgs += "--no-archive" }
Run $PythonExe $builderArgs
