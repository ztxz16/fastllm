[CmdletBinding()]
param(
    [string]$Wheel = "",
    [string]$Version = "",
    [switch]$CpuOnly,
    [string]$CudaArch = "",
    [string]$BuildDirectory = "build-fastllm-windows",
    [string]$OutputDirectory = "build-windows-release",
    [string]$CacheDirectory = "build-portable-cache",
    [string]$Constraints = "portable/constraints.txt",
    [string]$CMakePath = "",
    [string[]]$CMakeArgs = @(),
    [string]$CuobjdumpPath = "",
    [string]$SmokeModel = "",
    [ValidateRange(1, 64)][int]$SmokeTp = 1,
    [int]$Jobs = 12,
    [switch]$Offline,
    [switch]$RequireCuda
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
if ($env:OS -ne "Windows_NT" -or -not [Environment]::Is64BitProcess) {
    throw "Run make_release.ps1 in 64-bit Windows PowerShell."
}
$RepoRoot = $PSScriptRoot
. (Join-Path $RepoRoot "tools/windows/common.ps1")
if (-not $CudaArch) { $CudaArch = Get-FastllmCudaArch }
if ($Jobs -lt 1) { throw "Jobs must be at least 1." }
if ($CpuOnly -and ($RequireCuda -or $SmokeModel)) {
    throw "CpuOnly cannot be combined with RequireCuda or a GPU SmokeModel."
}
if ($SmokeModel -and -not (Test-Path -LiteralPath (Full-Path $SmokeModel))) {
    throw "SmokeModel does not exist: $SmokeModel"
}
if ($SmokeModel -and (Test-Path -LiteralPath (Full-Path $SmokeModel) -PathType Container) -and
    -not (Get-ChildItem -LiteralPath (Full-Path $SmokeModel) -Force | Select-Object -First 1)) {
    throw "SmokeModel directory is empty: $SmokeModel"
}
$Output = Full-Path $OutputDirectory
$Cache = Full-Path $CacheDirectory
foreach ($existing in @("FastLLM", "release-verification.json")) {
    if (Test-Path -LiteralPath (Join-Path $Output $existing)) {
        throw "Release output already exists. Choose another -OutputDirectory: $Output"
    }
}
$CMakePath = Get-FastllmCMake $CMakePath
New-Item -ItemType Directory -Force -Path $Output | Out-Null
$log = Join-Path $Output ("release-" + (Get-Date -Format "yyyyMMdd-HHmmss") + ".log")
Start-Transcript -Path $log | Out-Null
try {
    $Python = Initialize-FastllmPython $Cache -Offline:$Offline
    $builtWheel = -not $Wheel
    $backend = if ($CpuOnly) { "cpu" } else { "cuda" }
    if ($builtWheel) {
        Write-Host "[release] Build wheel (incremental)"
        $buildArgs = @{
            Incremental = $true; BuildDirectory = $BuildDirectory; CudaArch = $CudaArch
            Python = $Python; CMakePath = $CMakePath; CMakeArgs = $CMakeArgs
            Jobs = $Jobs; CpuOnly = $CpuOnly; UseNccl = "OFF"
        }
        & (Join-Path $RepoRoot "make_whl.ps1") @buildArgs
        $dist = Join-Path (Full-Path $BuildDirectory) "$backend/tools/dist"
        $Wheel = (Get-ChildItem -LiteralPath $dist -Filter '*-win_amd64.whl' |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1).FullName
    }
    $Wheel = Full-Path $Wheel
    Write-Host "[release] Verify wheel and embedded CUDA architectures"
    $verifyArgs = @("-I", "-B", "-X", "utf8", (Join-Path $RepoRoot "tools/windows/verify_wheel.py"),
        $Wheel, "--backend", $backend, "--report", (Join-Path $Output "wheel-verification.json"))
    if ($Version) { $verifyArgs += @("--version", $Version) }
    if (-not $CpuOnly) { $verifyArgs += @("--cuda-arch", $CudaArch) }
    if ($CuobjdumpPath) { $verifyArgs += @("--cuobjdump", $CuobjdumpPath) }
    if ($builtWheel) {
        $verifyArgs += @("--source-root", $RepoRoot, "--native-root", (Full-Path $BuildDirectory))
    }
    Run $Python $verifyArgs
    Write-Host "[release] Build and test Electron application"
    $desktopArgs = @{
        Wheel = $Wheel; CpuOnly = $CpuOnly; CudaArch = $CudaArch
        OutputDirectory = $Output; CacheDirectory = $Cache; Constraints = $Constraints
        CMakePath = $CMakePath; Jobs = $Jobs; Offline = $Offline
        RequireCuda = $RequireCuda; SmokeModel = $SmokeModel; SmokeTp = $SmokeTp
    }
    & (Join-Path $RepoRoot "desktop/package.ps1") @desktopArgs
    Write-Host "[release] Collect artifacts, checksums and verification report"
    Run $Python @("-I", "-B", "-X", "utf8", (Join-Path $RepoRoot "tools/windows/collect_release.py"),
        "--wheel", $Wheel, "--output", $Output)
    Write-Host "Release complete: $Output"
    Write-Host "Build log: $log"
} finally {
    Stop-Transcript | Out-Null
}
