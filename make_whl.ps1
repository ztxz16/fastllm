[CmdletBinding()]
param(
    [switch]$Nightly,
    [switch]$CpuOnly,
    [switch]$Incremental,
    [string]$CudaArch = "",
    [ValidateSet("Auto", "ON", "OFF")]
    [string]$UseNccl = "Auto",
    [string]$NcclRoot = "",
    [string]$BuildDirectory = "build-fastllm-windows",
    [int]$Jobs = [Environment]::ProcessorCount,
    [string]$CMakePath = "",
    [string]$Python = "python",
    [string]$Generator = "Visual Studio 17 2022",
    [string[]]$CMakeArgs = @()
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($env:OS -ne "Windows_NT") {
    throw "make_whl.ps1 must be run on Windows."
}
if ($Jobs -lt 1) {
    throw "Jobs must be at least 1."
}

$RepoRoot = [IO.Path]::GetFullPath($PSScriptRoot)

. (Join-Path $RepoRoot "tools/windows/common.ps1")
if (-not $CudaArch) { $CudaArch = Get-FastllmCudaArch }

function Get-SafeBuildPath {
    param([Parameter(Mandatory = $true)][string]$RequestedPath)

    $fullPath = Full-Path $RequestedPath
    $repoPrefix = $RepoRoot.TrimEnd([IO.Path]::DirectorySeparatorChar,
                                    [IO.Path]::AltDirectorySeparatorChar) +
                  [IO.Path]::DirectorySeparatorChar
    if (-not $fullPath.StartsWith($repoPrefix,
            [StringComparison]::OrdinalIgnoreCase)) {
        throw "BuildDirectory must resolve to a child of the repository: $fullPath"
    }
    $relativePath = $fullPath.Substring($repoPrefix.Length)
    $firstSegment = $relativePath.Split(
        @([IO.Path]::DirectorySeparatorChar, [IO.Path]::AltDirectorySeparatorChar),
        [StringSplitOptions]::RemoveEmptyEntries)[0]
    if ($firstSegment -notmatch '^build(?:[-_].*)?$') {
        throw "BuildDirectory must be under a top-level build, build-*, or build_* directory: $fullPath"
    }
    # Remove-Item below is recursive. Reject junctions/symlinks anywhere below
    # the repository root so a lexically safe path cannot redirect elsewhere.
    $currentPath = $RepoRoot
    foreach ($segment in $relativePath.Split(
            @([IO.Path]::DirectorySeparatorChar,
              [IO.Path]::AltDirectorySeparatorChar),
            [StringSplitOptions]::RemoveEmptyEntries)) {
        $currentPath = Join-Path $currentPath $segment
        if (-not (Test-Path -LiteralPath $currentPath)) {
            break
        }
        $item = Get-Item -LiteralPath $currentPath -Force
        if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) {
            throw "BuildDirectory cannot pass through a reparse point: $currentPath"
        }
    }
    return $fullPath
}

function Test-NcclSdk {
    param([string]$Root)

    if (-not $Root) {
        return $false
    }
    $header = Join-Path $Root "include\nccl.h"
    $libraries = @(
        (Join-Path $Root "lib\nccl.lib"),
        (Join-Path $Root "lib64\nccl.lib"),
        (Join-Path $Root "lib\x64\nccl.lib")
    )
    if (-not (Test-Path -LiteralPath $header -PathType Leaf)) {
        return $false
    }
    return $null -ne ($libraries | Where-Object {
        Test-Path -LiteralPath $_ -PathType Leaf
    } | Select-Object -First 1)
}

function Invoke-CMakeBuild {
    param(
        [Parameter(Mandatory = $true)][string]$BuildPath,
        [Parameter(Mandatory = $true)][bool]$Cuda,
        [Parameter(Mandatory = $true)][string]$NcclSetting
    )

    $configureArgs = @(
        "-S", $RepoRoot,
        "-B", $BuildPath,
        "-G", $Generator
    )
    if ($Generator -match "^Visual Studio") {
        $configureArgs += @("-A", "x64")
    } else {
        $configureArgs += "-DCMAKE_BUILD_TYPE=Release"
    }
    $configureArgs += @(
        "-DMAKE_WHL_X86=ON",
        "-DUSE_NUMAS=OFF"
    )
    if ($Cuda) {
        $configureArgs += @(
            "-DUSE_CUDA=ON",
            "-DUSE_NCCL=$NcclSetting",
            "-DCUDA_ARCH=$CudaArch",
            "-DCMAKE_CUDA_ARCHITECTURES=$CudaArch"
        )
        if ($NcclRoot) {
            $configureArgs += "-DNCCL_ROOT=$([IO.Path]::GetFullPath($NcclRoot))"
        }
    } else {
        $configureArgs += @(
            "-DUSE_CUDA=OFF",
            "-DUSE_NCCL=OFF"
        )
    }
    $configureArgs += $CMakeArgs

    Run -Program $CMakeExe -Arguments $configureArgs
    Run -Program $CMakeExe -Arguments @(
        "--build", $BuildPath,
        "--config", "Release",
        "--target", "fastllm_python_package",
        "--parallel", "$Jobs"
    )

    $stagedDll = Join-Path $BuildPath "tools\ftllm\fastllm_tools.dll"
    if (-not (Test-Path -LiteralPath $stagedDll -PathType Leaf)) {
        throw "CMake completed but did not stage $stagedDll"
    }
    return $stagedDll
}

$CMakeExe = Get-FastllmCMake $CMakePath
$PythonExe = Resolve-Program -Name $Python

$pythonBits = & $PythonExe -c "import struct; print(struct.calcsize('P') * 8)"
if ($LASTEXITCODE -ne 0 -or "$pythonBits".Trim() -ne "64") {
    throw "A 64-bit Windows Python interpreter is required."
}
Run -Program $PythonExe -Arguments @(
    "-c", "import setuptools, wheel; print('Python wheel toolchain is available')"
)

if (-not $NcclRoot -and $env:NCCL_ROOT) {
    $NcclRoot = $env:NCCL_ROOT
}
$resolvedNccl = $UseNccl
if ($UseNccl -eq "Auto") {
    if (Test-NcclSdk -Root $NcclRoot) {
        $resolvedNccl = "ON"
    } else {
        $resolvedNccl = "OFF"
    }
}
if ($resolvedNccl -eq "ON" -and $NcclRoot -and
    -not (Test-NcclSdk -Root $NcclRoot)) {
    throw "NCCL_ROOT does not contain include\nccl.h and nccl.lib: $NcclRoot"
}

$BuildRoot = Get-SafeBuildPath -RequestedPath $BuildDirectory
if ((Test-Path -LiteralPath $BuildRoot) -and -not $Incremental) {
    Write-Host "Removing previous build directory: $BuildRoot"
    Remove-Item -LiteralPath $BuildRoot -Recurse -Force
}
New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null

Write-Host "Using CMake: $CMakeExe"
Write-Host "Using Python: $PythonExe"
Write-Host "Build root: $BuildRoot"
if (-not $CpuOnly) {
    Write-Host "CUDA architectures: $CudaArch"
    if ($resolvedNccl -eq "ON") {
        Write-Host "NCCL: enabled"
    } else {
        Write-Host "NCCL: unavailable/disabled; building the P2P + host-staged fallback"
    }
}

$CpuBuild = Join-Path $BuildRoot "cpu"
$CpuDll = Invoke-CMakeBuild -BuildPath $CpuBuild -Cuda $false -NcclSetting "OFF"

if ($CpuOnly) {
    $PackageRoot = Join-Path $CpuBuild "tools"
} else {
    $CudaBuild = Join-Path $BuildRoot "cuda"
    $null = Invoke-CMakeBuild -BuildPath $CudaBuild -Cuda $true -NcclSetting $resolvedNccl
    $PackageRoot = Join-Path $CudaBuild "tools"
    $CpuFallbackDll = Join-Path $PackageRoot "ftllm\fastllm_tools-cpu.dll"
    Copy-Item -LiteralPath $CpuDll -Destination $CpuFallbackDll -Force
}

$SetupPy = Join-Path $PackageRoot "setup.py"
if (-not (Test-Path -LiteralPath $SetupPy -PathType Leaf)) {
    throw "Packaging stage is incomplete: $SetupPy was not found."
}
# An incremental build may retain metadata from a previous release. setup.py
# intentionally trusts PKG-INFO when rebuilding an sdist, so remove only the
# generated staging metadata before packaging the current checkout.
foreach ($metadataName in @("PKG-INFO", "ftllm.egg-info", "ftllm_nightly.egg-info")) {
    $metadataPath = Get-SafeBuildPath -RequestedPath (Join-Path $PackageRoot $metadataName)
    if (Test-Path -LiteralPath $metadataPath) {
        Remove-Item -LiteralPath $metadataPath -Recurse -Force
    }
}

$oldNightly = [Environment]::GetEnvironmentVariable(
    "FASTLLM_NIGHTLY", [EnvironmentVariableTarget]::Process)
try {
    if ($Nightly) {
        $env:FASTLLM_NIGHTLY = "1"
    } else {
        $env:FASTLLM_NIGHTLY = "0"
    }
    Push-Location $PackageRoot
    try {
        Run -Program $PythonExe -Arguments @(
            "setup.py", "sdist", "build"
        )
        Run -Program $PythonExe -Arguments @(
            "setup.py", "bdist_wheel", "--plat-name", "win_amd64"
        )
    } finally {
        Pop-Location
    }
} finally {
    if ($null -eq $oldNightly) {
        Remove-Item Env:FASTLLM_NIGHTLY -ErrorAction SilentlyContinue
    } else {
        $env:FASTLLM_NIGHTLY = $oldNightly
    }
}

$DistDirectory = Join-Path $PackageRoot "dist"
$Wheel = Get-ChildItem -LiteralPath $DistDirectory -Filter "*.whl" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -eq $Wheel -or $Wheel.Name -notmatch "-win_amd64\.whl$") {
    throw "The expected win_amd64 wheel was not produced in $DistDirectory"
}

$backend = if ($CpuOnly) { "cpu" } else { "cuda" }
Run $PythonExe @("-I", "-B", "-X", "utf8", (Join-Path $RepoRoot "tools/windows/verify_wheel.py"),
    $Wheel.FullName, "--backend", $backend, "--source-root", $RepoRoot, "--native-root", $BuildRoot)

Write-Host ""
Write-Host "Windows wheel created and verified:"
Write-Host "  $($Wheel.FullName)"
$sourceArchive = Get-ChildItem -LiteralPath $DistDirectory -Filter "*.tar.gz" |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($null -ne $sourceArchive) {
    Write-Host "Source archive:"
    Write-Host "  $($sourceArchive.FullName)"
}
