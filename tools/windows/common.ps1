# Shared helpers; callers set RepoRoot before dot-sourcing this file.
function Full-Path([string]$Path) {
    if ([IO.Path]::IsPathRooted($Path)) { return [IO.Path]::GetFullPath($Path) }
    return [IO.Path]::GetFullPath((Join-Path $RepoRoot $Path))
}

function Run([string]$Program, [string[]]$Arguments) {
    Write-Host "> $Program $($Arguments -join ' ')"
    & $Program @Arguments | Out-Host
    if ($LASTEXITCODE -ne 0) { throw "Command failed ($LASTEXITCODE): $Program" }
}

function Resolve-Program([string]$Name) {
    if ([IO.Path]::IsPathRooted($Name) -and (Test-Path -LiteralPath $Name -PathType Leaf)) {
        return [IO.Path]::GetFullPath($Name)
    }
    $command = Get-Command $Name -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    throw "Unable to find executable '$Name'."
}

function Get-FastllmCMake([string]$Requested = "") {
    if ($Requested) { return Resolve-Program $Requested }
    $command = Get-Command cmake.exe -ErrorAction SilentlyContinue
    if ($command) { return $command.Source }
    $vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio/Installer/vswhere.exe"
    if (Test-Path -LiteralPath $vswhere) {
        $vsRoot = & $vswhere -latest -products '*' -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($LASTEXITCODE -eq 0 -and $vsRoot) {
            return Resolve-Program (Join-Path $vsRoot "Common7/IDE/CommonExtensions/Microsoft/CMake/CMake/bin/cmake.exe")
        }
    }
    throw "CMake/Visual Studio C++ toolchain not found. Use -CMakePath to specify cmake.exe."
}

function Get-FastllmCudaArch {
    # Keep the Windows release defaults aligned with the Linux wheel script.
    $source = Get-Content -LiteralPath (Join-Path $RepoRoot "make_whl.sh") -Raw -Encoding UTF8
    $match = [regex]::Match($source, '(?m)^CUDA_ARCH_LIST="([^"]+)"')
    if (-not $match.Success) { throw "Cannot read CUDA_ARCH_LIST from make_whl.sh." }
    return $match.Groups[1].Value
}

function Get-FastllmRuntimeLock {
    return Get-Content -LiteralPath (Join-Path $RepoRoot "portable/windows/runtime-lock.json") -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-FastllmAsset([string]$Name, [string]$Cache, [switch]$Offline) {
    $asset = (Get-FastllmRuntimeLock).$Name
    $destination = Join-Path $Cache $asset.filename
    New-Item -ItemType Directory -Force -Path $Cache | Out-Null
    if (Test-Path -LiteralPath $destination) {
        if ((Get-FileHash -LiteralPath $destination -Algorithm SHA256).Hash -ne $asset.sha256) {
            throw "Cached download has an invalid SHA256: $destination"
        }
        return $destination
    }
    if ($Offline) { throw "Offline cache missing: $destination" }
    Run "curl.exe" @("--fail", "--location", "--retry", "3", "--output", "$destination.part", $asset.url)
    if ((Get-FileHash -LiteralPath "$destination.part" -Algorithm SHA256).Hash -ne $asset.sha256) {
        throw "Download SHA256 mismatch: $($asset.url)"
    }
    Move-Item -LiteralPath "$destination.part" -Destination $destination
    return $destination
}

function Initialize-FastllmPython([string]$Cache, [switch]$Offline) {
    $archive = Get-FastllmAsset "python" $Cache -Offline:$Offline
    $bootstrap = Join-Path $Cache "bootstrap"
    $python = Join-Path $bootstrap "python/python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        New-Item -ItemType Directory -Force -Path $bootstrap | Out-Null
        Run "tar.exe" @("-xzf", $archive, "-C", $bootstrap)
    }
    $wheelhouse = Join-Path $Cache "wheelhouse-cp311-win_amd64"
    New-Item -ItemType Directory -Force -Path $wheelhouse | Out-Null
    if (-not $Offline) {
        Run $python @("-I", "-m", "pip", "download", "--only-binary=:all:", "--dest", $wheelhouse,
            "--find-links", $wheelhouse, "wheel", "setuptools")
    }
    Run $python @("-I", "-m", "pip", "install", "--no-index", "--find-links", $wheelhouse, "wheel", "setuptools")
    return $python
}
