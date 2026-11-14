param(
    [string]$Bucket = "devilson/fastllm-backup",
    [string]$Prefix = "fastllm-cu126-image",
    [string]$Destination = "C:\fastllm-cu126-image"
)

$ErrorActionPreference = "Stop"
$ariaCommand = Get-Command aria2c -ErrorAction Stop
if ($Bucket -notmatch '^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$') {
    throw "Invalid bucket ID. Expected owner/bucket."
}
if ($Prefix -notmatch '^[A-Za-z0-9_./+-]+$' -or $Prefix.Split('/') -contains '..') {
    throw "Invalid bucket directory."
}
$Prefix = $Prefix.Trim('/')
$Destination = [IO.Path]::GetFullPath($Destination)
$previousOutputEncoding = $OutputEncoding
$OutputEncoding = [Text.UTF8Encoding]::new($false)
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Get-HfSignedUrl([string]$RelativePath) {
    $encoded = ($RelativePath.Split('/') | ForEach-Object { [Uri]::EscapeDataString($_) }) -join '/'
    $url = "https://huggingface.co/buckets/$Bucket/resolve/$encoded"
    $request = [Net.HttpWebRequest][Net.WebRequest]::Create($url)
    $request.Method = "HEAD"
    $request.AllowAutoRedirect = $false
    $request.Timeout = 60000
    $request.Headers["Authorization"] = "Bearer $hfToken"

    try {
        $response = $request.GetResponse()
    } catch {
        throw "Cannot access $RelativePath. Check the HF token, bucket read permission, and uploaded directory."
    }
    try {
        $status = [int]$response.StatusCode
        $location = $response.Headers["Location"]
    } finally {
        $response.Close()
    }
    if ($status -notin @(301, 302, 303, 307, 308) -or [string]::IsNullOrWhiteSpace($location)) {
        throw "HF did not return a signed download URL for $RelativePath (HTTP $status)."
    }
    $signed = [Uri]::new([Uri]$url, $location)
    if ($signed.Scheme -ne "https" -or $signed.AbsoluteUri -match '[\r\n]') {
        throw "Invalid HF download redirect."
    }
    return $signed.AbsoluteUri
}

function Invoke-AriaDownload([string[]]$Entries) {
    $Entries -join "`n" | & $ariaCommand.Source --input-file=- -c -x 16 -s 16 -j 2 -k 1M `
        --file-allocation=none --auto-file-renaming=false --allow-overwrite=true `
        --check-integrity=true --max-tries=5 --retry-wait=3 --console-log-level=warn
    if ($LASTEXITCODE -ne 0) {
        throw "Download incomplete. Keep existing files and .aria2 files, then rerun this script."
    }
}

$secret = Read-Host "HF Token with read access to this bucket (hidden input)" -AsSecureString
$credential = [Management.Automation.PSCredential]::new("hf", $secret)
$hfToken = $credential.GetNetworkCredential().Password.Trim()
if ([string]::IsNullOrWhiteSpace($hfToken)) { throw "HF token is empty." }
$scratch = Join-Path ([IO.Path]::GetTempPath()) ("fastllm-hf-" + [Guid]::NewGuid().ToString("N"))

try {
    New-Item -ItemType Directory -Force $scratch | Out-Null
    $sumUrl = Get-HfSignedUrl "$Prefix/SHA256SUMS"
    Invoke-AriaDownload @($sumUrl, "  dir=$scratch", "  out=SHA256SUMS")

    $records = @()
    $seen = [Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    foreach ($line in [IO.File]::ReadAllLines((Join-Path $scratch "SHA256SUMS"))) {
        if ($line -notmatch '^([a-fA-F0-9]{64})  (.+)$') {
            throw "Invalid SHA256SUMS line."
        }
        $digest = $Matches[1].ToLowerInvariant()
        $relative = $Matches[2]
        $rootFiles = @("Dockerfile", "export_env.py", "README.md", "manifest.json", "packages.lock", `
                       "system-packages.txt", "python-env.tar.gz", ".dockerignore")
        if ($relative -cnotin $rootFiles -and $relative -cnotmatch '^debs/[A-Za-z0-9][A-Za-z0-9_.+~-]*\.deb$') {
            throw "Unexpected export file in SHA256SUMS: $relative"
        }
        if (-not $seen.Add($relative)) { throw "Duplicate export file: $relative" }
        $records += [PSCustomObject]@{ Path = $relative; Hash = $digest }
    }
    foreach ($required in @("Dockerfile", "export_env.py", "manifest.json", "packages.lock", `
                            "python-env.tar.gz", ".dockerignore")) {
        if (-not $seen.Contains($required)) { throw "Missing required export file: $required" }
    }

    New-Item -ItemType Directory -Force $Destination | Out-Null
    $entries = @()
    foreach ($record in $records) {
        $target = Join-Path $Destination $record.Path
        if ((Test-Path -LiteralPath $target -PathType Leaf) -and `
            (Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -eq $record.Hash) {
            Write-Host "Verified cache: $($record.Path)"
            continue
        }
        $directory = [IO.Path]::GetDirectoryName($target)
        New-Item -ItemType Directory -Force $directory | Out-Null
        $entries += Get-HfSignedUrl "$Prefix/$($record.Path)"
        $entries += "  dir=$directory"
        $entries += "  out=$([IO.Path]::GetFileName($target))"
        $entries += "  checksum=sha-256=$($record.Hash)"
    }
    if ($entries.Count -gt 0) { Invoke-AriaDownload $entries }

    foreach ($record in $records) {
        $target = Join-Path $Destination $record.Path
        if ((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash -ne $record.Hash) {
            throw "SHA256 mismatch: $($record.Path)"
        }
    }
    Copy-Item -LiteralPath (Join-Path $scratch "SHA256SUMS") -Destination $Destination -Force
    Write-Host "Verified $($records.Count) files. Docker build directory: $Destination"
    Write-Host "Keep python-env.tar.gz compressed. It will be restored inside the Docker build."
} finally {
    $hfToken = $null
    $credential = $null
    $secret.Dispose()
    $OutputEncoding = $previousOutputEncoding
    if (Test-Path -LiteralPath $scratch) { Remove-Item -LiteralPath $scratch -Recurse -Force }
}
