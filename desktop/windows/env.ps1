# Scope all changes to this terminal; no system PATH or registry modifications.
$support = $PSScriptRoot
$bundle = Split-Path -Parent $support
$env:PATH = "$bundle;$support;$support\runtime;$support\tools;$env:PATH"
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONDONTWRITEBYTECODE = '1'
Write-Host 'FastLLM portable environment ready. Run ftllm --help; exit closes this terminal.'
