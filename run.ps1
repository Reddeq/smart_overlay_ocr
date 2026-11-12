param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArguments)
$ErrorActionPreference = 'Stop'
$projectPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $projectPython)) {
    throw 'Create .venv and install requirements first. See README.md.'
}
if (-not $env:PADDLE_PDX_CACHE_HOME) {
    $env:PADDLE_PDX_CACHE_HOME = Join-Path $PSScriptRoot '.cache\paddlex'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $projectPython -m smart_overlay @AppArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
