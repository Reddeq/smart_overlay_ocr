param([Parameter(ValueFromRemainingArguments = $true)][string[]]$AppArguments)
$ErrorActionPreference = 'Stop'
$python = Join-Path $PSScriptRoot '.venv-vl\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python)) {
    throw 'Local PaddleOCR-VL environment missing. Follow docs/local-vl.md before using run-vl.ps1.'
}
if (-not $env:PADDLE_PDX_CACHE_HOME) {
    $env:PADDLE_PDX_CACHE_HOME = Join-Path $PSScriptRoot '.cache\paddlex'
}
Push-Location -LiteralPath $PSScriptRoot
try {
    & $python -m smart_overlay @AppArguments
    exit $LASTEXITCODE
}
finally {
    Pop-Location
}
