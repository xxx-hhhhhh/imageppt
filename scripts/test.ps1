$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $env:LOCALAPPDATA 'Image2EditablePPT'
$venvPython = Join-Path $runtimeRoot 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) { $venvPython = (Get-Command python).Source }
$env:PYTHONPATH = Join-Path $projectRoot 'backend'
& $venvPython -m pytest (Join-Path $projectRoot 'backend\tests') -q
$node = Get-Command node -ErrorAction SilentlyContinue
$packageManager = Get-Command pnpm -ErrorAction SilentlyContinue
if (-not $packageManager) { $packageManager = Get-Command npm -ErrorAction SilentlyContinue }
$frontendDir = Join-Path $projectRoot 'frontend'
$tscEntry = Join-Path $frontendDir 'node_modules\typescript\bin\tsc'
$viteEntry = Join-Path $frontendDir 'node_modules\vite\bin\vite.js'
if ($node -and (Test-Path $tscEntry) -and (Test-Path $viteEntry)) {
  Push-Location $frontendDir
  try { & $node.Source $tscEntry --noEmit; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; & $node.Source $viteEntry build; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } finally { Pop-Location }
} else {
  if (-not $packageManager) { throw '未找到 pnpm 或 npm。' }
  Push-Location $frontendDir
  try { & $packageManager.Source run build } finally { Pop-Location }
}
