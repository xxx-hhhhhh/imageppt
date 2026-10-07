$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $env:LOCALAPPDATA 'Image2EditablePPT'
$venvPython = Join-Path $runtimeRoot 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) { $venvPython = (Get-Command python).Source }
$env:PYTHONPATH = Join-Path $projectRoot 'backend'
& $venvPython -m pytest (Join-Path $projectRoot 'backend\tests') -q
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
$node = Get-Command node -ErrorAction SilentlyContinue
$packageManager = Get-Command pnpm -ErrorAction SilentlyContinue
if (-not $packageManager) { $packageManager = Get-Command npm -ErrorAction SilentlyContinue }
$frontendDir = Join-Path $projectRoot 'frontend'
if (-not (Test-Path (Join-Path $frontendDir 'node_modules\vite\bin\vite.js'))) {
  . (Join-Path $PSScriptRoot 'frontend-runtime.ps1')
  $frontendDir = Sync-FrontendRuntime $projectRoot
}
$tscEntry = Join-Path $frontendDir 'node_modules\typescript\bin\tsc'
$viteEntry = Join-Path $frontendDir 'node_modules\vite\bin\vite.js'
if ($node -and (Test-Path $tscEntry) -and (Test-Path $viteEntry)) {
  Push-Location $frontendDir
  try { & $node.Source $tscEntry --noEmit; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }; & $node.Source $viteEntry build; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } finally { Pop-Location }
} else {
  if (-not $packageManager) { throw '未找到 pnpm 或 npm。' }
  Push-Location $frontendDir
  try { & $packageManager.Source run build; if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE } } finally { Pop-Location }
}
