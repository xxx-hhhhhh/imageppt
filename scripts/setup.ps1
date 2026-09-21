$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $env:LOCALAPPDATA 'Image2EditablePPT'
$venvPath = Join-Path $runtimeRoot 'venv'
$pyLauncher = Get-Command py -ErrorAction SilentlyContinue
$python = (Get-Command python -ErrorAction SilentlyContinue).Source
if (-not $pyLauncher -and -not $python) { throw '未找到 Python。请先安装 Python 3.10+ 并加入 PATH。' }
New-Item -ItemType Directory -Force -Path $runtimeRoot | Out-Null
if (-not (Test-Path (Join-Path $venvPath 'Scripts\python.exe'))) {
  if ($pyLauncher) { & $pyLauncher.Source -3 -m venv $venvPath } else { & $python -m venv $venvPath }
}
$venvPython = Join-Path $venvPath 'Scripts\python.exe'
& $venvPython -m pip install --upgrade pip
& $venvPython -m pip install -r (Join-Path $projectRoot 'backend\requirements.txt')
try { & $venvPython -m pip install -r (Join-Path $projectRoot 'backend\requirements-optional.txt') } catch { Write-Warning "可选 PaddleOCR/LAMA 安装失败，应用仍会使用 RapidOCR/OpenCV fallback。" }
$packageManager = Get-Command pnpm -ErrorAction SilentlyContinue
if (-not $packageManager) { $packageManager = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $packageManager) { throw '未找到 pnpm 或 npm。请先安装 Node.js。' }
Push-Location (Join-Path $projectRoot 'frontend')
try { & $packageManager.Source install } finally { Pop-Location }
Write-Host "安装完成。Python 环境位于 $venvPath，未写入项目目录。"
