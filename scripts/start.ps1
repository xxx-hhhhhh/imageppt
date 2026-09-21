$projectRoot = Split-Path -Parent $PSScriptRoot
$runtimeRoot = Join-Path $env:LOCALAPPDATA 'Image2EditablePPT'
$venvPython = Join-Path $runtimeRoot 'venv\Scripts\python.exe'
if (-not (Test-Path $venvPython)) { throw '请先运行 .\scripts\setup.ps1' }
$node = Get-Command node -ErrorAction SilentlyContinue
$packageManager = Get-Command pnpm -ErrorAction SilentlyContinue
if (-not $packageManager) { $packageManager = Get-Command npm -ErrorAction SilentlyContinue }
if (-not $node -and -not $packageManager) { throw '未找到 Node.js、pnpm 或 npm。' }
Start-Process powershell -WorkingDirectory $projectRoot -ArgumentList '-NoExit','-Command',"& '$venvPython' -m uvicorn app.main:app --app-dir '$projectRoot\backend' --host 127.0.0.1 --port 8000"
$frontendDir = Join-Path $projectRoot 'frontend'
$viteEntry = Join-Path $frontendDir 'node_modules\vite\bin\vite.js'
if ($node -and (Test-Path $viteEntry)) { $frontendCommand = "& '$($node.Source)' '$viteEntry' --host 127.0.0.1 --port 5173" } else { $frontendCommand = "& '$($packageManager.Source)' run dev -- --host 127.0.0.1" }
Start-Process powershell -WorkingDirectory $frontendDir -ArgumentList '-NoExit','-Command',$frontendCommand
Start-Process 'http://127.0.0.1:5173'
