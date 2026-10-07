param([switch]$NoBrowser)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$logs = Join-Path $root 'temp\launcher'
New-Item -ItemType Directory -Path $logs -Force | Out-Null

function PortOwner([int]$port) {
    @(Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1)
}
function HttpGet([string]$url) {
    try { return Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2 } catch { return $null }
}
function IsBackend {
    $r = HttpGet 'http://127.0.0.1:8000/api/health'
    if (-not $r) { return $false }
    try { return (($r.Content | ConvertFrom-Json).status -eq 'ok') } catch { return $false }
}
function IsLama {
    $r = HttpGet 'http://127.0.0.1:8080/api/v1/model'
    if (-not $r) { return $false }
    try { return (($r.Content | ConvertFrom-Json).name -eq 'lama') } catch { return $false }
}
function IsFrontend([int]$port) {
    $r = HttpGet "http://127.0.0.1:$port/"
    return ($null -ne $r -and $r.Content -match '<title>Image2EditablePPT</title>')
}
function WaitHealthy([string]$name, [scriptblock]$check, [int]$seconds, [System.Diagnostics.Process]$process) {
    $until = (Get-Date).AddSeconds($seconds)
    do {
        if (& $check) { Write-Host "[OK] $name is healthy."; return }
        if ($process.HasExited) { throw "$name exited before becoming healthy. See $logs for its error log." }
        Start-Sleep -Seconds 2
    } while ((Get-Date) -lt $until)
    throw "$name did not become healthy within $seconds seconds. See $logs for its error log."
}
function Launch([string]$name, [string]$exe, [string[]]$arguments, [string]$directory) {
    $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
    $stdout = Join-Path $logs "$name-$stamp.out.log"
    $stderr = Join-Path $logs "$name-$stamp.err.log"
    Write-Host "[START] $name"
    return Start-Process -FilePath $exe -ArgumentList $arguments -WorkingDirectory $directory -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru
}
function FindPython {
    $venv = Join-Path $env:LOCALAPPDATA 'Image2EditablePPT\venv\Scripts\python.exe'
    if (Test-Path $venv) { return $venv }
    $cmd = Get-Command python.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    $condaPython = 'D:\ProgramData\miniconda3\python.exe'
    if (Test-Path $condaPython) { return $condaPython }
    throw 'Backend Python not found. Run scripts\setup.ps1 first.'
}
function FindNode {
    $cmd = Get-Command node.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($cmd) { return $cmd.Source }
    foreach ($candidate in @("$env:ProgramFiles\nodejs\node.exe", "$env:LOCALAPPDATA\Programs\nodejs\node.exe", "$env:USERPROFILE\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe")) {
        if (Test-Path $candidate) { return $candidate }
    }
    throw 'Node.js not found. Install Node.js or run scripts\setup.ps1.'
}
function FindIopaint {
    $candidates = @()
    $conda = Get-Command conda.exe -ErrorAction SilentlyContinue | Select-Object -First 1
    if ($conda) {
        try {
            $envList = & $conda.Source env list --json | ConvertFrom-Json
            $candidates += @($envList.envs | Where-Object { (Split-Path $_ -Leaf) -eq 'lama-inpaint' } | ForEach-Object { Join-Path $_ 'Scripts\iopaint.exe' })
        } catch { }
    }
    $candidates += @(
        (Join-Path $env:USERPROFILE 'miniconda3\envs\lama-inpaint\Scripts\iopaint.exe'),
        (Join-Path $env:LOCALAPPDATA 'miniconda3\envs\lama-inpaint\Scripts\iopaint.exe'),
        'D:\ProgramData\miniconda3\envs\lama-inpaint\Scripts\iopaint.exe'
    )
    foreach ($candidate in $candidates) { if (Test-Path $candidate) { return $candidate } }
    throw 'LaMa conda environment lama-inpaint or its iopaint.exe was not found.'
}

try {
    Write-Host "Image2EditablePPT one-click startup: $root"
    $samWeights = Join-Path $env:LOCALAPPDATA 'Image2EditablePPT\models\sam2.1_t.pt'
    if (-not $env:SAM_MODEL_PATH -and (Test-Path -LiteralPath $samWeights)) { $env:SAM_MODEL_PATH = $samWeights }
    if (PortOwner 8080) {
        if (-not (IsLama)) { Write-Warning 'Optional LaMa port 8080 is occupied by another service; continuing with OpenCV fallback.' }
        else { Write-Host '[SKIP] LaMa is already running on 8080.' }
    } else {
        try {
            $iopaint = FindIopaint
            $p = Launch 'lama' $iopaint @('start','--model=lama','--device=cpu','--port=8080') $root
            WaitHealthy 'LaMa' ${function:IsLama} 30 $p
        } catch { Write-Warning 'LaMa is unavailable; continue with protected local OpenCV fallback.' }
    }
    if (PortOwner 8000) {
        if (-not (IsBackend)) { throw 'Backend port 8000 is occupied, but /api/health is not healthy.' }
        Write-Host '[SKIP] Backend is already running on 8000.'
    } else {
        $python = FindPython
        $backend = Join-Path $root 'backend'
        $p = Launch 'backend' $python @('-m','uvicorn','app.main:app','--host','127.0.0.1','--port','8000') $backend
        WaitHealthy 'Backend' ${function:IsBackend} 60 $p
    }
    $frontendPort = 0
    foreach ($port in @(5173,5174)) {
        if ((PortOwner $port) -and (IsFrontend $port)) { $frontendPort = $port; break }
    }
    if ($frontendPort) {
        Write-Host "[SKIP] Frontend is already running on $frontendPort."
    } else {
        if (PortOwner 5173) { throw 'Frontend port 5173 is occupied by another service.' }
        $frontend = Join-Path $root 'frontend'
        $vite = Join-Path $frontend 'node_modules\vite\bin\vite.js'
        if (-not (Test-Path $vite)) {
            . (Join-Path $PSScriptRoot 'frontend-runtime.ps1')
            $frontend = Sync-FrontendRuntime $root
            $vite = Join-Path $frontend 'node_modules\vite\bin\vite.js'
        }
        if (-not (Test-Path $vite)) { throw 'Frontend dependencies are missing. Run scripts\setup.ps1 first.' }
        $node = FindNode
        $frontendPort = 5173
        $p = Launch 'frontend' $node @($vite,'--host','127.0.0.1','--port','5173','--strictPort') $frontend
        WaitHealthy 'Frontend' { IsFrontend 5173 } 60 $p
    }
    if (-not (IsBackend) -or -not (IsFrontend $frontendPort)) { throw 'A required service failed the final health check.' }
    $url = "http://127.0.0.1:$frontendPort/"
    Write-Host "[READY] $url"
    if (-not $NoBrowser) { Start-Process $url }
    exit 0
} catch {
    Write-Host "[FAILED] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
