$ErrorActionPreference = 'Stop'
function Healthy([string]$url, [string]$marker) {
    try { return ((Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 2).Content -match $marker) } catch { return $false }
}
function StopService([string]$name, [int]$port, [string]$url, [string]$marker) {
    $listener = Get-NetTCPConnection -LocalPort $port -State Listen -ErrorAction SilentlyContinue | Select-Object -First 1
    if (-not $listener) { Write-Host "[SKIP] $name is not running."; return }
    if (-not (Healthy $url $marker)) { Write-Host "[WARN] $name port $port belongs to an unrecognized service; it was left running."; return }
    try {
        Stop-Process -Id $listener.OwningProcess -ErrorAction Stop
        Write-Host "[STOP] $name on $port (PID $($listener.OwningProcess))."
    } catch {
        Write-Host "[WARN] Cannot stop $name on $port (PID $($listener.OwningProcess)): $($_.Exception.Message). Close its original terminal or run stop.bat as Administrator." -ForegroundColor Yellow
        $script:stopFailed = $true
    }
}
try {
    StopService 'Frontend' 5173 'http://127.0.0.1:5173/' '<title>Image2EditablePPT</title>'
    StopService 'Frontend' 5174 'http://127.0.0.1:5174/' '<title>Image2EditablePPT</title>'
    StopService 'Backend' 8000 'http://127.0.0.1:8000/api/health' '"status"\s*:\s*"ok"'
    StopService 'LaMa' 8080 'http://127.0.0.1:8080/api/v1/model' '"name"\s*:\s*"lama"'
    if ($script:stopFailed) { exit 1 }
    exit 0
} catch {
    Write-Host "[FAILED] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
