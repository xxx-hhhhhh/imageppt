function Sync-FrontendRuntime([string]$projectRoot) {
    # Drive filesystems may not support junctions. Keep both dependencies and
    # the Vite execution copy outside Drive; the project remains the source.
    $source = Join-Path $projectRoot 'frontend'
    $runtime = Join-Path $env:LOCALAPPDATA 'Image2EditablePPT\frontend-dependencies'
    New-Item -ItemType Directory -Path $runtime -Force | Out-Null
    foreach ($name in @('package.json', 'package-lock.json', 'pnpm-lock.yaml', 'pnpm-workspace.yaml', 'index.html', 'tsconfig.json', 'vite.config.ts')) {
        $file = Join-Path $source $name
        if (Test-Path -LiteralPath $file) { Copy-Item -LiteralPath $file -Destination (Join-Path $runtime $name) -Force }
    }
    Copy-Item -LiteralPath (Join-Path $source 'src') -Destination $runtime -Recurse -Force
    return $runtime
}
