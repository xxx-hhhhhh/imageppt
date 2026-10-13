param(
    [Parameter(Mandatory=$true)][string]$ProjectPath
)
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectPath).Path
$layout = Get-Content -LiteralPath (Join-Path $project 'slides\page_1.json') -Raw -Encoding UTF8 | ConvertFrom-Json
$pptx = Join-Path $project 'editable.pptx'
if (-not (Test-Path -LiteralPath $pptx)) { throw 'PPTX not found' }
# PowerPoint COM can fail with deep output paths; use a short unique temporary
# filename, then copy our generated render back. Do not quit the user's app.
$render = Join-Path $env:TEMP ('i2e-' + [guid]::NewGuid().ToString('N') + '.png')
$application = New-Object -ComObject PowerPoint.Application
$deck = $application.Presentations.Open($pptx,-1,0,0)
try {
    $deck.Slides.Item(1).Export($render,'PNG',[int]$layout.slide.width,[int]$layout.slide.height)
    $destination = Join-Path $project 'powerpoint_native.png'
    Copy-Item -LiteralPath $render -Destination $destination
    Write-Output $destination
} finally { $deck.Close() }
