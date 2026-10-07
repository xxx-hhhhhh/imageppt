param([Parameter(Mandatory=$true)][string]$ProjectDirectory,[string]$PptFile='editable.pptx')
$ErrorActionPreference = 'Stop'
$project = (Resolve-Path -LiteralPath $ProjectDirectory).Path
$pptPath = Join-Path $project $PptFile
$blankPath = Join-Path $project 'text_removed_test\editable.pptx'
if (-not (Test-Path -LiteralPath $pptPath)) { throw 'PPTX missing' }
$app = New-Object -ComObject PowerPoint.Application
# Never quit PowerPoint globally: another presentation may belong to the user.
foreach ($pair in @(@($pptPath,'powerpoint_native.png'),@($blankPath,'powerpoint_text_removed.png'))) {
    if (-not (Test-Path -LiteralPath $pair[0])) { continue }
    $deck = $null
    try {
        $deck = $app.Presentations.Open($pair[0],-1,0,0)
        $deck.Slides.Item(1).Export((Join-Path $project $pair[1]),'PNG',1672,941)
        Write-Output ('Read-only rendered: ' + $pair[1])
    } finally {
        if ($null -ne $deck) { $deck.Close() }
    }
}
