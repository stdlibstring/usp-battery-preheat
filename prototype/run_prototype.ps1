param(
    [string]$Route = "$PSScriptRoot\example_route_synthetic.csv",
    [double]$InitialTemperatureC = 0.0,
    [double]$InitialSocPercent = 65.0
)

$ErrorActionPreference = "Stop"
$buildDir = Join-Path $PSScriptRoot "build"
$outputDir = Join-Path $PSScriptRoot "output"
$exe = Join-Path $buildDir "preheat_prototype.exe"
New-Item -ItemType Directory -Force -Path $buildDir, $outputDir | Out-Null

$gcc = (Get-Command gcc -ErrorAction Stop).Source
& $gcc -Wall -Wextra -O2 -std=c11 (Join-Path $PSScriptRoot "preheat_prototype.c") -lm -o $exe
if ($LASTEXITCODE -ne 0) { throw "C prototype compilation failed." }

& $exe $Route $outputDir $InitialTemperatureC $InitialSocPercent
if ($LASTEXITCODE -ne 0) { throw "Simulation or optimization failed." }

$pythonCandidates = @(
    (Join-Path $env:USERPROFILE ".cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"),
    ((Get-Command python -ErrorAction SilentlyContinue).Source)
) | Where-Object { $_ -and (Test-Path -LiteralPath $_) }

$plotPython = $null
foreach ($candidate in $pythonCandidates) {
    & $candidate -c "from PIL import Image" 2>$null
    if ($LASTEXITCODE -eq 0) {
        $plotPython = $candidate
        break
    }
}
if (-not $plotPython) {
    throw "No Python environment with Pillow was found. The C results and CSV trajectory were still generated."
}

& $plotPython (Join-Path $PSScriptRoot "plot_results.py") $outputDir
if ($LASTEXITCODE -ne 0) { throw "Plot generation failed. Ensure Pillow is installed." }

Write-Host "Results: $outputDir"
