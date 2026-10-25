param([string]$ProjectRoot = '.')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath ([IO.Path]::GetFullPath($ProjectRoot))
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$run = 'experiments/results/algorithms/research_full/20261024T181421Z_96b32502d9344be6b2d14bd79f189831'
$state = Get-Content -Raw -Encoding UTF8 -LiteralPath "$run/raw/state.json" | ConvertFrom-Json
$jobs = @($state.jobs | Where-Object { $_.id -eq 'sd_cold_evaluation' })
if ($jobs.Count -ne 1 -or -not $jobs[0].result) { throw 'Expected one recorded SD Evaluation attempt.' }
$source = Join-Path (Join-Path $run $jobs[0].result) 'raw/benchmark'
if (-not (Test-Path -LiteralPath "$source/results.jsonl" -PathType Leaf)) { throw "Missing source results: $source" }
# Reuse the exact effective config recorded by that failed benchmark. The Python
# diagnostic also compares it to run.json; no settings are guessed or changed.
$output = 'experiments/results/algorithms/sd_logits_' + [guid]::NewGuid().ToString('N')
$settings = Get-Content -Raw -Encoding UTF8 -LiteralPath 'experiments/models/provisional_full_20261025_021058_666b6611/run_all.provisional.config.json' | ConvertFrom-Json
if (-not $settings.configs.sd) { throw 'Missing original SD config path.' }
& python -m experiments.algorithms.speculative_decoding.diagnose_logits --project-root . `
    --config $settings.configs.sd --source-run $source --run-dir $output
if ($LASTEXITCODE -ne 0) { throw "Logit diagnostic incomplete; inspect $output/failed.json. Original run unchanged." }
Write-Output "Inspect $output/diagnostic_summary.json and the per-mode Target logits. No resume started."
