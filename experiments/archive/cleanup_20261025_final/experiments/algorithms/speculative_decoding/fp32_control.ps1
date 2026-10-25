param([string]$ProjectRoot = '.')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath ([IO.Path]::GetFullPath($ProjectRoot))
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$run = 'experiments/results/algorithms/research_full/20261024T181421Z_96b32502d9344be6b2d14bd79f189831'
$state = Get-Content -Raw -Encoding UTF8 -LiteralPath "$run/raw/state.json" | ConvertFrom-Json
$jobs = @($state.jobs | Where-Object { $_.id -eq 'sd_cold_evaluation' })
if ($jobs.Count -ne 1 -or -not $jobs[0].result) { throw 'Original SD Evaluation evidence not found.' }
$attempt = Join-Path $run $jobs[0].result
$source = Join-Path $attempt 'raw/benchmark'
$output = 'experiments/results/algorithms/sd_fp32_single_' + [guid]::NewGuid().ToString('N')
& python -m experiments.algorithms.speculative_decoding.fp32_control --project-root . `
    --source-run $source --run-dir $output --repeats 2
if ($LASTEXITCODE -ne 0) {
    throw "FP32 single-query failed/unverified or insufficient memory. Inspect $output; no fallback, full Evaluation NOT started."
}
$summary = Get-Content -Raw -Encoding UTF8 -LiteralPath "$output/summary.json" | ConvertFrom-Json
if ($summary.status -ne 'passed_single_query') { throw 'No verified single-query pass; stop.' }
$scenarios = Join-Path $attempt 'raw/adapter/scenarios.jsonl'
$full = 'experiments/results/algorithms/sd_fp32_evaluation_' + [guid]::NewGuid().ToString('N')
Write-Output "Single query passed only. Original FP16 Run remains unchanged. Full Evaluation is NOT started automatically."
Write-Output "python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config `"$output/fp32.config.json`" --scenarios `"$scenarios`" --mode all --fp32-control --allow-unreviewed --run-dir `"$full`""
$lookup = 'experiments/results/algorithms/sd_fp32_lookup_evaluation_' + [guid]::NewGuid().ToString('N')
Write-Output 'Optional Prompt Lookup is a separate paired substudy; legacy mode all excludes it:'
Write-Output "python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config `"$output/fp32.config.json`" --scenarios `"$scenarios`" --mode sd2_4 --fp32-control --allow-unreviewed --run-dir `"$lookup`""
