param([string]$ProjectRoot = '.')
$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath ([IO.Path]::GetFullPath($ProjectRoot))
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$run = 'experiments/results/algorithms/research_full/20261024T181421Z_96b32502d9344be6b2d14bd79f189831'
$allConfig = 'experiments/models/provisional_full_20261025_021058_666b6611/run_all.provisional.config.json'
$state = Get-Content -Raw -Encoding UTF8 -LiteralPath "$run/raw/state.json" | ConvertFrom-Json
$job = @($state.jobs | Where-Object { $_.id -eq 'sd_cold_evaluation' })
if ($job.Count -ne 1 -or -not $job[0].result) { throw 'Expected one recorded SD Evaluation attempt.' }
$attempt = Join-Path $run $job[0].result
$scenarios = Join-Path $attempt 'raw/adapter/scenarios.jsonl'
$settings = Get-Content -Raw -Encoding UTF8 -LiteralPath $allConfig | ConvertFrom-Json
$sdConfig = $settings.configs.sd
if (-not $sdConfig) { throw 'SD config missing from the original run_all settings.' }
$oldRun = Get-Content -Raw -Encoding UTF8 -LiteralPath (Join-Path $attempt 'raw/benchmark/run.json') | ConvertFrom-Json
$diagnostic = 'experiments/results/algorithms/sd_diagnostic_' + [guid]::NewGuid().ToString('N')
# Same source file/query, seed/config, full set of k values, original scenario scheduling offset.
& python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config $sdConfig `
    --scenarios $scenarios --scenario-id SC-E16-N --mode all `
    --repeats $oldRun.config.repeats --warmups $oldRun.config.warmups `
    --allow-unreviewed --run-dir $diagnostic
if ($LASTEXITCODE -ne 0) {
    throw "Strict diagnostic failed. Inspect $diagnostic/results.jsonl token_difference and installed_generation_sources.json. Original research evidence is unchanged; resume was NOT started."
}
Write-Output 'Single-query equivalence passed; this does not certify the complete Evaluation.'
# Existing resume remains authoritative. Any model/data/config/code identity rejection is NOT bypassed.
& python -m experiments.research_tools.run_all --mode provisional --config $allConfig --resume $run
if ($LASTEXITCODE -ne 0) { throw 'Full resume did not complete; inspect its preserved failure evidence.' }
