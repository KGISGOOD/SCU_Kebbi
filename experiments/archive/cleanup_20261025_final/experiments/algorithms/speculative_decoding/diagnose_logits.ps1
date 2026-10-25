param([string]$ProjectRoot = '.', [switch]$V2Only)
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
$diagnosticExit = 2
if (-not $V2Only) {
    & python -m experiments.algorithms.speculative_decoding.diagnose_logits --project-root . `
        --config $settings.configs.sd --source-run $source --run-dir $output
    $diagnosticExit = $LASTEXITCODE
}
$diagnosticPassed = $false
if ($diagnosticExit -eq 0 -and (Test-Path -LiteralPath "$output/diagnostic_summary.json")) {
    $result = Get-Content -Raw -Encoding UTF8 -LiteralPath "$output/diagnostic_summary.json" | ConvertFrom-Json
    $diagnosticPassed = $result.equivalence_passed -eq $true
}
# A successful capture is NOT a successful token test. A clean, uninstrumented
# replay must pass the original benchmark Guard before even checking resume.
$cleanPassed = $false
if ($diagnosticPassed -and -not $V2Only) {
    $clean = 'experiments/results/algorithms/sd_clean_acceptance_' + [guid]::NewGuid().ToString('N')
    & python -m experiments.algorithms.speculative_decoding.benchmark --project-root . `
        --config $settings.configs.sd --scenarios (Join-Path (Split-Path $source -Parent) 'adapter/scenarios.jsonl') `
        --scenario-id SC-E16-N --mode all --repeats 2 --warmups 1 --allow-unreviewed --run-dir $clean
    $cleanPassed = $LASTEXITCODE -eq 0
}
if ($cleanPassed) {
    & python -m experiments.algorithms.speculative_decoding.acceptance --project-root . `
        --source-run $run --config 'experiments/models/provisional_full_20261025_021058_666b6611/run_all.provisional.config.json' --check-resume
    if ($LASTEXITCODE -eq 0) {
        & python -m experiments.research_tools.run_all --project-root . --mode provisional `
            --config 'experiments/models/provisional_full_20261025_021058_666b6611/run_all.provisional.config.json' --resume $run
        if ($LASTEXITCODE -ne 0) { Write-Warning 'Full Resume failed. Original Guard remains authoritative; evidence retained.' }
    } else {
        Write-Warning 'Resume identity differs. Inspect resume_identity_diff.json. No bypass and no original state edits.'
    }
} else {
    # Exactly one independent V2 attempt. Its results NEVER replace V1 and never
    # authorize a resume. Failure here is a reported limitation, not another loop.
    $v2 = 'experiments/results/algorithms/sd_v2_acceptance_' + [guid]::NewGuid().ToString('N')
    & python -m experiments.algorithms.speculative_decoding.v2 --project-root . --source-run $source --run-dir $v2 --repeats 2
    if ($LASTEXITCODE -ne 0) { Write-Warning 'V2 failed/unverifiable. Stop SD retries; preserve this technical limitation.' }
    Write-Warning 'V1 did not pass acceptance. V2 is independent evidence, not a replacement for the failed Evaluation.'
}
& python -m experiments.algorithms.speculative_decoding.acceptance --project-root . --source-run $run
$acceptanceExit = $LASTEXITCODE
& python -m experiments.research_tools.collect_provisional --project-root . --source-run $run
if ($LASTEXITCODE -ne 0) { throw 'Result collection failed; inspect its new output directory. No evidence was deleted.' }
if ($acceptanceExit -ne 0) { throw 'Reports generated, but not all original technical acceptance checks passed. See technical_acceptance_*.' }
