param(
    [string]$ProjectRoot = '.',
    [string]$Config = 'experiments/configs/basic_tests.local.json',
    [ValidateSet('provisional','formal')][string]$Mode = 'provisional',
    [string]$View,
    [string]$GoldCertification,
    [switch]$ReportOnly,
    [string]$SourceRun
)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($ProjectRoot)
Set-Location -LiteralPath $root
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
function Locate([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $root $Value))
}
function Write-NewJson([string]$Path, $Value) {
    $target = Locate $Path
    if (-not $target.StartsWith($root.TrimEnd('\') + '\', [StringComparison]::OrdinalIgnoreCase)) { throw 'New config outside project' }
    if (Test-Path -LiteralPath $target) { throw "Refusing overwrite: $target" }
    [IO.File]::WriteAllText($target, ($Value | ConvertTo-Json -Depth 30) + "`n", [Text.UTF8Encoding]::new($false))
}
if ($ReportOnly) {
    if (-not $SourceRun) { throw '-ReportOnly requires -SourceRun; no model execution is performed.' }
    & python -m experiments.research_tools.collect_provisional --project-root . --source-run $SourceRun
    if ($LASTEXITCODE -ne 0) { throw 'Evidence report collection failed' }
    Write-Output 'One independent report set created; inspect its Technical Acceptance sheet. Collection success is not experiment certification.'
    return
}
if ($SourceRun) { throw 'New basic tests never resume an old Run. Use -ReportOnly for old evidence.' }
$configPath = Locate $Config
if (-not (Test-Path -LiteralPath $configPath)) {
    if ($Config -ne 'experiments/configs/basic_tests.local.json') { throw "Config not found: $Config" }
    $profile = Get-Content -LiteralPath 'experiments/configs/basic_tests.example.json' -Raw -Encoding UTF8 | ConvertFrom-Json
    Write-NewJson $Config $profile
}
$settings = Get-Content -LiteralPath $configPath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($settings.sd_fp32 -ne $true -or $settings.require_full_training -ne $true) { throw 'Basic test profile must explicitly enable sd_fp32 and require_full_training.' }
$sdPath = Locate $settings.configs.sd
if (-not (Test-Path -LiteralPath $sdPath)) {
    if ($settings.configs.sd -ne 'experiments/configs/sd_fp32.local.json') { throw "Missing SD config: $sdPath" }
    $existing = 'experiments/algorithms/speculative_decoding/config.gpu.local.json'
    if (-not (Test-Path -LiteralPath $existing)) { throw 'Supply local model paths in experiments/configs/sd_fp32.local.json using its example. No model download.' }
    $sd = Get-Content -LiteralPath $existing -Raw -Encoding UTF8 | ConvertFrom-Json
    $sd.precision = 'fp32'
    Write-NewJson $settings.configs.sd $sd
}
foreach ($name in @('qa','sc','sd','ti','int')) {
    $path = Locate $settings.configs.$name
    if (-not (Test-Path -LiteralPath $path -PathType Leaf)) { throw "Missing $name config: $path. Edit $Config; no fallback to Smoke models or old Run settings." }
    if ($path -match '[\\/]experiments[\\/]results[\\/]') { throw "$name config must not depend on an old results Run." }
}
$run = 'experiments/results/algorithms/research_full/basic_' + [DateTime]::UtcNow.ToString('yyyyMMddTHHmmssZ_') + [guid]::NewGuid().ToString('N')
$arguments = @('-m','experiments.research_tools.run_all','--project-root','.', '--mode',$Mode,
    '--config',$Config,'--run-dir',$run,'--sd-fp32','--require-full-training')
if ($View) { $arguments += @('--view',$View) }
if ($GoldCertification) { $arguments += @('--gold-certification',$GoldCertification) }
& python @arguments
$experimentExit = $LASTEXITCODE
Write-Output "Run: $run; execution exit=$experimentExit; Provisional is not Formal certification."
if (-not (Test-Path -LiteralPath "$run/raw/identity.json" -PathType Leaf)) {
    throw "Preflight did not establish a valid experiment identity. No complete experiment can be claimed; inspect $run/report.md and raw failure records."
}
$acceptancePath = "$run/raw/technical_acceptance.json"
if (-not (Test-Path -LiteralPath $acceptancePath -PathType Leaf)) { throw 'Technical acceptance evidence missing; do not claim complete validation.' }
$acceptance = Get-Content -LiteralPath $acceptancePath -Raw -Encoding UTF8 | ConvertFrom-Json
if ($experimentExit -ne 0 -or $acceptance.technical_passed -ne $true) {
    throw 'Some stages/report/technical checks failed or remain unverified. Evidence retained; do not claim all passed.'
}
Write-Output "All technical stages committed. Formal research qualification still follows original Gold/training Guards: $run"
