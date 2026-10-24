param(
    [string]$ProjectRoot = '.',
    [string]$BgeModel = '',
    [string]$LlamaModel = '',
    [ValidateRange(3,42)][int]$A2BatchSize = 8,
    [switch]$RunAll
)
# GPU-only orchestration. No approval creation, downloads, deletions or Git commands.
$ErrorActionPreference = 'Stop'
$projectPath = [IO.Path]::GetFullPath($ProjectRoot)
Set-Location -LiteralPath $projectPath
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'

function Project-Path([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $projectPath $Value))
}
function Read-Json([string]$Value) {
    return Get-Content -Raw -Encoding UTF8 -LiteralPath (Project-Path $Value) | ConvertFrom-Json
}
function Write-NewJson([string]$Path, $Value) {
    $stream = [IO.File]::Open($Path, [IO.FileMode]::CreateNew, [IO.FileAccess]::Write)
    try {
        $writer = [IO.StreamWriter]::new($stream, [Text.UTF8Encoding]::new($false))
        try { $writer.WriteLine(($Value | ConvertTo-Json -Depth 100)) } finally { $writer.Dispose() }
    } finally { $stream.Dispose() }
}
function Invoke-Python([string[]]$Arguments) {
    & python @Arguments
    if ($LASTEXITCODE -ne 0) { throw "Python exit=$LASTEXITCODE : $($Arguments -join ' ')" }
}

$configs = [ordered]@{
    qa = 'experiments/evaluation/config.json'
    sc = 'experiments/algorithms/semantic_cache/config.gpu.local.json'
    sd = 'experiments/algorithms/speculative_decoding/config.gpu.local.json'
    ti = 'experiments/algorithms/topic_identification/config.gpu.local.json'
    int = 'experiments/algorithms/integrated_efficiency/config.gpu.local.json'
}
if (Test-Path -LiteralPath 'experiments/evaluation/config.gpu.local.json') {
    $configs.qa = 'experiments/evaluation/config.gpu.local.json'
}
if (Test-Path -LiteralPath 'experiments/research_tools/run_all.local.json') {
    $localSettings = Read-Json 'experiments/research_tools/run_all.local.json'
    foreach ($property in $localSettings.configs.PSObject.Properties) {
        if ($configs.Contains($property.Name)) { $configs[$property.Name] = $property.Value }
    }
}
$sc = Read-Json $configs.sc
if (-not $BgeModel) { $BgeModel = $sc.bge }
if (-not $LlamaModel) { $LlamaModel = $sc.model }
if (-not $BgeModel -or -not $LlamaModel) { throw 'Set local BGE and Llama paths in SC config or pass -BgeModel/-LlamaModel.' }
$BgeModel = Project-Path $BgeModel
$LlamaModel = Project-Path $LlamaModel
foreach ($modelPath in @($BgeModel, $LlamaModel)) {
    if (-not (Test-Path -LiteralPath (Join-Path $modelPath 'config.json') -PathType Leaf)) {
        throw "Incomplete local model path: $modelPath"
    }
}

$a2Data = 'experiments/datasets/a2_review_20261024_235354'
$a3Data = 'experiments/datasets/a3_spoken_20261025_001847'
$answers = 'experiments/datasets/faq_finetuning/a3_answer_revisions.spoken_candidate_v1.jsonl'
$tag = (Get-Date -Format 'yyyyMMdd_HHmmss') + '_' + [guid]::NewGuid().ToString('N').Substring(0, 8)
$output = Project-Path "experiments/models/provisional_full_$tag"
New-Item -ItemType Directory -Path $output -ErrorAction Stop | Out-Null
$stage = 'data_validation'
Write-NewJson (Join-Path $output 'started.json') @{status='started'; provisional=$true; output=$output}
try {
    if (-not (Test-Path -LiteralPath $a2Data)) {
        Invoke-Python -Arguments @('-m','experiments.a2_faq_finetuning.prepare_review','--project-root','.', '--output-dir',$a2Data)
    }
    if (-not (Test-Path -LiteralPath $a3Data)) {
        Invoke-Python -Arguments @('-m','experiments.a3_faq_sft.prepare_data','--project-root','.',
            '--data-dir',$a3Data,'--answer-revisions',$answers)
    }
    # Existing malformed directories fail validation; never overwrite or repair them silently.
    Invoke-Python -Arguments @('-m','experiments.a2_faq_finetuning.prepare_review','--project-root','.',
        '--output-dir',$a2Data,'--check-only')
    Invoke-Python -Arguments @('-m','experiments.a3_faq_sft.prepare_data','--project-root','.',
        '--data-dir',$a3Data,'--check-only')
    $manifest = Read-Json "$a3Data/dataset_manifest.json"
    if ($manifest.schema_version -ne 'a3_sft_data_v2' -or $manifest.answer_revisions -ne $answers) {
        throw 'A3 must use the latest spoken v2 revisions, not the original FAQ answers.'
    }
    $stage = 'a3_tokenization'
    Invoke-Python -Arguments @('-m','experiments.a3_faq_sft.train','--project-root','.', '--data-dir',$a3Data,
        '--base-model',$LlamaModel,'--offline','--max-length','512','--check-only')
    foreach ($mode in @('a2_1','a2_2')) {
        $stage = $mode
        Invoke-Python -Arguments @('-m','experiments.a2_faq_finetuning.train_faq_embedding','--project-root','.',
            '--mode',$mode,'--model',$BgeModel,'--training',"$a2Data/train.jsonl",'--device','cuda',
            '--epochs','3','--batch-size',([string]$A2BatchSize),'--seed','42','--max-length','512','--learning-rate','0.00002',
            '--offline','--allow-unreviewed','--gradient-checkpointing','--verify-save-load','--final-only',
            '--output-dir',(Join-Path $output $mode))
    }
    $stage = 'a3'
    Invoke-Python -Arguments @('-m','experiments.a3_faq_sft.train','--project-root','.', '--data-dir',$a3Data,
        '--base-model',$LlamaModel,'--device','cuda','--precision','fp16','--epochs','3','--batch-size','1',
        '--gradient-accumulation','4','--max-length','512','--seed','42','--offline','--allow-unreviewed',
        '--final-only','--output-dir',(Join-Path $output 'a3'))
    $stage = 'checkpoint_verification'
    $a2Summaries = @()
    $a2Provenances = @()
    foreach ($mode in @('a2_1','a2_2')) {
        $summary = Read-Json (Join-Path $output "$mode/training_summary.json")
        $provenance = Read-Json (Join-Path $output "$mode/final_model/a2_training_provenance.json")
        if ($summary.status -ne 'completed' -or $summary.provisional -ne $true -or
            $summary.final_model_reload_verified -ne $true -or $provenance.provisional -ne $true -or
            $provenance.completed_epoch -ne 3 -or $provenance.mode -ne $mode) {
            throw "$mode missing provisional full-training/reload evidence"
        }
        $a2Summaries += $summary
        $a2Provenances += $provenance
    }
    if ($a2Summaries[0].optimizer_steps -ne $a2Summaries[1].optimizer_steps -or
        $a2Provenances[0].positive_pair_schedule_sha256 -ne $a2Provenances[1].positive_pair_schedule_sha256) {
        throw 'A2 schedules/optimizer steps differ; do not mark the comparison ready.'
    }
    $a3Complete = Read-Json (Join-Path $output 'a3/completed.json')
    $a3Provenance = Read-Json (Join-Path $output 'a3/final_adapter/a3_provenance.json')
    if ($a3Complete.status -ne 'completed_training' -or $a3Complete.provisional -ne $true -or
        $a3Complete.completed_epochs -ne 3 -or $a3Complete.examples_consumed -ne 126 -or
        $a3Provenance.provisional -ne $true -or $null -ne $a3Provenance.smoke_step_limit) {
        throw 'A3 did not complete all 3 epochs as provisional training.'
    }
    # Separate process reloads base+adapter and tokenizer; exit only after loader succeeds.
    'exit' | & python -m experiments.a3_faq_sft.chat --project-root . --base-model $LlamaModel `
        --adapter (Join-Path $output 'a3/final_adapter') --device cuda --precision fp16 --offline --allow-unreviewed `
        | Tee-Object -FilePath (Join-Path $output 'a3/reload.log')
    if ($LASTEXITCODE -ne 0) { throw 'A3 saved adapter reload failed' }
    Write-NewJson (Join-Path $output 'a3/reload_verification.json') @{
        status='passed'; provisional=$true; model_and_tokenizer_reloaded=$true
        generation_test_performed=$false
        adapter_completion_sha256=(Get-FileHash -Algorithm SHA256 -LiteralPath (Join-Path $output 'a3/final_adapter/a3_complete.json')).Hash.ToLowerInvariant()
    }
    $stage = 'provisional_qa_configuration'
    $qa = Read-Json $configs.qa
    $qa.a2_1_model = Join-Path $output 'a2_1/final_model'
    $qa.a2_2_model = Join-Path $output 'a2_2/final_model'
    $qa.a3_adapter = Join-Path $output 'a3/final_adapter'
    $qa.base_model = $LlamaModel
    $qa.a2_max_length = 512 # Match both newly trained embedding checkpoints; record in this NEW config.
    $qa.allow_unreviewed = $true
    $qaPath = Join-Path $output 'qa.provisional.config.json'
    Write-NewJson $qaPath $qa
    $configs.qa = $qaPath
    $allPath = Join-Path $output 'run_all.provisional.config.json'
    Write-NewJson $allPath @{configs=$configs; view=$null; gold_certification=$null}
    Write-NewJson (Join-Path $output 'completed.json') @{
        status='completed_provisional_training'; provisional=$true; formal_approved=$false
        run_all_config=$allPath; qa_config=$qaPath
        next_command="python -m experiments.research_tools.run_all --mode provisional --config `"$allPath`""
    }
    Write-Output "Training ready (PROVISIONAL): $output"
    Write-Output "python -m experiments.research_tools.run_all --mode provisional --config `"$allPath`""
} catch {
    Write-NewJson (Join-Path $output 'failure.json') @{status='failed'; stage=$stage; error=$_.ToString(); provisional=$true}
    throw
}
if ($RunAll) {
    Invoke-Python -Arguments @('-m','experiments.research_tools.run_all','--mode','provisional','--config',$allPath)
}
