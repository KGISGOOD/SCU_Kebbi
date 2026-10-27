param(
    [Parameter(Mandatory=$true)][string]$BgeModelPath,
    [Parameter(Mandatory=$true)][string]$LlamaModelPath,
    [string]$ProjectRoot=(Join-Path $PSScriptRoot '../../..'),
    [ValidateSet('bf16','fp16')][string]$Precision='bf16',
    [switch]$AllowUnreviewed,
    [string]$A2Review,
    [string]$A3Review
)
$ErrorActionPreference='Stop'
$root=[IO.Path]::GetFullPath($ProjectRoot)
Set-Location -LiteralPath $root
foreach($path in @($BgeModelPath,$LlamaModelPath)){
    if(!(Test-Path -LiteralPath $path -PathType Container)){throw "Local base model directory missing: $path"}
}
if(!$AllowUnreviewed -and (!$A2Review -or !$A3Review)){
    throw 'Formal training requires genuine A2Review and A3Review. Explicit AllowUnreviewed produces provisional models only.'
}
function RunPython([string[]]$Arguments){
    & python @Arguments
    if($LASTEXITCODE -ne 0){throw "Python stage failed (exit $LASTEXITCODE): $($Arguments -join ' ')"}
}
$data='experiments/datasets/training_v2/converted'
RunPython @('-B','-m','experiments.datasets.training_v2.prepare_training','--project-root','.','--check-only')
RunPython @('-B','-m','experiments.datasets.training_v2.check_preparation')
RunPython @('-B','-m','experiments.a3_faq_sft.train','--project-root','.','--data-dir',$data,
    '--base-model',$LlamaModelPath,'--offline','--check-only','--max-length','512')
$tag=(Get-Date -Format 'yyyyMMdd_HHmmss')+'_'+[Guid]::NewGuid().ToString('N').Substring(0,8)
$output='experiments/models/training_v2/'+$tag
$review2=@();$review3=@()
if($AllowUnreviewed){$review2=@('--allow-unreviewed');$review3=@('--allow-unreviewed')}
else{
    $review2=@('--training-review',$A2Review,'--training-manifest',($data+'/dataset_manifest.json'))
    $review3=@('--training-review',$A3Review)
}
foreach($mode in @('a2_1','a2_2')){
    RunPython (@('-B','-m','experiments.a2_faq_finetuning.train_faq_embedding','--project-root','.',
        '--config','experiments/a2_faq_finetuning/training_v2_config.json',
        '--v2-data-dir',$data,'--mode',$mode,'--model',$BgeModelPath,'--device','cuda',
        '--epochs','3','--batch-size','8','--seed','42','--learning-rate','0.00002',
        '--gradient-checkpointing','--offline','--final-only','--verify-save-load',
        '--output-dir',($output+'/'+$mode))+$review2)
}
RunPython (@('-B','-m','experiments.a3_faq_sft.train','--project-root','.','--data-dir',$data,
    '--base-model',$LlamaModelPath,'--device','cuda','--precision',$Precision,
    '--epochs','3','--batch-size','1','--gradient-accumulation','4','--seed','42',
    '--learning-rate','0.0001','--max-length','512','--offline','--final-only',
    '--output-dir',($output+'/a3'))+$review3)
Write-Output "Training commands completed; models: $output. Inspect validation losses before accepting the final epochs."
