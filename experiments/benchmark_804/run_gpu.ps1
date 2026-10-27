param(
    [string]$ProjectRoot=(Join-Path $PSScriptRoot '../..'),
    [string]$BaseModel='', [string]$Bge='', [string]$Draft='', [string]$Reranker='',
    [switch]$CheckOnly, [switch]$SmokeOnly, [switch]$AllowProvisional,
    [switch]$AllowOverlapForMeasurement, [string]$Resume='', [string]$ReportOnly=''
)
$ErrorActionPreference='Stop'
Set-Location -LiteralPath ([IO.Path]::GetFullPath($ProjectRoot))
$config='experiments/benchmark_804/config.local.json'
if(!(Test-Path -LiteralPath $config)){
    # Prefer actual training provenance and retained GPU settings. These lookups
    # read files/cache metadata only, never download or load model weights.
    $trained='experiments/models/training_v2/20261028_003944_e3c6657d'
    if(!$BaseModel -and (Test-Path -LiteralPath ($trained+'/a3/final_adapter/a3_provenance.json'))){
        $BaseModel=(Get-Content -LiteralPath ($trained+'/a3/final_adapter/a3_provenance.json') -Raw -Encoding UTF8|ConvertFrom-Json).base_argument
    }
    if(!$Bge -and (Test-Path -LiteralPath ($trained+'/a2_1/final_model/a2_training_provenance.json'))){
        $Bge=(Get-Content -LiteralPath ($trained+'/a2_1/final_model/a2_training_provenance.json') -Raw -Encoding UTF8|ConvertFrom-Json).initial_model
    }
    $sdLocal='experiments/algorithms/speculative_decoding/config.gpu.local.json'
    if(!$Draft -and (Test-Path -LiteralPath $sdLocal)){
        $Draft=(Get-Content -LiteralPath $sdLocal -Raw -Encoding UTF8|ConvertFrom-Json).draft_model
    }
    function ResolveLocal([string]$Reference,[string]$Fallback){
        if($Reference -and (Test-Path -LiteralPath $Reference -PathType Container)){return [IO.Path]::GetFullPath($Reference)}
        $name=if($Reference){$Reference}else{$Fallback}
        if($name -notin @('BAAI/bge-m3','BAAI/bge-reranker-v2-m3','meta-llama/Llama-3.2-3B-Instruct','meta-llama/Llama-3.2-1B-Instruct')){
            throw "Local model path missing: $name. Pass the correct model path explicitly; no fallback."
        }
        $env:SCU_B804_MODEL_REFERENCE=$name
        try{
            $code='import os; from pathlib import Path; from huggingface_hub import try_to_load_from_cache; p=try_to_load_from_cache(os.environ["SCU_B804_MODEL_REFERENCE"],"config.json"); assert isinstance(p,str), "Required model not locally cached; pass explicit directory (no download)"; print(Path(p).parent)'
            $resolved=($code|& python -B - |Out-String).Trim()
            if($LASTEXITCODE -ne 0 -or !(Test-Path -LiteralPath $resolved -PathType Container)){throw "Local cached model unavailable: $name"}
            return $resolved
        }finally{Remove-Item Env:SCU_B804_MODEL_REFERENCE -ErrorAction SilentlyContinue}
    }
    $BaseModel=ResolveLocal $BaseModel 'meta-llama/Llama-3.2-3B-Instruct'
    $Bge=ResolveLocal $Bge 'BAAI/bge-m3'
    $Draft=ResolveLocal $Draft 'meta-llama/Llama-3.2-1B-Instruct'
    $Reranker=ResolveLocal $Reranker 'BAAI/bge-reranker-v2-m3'
    $models=@{base=$BaseModel;bge=$Bge;draft=$Draft;reranker=$Reranker}
    foreach($key in $models.Keys){
        if(!$models[$key] -or !(Test-Path -LiteralPath $models[$key] -PathType Container)){
            throw "First execution requires local -BaseModel, -Bge, -Draft and -Reranker directories. Missing: $key"
        }
    }
    $value=Get-Content -LiteralPath 'experiments/benchmark_804/config.example.json' -Raw -Encoding UTF8|ConvertFrom-Json
    foreach($key in $models.Keys){$value.models.$key=[IO.Path]::GetFullPath($models[$key])}
    [IO.File]::WriteAllText((Join-Path (Get-Location) $config),($value|ConvertTo-Json -Depth 12)+[char]10,[Text.UTF8Encoding]::new($false))
    Write-Output "Created $config. Confirm dense_index and trained model paths before a long run; existing configs are never overwritten."
}
$arguments=@('-u','-B','-m','experiments.benchmark_804.run_all','--project-root','.','--config',$config)
if($CheckOnly){$arguments+='--check-only'}
if($SmokeOnly){$arguments+='--smoke-only'}
if($AllowProvisional){$arguments+='--allow-provisional'}
if($AllowOverlapForMeasurement){$arguments+='--allow-overlap-for-measurement'}
if($Resume){$arguments+=@('--resume',$Resume)}
if($ReportOnly){$arguments+=@('--report-only',$ReportOnly)}
& python @arguments
if($LASTEXITCODE -ne 0){throw "Benchmark stopped/partial (exit $LASTEXITCODE); preserve Run and inspect summary.xlsx/report.md/raw. Resume with the SAME qualification flags."}
