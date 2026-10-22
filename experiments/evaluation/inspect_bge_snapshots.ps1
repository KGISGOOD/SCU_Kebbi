# Read-only inventory of the two reported snapshots. Never downloads/repairs.
[CmdletBinding()]
param([string]$CacheRoot, [switch]$HashWeights)
$ErrorActionPreference = 'Stop'
if (-not $CacheRoot) {
    if ($env:HF_HUB_CACHE) { $CacheRoot = $env:HF_HUB_CACHE }
    elseif ($env:HUGGINGFACE_HUB_CACHE) { $CacheRoot = $env:HUGGINGFACE_HUB_CACHE }
    elseif ($env:HF_HOME) { $CacheRoot = Join-Path $env:HF_HOME 'hub' }
    else { $CacheRoot = Join-Path $env:USERPROFILE '.cache/huggingface/hub' }
}
$repoPath = Join-Path $CacheRoot 'models--BAAI--bge-m3'
Write-Output "Cache repository: $repoPath"
$refPath = Join-Path $repoPath 'refs/main'
if (Test-Path -LiteralPath $refPath -PathType Leaf) {
    Write-Output ('Cached refs/main: ' + (Get-Content -LiteralPath $refPath -Raw -Encoding UTF8).Trim())
}
$anyMissing = $false
foreach ($revision in @('5617a9f61b028005a4858fdac845db406aefb181', '9a0624b896d81da7492a910ffa53731274b6cf3d')) {
    $snapshotPath = Join-Path $repoPath "snapshots/$revision"
    Write-Output "Snapshot: $snapshotPath"
    if (-not (Test-Path -LiteralPath $snapshotPath -PathType Container)) {
        Write-Output 'SNAPSHOT DIRECTORY MISSING'
        $anyMissing = $true
        continue
    }
    Get-ChildItem -LiteralPath $snapshotPath -Force | Select-Object Name,Length,LinkType,Target | Format-Table -AutoSize
    $dependencies = @('config.json', 'tokenizer_config.json', 'modules.json', '1_Pooling/config.json')
    $vocabularies = @('tokenizer.json', 'sentencepiece.bpe.model') | Where-Object {
        Test-Path -LiteralPath (Join-Path $snapshotPath $_) -PathType Leaf
    }
    if (-not $vocabularies) { Write-Output 'TOKENIZER VOCABULARY MISSING'; $anyMissing = $true }
    $dependencies += $vocabularies
    foreach ($optional in @('config_sentence_transformers.json','sentence_bert_config.json','special_tokens_map.json','added_tokens.json')) {
        if (Test-Path -LiteralPath (Join-Path $snapshotPath $optional) -PathType Leaf) { $dependencies += $optional }
    }
    $weights = @()
    foreach ($candidate in @('model.safetensors','model.safetensors.index.json','pytorch_model.bin','pytorch_model.bin.index.json')) {
        $candidatePath = Join-Path $snapshotPath $candidate
        if (-not (Test-Path -LiteralPath $candidatePath -PathType Leaf)) { continue }
        if ($candidate.EndsWith('.index.json')) {
            $dependencies += $candidate
            $indexData = Get-Content -LiteralPath $candidatePath -Raw -Encoding UTF8 | ConvertFrom-Json
            $weights = @($indexData.weight_map.PSObject.Properties.Value | Sort-Object -Unique)
        } else { $weights = @($candidate) }
        break
    }
    if (-not $weights) { Write-Output 'NATIVE PYTORCH WEIGHTS MISSING'; $anyMissing = $true }
    $dependencies += $weights
    foreach ($relative in ($dependencies | Sort-Object -Unique)) {
        # A malicious weight index must not redirect this inventory outside snapshot.
        if ([IO.Path]::IsPathRooted($relative) -or $relative -match '(^|[\\/])\.\.([\\/]|$)|:') {
            throw "Unsafe dependency path: $relative"
        }
        $filePath = Join-Path $snapshotPath $relative
        if (-not (Test-Path -LiteralPath $filePath -PathType Leaf)) {
            Write-Output "MISSING/BROKEN LINK: $relative"; $anyMissing = $true; continue
        }
        $stream = [IO.File]::OpenRead($filePath)
        try {
            $byteCount = $stream.Length  # Follow cache symlink; do not measure link metadata.
            $prefix = New-Object byte[] 256
            $readCount = $stream.Read($prefix, 0, $prefix.Length)
            $isPointer = [Text.Encoding]::ASCII.GetString($prefix,0,$readCount).StartsWith('version https://git-lfs.github.com/spec/')
        } finally { $stream.Dispose() }
        if ($byteCount -eq 0 -or $isPointer -or (($weights -contains $relative) -and $byteCount -lt 1024)) {
            Write-Output "EMPTY/LFS POINTER/SMALL WEIGHT: $relative"; $anyMissing = $true
        }
        $fileHash = 'not requested for weights'
        if ($HashWeights -or ($weights -notcontains $relative)) {
            $fileHash = (Get-FileHash -LiteralPath $filePath -Algorithm SHA256).Hash
        }
        [pscustomobject]@{File=$relative; Bytes=$byteCount; SHA256=$fileHash} | Format-List
    }
}
Write-Output 'Inventory only: no tensor/tokenizer/pooling loader validation. Confirm using prepare_a1 index and preflight --load-models.'
if ($anyMissing) { exit 2 }
