# LEGACY_FIXED60 來源 Hash 調查

本輪只做靜態檢查與唯讀 PowerShell 檔案檢視，沒有執行 Python、專案測試、模型或 Git 操作。
沒有修改固定60題、FAQ60、主要230題、Calibration/Evaluation、模型或索引，也沒有改登錄Hash。

## 本機已確認

- registry source ID：LEGACY_FIXED60。
- 專案相對路徑：experiments/datasets/scu_fixed_questions_v1_1.jsonl。
- 本機絕對路徑：D:\Ray DB\SCU\SCU_Kebbi\experiments\datasets\scu_fixed_questions_v1_1.jsonl。
- 來源登錄與目前原始位元組SHA256均為
  2859667d72ea51a33cc53dffed04a3545709765c9507c0a04158e3cd041b2820。
- 11434 bytes，UTF-8無BOM，59個LF、0個CRLF，最後一題後無換行。
- 唯讀比對：60題、60個唯一Question ID；和questions_legacy_fixed60.jsonl的ID及題目文字無差異。
- 僅在記憶體把59個LF改為CRLF，得到11493 bytes及
  cfb01bb816b500c074bc9eb96d4c6308bd26f6f70abfeb44baf4960093cc43b6。
  這只是可對照的診斷值，**不是第二個允許的來源Hash**。

registry指向正確的固定題庫；但現行experiments/evaluation/common.py的dataset()從A3 snapshot
取得兩CSV的P/S60，並非讀此A/B/...固定題庫。缺少舊GPU Run快照時，不能宣稱已證明該次
八架構實際使用的是這份固定題庫；本輪只確認本機來源與保留副本一致。

## 原因判斷界線

本機Hash沒有登錄錯誤，sha()以rb分塊送入hashlib.sha256，和Get-FileHash一致。
encoding=utf-8-sig是後續文字解析策略，不參與原始位元組Hash；不會把BOM或CRLF自動排除。
root原有.gitattributes只保障faq子目錄，未涵蓋這個位於datasets根下的檔案；Windows
core.autocrlf造成CRLF是合理風險，但目前GPU只回報mismatch，沒有actual Hash，尚未證實原因。

新增精確路徑的text eol=lf規則，保留原有規則。它約束後續Git checkout，不會立即改寫目前
GPU工作目錄中的檔案；本輪不執行checkout/renormalize或自動修復來源。

validation.py現在回報完整expected/actual Hash、實際絕對路徑、大小、BOM、CRLF/LF/bare CR，
以及僅作診斷的CRLF→LF Hash。即使診斷值吻合，來源仍以原始Hash判定失敗，不自動接受。
若GPU是其他Hash，即使ID/題目一樣，也可能改了日期、題型、pair metadata、JSON格式、BOM或
結尾換行；不能只看題目相同就改registry。

## GPU唯讀診斷

先同步本輪程式／文件／manifest／換行規則；不要覆寫任何原題。以下只讀檔案，不載模型：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
$root = (Get-Location).Path
$registry = Get-Content -Raw -Encoding utf8 experiments/datasets/unified_research_v1/source_registry.json | ConvertFrom-Json
$source = Join-Path $root $registry.sources.LEGACY_FIXED60.path
Get-FileHash -LiteralPath $source -Algorithm SHA256
$bytes = [IO.File]::ReadAllBytes($source)
$text = [Text.UTF8Encoding]::new($false, $true).GetString($bytes)
$sha = [Security.Cryptography.SHA256]::Create()
try {
    $lfBytes = [Text.UTF8Encoding]::new($false).GetBytes($text.Replace("`r`n", "`n"))
    [pscustomobject]@{
        Path = $source
        Expected = $registry.sources.LEGACY_FIXED60.sha256
        Bytes = $bytes.Length
        BOM = ($bytes.Length -ge 3 -and $bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191)
        CRLF = ([regex]::Matches($text,"`r`n")).Count
        LF = ([regex]::Matches($text,"`n")).Count
        DiagnosticLFHash = ([BitConverter]::ToString($sha.ComputeHash($lfBytes))).Replace('-','').ToLower()
    } | Format-List
} finally { $sha.Dispose() }
```

若raw=cfb01...且DiagnosticLFHash=285966...，即支持「只有CRLF差異」；仍不接受raw檔。
若其他情況，保留GPU原檔，先比較保留副本的完整provenance.original_metadata，包含as_of、
dataset_version、pair_group/repeat_of/test_type，不只Question ID和question。
沒有原始版本證據時不得更新Hash。後續部署應傳送本機已驗證的原始位元組，或使用LF checkout
流程；任何修復既有GPU原檔的動作都須另外安排，本輪不提供自動改寫或放寬驗證的工具。

## 重新驗證

```powershell
$auditRun = 'experiments/results/dataset_audits/unified_v1_source_' + [guid]::NewGuid().ToString('N')
python -m experiments.datasets.unified_research_v1.validation --project-root . --report-dir $auditRun
```

這是GPU/分析電腦的後續指令，本輪未執行。若來源仍不同，預期繼續exit2，新增診斷會說明差異；
不是套上換行規則就宣稱舊工作檔已修好。不得重用舊報告目錄或覆寫舊紀錄。
