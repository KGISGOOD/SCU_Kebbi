# 正式 Run：保存既有結果、A1 LF 防護及 A5 P11 獨立補測

本次僅閱讀原碼、編輯工具與文件，以及唯讀 Git 設定查詢。沒有執行 Python、模型、測試、訓練、下載或 Git 變更。開發機沒有這次正式 Run 的原始紀錄；59 筆是否通過 Hash、post verify 的確切差異，仍須 GPU 執行下列唯讀 audit 確認，不能只依 log 行數宣稱通過。

## 必須分開的結論

1. **已提交紀錄**：Worker 每題 save_result 原子寫入後才印 success，之後才 post_execution_verify_inputs。結束檢查失敗不會刪除或撤銷先前的 success。read_attempts／latest_results 會核對 Run/record Hash、FAQ 原文歸屬、嘗試序號及禁止成功後重寫；59 筆若通過，即為有效提交的歷史紀錄。
2. **實驗完整性**：有 integrity_violations 的舊 Run 仍被標記，不清除、不冒稱正常。若證據僅為無關 A1 canonical 變動，能支持 A5 的依賴未在該項差異中改變，但不能無紀錄地刪警示、重寫舊報表或自動宣稱所有結果為乾淨正式成績。
3. **補 P11 的設定**：原 2048 cap 必然拒絕 2256-token prompt。提高 cap 是設定變更，所以不能以「同設定 resume」成功補上。保留原 59 題、原 P11 失敗與耗時；新 Run 只跑 P11，另標記 cap-only supplement，不能稱 60 題皆在原 2048 設定成功。

## A5 輸入與容量

evaluation config 的 a5_max_input_tokens 預設 2048，max_new_tokens 預設 256。A5 Pipeline 保留完整 Top5 文件，套用 chat template 並 tokenize 後檢查：

- actual_prompt_tokens <= a5_max_input_tokens；
- actual_prompt_tokens + max_new_tokens <= loaded_model.config.max_position_embeddings。

P11 已知 2256 tokens，至少需要 2512 的總 context（若原 Run max_new_tokens=256）。建議新 cap 3072，但必須確認模型可容納 **3072 + 原 max_new_tokens**。不是修改 tokenizer.model_max_length、RoPE、文件數或偷偷截斷。新 Wrapper 會在初始化核對 cap+生成預算不超過實際載入模型 context，並記錄 model_context_window；容量不足直接拒絕。實際 GPU/VRAM 仍需執行確認。

audit_run 會讀凍結 Base config 的 max_position_embeddings，先核對該 config 的 SHA；這是容量的檔案證據，不等於實際載入驗證。模型載入後的值才是 Wrapper 的硬限制。本次不把宣傳的模型 context 數字寫死在程式中。

## GPU 步驟 1：先同步程式，核對舊 Run

同步：新增 audit_run.py、修改 run.py/wrappers.py/report.py/prepare_a1.py、本文件與 README。既有 common.py/worker.py/preflight.py 等沿用目前完整版本。只同步程式，不把 datasets/models/vector_stores/results 目錄覆蓋到 GPU。

**先同步與完成所有新設定／canonical 準備，再開始任何模型工作；Run 執行期間停止同步與編輯。** 新 config 放在 evaluation 目錄也會影響 code fingerprint，不可在另一個 Worker 執行中建立。

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
# 這是唯一需要輸入的既有路徑；應含 run.json，不是 reports 子目錄
$oldRun = Read-Host '貼上本次正式 Run 根目錄（含 run.json）'
python -m experiments.evaluation.audit_run --run-dir "$oldRun" --architecture A5
```

預期核對：committed_record_validation=passed、latest_success_count=59、latest_failure_ids=[P11]、missing_ids=[]；committed_attempt_count 若非 60，檢查是否曾經有重試。每題列出原 attempt ID、record/file Hash、答案摘要 Hash、latency、模型 identity Hash。整體 exit=2 在仍有 P11／原完整性警示時是預期，不代表 59 筆驗證失敗。

`integrity_violations[].differences` 才是結束失敗的直接證據。若 A5 的 post_execution_changes_only_a1=true，且列出的欄位僅 resources.a1_canonical.*／resources.a1_index.*，可確認該記錄的失敗來自 A1 盤點變動。原因是 verify_inputs 目前對**所有架構資源**重算，A5 不使用 A1 canonical，但全 Run 的 A1 變動仍會使其失敗。本次沒有關閉這項驗證。若涉及模型、資料、code 或 environment，不可歸咎於 A1；補測入口將阻止把它標為 cap-only。

保存舊 Attempt 的原始 bytes 證據到記憶體，不更動檔案：

```powershell
$beforeA5 = @(Get-ChildItem -LiteralPath (Join-Path $oldRun 'attempts/A5') -Filter '*.json' -File | ForEach-Object {
    [pscustomobject]@{Name=$_.Name; Hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
})
```

## GPU 步驟 2：先恢復並固定 A1 資料

canonical 的 LF/CRLF Hash 證據與 docstore 安全核對見 A1_CANONICAL_RECOVERY.md。這次 restore 支援 --from-run，繼承正式測試的凍結設定，而非假設 default config 相同。使用全新名字，不改原 canonical/index/receipt/config：

```powershell
$a1Tag = [guid]::NewGuid().ToString('N')
$a1Canonical = "experiments/datasets/faq/faq_canonical_60_lf_$a1Tag.jsonl"
$a1Config = "experiments/evaluation/config.a1_lf_$a1Tag.json"
python -m experiments.evaluation.prepare_a1 --project-root . restore-canonical --from-run "$oldRun" --output "$a1Canonical" --output-config "$a1Config" --trust-local-docstore
python -m experiments.evaluation.prepare_a1 --project-root . audit-index --config "$a1Config" --inspect-docstore --trust-local-docstore
```

只對自己建立且可信任的 docstore opt-in。每步需成功；新 LF 檔必須精確還原原 receipt Hash，三個檔案 Hash／60 題原文都通過。此後不要再改寫它或用整個開發工作樹覆蓋 GPU datasets。**不要把新 LF 檔複製回舊路徑來洗掉舊 Run 的警示**；原 CRLF 與舊 Run 保留，新 A1 設定改指新 LF 檔。

## GPU 步驟 3：只補 A5 P11，全新 Run

先確認步驟 1 的 model_context 有足夠容量（3072+原生成預算），記錄檢查結果。以下繼承原 Run config、seed、模型、輸出預算、precision 等，只提高 a5_max_input_tokens；保留相同 60 題 FAQ/CSV 與原始 Top5 檢索。--faq-ids 只縮小**待問的查詢集合**，不篩選候選庫或文件。

```powershell
$a5Tag = [guid]::NewGuid().ToString('N')
$a5Supplement = "experiments/results/unified_faq/a5_p11_supplement_$a5Tag"
python -m experiments.evaluation.run full --project-root . --from-run "$oldRun" --architectures A5 --faq-ids P11 --a5-max-input-tokens 3072 --run-dir "$a5Supplement"
```

這不是 resume。新 Run 位於舊 Run 外，只能選舊 A5 status=error 的 ID；不能重做原 success。入口核對原始 CSV/split/Run 原文、Python/套件/host、A5 所需 Base/BGE/Dense/A3 Adapter identity，以及全部非 evaluation 核心原碼。與舊設定相比只允許增加 A5 cap；其他變更拒絕。舊 integrity violation 必須有可證明僅 A1 相關的差異，未知或其他差異會阻止。若核對失敗先調查，不放寬、不替換權重。

新程式 hash、舊程式 hash 差異與 cap 差異均記入 supplement Metadata。父 Run ID、contract Hash、原 P11 attempt ID／Hash、父警示狀態保留；same_experiment_settings=false、automatic_merge_permitted=false。新報表的 HTML/CSV/JSON 標明只選 P11、設定變更，其他題為 not_selected，不匯入或重算原 59 題的時間。

若仍有 OOM、未知模型容量或其他錯誤，只保留新錯誤，不自動截斷、降低 TopK 或改參數重試。如果要報「同一個 3072-cap 設定的完整 60 題」成績，必須另跑 A5 60 題，而不是把這一題和原 59 題無標記相加；不必重跑其他六架構。本次預設採最小單題補測。

再次確認舊 Attempts 未動：

```powershell
$afterA5 = @(Get-ChildItem -LiteralPath (Join-Path $oldRun 'attempts/A5') -Filter '*.json' -File | ForEach-Object {
    [pscustomobject]@{Name=$_.Name; Hash=(Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash}
})
Compare-Object -ReferenceObject $beforeA5 -DifferenceObject $afterA5 -Property Name,Hash
python -m experiments.evaluation.audit_run --run-dir "$oldRun" --architecture A5
python -m experiments.evaluation.audit_run --run-dir "$a5Supplement" --architecture A5
```

Compare-Object 應無輸出。舊 audit 仍為 59 success+P11 error+原警示；新 audit 應只有 P11 success，不是 60 success。原答案、latency、attempt IDs、模型與版本紀錄仍在原檔案；補測另有獨立 latency，無法拿來改寫原失敗耗時或合併成公平的同設定延遲分布。

## GPU 步驟 4：只重做 A1 的 60 題

新 LF config 在步驟 2 已建立。沒有改模型、索引與原題，只恢復 receipt 指定的 byte 內容：

```powershell
$a1RunTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run full --project-root . --config "$a1Config" --architectures A1 --run-dir "experiments/results/unified_faq/a1_full_recovered_$a1RunTag"
```

繼承 oldRun 的 allow_unreviewed 與 bge_snapshot 等設定，不額外改模型路徑；如原 config 不具備必要授權或資源則明確報錯。新 A1 與原六架構、新 A5 補測各自保存，沒有把其他六架構重新執行或覆蓋。

## 為什麼不直接 resume

原 resume --architectures A5 --retry-failed 確實只選未成功題，原成功題不會被覆寫；但它嚴格要求凍結 config、code、資料、資源、runtime contract 都相同。原 cap=2048 依然會拒絕 P11；改 cap 或修改 run.json 不能當成合法 resume。同步這次工具也會改 evaluation code hash，舊 Run 不能繼續假裝是相同程式。此入口沒有增加忽略完整性的選項。

## A1 換行：已找到的風險與防護

開發機唯讀實際結果：

- `git config --show-origin --get-regexp core...` 顯示 `C:/Program Files/Git/etc/gitconfig` 的 core.autocrlf=true。
- `git check-attr text eol -- canonical...` 兩屬性均 unspecified。
- 工作區沒有 .gitattributes/.editorconfig/.vscode 同步設定、也未找到部署同步腳本；除現有唯讀 inspect_bge_snapshots.ps1 外無相關腳本。

此組合足以造成 Git checkout 將文字檔工作副本變為 CRLF，符合兩個已精確重現的 hash。但**不能據此斷言 GPU 某次操作一定是 Git，而不是文字同步或編輯器保存**。GPU 可用下列唯讀查詢補足證據，沒有 Git 設定或分支變更：

```powershell
git config --show-origin --get core.autocrlf
git check-attr text eol -- experiments/datasets/faq/faq_canonical_60_v2.jsonl
Get-Item -LiteralPath "$a1Canonical" | Select-Object FullName,Length,LastWriteTimeUtc
Get-FileHash -LiteralPath "$a1Canonical" -Algorithm SHA256
```

在「恢復後、同步前、同步後、編輯器保存後」分別保存來源／目的 Hash 與工具紀錄，便能定位哪一步改 bytes；時間戳只輔助，不能單獨當因果證據。若同步前檔案在來源端已是 CRLF，不能怪 byte-preserving copy。

本次不修改 Git 設定或 attributes，採以下不變更 Git 的流程：

1. GPU 上經驗證的新 LF canonical 作為該版本 runtime artifact，程式更新只複製 evaluation 的程式／文件，不同步 datasets、models、vector_stores、results，也不在 Run 執行中更新任何 code/config。
2. 使用 byte-preserving copy／ZIP 傳輸資料；傳輸前後核對原始 SHA。不要用 Get-Content | Set-Content、Out-File、格式化 JSON 或編輯器重存來搬資料。ZIP 只能保留打包時的 bytes，不能修正打包前已錯的 CRLF。
3. 目前未配置工具自動同步，因此不能代填特定同步工具的 exclude；需在實際工具中明確排除上述 artifact 路徑。原有全專案 Git checkout 流程若仍觸及這些文字檔，僅「手動恢復 LF」不能永久防止復發。
4. 團隊後續若允許 repository 行尾政策，可採用僅針對 canonical 的 `.gitattributes` 規則：`experiments/datasets/faq/faq_canonical_60_*.jsonl text eol=lf`。這會約束日後 checkout 的行尾；**本次僅提供建議，未建立該檔或執行 renormalize/checkout/commit**。EditorConfig 的 end_of_line=lf 可另防編輯器重存，但不能代替 Git 的規則。

參考 [Git attributes 官方文件](https://git-scm.com/docs/gitattributes/)：未指定 text/eol 時使用 core.autocrlf/core.eol；明確 eol=lf 可固定文字工作副本行尾。

## 尚待執行確認

本次原 Run 的實際 attempts/hash、post-execution 差異與真實 context，須 GPU audit；新 subset/supplement CLI、依賴核對、恢復、報表與模型啟動均只經靜態審查。不能宣稱已補測成功、原警示已解除或八架構都有乾淨 60 題成績。
