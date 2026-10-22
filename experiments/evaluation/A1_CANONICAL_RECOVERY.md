# A1 canonical 換行診斷與不覆寫恢復

## 已確認的 byte 差異

GPU 唯一不符欄位為 canonical SHA-256，CSV 核對通過，FAISS/docstore hash 相符。本次在開發機僅以唯讀 PowerShell 讀取已同步的 canonical，**在記憶體內**計算 LF／CRLF bytes 的 SHA-256，沒有寫入或執行 Python／模型／測試：

| 格式 | Bytes | SHA-256 |
|---|---:|---|
| 原始 LF | 33573 | `7ff63fb1a27558345f33b034023243a8a35e774019f53397108861d40679f8ac` |
| 僅轉為 CRLF | 33633 | `60b4fad2ff93b16b94b5f980de6d883eaccd64ad1942e5d36dd2850303869a63` |

兩者精確重現 receipt 與 GPU 回報的 hash；差 60 bytes 對應 60 個行尾多出的 CR。故 **LF→CRLF 已足以完整解釋本次 checksum 差異**。不能因此指認是哪個編輯器、Git 行尾設定或同步工具做了轉換；本次未執行 Git 操作。不同 JSON 欄位順序、空白、BOM、ensure_ascii 等也可能讓語意相同資料有不同 byte hash，但不是解釋這兩個指定 hash 所必需的假設。

canonical 子命令只有在輸出不存在時以 `open("x", encoding="utf-8", newline="\n")` 寫入，明確使用 LF。index 子命令只讀 canonical：先取其 raw-byte hash → 驗證 CSV → 建置、儲存並重新載入索引 → 核對 docstore → 再確認 canonical/source/model 沒變 → 寫入全新 version.json → 自查 receipt。這條流程沒有在建索引後重新寫入 canonical。Wrapper 也只載入，不會 save/build canonical 或索引。

## 修改與不變條件

- `prepare_a1.py`：audit-index 增加 bytes/BOM/CRLF/LF 還原 hash 證據；可明確 opt-in 讀取本機 trusted docstore，逐筆核對 60 題 ID/Q/A 與完整 mapping。新增 restore-canonical，只產生新 canonical、新操作 receipt、新 evaluation config。
- `run.py`：Preflight 顯示各架構資源錯誤、Worker SKIPPED 與總退出碼，明確區分僅盤點與實際 loader。
- README 與本文件：完整操作順序。

未改變原本 raw-byte hash 驗證、A1 Top1 演算法、模型、索引、CSV、任何舊結果；不把 semantic equality 自動當成 byte equality。**絕不修改既有 version.json 的 hash**。

## GPU：先做唯讀檢查

同步上述程式、文件到 GPU，勿用開發機資料覆蓋目前 GPU 資料。從專案根目錄執行：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
python -m experiments.evaluation.prepare_a1 --project-root . audit-index --inspect-docstore --trust-local-docstore
$LASTEXITCODE
```

只對自己建立、可信任的 index.pkl 使用 opt-in；程式先核對 receipt 的 index/docstore hash，再用僅允許原 LangChain Document class 的 unpickler。限制 unpickler 並不代表能信任外部提供的任意 pickle／receipt。這步不載入 FAISS、不 encode 或推論，但 GPU 需有原 docstore 的 LangChain/Pydantic 相容環境；未知 pickle global 會阻止，不允許靜默退回不受限反序列化。

此階段 raw hash 尚未修復，整體仍預期 failed／退出 2。應同時看到：

- 唯一 receipt 差異 `a1_index_completion.canonical_sha256`。
- `canonical_representation.crlf_count=60`、`lf_bytes=33573`，`lf_matches_receipt=true`。
- `docstore_validation.status=passed`、faq_count=60、id_set_matches=true、differences=[]。
- `csv_validation.status=passed`。

不能因最終退出 2 就忽略這些子結果，也不能因 docstore 通過就忽略 raw hash。比較是原字串相等，不 strip、不做 Unicode 正規化或近似比對；page_content、metadata.question、metadata.answer 都檢查。index.pkl 與 index.faiss 的原 hash 必須符合 receipt。

## 分支 A：內容相同且 LF bytes 可還原原 Hash

使用全新名稱輸出，不修改目前 CRLF 檔、不複製或改寫 FAISS、不碰原 version.json：

```powershell
python -m experiments.evaluation.prepare_a1 --project-root . restore-canonical --output experiments/datasets/faq/faq_canonical_60_v2_lf_restored.jsonl --output-config experiments/evaluation/config.a1_lf_restored.json --trust-local-docstore
```

若使用自訂原 config，額外傳 `--config 你的原設定檔.json`。命令會再次核對 CSV/docstore/receipt，僅做 CRLF bytes → LF bytes；**輸出 hash 必須精確等於原 version.json 的 7ff63...9f8ac 才能發布新設定**。也會比對操作前後的原檔 hash。任何輸出已存在即拒絕，請選其他全新名字，勿刪除或覆寫。

新增三個檔案：

1. 新 LF canonical（UTF-8 原 bytes，不重新序列化 JSON）。
2. 同名 `.restoration.json`，記錄舊輸入 hash、轉換及逐筆檢查結果。
3. 新 evaluation config，只更改 a1_canonical 指向新檔、a1_index 指向原 v2 目錄，其餘設定承襲輸入 config。

原 receipt 已記錄這份 LF canonical 的 hash，所以不需要新的索引 receipt，也沒有「放寬」驗證。若中斷留下新的未完成輸出，保留調查，不覆寫它；沒有成功發布 config 就不能當成恢復完成。

再驗證新的設定：

```powershell
python -m experiments.evaluation.prepare_a1 --project-root . audit-index --config experiments/evaluation/config.a1_lf_restored.json --inspect-docstore --trust-local-docstore
$LASTEXITCODE
```

這次應為整體 passed、退出 0，並且三個 raw hash 全部通過。不要再用編輯器或會轉換換行的流程重新保存新 LF 檔；若再次轉換，完整性驗證仍應阻擋。

## 分支 B：內容不同或無法精確還原原 Hash

restore-canonical 會阻止。先保留所有原檔及差異報告：

- CSV 核對不通過：先確認選到正確 canonical/CSV 與資料版本；不猜測或修改原 CSV。
- CSV 通過、docstore 的某個 FAQ ID/Q/A 不同：不能繼續沿用舊向量與新的內容。使用正確且通過 CSV 驗證的 canonical，另建全新 v3 索引。
- Q/A 相同、LF hash 仍無法還原：可能有其他序列化或 metadata 差異；優先找與 receipt 相符的原始 byte 備份。找不到時亦可明確建新版本索引；不可直接把新 hash 填回舊 receipt。

僅在確認需要後，由使用者在 GPU 明確執行，例如：

```powershell
$bgeSnapshot = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
python -m experiments.evaluation.prepare_a1 --project-root . index --canonical experiments/datasets/faq/faq_canonical_60_v2.jsonl --output-dir experiments/vector_stores/faq_bge_m3_60_v3 --bge-snapshot "$bgeSnapshot" --device cuda
```

v3 必須不存在。這只是新 A1 embedding／索引建置，不重新訓練，也不重建其他架構。成功後複製原 evaluation config 成**新的** config，將 a1_index 改為 v3，a1_canonical 保留經驗證的來源路徑，再用 audit-index --config 核對。不得覆寫原設定或 v2 索引。本次已知 hash 是可還原的 LF/CRLF 差異，預期優先使用分支 A，而不是重建。

## 舊 Smoke 缺少 run.json

audit-index --run-dir 必須指向 Run 根目錄，不能是 reports/某快照、jobs/A1，亦不能照抄文件中的「實際失敗的Smoke目錄」佔位名稱。

```powershell
Get-ChildItem -LiteralPath experiments/results/unified_faq -Filter run.json -Recurse -File | Select-Object FullName
```

從找到的實際 run.json 的父目錄選定 Run；程式不會自動挑一個或 fallback 到目前設定，避免誤稱已驗證歷史設定。

若確實沒有 run.json，修正版會列出所指定完整路徑、目錄是否存在，以及 `preflight/*_dataset_failed.json`：

- 錯誤目錄、漏同步或檔案被外部移除：需要用實際檔案紀錄確認，目前不能斷言哪一種。
- create_run 的原始 dataset 驗證失敗：程式確實會建立只有 dataset_failed 報告、沒有 run.json 的部分目錄。
- **A1 canonical 資源 hash 失敗不是上述 dataset 失敗**：run.json 仍會建立，Preflight 再報資源失敗。若同一 Run 真有七個 Worker 正常完成，其 Worker 啟動時都必須讀過 run.json；不能用「一開始沒建立」解釋這份 Run。

缺少 run.json 時 audit-index 明確退出 2，標記 frozen_config_verified=false，沒有驗證歷史 Run；仍可不帶 --run-dir，明確查目前 canonical/index。不要補造 run.json。

## Preflight 為何只有 Run／Report、沒有 Worker

兩條不同路徑：

1. 未加 `--load-models` 或 `--verify-workers`：本來就只盤點，不啟動 Worker。即使 exit=0，也只證明資源盤點通過。
2. 有 `--load-models`，但 A1 資源盤點失敗：ready 清單為空，舊程式直接略過 launch，因此沒有 Worker log；仍應 exit=2。Run/Report 是產物路徑，不是成功指示。

修正版會列出 A1 資源錯誤、Worker SKIPPED、No model Workers launched、Preflight summary/exit。請在命令後立即看 `$LASTEXITCODE`；真正載入成功仍以 job 的 loaded_no_inference／model_load_verified 為準。

## 恢復後：全新 A1 loader 與 Smoke

只有分支 A 的新設定通過 audit 後才執行：

```powershell
$bgeSnapshot = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
$loadTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run preflight --project-root . --config experiments/evaluation/config.a1_lf_restored.json --architectures A1 --run-dir "experiments/results/unified_faq/a1_lf_load_$loadTag" --bge-snapshot "$bgeSnapshot" --allow-unreviewed --load-models
$LASTEXITCODE
```

確認 loaded_no_inference／exit=0，再做三題：

```powershell
$smokeTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run smoke --project-root . --config experiments/evaluation/config.a1_lf_restored.json --architectures A1 --run-dir "experiments/results/unified_faq/a1_lf_smoke_$smokeTag" --bge-snapshot "$bgeSnapshot" --allow-unreviewed
$LASTEXITCODE
```

不續跑舊 Run，不改成功答案，不重跑其他七架構。新工具的 Python 語法／套件、受限 pickle 讀取、新檔寫入及 loader/Smoke 尚待 GPU 實測；上述指令都未在開發機執行。
