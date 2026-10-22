# Worker 與 FAQ 編碼靜態檢查

本次僅閱讀、編輯與 PowerShell 唯讀檔案檢查；未執行 Python、測試或模型。

## 已確認與尚未確認

1. **空 Log 的確定原因**：worker 外層捕捉 `BaseException` 後只寫 job/attempt JSON，未向 stdout/stderr 印出例外。`verify_inputs` 在模型初始化前失敗，迴圈內唯一的進度 print 尚未執行，因而可產生 0-byte log。現在啟動即印架構與階段，捕捉錯誤即輸出原因及完整 traceback；parent 顯示本次新 job 的失敗摘要。早期未捕捉例外仍由 Python stderr 記錄，native crash 沒有 job 時顯示 log 尾端。
2. **歷史 fingerprint 不一致欄位尚未確定**：本機沒有 GPU 的 `smoke_v1/run.json`、jobs 或 integrity JSON。舊錯誤把 code 與 environment 用 `or` 合併，既不能判定是否真為環境，也沒有保存各欄差異。不能據此宣稱是某個套件、Windows、時間戳或 PID。
3. 父行程與 Worker 實際都呼叫同一個 `code_hashes(root)` 與 `environment()`；比較的是字典值，不是不同排序的 JSON 字串。environment 包含 Python 版本、`platform.platform()`、host、14 個套件版本。沒有時間、PID、暫存目錄。code 只收指定六個程式目錄下的 `.py`／`.json`；預設 `experiments/results/` 不在掃描內，`.pyc` 也不在內，且 runner 禁止輸出位於這些程式目錄。
4. 父行程先盤點資源再取 environment；Worker 先核對 environment 再盤點資源。子行程額外指定 HF 離線、hash seed、UTF-8 與 unbuffered 環境變數；這些變數本身不在現有 environment 字典中。靜態閱讀未證明它們導致哪個既有欄位改變，故未以移除欄位或忽略差異來掩蓋問題。
5. **本機原始 CSV 正常**：家長檔 12,077 bytes、高中生檔 11,747 bytes；皆無 BOM，開頭 UTF-8 bytes `E7-B7-A8-E8-99-9F`（編號），以 .NET 嚴格 UTF-8 解碼成功，各 30 列。未見 U+FFFD 或 BMP 私用區字元。P01 為「畢業後具體是去哪些企業上班？有沒有知名大廠？」；S01 為「高中完全沒有寫過程式，進來會不會跟不上？」。這只證明本機來源，不能證明 GPU 副本與歷史 attempt 正常。
6. 資料流為 A3 `snapshot()` 以 `utf-8-sig` 嚴格讀 CSV → evaluation `dataset()` 直接取 messages → UTF-8 `atomic_json` 保存 run → Worker 嚴格 UTF-8 讀 run → `save_result` 保存同一 question/reference/source。沒有本地碼頁解碼，也沒有 `errors="replace"`。A2 答案載入同樣使用 `utf-8-sig`。`source` 來自 A3 Python 路徑常數，不是 CSV 內容；source 也亂碼時必須同時檢查檔案傳輸與顯示方式，不能只改 CSV decoder。

## 本次修改

- `common.py`：遞迴欄位差異（含 missing/null 區別、清單位置、原值／現值）；60 題 question/reference/source 的 Unicode 結構 gate 與逐欄 UTF-8 SHA-256 證據。
- `worker.py`：保持原本全部 identity 比較；改為詳細 `InputChanged`，包含模型資源與 runtime contract 差異；記錄階段、stderr traceback 與 integrity 差異；新增 `--verify-only` 私有模式。
- `run.py`：`check-data`、唯讀 `check-inputs`、`preflight --verify-workers`；preflight 保存文字證據；parent 只採本次 job 的錯誤摘要，保留 stdout/stderr 合併 log 與原有錯誤 attempts。
- `README.md` 與本文件：操作與判讀說明。

未修改 config、原始 CSV、A0～A5、模型、索引、訓練或評估題庫、既有 results。**這是確定的紀錄缺陷修復及診斷／資料 gate 補強，不能宣稱歷史 fingerprint 根因已定位或八架構已跑通。**

## GPU 上的定位與驗證順序

先同步上述 evaluation 程式。從 GPU 專案根目錄執行，舊目錄只讀，不續跑：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
python -m experiments.evaluation.run check-inputs --run-dir experiments/results/unified_faq/smoke_v1
python -m experiments.evaluation.run check-data --project-root .
```

`check-inputs` 不寫入舊目錄，不載入模型，只比較 code/environment/FAQ。同步修正版後 code 差異是預期的，退出碼 2 不代表所有差異都是舊錯誤原因；重點讀 `environment.*` 的原值／現值、來源 hash 與 `dataset[i].question/reference_answer/source`。它只能比較現在與凍結值，不能重建已結束的舊 Worker 狀態。

`check-data` 的 JSON 刻意全為 ASCII（中文為 `\uXXXX`），不依賴 PowerShell 顯示碼頁；含完整 60 題、split、問題、答案與 source 及各欄 hash。用編輯器或 JSON 檢視器還原 escape 逐題查看。應有 60 個唯一 P/S ID，42/9/9 切分且文字符合來源。若舊 run 與目前 CSV 字串相同、escape 也正確，只有終端機顯示錯，才可歸類為顯示問題。若舊 run 的 escape 已包含 U+FFFD／私用字元，則其儲存內容有問題，不可改檔救分數。

文字 gate 保留原字串，拒絕替代字元、私用區、孤立 surrogate、異常控制字元、沒有 CJK 字元的中文欄位；嚴格 decoder 失敗也直接停止。Preflight、Smoke、Full 與 Worker 都先經此 gate，無 bypass。這是保守的結構檢查，**不等於能辨認所有有效 Unicode 的亂碼或證明答案語意正確**；有疑慮仍應人工核對，不能直接 Full。

接著使用全新目錄，先測八個 Worker 的實際父子行程完整性，不載入模型、不生成 attempt：

```powershell
python -m experiments.evaluation.run preflight --project-root . --run-dir experiments/results/unified_faq/worker_verify_v2 --allow-unreviewed --verify-workers
```

八份新 job 應為 `verified_no_model_load`，`model_load_verified` 仍為 false。任何失敗查看 log、job.stage、integrity_violations.differences 的具體欄位。資源盤點若失敗，命令仍返回 2，即使指紋一致也不能當成架構可用。

中文與 Worker 檢查都通過、必要資源盤點正常後，用另一個全新目錄測固定三題 P01/P22/P26：

```powershell
python -m experiments.evaluation.run smoke --project-root . --run-dir experiments/results/unified_faq/smoke_v2 --allow-unreviewed
```

若 v2 已存在，換新的 v3 等名稱；不要刪除、修改或 resume `smoke_v1`，也不要混合新舊結果。確認八架構各有三筆 success、非空答案，中文正確、jobs 與 logs 無完整性違規；程式 exit=0 不代表回答內容正確。此階段不執行 Full。

尚待 GPU 確認：真正不一致的 fingerprint 欄位、GPU 原始 CSV bytes／舊 JSON Unicode、父子行程驗證、Python 語法與 runtime、模型初始化及三題 Smoke。取得具體差異後才能針對已證實的環境來源做後續最小修正，不應先刪除驗證。
