# 第一輪 Smoke 後的資源準備

本次僅編輯與靜態審查，未執行 Python、PowerShell 腳本、測試、模型載入或資料產生。以下由使用者在 GPU 電腦操作；未覆寫任何 CSV、舊 canonical、索引、模型、訓練資料或結果。沒有重新訓練。

## 原因與修正

### BGE-M3

舊 Preflight 用 `snapshot_download(repo_id="BAAI/bge-m3", revision=None, local_files_only=True)`，由 HF cache 的 `refs/main` 決定 revision，不是按兩個 snapshot 的目錄順序選第一個，也不保證選中先前 A5 使用的那個。沒有 allow_patterns 的 snapshot 請求範圍是整個 repo；新版 Hub 的 offline 完整性檢查可因此要求尚未快取的非推論檔案。上游該 commit 確有 README、.gitattributes、圖片、ONNX 等檔案，但沒有 GPU 的 19 個缺檔清單，**不能斷言 19 個全都可忽略**。

舊 A0/A1/A4 Wrapper 用 Preflight 返回的 snapshot 路徑；A5 pipeline 內部卻用 `BAAI/bge-m3` 重新解析，所以原先 A5 能跑不等於舊 Preflight 的全 repo 檢查會通過。

現在採 `try_to_load_from_cache(..., filename="config.json", revision="main")` 讀本機快取引用，或 `--bge-snapshot` 明確指定本機目錄。沒有下載、refs 修改、模型覆寫、跨 revision 自動湊檔或靜默 fallback。兩份 snapshot 哪個可用，由下面盤點與實際 loader 決定；不能憑 hash 新舊推定。

新 `model_resources.py` 驗證 native PyTorch 推論依賴：非空 config/tokenizer JSON、可用 tokenizer vocabulary、root weights 或 weight index 指向的**全部 shards**、LFS pointer、BGE modules.json 與實際 Pooling config；未知 module type 或缺必要檔直接失敗。保留可影響推論的附加 tokenizer／SentenceTransformer config hash。只對 BGE dense 推論所需檔案計算 identity，不要求 README、圖片、ONNX、ColBERT／sparse head 等本流程未用檔案。Base／reranker 仍保留全部既有檔案的 identity，兼容 A3 local-base provenance。檔案存在與 hash **不等於** tensor 可載入，仍須 loader 驗證。

A5 Wrapper 僅在該架構獨立 Worker 的 constructor 期間，把 shared BGEM3Embedder 的模型路徑注入為選定的 snapshot，隨後還原 class；A5 pipeline 本身、模型 ID／舊索引檢查、Pooling／正規化／Top5／prompt 不變。A0/A1/A4/A5 的 runtime contract 都記錄同一份 `embedding_model_version`。Wrapper 不下載或轉存模型。

兩份 snapshot 不能只因同名 BGE-M3 就當成權重相同。優先選先前 A0 Dense 索引建置時使用的版本，並核對兩份必要檔 hash；舊 A0 索引若僅記錄 Hub ID、沒有 commit／權重 hash，無法追溯證明選定 snapshot 與原建置版本完全一致。此限制仍須記錄，不能因 loader 成功就宣稱來源一致；本工具不重建或改寫舊 Dense 索引。

參考：[HF cache lookup API](https://huggingface.co/docs/huggingface_hub/package_reference/cache#huggingface_hub.try_to_load_from_cache)、[snapshot_download API](https://huggingface.co/docs/huggingface_hub/package_reference/file_download#huggingface_hub.snapshot_download)、[該 BGE snapshot 上游檔案](https://huggingface.co/BAAI/bge-m3/tree/5617a9f61b028005a4858fdac845db406aefb181)。上游資訊不證明 GPU 本機快取完整。

### A1

原 v1 canonical 只有 P01、P02、S01 三筆。原 FAQ loader 會在已帶 P/S 的 CSV ID 前再次加前綴，且 strip 文字並直接寫出 approved，故本任務不呼叫它生成正式 60 題，也不修改它。

`prepare_a1 canonical` 重用 evaluation/A3 的嚴格 UTF-8 source snapshot 與 split 驗證，逐字保存 question/answer，保留已存在的 ID，建立 `faq_canonical_60_v2.jsonl` 及來源/hash sidecar，標記 needs_review。不以測試 ID 答題。

**只有新 canonical 不夠，舊 3 題索引也不能配新 60 題。** `prepare_a1 index` 明確呼叫既有 FAQMatcher.build_index，將新 60 題原題 embedding 保存到全新的 `faq_bge_m3_60_v2`。原有 normalize + IndexFlatIP／Top1 Matcher 不變，沒有訓練。重新載入後核對 60 筆 docstore、ID／Q/A、維度與 mapping，再寫 completion/version.json。Preflight 對這種新 version 核對 canonical/index/docstore hash 及實際 BGE 推論檔案 hash，避免索引與模型不一致。

工具拒絕任何已存在輸出。index 失敗可能留下新的不完整目錄，不刪除、不重用，請選 v3 等全新路徑並在新評估 config 中對應。不要使用舊 builder 的 --force-rebuild。

### A2

`faq_training_candidates.jsonl` 是 42 個 Train 意圖的 positive/negative/hard_negative 訓練候選，不含完整 60 題庫，也不是答案來源。

既有 `experiments.a2_faq_finetuning.build_faq_candidates` 已能從兩份 CSV 生成完整 60 個 faq_id/question/source_group，拒絕 PP/SS、缺漏、重複、錯誤編碼，exclusive 寫出且附來源/hash metadata。直接重用，無需新工具或重訓。A2 的原 `load_answering_data` 再從 CSV 取得 60 答案，核對候選 question/group。`prepare_a1 check` 額外核對 A1/A2 60 題 ID、問題與答案一致。

## 同步檔案

同步 `experiments/evaluation/` 的：

- 新增：`model_resources.py`、`prepare_a1.py`、`inspect_bge_snapshots.ps1`、本文件。
- 修改：`preflight.py`、`run.py`、`wrappers.py`、`config.json`、`README.md`。

GPU 應保有最新前階段 common.py/worker.py（stable environment v2），及既有 A2 build_faq_candidates.py/common.py/faq_answering.py；A2 程式本次未修改。不要以同步方式覆蓋 GPU models、datasets、vector_stores 或 results。

config 預設改用 A1 新 canonical／新 index，新增可選 `bge_snapshot: null`。舊自訂 config 缺少此新欄仍可讀，但必須自行把 a1_canonical/a1_index 指向新檔；`bge_model` 仍必須是 `BAAI/bge-m3`，不能偷換微調模型。

## GPU 操作順序

以下命令從專案根目錄執行，每步成功後才進下一步。

### 1. 唯讀 PowerShell 盤點兩份 Snapshot

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
& .\experiments\evaluation\inspect_bge_snapshots.ps1 -HashWeights
```

腳本依 HF_HUB_CACHE、HUGGINGFACE_HUB_CACHE、HF_HOME 或預設 USERPROFILE cache 定位，印 refs/main 與兩份 snapshot 的必要檔案、size、SHA256、missing/broken link/LFS pointer。若實際快取不在預設位置：

```powershell
& .\experiments\evaluation\inspect_bge_snapshots.ps1 -CacheRoot '實際的HF快取hub目錄' -HashWeights
```

這只是檔案盤點；任一份缺檔退出 2，不表示另一份也不能用。large weight hash 可能需時。腳本不評估 tensor 內容或任意 ST module 行為。必要檔缺少時使用另一份完整的既有 snapshot；兩份都缺時停止並從既有備份取得，工具不下載、不拼接。

### 2. 指定一份確認完整的原始 BGE snapshot

以下用第一份示範，**不是斷言第一份已完整**；若第二份才完整，將尾端 hash 改為第二份。

```powershell
# 若有自訂 HF cache，將此處改為步驟1印出的實際 snapshot 絕對路徑
$bgeSnapshot = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
```

### 3. 產生 A1 新 canonical 與 A2 檢索庫（不載入模型）

```powershell
python -m experiments.evaluation.prepare_a1 --project-root . canonical
python -m experiments.a2_faq_finetuning.build_faq_candidates --project-root .
python -m experiments.evaluation.prepare_a1 --project-root . check
```

最後應明確輸出 A1=60、A2 candidates=60、A2 answers=60，且 unique IDs／Q/A 與 CSV 一致。不是只數行數。任一輸出已存在時命令會拒絕；若是先前已成功產生的同一檔案，可先用 check 驗證後沿用，不要刪除重建。若需新版本，使用各自 --output 與新的 config 路徑。

### 4. 明確建立全新 A1 索引（GPU embedding，無訓練）

```powershell
python -m experiments.evaluation.prepare_a1 --project-root . index --bge-snapshot "$bgeSnapshot" --device cuda
```

這步才實際載入原始 BGE-M3、encode 60 題、建新索引並重新載入；不是模型訓練。若成功，應有新的 index.faiss/index.pkl/version.json 且 faq_count=60。舊 `faq_bge_m3_v1`、3 題 canonical 完全保留。

### 5. 資源與實際模型載入檢查

```powershell
$loadTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run preflight --project-root . --run-dir "experiments/results/unified_faq/resources_load_$loadTag" --bge-snapshot "$bgeSnapshot" --allow-unreviewed --load-models
```

八個 ready Worker 應分別完成 `loaded_no_inference`，沒有生成 FAQ 答案。若架構資源失敗會被排除載入且命令退出 2，不能只看 A3 成功就視為全部通過。確認 bge.value.path、files_sha256、A1/A2 count=60、各 jobs.model_load_verified，以及 contracts 的 embedding_model_version。A4 文件／Dense 相符性、reranker 或其他資源若另有錯誤仍須照訊息處理，不保證這三項能解決尚未提供的所有資源問題。

### 6. 全新三題 Smoke

```powershell
$smokeTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run smoke --project-root . --run-dir "experiments/results/unified_faq/smoke_resources_$smokeTag" --bge-snapshot "$bgeSnapshot" --allow-unreviewed
```

驗收八架構各三筆 success、非空答案、正確中文、沒有 integrity violation，檢查實際回答，不僅是 worker exit=0。

### 7. 使用者確認 Smoke 通過後才執行 60 題

```powershell
$fullTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run full --project-root . --run-dir "experiments/results/unified_faq/full_resources_$fullTag" --bge-snapshot "$bgeSnapshot" --allow-unreviewed
```

Full 是八架構各 60 題、最多 480 筆成功回答；會曝光保留 Test，仍適用 README 的研究限制。Codex 本次未執行上述任何命令，不能宣稱 GPU 快取、語法、套件、索引或八架構已驗證成功。
