# 資科萬事通 V6

此 `v6-only` 分支僅提供 **FAQ → Semantic Cache → Dense RAG → Speculative Decoding** 單輪問答。歷史 V1～V5 與研究資料留在 `rh`，本分支不提供其他版本啟動模式。

獨立部署版本不等於答案已人工核准或完整 SD Lossless 通過：FAQ、Cache 維持 Experimental 安全規則及尚未校準的門檻，自動快取標記為 `unverified`。

## 架構與必要模組

```text
POST /chat → 共用 BGE-M3（同題只計算一次 Embedding）
           → FAQ / 記憶體 FAISS → 命中直接回答
           → Semantic Cache   → 命中直接回答
           → Type1、Type2 Dense RAG
           → Llama 1B Draft 候選 → Llama 3B Target 驗證
           → 相同輸入的額外 Target Greedy Token 一致性檢查
           → 回答；符合原有寫入條件才自動加入 Cache
```

FAQ 命中不查 Cache，也不呼叫 Draft/Target；Cache 命中不呼叫生成模型。Token 對照失敗拒絕回傳與快取，沒有複製 Target Greedy 答案來假裝通過。

```text
baseline_api/
  server.py                    # 唯一 API、六項終端機輸出、固定 CSV
  integrated_chat.py           # V6 共用路由 runtime
  speculative_chat.py          # 選用 V6 CLI，與 API 共用流程
  speculative.py               # SD Adapter 與 Greedy 對照
  faq.py / faq_review.py        # FAQ CSV、審核、接受規則
  cache.py                     # 快取安全規則、期限、資源保護
  client_example.py            # HTTP 呼叫範例，不操作硬體
experiments/
  baselines/rag_baseline/       # RAG、完整 Prompt、索引建置
  baselines/faq_semantic_matching/ # FAQMatcher 及必要載入依賴
  algorithms/semantic_cache/    # AnswerCache
  algorithms/speculative_decoding/ # 原有 Target 驗證引擎
  shared/                      # 模型、Embedding、FAISS、資料與身分驗證
  configs/integrated_baseline_v1.experimental.speculative.json
  datasets/                    # 138筆 FAQ CSV 與獨立審核檔
  models/                      # 本機模型，Git 忽略
  vector_stores/                # 本機索引，Git 忽略
  results/                     # 執行紀錄，Git 忽略
畢業專題_爬蟲/type1、type2/       # 原始知識來源
```

`shared/index_chunk_reference.py` 保留從舊 A4 建置器逐字抽取的分塊參考函式，供 Dense Index 完整性檢查使用，不是 Hybrid RAG 推論。FAQ Matcher 仍 import `config.py` 與 `faq_loader.py`，因此保留；V6 由 `FAQFallback` 注入 CSV 與共用 Embedding，不走舊 JSONL constructor。SD 的 `prompt_lookup.py` 是引擎條件式依賴；V6 不啟用 Prompt Lookup。

## 模型、環境與安裝

- 建議 Python 3.11，優先使用原本已驗證的 Windows GPU 環境。
- 原始 BAAI/bge-m3：FP32、L2 正規化、512 tokens；維持範本指定 revision 與內容 Hash。
- Target：本機 Meta Llama-3.2-3B-Instruct。
- Draft：本機 Meta Llama-3.2-1B-Instruct。
- 現有 Type1/Type2 Dense FAISS 索引：`index.faiss`、`index.pkl`，以及原有版本與建置身分檔。
- 現有 SD Guard 僅接受 Transformers **5.16.1**；不相容直接報錯，不回退成一般生成。

新環境安裝範例；PyTorch CUDA wheel 須配合驅動，不必重裝原本已驗證的環境：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r experiments/requirements.txt -r baseline_api/requirements.txt
python -m pip check
```

開發端未實際安裝此 GPU 套件組合。模型須先依授權取得；程式只離線載入，不自動下載。Target、Draft、BGE 與 Cache 同時駐留，不保證 24GB VRAM 足夠，OOM 不會偷偷改精度。

### 第一次部署：取得模型與建立 FAQ 索引

已有原本驗證過的模型時，直接複製完整模型目錄並填入 local 設定，不必重新下載。新部署須先在 Hugging Face 取得兩個 Meta Llama 模型的存取授權，再於**部署電腦**執行以下下載步驟；這與服務本身的離線載入分開：

```powershell
python -c "from huggingface_hub import login; login()"
python -c "from huggingface_hub import snapshot_download; snapshot_download('BAAI/bge-m3', revision='5617a9f61b028005a4858fdac845db406aefb181', local_dir='LOCAL_MODELS/bge-m3')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.2-3B-Instruct', local_dir='LOCAL_MODELS/Llama-3.2-3B-Instruct')"
python -c "from huggingface_hub import snapshot_download; snapshot_download('meta-llama/Llama-3.2-1B-Instruct', local_dir='LOCAL_MODELS/Llama-3.2-1B-Instruct')"
```

新下載的 Llama 預設使用模型儲存庫當時版本；若要重現既有部署，使用既有模型副本或明確加入既有身分紀錄中的 `revision`，不要把新下載等同於舊模型版本。BGE 仍須通過既有內容 Hash，不能降低驗證以配合下載結果。不要把 Hugging Face 存取權杖寫入設定或 Git。

FAQ 不需要另跑舊版建置器。啟動 V6 時，`FAQFallback` 讀取兩欄 `experiments/datasets/faq_精簡問答_138筆.csv` 及 `faq_csv_review.csv`，驗證138筆與審核 Hash，使用共用 BGE-M3 建立一次**記憶體 IndexFlatIP**；不從 Frozen 母資料取題，不保存 FAQ 磁碟索引。可先做不載入模型的資料檢查：

```powershell
python -m baseline_api.faq_review --check-only
```

保留 `畢業專題_爬蟲/type1/`、`type2/` 的原始檔案與相對路徑。其餘爬蟲蒐集程式不被 V6 載入，暫作來源追溯保留，部署服務只需要這兩個知識資料目錄。

## 本機設定與啟動

保留範本，只建立本機設定；若 local 已存在，不要覆寫：

```powershell
$local = 'experiments/configs/integrated_baseline_v1.experimental.speculative.local.json'
if (Test-Path $local) { throw '本機設定已存在，請直接檢查原檔，勿覆寫' }
Copy-Item experiments/configs/integrated_baseline_v1.experimental.speculative.json $local
```

編輯 local 的 `model`、`bge`、`knowledge_index`、`speculative.draft_model` 指向本機資源，路徑可相對於專案根目錄。保留以下參數：

- Prompt：`baseline_senior_v2`。
- FAQ：Experimental、threshold 0.9、min_margin 0.05。
- Cache：Experimental、自動寫入、threshold 0.9、容量128、TTL3600秒、不持久化。
- Draft tokens：4；`verify_greedy=true`，不可停用 Token 一致性 Guard。

完整 System Prompt 在 `experiments/baselines/rag_baseline/prompts.py`，以 `{context}` 填入檢索內容，沒有傳入對話歷史。100字目前是 Prompt 約束及超長觀測，不會直接截斷回答。

預設 FP16 沿用原設定；不同前向路徑仍可能產生 Token 分歧，由 Guard 阻擋。另行使用 FP32 時，既有政策會停用 TF32，但仍須驗證 VRAM 與 Token 等價性。

GPU PowerShell：

```powershell
Set-Location "C:\Users\rsy\Desktop\SCU_Kebbi"
python -m baseline_api.server --config experiments/configs/integrated_baseline_v1.experimental.speculative.local.json --host 127.0.0.1 --port 8000
```

另一個視窗測試：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
$body = @{question="東吳大學資料科學系的聯絡信箱是什麼？"} | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```

`/health` 載入時503，ready 後200。`/chat` 只接收 `question`，保留 `answer`、`success`、`answer_source`、`stage_timings`、`speculative` 等原有欄位。單 worker 依序執行，忙碌時503，不堆積 GPU 請求。Ctrl+C 停止。不要使用多 worker 或 reload 重複載入模型。外部電腦連線的監聽位址與防火牆由部署人員另行設定。

選用非 API 的 V6 CLI：

```powershell
python -m baseline_api.speculative_chat --config experiments/configs/integrated_baseline_v1.experimental.speculative.local.json
```

## 紀錄與計時

每筆成功 HTTP 問答自動追加至 **`experiments/results/api_logs/v6_chat_history.csv`**，無須 `--run-dir`。UTF-8 BOM、12欄，處理逗號、引號及換行，重啟不清空。終端機顯示問題、答案、來源、各階段耗時、總耗時及 Output Tokens。

FAQ/Cache 命中沒有新 LLM Token，Output Tokens 與未執行的生成階段留空。沿用既有測量，Cache 欄為查詢加寫入耗時。總耗時使用共用引擎的 `total_seconds`，排除初次載入及 CSV 寫入，包含額外 Greedy 驗證；不能將它當成純 SD 效能。

CSV 有寫入鎖、防止同一請求重複追加與欄位檢查。失敗回傳 `csv_recording_failed`，不宣稱紀錄成功。HTTP timeout 後才完成的推論不會當成成功 HTTP 問答追加 CSV。

需要詳細證據時，啟動命令加上 `--run-dir experiments/results/v6_api_<新的唯一名稱>`，保存設定、身分、Token、錯誤及原始 JSONL。不可用舊目錄覆寫。候選總數、接受 Token 數及接受率無可靠統計時為不可用，不以 forward 次數推估。

## 索引重建與保護範圍

已有可用索引時不重建。必要時沿用原建置器及512/128分塊設定：

```powershell
python -m experiments.baselines.rag_baseline.build_index --offline --embedding-model-path "<本機BGE目錄>" --device cuda --index-dir experiments/vector_stores/type1_type2_bge_m3_v1
```

增加 `--check-only` 只驗證現有索引。原始知識、FAQ及審核狀態、Prompt、模型與索引不因整理而修改。`.local.json`、模型、索引及結果不得加入 Git；忽略檔案不表示已備份。

## 驗證界線

開發端只做 CPU 語法、Import、CLI、設定、來源完整性、替身分流／HTTP／CSV 測試。真實 Llama 相容性、VRAM、Token 完全一致、回答品質及速度仍須 GPU 實測；CPU 通過不代表這些項目通過。
