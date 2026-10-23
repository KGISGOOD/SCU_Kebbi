# 演算法第一階段：單輪 Semantic Cache

這個目錄目前只實作答案 Cache。Speculative Decoding、Attention Topic 保持規劃狀態。
不使用 Ollama，不改寫既有八架構，不建立／覆寫知識庫，不儲存 Cache。

## 實際共用與限制

- 重用 `experiments.baselines.rag_baseline.pipeline.RAGPipeline` 的檢索與生成流程。
- 重用 A0 Prompt、`Retriever`、`FAISSVectorStore`、`BGEM3Embedder` 編碼方法、
  `LLMLoader.generate`。為強制 local-only，adapter 明確初始化既有類別所需屬性，
  不呼叫會依預設設定載入的便利 factory；這個薄介面仍須實機確認。
- Transformers 與 SentenceTransformer 載入均使用本機目錄與 `local_files_only=True`；
  啟用 offline 環境變數，缺檔直接失敗。不要把 Hub ID 填入模型路徑。
- 使用現有 dense A0 Top-5，greedy、指定 seed／dtype，無量化或 CPU offload。
- 不截斷文件。沿用現有 A0 生成器的 2048 input-token 上限，但先檢查完整 Prompt，
  超長直接失敗；不得以這次失敗認定 Cache 演算法錯誤。若實際資料經常超長，
  需要另外確認 A0 輸入政策，不在本任務默默改動它。
- 原生 A0 類別載入知識庫仍使用可信專案的 pickle，不能指向不可信檔案。
- 開發機沒有執行 Python、測試或模型。套件相容性、生成與數值均未驗證。

## 模式與資料流

`none`：不查／不存 Cache，每題進 A0。

`exact`：依原始問題字串完全比對，不 strip、大小寫轉換或 Unicode 正規化。

`semantic`：問題經本機 BGE-M3、L2 正規化，搜尋獨立 `IndexIDMap2(IndexFlatIP)`。
分數達門檻才回傳答案；否則進 A0。0.75 只是可覆寫的候選值，不是最佳值或正確性保證。

每筆 ID 是 namespace 與原始問題的 SHA-256。FAISS int64 label 透過獨立映射找到
cache_id，再由字典取得答案；與任何容器順序無關。沒有 LRU、淘汰、Semantic Forest、
KV Cache、跨 Session 共享或持久化。程式為單執行緒順序入口，不支援併發服務。

namespace 包含本機模型／BGE／知識庫內容 hash、相關 Python 原始碼 hash、Prompt 版本、
生成設定、Tokenizer 特殊 Token、套件資訊。每次請求核對 namespace；執行前後另外核對
磁碟來源。來源在執行中應保持唯讀，若最後核對失敗，不產生 completed.json。
這不是每題重新雜湊整份模型，亦不保證偵測執行中短暫改動後又還原的外部操作。

## GPU 電腦準備

使用既有 GPU 環境，不安裝／下載。需要 torch、transformers、sentence-transformers、
numpy、FAISS、langchain-core 及 A0 原有依賴。SentenceTransformer 的 local_files_only
參數預期對應先前回報的 6.1.0，但尚未實際呼叫確認。

在專案根目錄執行，先複製範本成新的個人設定（不改八架構設定）：

```powershell
if (Test-Path experiments/algorithms/semantic_cache/config.gpu.local.json) { throw 'Local config already exists; edit it instead.' }
Copy-Item experiments/algorithms/semantic_cache/config.example.json experiments/algorithms/semantic_cache/config.gpu.local.json
notepad experiments/algorithms/semantic_cache/config.gpu.local.json
```

若個人設定已存在，不要覆寫。將 model、bge 改成既有本機模型完整目錄或專案相對路徑，
確認 knowledge_index 指向原 A0 dense 索引。JSON 中 Windows 路徑可使用 `/`。
CPU 必須同時設定 device=cpu、precision=fp32；CUDA 可使用 fp16 或硬體支援的 bf16。
模型與知識庫會完整讀取並雜湊，初次準備可能耗時；不計入單題延遲。

先跑容易核對的 Exact 模式，使用從未存在的新 Run：

```powershell
$cacheRun = "experiments/results/algorithms/exact_" + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.benchmark --project-root . --config experiments/algorithms/semantic_cache/config.gpu.local.json --mode exact --run-dir $cacheRun --allow-unreviewed
```

再以另一個全新 Run 比較三模式：

```powershell
$cacheRun = "experiments/results/algorithms/cache_all_" + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.benchmark --project-root . --config experiments/algorithms/semantic_cache/config.gpu.local.json --mode all --threshold 0.75 --run-dir $cacheRun --allow-unreviewed
```

不要重用失敗或成功目錄；程式不 resume、不覆寫。每種模式重新建立空 Cache，單獨暖機，
每種模式固定順序；這只是小型開發診斷，不能以單次執行宣稱正式加速效果。

## 八筆待人工審核情境

`semantic_cache/scenarios.jsonl` 包含 P05 制度／名額、P06 就業是否需要研究所、
P07 推甄／考試，以及一題範圍外天氣探測。不是新增 FAQ 事實或正式 Test。

1. SC01：P05 首次插入。
2. SC02：完全重複 P05。
3. SC03：同義改寫 P05，保留制度與名額。
4. SC04：相近主題但不同 P06 意圖。
5. SC05：完全重複 P06。
6. SC06：相近主題但不同 P07 意圖。
7. SC07：應 Miss 的系務外天氣問題。
8. SC08：完全重複 P07。

全部 needs_review；未允許未審核資料時直接阻擋。預設情境需 --allow-unreviewed，
結果標 provisional。Gold／意圖資訊只在 benchmark 事後評分，不傳進 Cache 或 A0。

GPU 人工核對：none 應為 0 Hit／8 Miss；exact 應為 3 Hit／5 Miss，SC02/05/08
分別取回 SC01/04/06 的 cache_id 與完全相同答案。Semantic 不預設固定命中數：
檢查 SC03 是否正確重用，SC04/06/07 是否錯誤命中；高分不代表答案可共用。

## 輸出與計時

- run.json：有效設定、資料來源 hash、用途與 provisional。
- identity.json：模型／資料／程式來源識別。
- 每模式 0001.json 等：答案、候選問題／ID、相似度、Hit、原始插入題 ID、時間。
- summary.json：Hit/Miss、平均時間與意圖不相容命中數。
- completed.json：只有全程完成且來源核對通過才建立。
- failed.json／逐題 .failed.json：例外與 traceback。既有成功逐題檔保留，但不能因而
  將整次失敗的 Run 當作完成實驗。

端到端包含 identity、Cache Embedding、搜尋、A0、插入更新、必要 GPU 同步；不把 Hit
設成零。分項包含 A0 retrieval、Prompt 長度檢查、生成呼叫。生成呼叫包含既有 A0 的
Prompt 建構、tokenize、generate、decode，不是純 GPU decoding。端到端不含結果寫檔／
終端輸出、模型載入與暖機。錯誤情況仍保存失敗階段與總耗時。

`intent_reuse_proxy` 只比較來源意圖，不驗證回答品質；正確意圖仍可能有錯誤生成。
所有模式的答案需人工審核。僅非空、成功返回的字串可插入，但本版沒有事實驗證器；
拒答也可能被保存。沒有檢索命中、模型成功載入，都不能直接算回答正確。

## 文獻定位

這是自行實作的問題層級答案 Cache，參考 GPTCache 的先查快取再呼叫模型流程：
https://aclanthology.org/2023.nlposs-1.24/

固定門檻沒有 vCache 的錯誤率保證：
https://proceedings.iclr.cc/paper_files/paper/2026/hash/9559cd2116de7a8f5672eac3fcd232cc-Abstract-Conference.html

也沒有論文 SmartCache 的上下文索引與跨 Session KV 管理：
https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb74b63d225f846e6032bf3e3ab0f4ec-Abstract-Conference.html
