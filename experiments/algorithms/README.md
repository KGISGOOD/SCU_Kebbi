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
- A0 知識庫沿用 `FAISSVectorStore.load()` 的 `index.faiss`＋`index.pkl` 格式，
  不要求 `version.json`，也不借用 FAQ Matching 索引的版本檔。保留索引類型／IP metric、
  Embedding 維度、非空文件數及完整映射檢查。`identity.json` 的
  `resources.knowledge_index` 分別保存兩個檔案的 SHA-256，結束時重新核對；
  這是內容識別，不代表已證明原建置模型或向量正規化來源，仍須使用已確認的 A0 索引。
- 開發機沒有執行 Python、測試或模型。先前 Exact 的 8 題 3 Hit／5 Miss 為使用者
  回報的 GPU 結果；本輪正規化、映射檢查與新輸出尚未在 GPU 驗證。

## 模式與資料流

`none`：不查／不存 Cache，每題進 A0。

`exact`：先 NFC Unicode 正規化、移除首尾空白、將連續空白縮成一個半形空白，再完全
比對。保留大小寫、標點、數字、否定詞，不做簡繁轉換或模糊匹配。原始問題仍原樣保存，
Miss 時原樣傳給 A0；正規化僅用於 Exact key。這是自然語言問題的明確政策，不適合將
排版有意義的程式碼當成 Exact 重用輸入。版本為 `nfc_strip_collapse_whitespace_v1`。

`semantic`：問題經本機 BGE-M3、L2 正規化，搜尋獨立 `IndexIDMap2(IndexFlatIP)`。
分數達門檻才回傳答案；否則進 A0。0.75 只是可覆寫的候選值，不是最佳值或正確性保證。

每筆 ID 是 namespace 與 key 的 SHA-256（Exact 為正規化 key，Semantic 為原始問題）。
namespace 另外綁定模式與正規化版本，因此不同模式不共享 Cache 或 ID。
FAISS int64 label 透過獨立映射找到
cache_id，再由字典取得答案；與任何容器順序無關。沒有 LRU、淘汰、Semantic Forest、
KV Cache、跨 Session 共享或持久化。程式為單執行緒順序入口，不支援併發服務。
每次搜尋及插入都檢查 key／Entry／namespace；Semantic 另檢查 FAISS 實際 id_map、
向量數與標籤對應。插入後再核對。索引新增若失敗，Cache 失效並停止，不繼續提供答案。
同 key 不允許覆寫成另一個答案；這版採 append-only，不提供更新／刪除介面。

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

根目錄 `.gitignore` 已限定忽略 `/experiments/algorithms/**/config*.local.json`，
範本仍可追蹤。本機唯讀 `git ls-files` 未列出此類個人設定；GPU 若曾追蹤它，新增 ignore
不會自動解除追蹤，不應宣稱已防止 Pull 覆蓋。可在 GPU 唯讀確認：

```powershell
git check-ignore -v -- experiments/algorithms/semantic_cache/config.gpu.local.json
git ls-files -- experiments/algorithms/semantic_cache/config.gpu.local.json
```

第二個指令應無輸出；若列出檔案，先人工處理其追蹤與備份，不在本輪執行 Git 變更。
忽略規則不是備份，也不抵禦手動刪除；本工具從不寫入個人設定。

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

另新增 `semantic_cache/scenarios_normalization.jsonl`，保留原 8 題不變。這 4 筆使用
P05／P06 與首尾空白、Tab、換行變體；Exact 預期 2 Hit／2 Miss，None 為 0 Hit／4 Miss。
尚未實際執行。GPU 可使用新目錄驗證：

```powershell
$cacheRun = "experiments/results/algorithms/normalization_" + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.benchmark --project-root . --config experiments/algorithms/semantic_cache/config.gpu.local.json --mode exact --scenarios experiments/algorithms/semantic_cache/scenarios_normalization.jsonl --run-dir $cacheRun --allow-unreviewed
```

目前只支援獨立單輪問題。`CachedA0.answer(query, history=...)` 在非空歷史時明確失敗；
benchmark 也拒絕含非空 history/messages/context/session_id/dialogue_id 的情境，避免默默
忽略上下文。這不會自動辨識字串中的「那它呢」等隱含指代；呼叫方必須確認每題可獨立
理解，不能把多輪問題刪掉歷史後當作單輪輸入。沒有 context-aware 重用保證。

## 輸出與計時

- run.json：有效設定、資料來源 hash、用途與 provisional。
- identity.json：模型／資料／程式來源識別。
- scenarios.snapshot.json：本輪完整題序與人工標註快照。
- 每模式 0001.json 等：答案、候選問題／ID、相似度、Hit、原始插入題 ID、時間。
- 每模式 results.jsonl：相同逐題紀錄的 UTF-8 JSONL，依序保存，方便分析。
- summary.json：Hit/Miss、平均／中位數／總時間、分項平均及意圖相容／不相容命中數。
- comparisons.json：有 None 對照時，逐題保存答案文字是否相同與節省秒數；
  文字相同不代表答案正確，文字不同亦不必然表示錯誤重用。
- completed.json：只有全程完成且來源核對通過才建立。
- failed.json／逐題 .failed.json：例外與 traceback。既有成功逐題檔保留，但不能因而
  將整次失敗的 Run 當作完成實驗。

端到端包含正規化、identity、Cache Embedding、搜尋與一致性檢查、A0、插入更新、必要 GPU 同步；不把 Hit
設成零。分項包含 A0 retrieval、Prompt 長度檢查、生成呼叫。生成呼叫包含既有 A0 的
Prompt 建構、tokenize、generate、decode，不是純 GPU decoding。端到端不含結果寫檔／
終端輸出、模型載入與暖機。錯誤情況仍保存失敗階段與總耗時。
Hit 的本次 retrieval／generation 明確為 0，表示沒有執行 A0，不代表整題零延遲；
原始插入題 ID 與答案來源問題可追溯到先前 Miss 的完整檢索紀錄。
相似度使用正規化 float32 inner product；Exact／None／空 Cache 沒有 cosine 時記 null，
不虛構分數。近似向量浮點誤差仍需 GPU 驗證。

`intent_reuse_proxy` 只比較來源意圖，不驗證回答品質；正確意圖仍可能有錯誤生成。
所有模式的答案需人工審核。僅非空、成功返回的字串可插入，但本版沒有事實驗證器；
拒答也可能被保存。沒有檢索命中、模型成功載入，都不能直接算回答正確。
結果明確記錄 `semantic_safety_verified: false`。固定門檻不能防止相近不同意圖的錯誤
命中，本版沒有語意驗證器。相同意圖不等於相同答案需求，人工仍應核對條件與回答。

三模式使用同一個已載入的 A0、BGE、知識庫、生成參數、seed 與相同題序；各自空 Cache、
各自暖機。順序固定 none/exact/semantic，沒有多次重複／順序平衡，不足以支持正式效能
結論。完整一致性掃描的成本已包含計時，小題庫可用，但不是大規模 Cache 的最佳化實作。

## 文獻定位

這是自行實作的問題層級答案 Cache，參考 GPTCache 的先查快取再呼叫模型流程：
https://aclanthology.org/2023.nlposs-1.24/

固定門檻沒有 vCache 的錯誤率保證：
https://proceedings.iclr.cc/paper_files/paper/2026/hash/9559cd2116de7a8f5672eac3fcd232cc-Abstract-Conference.html

也沒有論文 SmartCache 的上下文索引與跨 Session KV 管理：
https://proceedings.neurips.cc/paper_files/paper/2025/hash/fb74b63d225f846e6032bf3e3ab0f4ec-Abstract-Conference.html
