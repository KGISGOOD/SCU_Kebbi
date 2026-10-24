# SC1-3：Semantic Cache Threshold Comparison

狀態：已實作及靜態閱讀，沒有在開發電腦執行 Python、測試、模型或 Git 操作。
使用者已回報原 SC0／SC1-1／SC1-2 GPU 功能成功；這不代表本次 SC1-3 已通過。
此任務只新增檔案；原 benchmark、Cache、Pipeline、A0 Adapter、8 題情境與設定不改動。
不修改其他 QA／SD／Attention 架構，不重建索引、不下載、不建立 ZIP。

## 真實獨立軌跡

入口 `threshold_benchmark.py` 重用既有 `CachedA0`／`AnswerCache`／`A0Adapter`。
預設 0.75、0.80、0.85、0.90、0.95，CLI `--thresholds` 可改為任意不重複的有限 cosine
門檻 [-1,1]。CLI 優先於設定檔 thresholds，再來是五個預設；舊 config 的單值 threshold
不會把 sweep 意外縮成一組。

每個門檻／Repeat 都建立全新空 Cache、FAISS IndexIDMap2 及插入來源對照表，namespace
另綁定門檻／Repeat。因此沒有跨門檻 Entry、向量或答案的共享。只有本機模型與唯讀 A0
文件索引共用。FAISS label → Cache ID → 答案仍走原有映射與一致性檢查。

每一組都依檔案原有 query_order 逐題完整執行：BGE → Cache FAISS → Hit 回舊答案；
Miss 真正做 A0 Top-5／原 Prompt／3B 生成，再插入 Cache。不重播其他門檻的相似度
紀錄，也不借用 SC0 已生成答案填 Cache。不同門檻會產生不同後續軌跡，這是量測的一部分。
有相似度才記實測值；空 Cache 沒有候選時為 null，不補假分數。

設定、seed、問題順序、模型、Prompt 與生成參數相同。每一組單獨做 A0 warmup，不填入
Cache且不計入指標。--repeats 預設 1，每次 Repeat 重新建立所有 Cache；門檻執行順序
按 Repeat 旋轉，問題順序不變。這仍不是完整的正式效能統計或隨機化實驗。

## 資料與人工標籤

- threshold_functional.jsonl：12 題功能序列，預設使用。
- threshold_calibration.jsonl：8 題獨立候選，供觀察及選門檻。
- threshold_evaluation.jsonl：8 題保留候選，只允許使用已選單一門檻。
- threshold_selection.example.json：人工凍結門檻的範本，不是真實選擇結果，threshold=null
  必須在 Calibration 完成及審查後才可填寫。

所有資料 needs_review、只屬功能驗證規模。包含重複、改寫、相近不同意圖、否定句、
年級／年度差異、不同主題及即時資料／可能無法回答的提問。年份、實習年級等是問題
條件，不是在新增或宣稱任何真實系所規定；拒答也可能被 Cache 保存。

每筆欄位：scenario_id、question、query_order、intent_group、family_id、
expected_answer_reuse、reference_scenario_id、review_status、split。
順序必須從 1 連續遞增，Reference 必須指向先前題。單輪限制不變，不接受對話歷史。

expected_answer_reuse 為 allowed／forbidden／unverified，僅描述「目前題與指定 Reference」
這個配對，不能推廣為可重用任何同意圖答案。標籤可人工修正，不能自動 approved。
如需審查更多可能命中的來源，加入：

```json
"reuse_judgments": [
  {"reference_scenario_id": "先前存在的 Scenario ID",
   "expected_answer_reuse": "forbidden",
   "review_status": "needs_review"}
]
```

評分依每個門檻當時實際命中的原始插入題，而不是字面 Reference 或前一次 Hit 題：

- 沒有該配對標籤、任一題／配對未 approved：unverified。
- 已確認 forbidden：計入 false_reuse_count，狀態為 confirmed_pair_incompatibility。
- 已確認 allowed：只記 pair_compatible_answer_unverified；不算答案品質正確。
- 沒有任何已確認的命中配對：false_reuse_count=null，不能把「沒標籤」寫成零錯誤。
- 不根據 intent_group 自動推定安全，不把高 cosine 當成正確答案。

False Reuse 是已確認配對標籤下的不相容命中數，不是自動驗證回答事實正確性的分數。
即使配對可重用，首次生成答案仍可能錯誤，逐題保留人工 answer_quality_review 欄位為
null。整體 answer_reuse_safety_status 未發現已確認違規時仍為 unverified。

## GPU 未來功能驗證

不需要新增 GPU 模型設定，直接使用原本已成功的 config.gpu.local.json。不要往該檔
加入 thresholds 後繼續交給舊 benchmark（舊 CLI 的欄位驗證不接受新欄位）。推薦直接用
--thresholds；若想採設定檔列表，另複製成 config.thresholds.local.json，只給新入口使用，
保留原 GPU 設定。既有 ignore 規則涵蓋這類 config*.local.json，工具不寫入設定檔。

在 GPU 專案根目錄，以新目錄執行功能序列及真正 SC0 對照：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
$sc13 = 'experiments/results/algorithms/sc13_functional_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.threshold_benchmark --project-root . --config experiments/algorithms/semantic_cache/config.gpu.local.json --thresholds "0.75,0.80,0.85,0.90,0.95" --split functional --with-baseline --repeats 1 --run-dir $sc13 --allow-unreviewed
```

不帶 --with-baseline 時不執行 SC0，加速比為 null。後續可提高 --repeats，每個 Repeat
重新做真實 SC0（若啟用）與各門檻；不能用第一次答案取代後續生成。

SC0 加速比 = 同題序、相同 Repeat 集合的 SC0 端到端時間總和 ÷ 該門檻時間總和。
不是從舊八題結果推算，也不是 Generation-only。計時包含既有 Pipeline 的實際成本，
不含模型載入、warmup、離線標籤評分與結果寫檔。FAISS lookup 使用既有 cache_lookup
分項，包含一致性檢查與 ID 映射，不冒充純 C++ search 時間。生成分項也含既有 A0 的
Prompt 建構／tokenize／decode，不能當純 GPU 解碼時間。

## Calibration → 獨立 Evaluation

先用另一次新 Run：

```powershell
$scCal = 'experiments/results/algorithms/sc13_cal_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.threshold_benchmark --project-root . --config experiments/algorithms/semantic_cache/config.gpu.local.json --split calibration --with-baseline --run-dir $scCal --allow-unreviewed
```

檢視 Calibration、人工配對標籤與品質，再複製 selection 範本成獨立個人檔案（例如
config.threshold_selection.local.json）。填入實際 threshold、已完成的 calibration_run、
選擇理由及審查狀態，不能事先把某個門檻說成最佳值。

```powershell
$scEval = 'experiments/results/algorithms/sc13_eval_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.threshold_benchmark --project-root . --config experiments/algorithms/semantic_cache/config.gpu.local.json --split evaluation --selection-file experiments/algorithms/semantic_cache/config.threshold_selection.local.json --with-baseline --run-dir $scEval --allow-unreviewed
```

Evaluation 禁止多門檻 sweep；若指定 --thresholds，必須只含凍結的同一個值。檢查校準
Run completed/source verification、門檻確實曾量測、模型／程式／Prompt／生成設定相同，
以及兩組 ID／family／正規化題文不重疊。仍需人工複查語意洩漏，不保證完全獨立。
不得看 Evaluation 後反覆挑門檻並稱為同一份固定測試。

即使將標籤審核完成，本版仍標 formal_research_result=false，不會自動變成正式研究
結論。若更新輸入標籤，應建立新 Run；不要覆寫舊結果。

## 輸出與檢查

```text
run.json / identity.json / scenarios.snapshot.json
SC0/repeat_000/queries.jsonl                 僅啟用 Baseline 時
T01/repeat_000/queries.jsonl                 對應 run.json thresholds[0]
T01/repeat_000/cache_id_mapping.json         FAISS label／Cache ID／原始題
T01/summary.json                            該門檻全部 Repeat
T02/... 至所有門檻
threshold_comparison.csv / summary.json
completed.json 或 failed.json
```

逐題包含 threshold、repeat、原問題、Hit/Miss、實測 cosine、候選 ID／label、實際插入來源、
答案、完整時間與安全狀態。暖機時間另存 mapping 診斷，不計正常指標。
exact_hit_count 是 Semantic 命中中，與實際插入問題逐字相同的子集；不是另跑 Exact
Cache。semantic_hit_count 包含這些逐字重複，另列 nonexact_semantic_hit_count 以免混淆。

summary 保存查詢總次數、唯一題數、Repeat 數、Hit/Miss、已確認／未確認配對數、
Mean/Median latency、各分項平均與可選 SC0 加速比。CSV 空白 false_reuse_count 代表
null／沒有已確認命中標籤，不是零。人工核對生成答案與來源映射，不要求各門檻命中率
單調（Cache 軌跡不同），也不預先宣稱 0.95 最安全或最快。

只在所有門檻與必要來源檢查完成後產生 completed.json。錯誤保存 failed.json、traceback、
當前 threshold/repeat/scenario 與既有逐題結果，不會補造其餘紀錄或成功標記。
不支援覆寫／resume。沒有離線 score replay 冒充端到端量測的功能。

## 尚待實測與 SC1-4 建議

SC1-3 尚待 GPU 驗證：新 JSON 解析、五組 Cache 隔離、數值、來源映射、實際不同軌跡、
標籤計數與完整輸出。固定門檻仍無語意安全保證。

SC1-4 可先研究「不確定就回退 A0」：以否定／時間／條件差異作拒絕重用線索，再用
獨立開發資料評估；必要時加人工審核配對或低成本驗證。所有拒絕／驗證成本都應納入
延遲，不能用 gold FAQ／答案標籤作線上決策。本任務沒有開始實作這些安全策略。
