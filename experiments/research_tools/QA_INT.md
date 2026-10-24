# QA／INT 共用研究題庫入口

本輪只完成程式實作與人工靜態審查，未執行Python或模型。既有SD／SC／TI Smoke成功不能視為這兩個新Runner已驗證。

## 入口與資料契約

- `experiments.research_tools.qa_benchmark`：八架構逐一在獨立子程序載入，重用`evaluation.wrappers.Architecture`、原資源Preflight、離線載入、Greedy生成擷取、種子及CUDA同步。使用既有`evaluation/config.json`格式。沒有改動模型、檢索或生成核心。
- `experiments.research_tools.int_benchmark`：重用`SDEngine`、`IntegratedPipeline`、既有`report()`的Token一致性及計時公式。使用既有`integrated_efficiency`設定格式，`--mode all|int0|int1|int2|int3`。INT2／INT3解碼器由設定檔`decoder=standard|draft_1b|prompt_lookup`指定；3B／1B與BGE均需既有本機路徑。
- 舊`experiments.evaluation.run`的原始CSV FAQ60與舊`integrated_efficiency.benchmark` functional入口完全保留，既有結果不修改。
- 先以`adapters --target qa|int`選Historical／Calibration／Evaluation、ID、題型或工作負載；新Runner讀取已完成、Hash相符的Adapter目錄，不直接接受任意JSONL。
- 原題庫、來源原始位元組SHA、Adapter各檔SHA、Review View、設定檔與工具程式Hash都保留並在執行前後重新比對。QA另驗證原FAQ60來源、八架構程式、套件、穩定環境及模型／索引資源；INT沿用引擎來源驗證。變更會阻止完成標記。
- `--view`須與匯出時相同；無View保持needs_review。功能測試需`--allow-unreviewed`，輸出永遠provisional。這個旗標不會把functional資料升級為正式結果。
- 正式Calibration／Evaluation必須經人工合併View、相同設定檔原始Hash及相同Adapter工具版本；Evaluation匯出還需非provisional且來源驗證通過的Calibration實測收據與不相交family／ID。Runner無門檻、解碼或題目篩選覆寫參數；換參數需先重新做Calibration再凍結Evaluation。

## QA評分與Prompt隔離

`requests.json`嚴格只允許`question_id`與`question`。Worker再次檢查欄位與Hash，實際只呼叫`Architecture.answer(question)`。Question ID只關聯結果，不傳入模型，更不作FAQ Gold ID。

`scorer_only.jsonl`不作Worker請求；父程序在模型執行後，才把參考答案、expected_behavior、來源與審核／曝光資訊附加到`results.scorer.jsonl`。`answer_review.csv`為UTF-8 BOM，方便Excel人工評分。模型Prompt及輸入Token可在各架構原始紀錄核查；不能因問題本身包含與答案重複詞語就認定發生程式標籤洩漏。

目前Review View未提供獨立、明確審核的唯一Gold FAQ映射。因此不把`intent_id`、`supporting_intent_ids`或Historical Question ID自動升格為Gold：`gold_faq_id=null`、FAQ Accuracy=`unavailable_no_human_gold_mapping`。若需檢索準確率，下一步先審核多Gold／無適用FAQ情形，再增加獨立Gold映射契約。答案品質仍需人工評分，非字串相等即正確。

A1、A2-1、A2-2只做FAQ匹配，`generation_seconds`、生成Token數與Tokens/sec為null，狀態`not_applicable`；其完整查詢延遲仍可描述，但不能與生成架構強制比較「生成速度」。其他架構也只是描述性時間，答案品質未審核前不宣稱品質不變的加速。

QA輸出：`run.json`、`identity.json`、`worker_contract.json`、`requests.json`、各架構`.log`／`.model.json`／`.raw.jsonl`／完成或失敗收據、`results.scorer.jsonl`、`answer_review.csv`、`summary.json`。Worker前後完整性通過才提交該架構結果；其餘raw保留但`result_committed=false`。任何Worker失敗則整體`failed.json`，沒有整體`completed.json`。

## INT Cache填充與量測

每個Variant／Repeat／Warm-up建立獨立`IntegratedPipeline`與空Cache，並按相同順序處理：

1. `initial_fill.jsonl`：用該Variant本身的真實A0／解碼器回答，按既有流程填充。INT0／INT2無Cache時也執行相同請求作控制；不把INT0答案灌入其他Variant。
2. `measured.jsonl`：沿用上述同一個Pipeline，所以可命中初始填充。不同Variant不共享答案、FAISS Cache或ID映射。
3. 下一Repeat重新從空Cache與相同填充計畫開始；Warm-up的Cache不帶入正式量測。

每一階段保留獨立`results.jsonl`、`comparison.csv`與`summary.json`；根目錄另存`measurements.raw.jsonl`、身分、來源、完整情境快照與分階段摘要。填充成本在`initial_fill/summary.json`的各Variant `total_online_seconds`，不進measured命中率或速度比分母；Warm-up另標示排除。若評估整段工作負載總成本，須另將兩階段相加，不可只報measured冒充總成本。

計時仍為原`CachedA0`量得的獨立online wall-clock，沒有改成共享檢索成本加總。Miss仍完整A0，Hit沿用原答案；既有Token／Prompt配對驗證與CUDA計時都保留。INT0對照下的Hit不是lossless decoding比較。門檻不保證安全：未審核配對保持unverified，已審核的相容配對也不代表生成答案已通過品質審核。

原工作負載使用既有`--workload-id`，保留原fill及measured計畫。新增`--initial-fill-ids`只供INT functional搭配明確`--ids`做Smoke，不可冒充凍結正式工作負載；source workload不能覆寫fill。不同Split、錯誤來源、重複事件ID、非先前配對、錯誤phase/order與被拒絕題目會被阻擋。現有INT仍為單輪；多輪history不會被靜默丟棄。

## GPU：QA Calibration三題

在GPU專案根目錄執行。`$qaConfig`指向GPU先前通過八架構的實際設定檔（相同格式）；若使用GPU本機檔，請替換此變數。每次GUID新目錄，不重用舊Run。

```powershell
$qaConfig = 'experiments/evaluation/config.json'
$qaData = 'experiments/results/research_tools/qa_cal3_data_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --target qa --split calibration --purpose functional --limit 3 --settings $qaConfig --output-dir $qaData
if ($LASTEXITCODE -ne 0) { throw 'QA adapter failed' }
$qaRun = 'experiments/results/research_tools/qa_cal3_run_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.qa_benchmark --project-root . --adapter-dir $qaData --config $qaConfig --architectures all --run-dir $qaRun --allow-unreviewed
```

檢查八架構逐題各3筆且`committed=true`；每個`.log`無錯誤，整體completed存在；Question ID為UR1 ID且gold_faq_id為null；Prompt未附帶評分欄位；A1/A2生成指標not_applicable。模型尚未準備好的架構會明確失敗，不自動建索引／訓練／下載。

## GPU：INT Calibration三題，含初始填充

使用既有GPU本機INT設定檔；檔案應已設定本機3B、1B、BGE、A0知識庫路徑與`decoder=draft_1b`。Warmups／Repeats沿用設定檔，不在CLI偷偷覆寫。Prompt Lookup另用事先選定且另存的設定檔、獨立新Run執行。

```powershell
$intConfig = 'experiments/algorithms/integrated_efficiency/config.gpu.local.json'
$intData = 'experiments/results/research_tools/int_cal3_data_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --target int --split calibration --purpose functional --ids UR1_A001,UR1_H001,UR1_R001 --initial-fill-ids UR1_A001 --settings $intConfig --output-dir $intData
if ($LASTEXITCODE -ne 0) { throw 'INT adapter failed' }
$intRun = 'experiments/results/research_tools/int_cal3_run_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.int_benchmark --project-root . --adapter-dir $intData --config $intConfig --mode all --run-dir $intRun --allow-unreviewed
```

這是三個不同Question ID、每Variant每Repeat **1次initial_fill＋3次measured**（另有Warm-up）。要做冷快取Smoke就省略`--initial-fill-ids`。檢查兩階段分開、每Variant measured題序一致、Cache namespace不同、填充來源可追溯、Hit安全性unverified、Miss配對INT0 Token一致。既有填充型工作負載則將`--ids ... --initial-fill-ids ...`換為適當`--workload-id`與相符來源Split，不可把Evaluation改名Calibration。

## 正式研究前仍需完成

- 在GPU實際驗證這兩個新Runner；本機未執行任何測試。
- 完成人工題目／答案或行為／所有可能Cache來源配對審核與仲裁，凍結View及輸入。
- 現有完整Cache workload主要來自Evaluation；Calibration三題明確序列僅是功能驗證。需另行授權設計、審核並凍結Calibration正式工作負載，再做門檻選擇，不可拿Evaluation反覆調參。
- QA需獨立審核Gold FAQ映射才可算FAQ準確率；所有生成答案需要人工品質評估，不自動宣稱正確。
- 多輪QA／INT能力不在這兩個單輪入口內；不適用項目須排除或另建受審查Adapter。
- 先完成Calibration設定選擇及來源驗證收據，再凍結Evaluation、一次性執行；本輪不啟動230題或正式實驗。
