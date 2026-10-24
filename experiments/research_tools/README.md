# 共用研究題庫：人工審核與 Benchmark Adapter

本輪只完成程式與靜態審查；沒有執行 Python、測試、模型、安裝或 Git。
使用者已回報來源題庫在 GPU 結構驗證通過，但這不是人工 Gold，也不是本輪新增工具已通過測試。
只新增 experiments/research_tools/；原230題、原固定60題、資料集manifest、八架構、演算法與索引均不修改。

所有工具只用標準庫；只讀題庫及實際引用來源的原始位元組Hash。輸出限於
experiments/results/research_tools/ 下的全新目錄；不覆寫、不resume，錯誤留failed.json與traceback。
completed.json只代表資料匯出／合併完成，不代表模型或實驗成功。

## A：人工審核

GPU/分析電腦根目錄（開發機本輪不執行）：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
$reviewBundle = 'experiments/results/research_tools/review_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.review export --project-root . --output-dir $reviewBundle
```

輸出questions.csv、cache_pairs.csv、turns.csv、export_manifest.json，CSV均為UTF-8 BOM。
Excel建議「從文字/CSV」以UTF-8匯入，ID、時間戳及JSON欄指定文字，避免自動改格式。
先保留原匯出檔，另存人審副本。可依family、split、題型及Train曝光篩選／一起檢視。
來源路徑／Hash、題目、候選答案、關聯、期望行為等展示欄唯讀；修改會使合併失敗，
不會把Excel改寫的題目偷偷當新版原始題庫。

### questions.csv

- question_status：verified / rejected / needs_review，審核問題合理性、題型、來源意圖與期望行為。
- answer_status：獨立的verified / rejected / needs_review，不能僅因候選答案非空就verified。
- answer_action：verified時必須明確填keep_candidate、override或unavailable。
  keep_candidate是人工確認引用來源支持候選；override須在answer_override填審核答案；
  unavailable代表人工確認沒有可用事實答案、採原expected_behavior評估拒答／釐清，不是捏造Gold。
- reviewer、reviewed_at、evidence：任何verified/rejected決定皆必填。
  時間範例格式為2026-10-24T14:00:00+08:00，應填**實際**審核時間；不是要照抄範例。
  evidence填來源版本／條款／適用年度，rejected時填駁回理由；notes記錄疑點與仲裁。
- verified答案同時要求question_status=verified。題型／意圖有錯可先rejected或needs_review，
  不在本工具偷偷改原始問題；需要修訂來源時另建授權版本。

### cache_pairs.csv

顯示origin/current兩題、各自候選答案、family、split、來源、Train曝光與候選reuse標籤。
包含既有關聯、同family同split候選，以及每個workload所有可能較早命中來源，包含跨family
和同Question ID重訪的self pair。可篩current_family或兩個Question ID，無須逐筆對照原JSON。

reuse_status使用三種狀態；verified時reuse_decision必須為reusable或not_reusable，不能是unknown。
配對有方向：原始答案足以回答目前問題，不表示反向也成立；工具不自動鏡像核准。
verified/rejected同樣需要審核人、時間及evidence。不能僅憑相似度、同FAQ或同family判安全。
此表核准的是「問題pair的重用相容性／候選答案」，**不是之後任何生成答案都安全**；
實際Benchmark仍須保留actual cache origin及model answer再人工評估品質。未標註的實際來源仍unverified。

### turns.csv

保留Conversation/Turn/Family/Split、完整腳本歷史、來源Question ID與必要歷史候選。
turn_status也用三態；verified時明確turn_action=keep_candidate或override。
override須填topic_label_override（initial/same_topic/shift）及required_history_override的JSON陣列；
只能引用該輪真正較早的user turn IDs，不可引用未來／不存在輪次。initial僅能是第一輪。
中性assistant歷史是原始腳本，不是模型輸出；本輪不把它改成事實Gold。

### 合併人審結果成資料視圖

若直接編輯匯出的三份CSV：

```powershell
$reviewView = 'experiments/results/research_tools/view_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.review merge --project-root . --bundle $reviewBundle --output-dir $reviewView
```

若另存副本，使用--questions-csv、--pairs-csv、--turns-csv分別指定；未指定者讀原bundle。
允許CSV只保留部分列，缺少者維持needs_review；未知ID、重複ID、變動展示欄及過期資料身分會失敗。
輸出questions.view.jsonl、pairs.view.jsonl、turns.view.jsonl、view_manifest.json。
原欄位保留；人審決定與effective_*另存。原題庫仍needs_review，沒有自動寫回或verified升級。
每次合併保留輸入CSV、原匯出manifest、來源題庫、視圖檔案及工具程式的Hash。
這是審核可追溯紀錄，不是審核人身分的數位簽章；姓名與evidence的真實性仍由研究者負責。

## B：Adapter選擇與輸出

```text
python -m experiments.research_tools.adapters
  --project-root . --target <qa|sc|sc_threshold|sd|prompt_lookup|ti|int>
  --split <historical|calibration|evaluation>
  --purpose <functional|calibration|evaluation>
  --output-dir <全新目錄>
```

可加--view、--ids逗號清單、--types A,B、--limit。
一般QA/SD維持原資料順序；Cache明確--ids維持使用者提供順序，或使用--workload-id保留既有順序。
Cache不可用type/ID篩掉既有workload事件；functional可用--limit截取前幾筆，明示非完整比例實驗。
TI的--limit單位是**整組對話**，不是任意截掉歷史的單輪數。

functional匯出的原生review_status一律needs_review，即使有部分人審也不能把Smoke變正式成績。
原始review_status與人工effective狀態皆保留於metadata/scorer。rejected項目明確拒絕，
不能無聲改題序；請改用明確接受的ID清單，或另審核完整對話／workload。

正式calibration只接受Calibration；正式evaluation只接受Evaluation，需人審view、
--settings釘選設定Hash、--calibration-export及--calibration-run。後兩者必須是同題庫／view／target、
同settings的Calibration匯出，以及非provisional、source_verification=passed的已完成實測Run；
Run來源Hash必須包含該Calibration輸入。不能用「匯出完成」冒充校準已測量。
另外拒絕家族／Question ID交集；語意交集和是否曾偷看Evaluation仍需人工研究紀錄，不能靠工具證明。
一旦改view或參數，需重新建立一致的校準輸出與研究紀錄，不可拿既有Evaluation反覆選參數。
所有CLI覆寫值也應納入最終settings／實際Run比對，不能只釘config後又悄悄改生成參數。

原生輸入不含reference_answer或expected_behavior（TI僅保留其本來所需分類標籤及歷史）；
答案及人審細節另外寫入scorer_only.jsonl，不供檢索／生成。question_id不能冒充gold FAQ ID；
模糊／多意圖題不自動製造單一Gold。A1/A2生成指標not_applicable，單輪架構的完整多輪QA為unavailable。

### 實際對接狀態

| target | 產物與既有入口 | 邊界 |
|---|---|---|
| sd | scenarios.jsonl → SD benchmark --scenarios | 可直接接既有欄位；sd0/sd2_1/sd2_2/all |
| prompt_lookup | 同格式 → SD benchmark --mode sd2_4 | 可直接接；不載1B，仍逐Token對照SD0 |
| sc | 冷序列scenarios.jsonl → SC benchmark --scenarios | 可直接接；intent_reuse_proxy不是人審安全性，需sidecar評估 |
| sc_threshold | 冷Calibration/Evaluation序列 → threshold_benchmark --split/--scenarios | 可直接接欄位；Evaluation仍需既有--selection-file，Historical無原生split |
| ti | calibration.jsonl/evaluation.jsonl → TI --calibration/--evaluation | 可直接接；保留整組歷史，跳過initial，需包含兩類gold_shift |
| qa | scenarios.jsonl + scorer_only | 現行八架構仍固定讀FAQ60，不能直接使用；需改獨立資料入口及Worker/scorer事件ID |
| int | measured.jsonl + initial_fill.jsonl | 現行INT硬編碼functional，不能假改split繞過；需增加研究split/來源契約入口 |
| 任何warm workload | initial_fill與measured分檔 | 目前原生Benchmark沒有相位入口，標direct_current_cli=false，不輸出誤導的scenarios.jsonl |

不相容匯出仍保留完整資料計畫與adapter_manifest.capability原因，不能稱已執行成功。
INT及warm runner後續只需加資料入口、phase、審核及manifest gate；不要重寫Cache或生成核心。
正式Cache輸出要求所有可能較早origin pair都有已驗證判斷，不能只審指定reference然後忽略其他命中。
100%兩次初始填充仍是各variant自己的真實請求，成本另列，不跨mode借答案、不計入20次分母。
本版現有完整workloads均在Evaluation；不得拿它們選門檻。Calibration Smoke可使用明確的3～5個ID序列。
完整正式Cache校準尚需要另建並審核Calibration workload；本輪不挪用Evaluation來填這個資料缺口。

已完成程式但尚未實測：直接接格式不等於新增工具或GPU執行已通過。
舊Benchmark的development/formal_research_result標記不在本輪更改；Adapter正式匯出本身也不是正式成果。

正式匯出參數範例（以已人審的SD資料視圖為前提，路徑需替換成真實產物）：

```text
python -m experiments.research_tools.adapters --target sd --split calibration --purpose calibration --view <已完成view> --settings <固定GPU設定> --output-dir <新Calibration匯出目錄>
```

只會匯出已選資料；未人審項目會阻擋。完成該資料的真實Calibration Run與設定選擇後才用：

```text
python -m experiments.research_tools.adapters --target sd --split evaluation --purpose evaluation --view <同一已凍結view> --settings <同一固定設定> --calibration-export <上述Calibration匯出> --calibration-run <已完成Calibration實測Run> --output-dir <新Evaluation匯出目錄>
```

若只核准部分題目，以--ids或--types明確指定；不自動刪除rejected列改變研究樣本數。

## 3～5題Smoke（先資料，再選擇GPU功能驗證）

以下指令都留給GPU電腦；本輪沒有執行。使用Calibration，避免先曝光Evaluation來調參。

### SD／Prompt Lookup：3題

```powershell
$sdData = 'experiments/results/research_tools/sd_smoke_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --target sd --split calibration --purpose functional --limit 3 --output-dir $sdData
$sdRun = 'experiments/results/algorithms/research_sd_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config experiments/algorithms/speculative_decoding/config.gpu.local.json --scenarios "$sdData/scenarios.jsonl" --mode sd2_1 --repeats 1 --warmups 1 --run-dir $sdRun --allow-unreviewed
```

要驗證Prompt Lookup，另用--target prompt_lookup匯出同一split／同一3個ID，並在**全新**sdRun
用--mode sd2_4。該模式包含SD0與k2/4/6/8，比較相同Top-5、Prompt及input IDs；不是只做3次生成。

### Semantic Cache：4題，含重複與近意圖

```powershell
$scData = 'experiments/results/research_tools/sc_smoke_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --target sc --split calibration --purpose functional --ids UR1_A001,UR1_H001,UR1_R001,UR1_B002 --output-dir $scData
$scRun = 'experiments/results/algorithms/research_sc_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.benchmark --project-root . --config experiments/algorithms/semantic_cache/config.gpu.local.json --scenarios "$scData/scenarios.jsonl" --mode all --run-dir $scRun --allow-unreviewed
```

不預填Hit或安全結果。None/Exact/Semantic使用各自Cache。不要把同意圖proxy當標準答案通過。

### TI：1組完整對話、3個判斷

```powershell
$tiData = 'experiments/results/research_tools/ti_smoke_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --target ti --split calibration --purpose functional --limit 1 --output-dir $tiData
$tiRun = 'experiments/results/algorithms/research_ti_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.topic_identification.benchmark --project-root . --config experiments/algorithms/topic_identification/config.gpu.local.json --phase calibrate --mode detection --calibration "$tiData/calibration.jsonl" --run-dir $tiRun --allow-unreviewed
```

這只是檢查資料接通與特徵；3個判斷不能用來選正式門檻或宣稱正式成績。

### 八架構與INT先只檢查匯出

```powershell
$qaData = 'experiments/results/research_tools/qa_smoke_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --target qa --split historical --purpose functional --limit 3 --output-dir $qaData
$intData = 'experiments/results/research_tools/int_smoke_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --target int --split calibration --purpose functional --ids UR1_A001,UR1_H001,UR1_R001 --output-dir $intData
```

兩份adapter_manifest應明確標示原生CLI尚不支援；不要把舊固定60題Smoke的成功當新題庫已接通。

## 正式Benchmark順序

1. 保留已通過的結構驗證紀錄，匯出審核表；核對來源、年代、Train曝光、多Gold及跨family交集。
2. 完成人工問題／答案／pair／對話標註、仲裁，合併為獨立view；核准不是工具自動決定。
3. 固定題庫與view身分、明確選定每個target的ID／family與指標，記錄排除及not_applicable。
4. 完成QA/INT/phase runner少量入口改動後，各先用Calibration 3～5題做Smoke，核對標籤未進入Prompt。
5. 只在Calibration選門檻／解碼設定；保存非provisional實測Run、來源Hash與參數選擇理由。
6. 用一致settings及該Calibration export/run身分匯出獨立Evaluation，不再挑參數。
7. 各variant用獨立Cache及相同題序；SD配對同Prompt，INT用獨立真實online E2E；暖填／warmup另計。
8. 先評估答案與重用安全性，再分析延遲／速度；一次模型成功執行不等於研究假設通過。

限制：未建立自動模型答案品質評分、審核人身分驗證、實驗自動啟動器或QA/INT新runner。
本輪不會重新建FAISS、不訓練模型，也不啟動230題GPU實驗。
