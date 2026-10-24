# 資科萬事通共用研究題庫 V1：待審核初稿

本輪只新增本目錄，沒有執行 Python、Validation、測試、模型、安裝或 Git。
**不是已凍結正式 Gold；沒有已通過的驗證／實驗結果。** 不修改既有資料、Benchmark、索引、模型或結果。

## 實際交付與目標差異

主要單輪題庫 **230題**：Historical60、Calibration60、Evaluation110；新增170題，距新增180題
目標尚缺I類10題。另保留 questions_legacy_fixed60.jsonl **60題**補充歷史資料，不拿它補足新題缺口。
不能把兩種不同原有60題混成同一份測試集。原有題型配置表需依這個來源差異重新確認。

| 題型 | Historical CSV60 | 新增候選 | 主要題庫合計 |
|---|---:|---:|---:|
| A 直接 |60|15|75|
| B 改寫 |0|15|15|
| C 近意圖 |0|20|20|
| D 多資訊 |0|14|14|
| U 資訊不足 |0|12|12|
| O 領域外 |0|9|9|
| T 時效 |0|14|14|
| R 重複 |0|6|6|
| E 條件／否定 |0|20|20|
| F 數字／計算 |0|15|15|
| G 模糊 |0|10|10|
| H 口語／錯字 |0|10|10|
| I 來源衝突 |0|0|0|
| J 干擾／注入 |0|10|10|

數量是檔案編製紀錄，尚不是執行 validation.py 的結果。I 類沒有建立已查證的真實衝突來源，
因此保留10題來源缺口；其餘新增類型已按規劃數量建立候選，仍須人審，不以筆數代表品質驗收。

20組×4輪=80輪對話，Calibration/Evaluation各10組；排除第一輪後各30個 Topic 判斷候選。
10組×20次=200次 Cache 量測查詢，另有100%兩組各2次初始填充，共4次，分檔不混入200次。

## 三種不同資料角色

1. 現行八架構 evaluation/common.py 實際用的是 A3 snapshot 從兩份原 CSV 讀出 P01～P30、
   S01～S30。questions_historical 原封保留這60題 ID、question、ai答覆；不稱新獨立測試。
2. FAQ canonical 是檢索知識庫，原始版3題與正式版60題都存在。其 Q/A 與同批 CSV 有重疊，
   不代表它本身是新問法評估題庫。現在 canonical60 與 CSV60 的 Q/A 逐項讀取對照一致。
3. scu_fixed_questions_v1_1.jsonl 另有 A/B/C/D/U/O/T/R 的60題，吻合需求中的舊題型配置，
   但不是現行 dataset() 所選的 P/S60。補充歷史檔保留其原ID、原文與原 metadata，不杜撰標準答案。

開啟頁籤所指的 full_60_1d5d4ab3d1cc4e728a6148d87a38bcde 在此工作區未找到，不能核實該次
run.json／attempt snapshot；使用者先前報告的 GPU 完成狀態與本機實際檔案證據分開看。

## 來源與答案

source_registry 提供實際路徑、大小、SHA256、JSONL物理行數、CSV selector，包含既有資料、
SC/SD/TI/INT情境及 Type1/Type2／索引檔；不是目前官網有效性證明。詳見 SOURCE_AUDIT.md。
保留原文，UTF-8-sig 嚴格讀取，不用 errors=replace 或人工修補亂碼。
本目錄另有局部 .gitattributes 固定新資料文字檔 LF，避免 Windows checkout 改變 artifact Hash；
未修改根目錄規則或執行任何 Git 操作。manifest 的 artifact_sha256 不包含 manifest 自身，避免循環。
實際引用的來源ID／答案文件Hash嚴格驗證；盤點用的舊副本／placeholder缺少或改變只列警告，
不要求GPU同步無關備份。正式閘門仍要求既有題庫曝光盤點可取得，不因檔案缺失宣稱已排除洩漏。

主要230題有116題含候選參考答案：60個原始AI FAQ答案、46個單一FAQ問法候選、10個基於
本機文件的數值／運算答案；全是 needs_review，不是 verified。其餘114題 reference_answer=null，
仍有明確 expected_behavior；不憑空寫規章、即時數字、個人或未來保證。知識日期未知就存null。
DOC_CREDITS 是112入學適用的本機文件；DOC_PROJECT 的115年6月公告只作候選，不能當永遠有效。
文件是否真正包含於目前 FAISS chunk 仍需另查索引建置來源；本輪未讀取 pickle 或執行檢索。

## Split 與已知重疊

新增 Calibration 的 career/degree/math/fees 與 Evaluation 的實習、師資、資源、專題等 family
分開。舊60和相關新題可同family，但一個新family不拆到兩種新split。R與原題同split，保留relation。
跨family仍可能語意重疊，必須人審；difflib候選不是語意無洩漏證明。

Historical42題屬Train原始意圖，另外18題雖為Validation/Test意圖也已是既有評估原問法。
新增170題有133題的支持來源涉及Train意圖；每題保留 train/validation/test ID 清單。
不能因為新增問法就聲稱對A2/A3/A5是unseen。133也不代表133筆查詢原文都曾被訓練。
實際 checkpoint 訓練記錄待核對；A3準備程式明確使用42/9/9原始Q/A，A5沿用A3 Adapter。

另須核對：原15題、A2 138題／Hard Diagnostic30、舊SC/SD/TI/INT情境與實際Run曝光。
leakage_report 是靜態風險登錄，status未執行；不宣稱已排除文字或語意重複。
Evaluation 只作未凍結候選，禁止拿去選Cache門檻或解碼設定後再把同一分數稱獨立測試。

## Cache 序列

五個目標重訪比例0/25/50/75/100%，各兩個題序候選；同一比例兩組使用相同seed池，逆序執行，
均取自本版Evaluation候選池，**不是門檻校準集**。未凍結前只供人工檢查，不應先測後調參。
重訪定義為帶有指定reference的預定重訪事件／20；新Question ID仍可能語意相近，
所以0%不保證0 Semantic Hit，100%也不保證100實際Hit或答案正確。這不是已測出的命中率。

25/50/75%分別15/10/5個初次查詢及5/10/15個指定重訪；明示exact和paraphrase，另有C/E
安全pair候選。100%從空Cache執行獨立初始填充後再量20次，填充成本另報，不跨variant共享答案。
記錄actual origin後才做安全判定；尚未有答案時不能把expected_reuse當已確認正確。
所有observed hit、hit kind、品質欄位都是null，沒有預填假命中。

## 格式、人工標註與共用

schema.json 定義 question/conversation_turn/cache_event，額外欄位可擴充。
source_hash 是 source_id→SHA256 object；intent_id 可以null，多來源用supporting_intent_ids；
G模糊題不強迫單gold。標註CSV含747筆問題／對話輪次／序列事件／關聯的標註列，
保留來源、候選答案及空白的人審欄位，詳見 ANNOTATION_GUIDE.md。
review_status用reviewed；經人審後才映射成既有演算法的approved。沒有自動核准／回寫工具。

reader.py 提供只讀共用載入與QA/SD/TI欄位映射，預設拒絕未審核資料。ADAPTER_SPEC.md
明列未來八架構及SC/INT對接需求；**本輪未修改Benchmark，現階段不是一個指令就能對新題庫跑八架構**。
單輪和真正多輪QA、分類品質和生成加速、FAQ Matching的N/A生成指標，各自區分。

## 執行電腦先做純資料檢查

開發電腦本輪不要執行。GPU/分析電腦根目錄：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
$dataAudit = 'experiments/results/dataset_audits/unified_v1_' + [guid]::NewGuid().ToString('N')
python -m experiments.datasets.unified_research_v1.validation --project-root . --report-dir $dataAudit
```

只用Python標準庫，不載模型；報告目錄必須全新且在資料目錄以外。驗證器檢查JSON/重複鍵/型別、
ID/關聯、CSV逐字一致、來源及artifact Hash、family split、重複／訓練原文字串、對話歷史與
Cache題序／比例。lexical threshold=0.72只產生人審候選，不當成語意判決。
失敗exit2且保存原因；成功狀態也是structural_checks_passed_semantics_unverified，非Gold認證。
人審凍結前的嚴格閘門加 `--require-reviewed`，**目前預期會阻擋**，不是現在應宣稱通過的測試。

待人審、來源補強、缺口整理、實際訓練receipt查核後，另凍結版本及Hash，再實作獨立adapters、
少量Smoke，最後固定Calibration選參數、封存設定並只使用一次正式Evaluation。現在不啟動GPU實驗。
