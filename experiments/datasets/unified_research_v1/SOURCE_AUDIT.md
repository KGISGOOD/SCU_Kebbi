# 現有資料盤點：本機檔案與使用歷史分開

盤點日期2026-10-24。只讀檔案／程式／雜湊，未執行任何專案程式、測試、模型或Git。
JSONL筆數是讀取行數；不代表此輪已由validation.py解析通過。所有實際路徑、大小及完整SHA256
見 source_registry.json 的 inventory；FAQ逐題來源以編號欄selector定位，不憑空重建缺少原文。

## 已確認來源與角色

| 實際路徑／元件 | 本機數量 | 角色及已知使用 |
|---|---:|---|
| 畢業專題_爬蟲/type2/資料擴充 - 家長 FAQ 資料集（30 題）.csv |30個邏輯Q/A|P01～P30；欄編號、關切維度、問題內容、ai答覆 |
| 同目錄／資料擴充 - 高中生 FAQ 資料集（30 題）.csv |30個邏輯Q/A|S01～S30；原始FAQ，AI擴充答案仍需查證 |
| experiments/datasets/scu_fixed_questions_v1_1.jsonl |60行|A/B/C各10、D6/U8/O6/T6/R4；有as_of/version，無標準答案 |
| experiments/datasets/scu_faq_matching_test_v1.jsonl |15行|舊診斷查詢，不能視為乾淨最終Test |
| experiments/datasets/faq/faq_canonical_v1.jsonl |3行|舊示範知識庫，不拿來填60題 |
| experiments/datasets/faq/faq_canonical_60_v2.jsonl |60行|正式知識Q/A；現行evaluation/config.json指向此檔 |
| 同目錄／faq_canonical_60_v2_lf_restored.jsonl |60行|換行修復副本；不自動當最新版或替換設定 |
| 同目錄／faq_canonical_60_lf_0cf90d186bbe4abaa4200c9f65036ddc.jsonl |60行|另一修復副本，旁有restoration資料 |
| 同目錄／faq_canonical_60_lf_f3b406159a3045a994e909f3b6c0623c.jsonl |60行|另一歷史副本；不能只看檔名判斷LF／內容 |
| experiments/datasets/faq_finetuning/faq_training_candidates.jsonl |42行|正式候選，有原question/positive/negative/hard_negative；既有before_task2a/2b各42行是備份 |
| 同目錄／faq_split_manifest.json |42/9/9意圖|A2切分；A3 snapshot直接重用，不是新的research split |
| 同目錄／faq_retrieval_candidates.jsonl |60行|完整FAQ檢索候選question/source_group；不是42題訓練資料，無標準答案欄 |
| 同目錄／faq_eval_queries.jsonl |138行|Seen84／UnseenValidation27／UnseenTest27依既有規劃；本輪不讀取結果調整模型，未聲稱已重新驗證各組筆數 |
| 同目錄／faq_hard_diagnostic.jsonl |30行|既有開發診斷，非新獨立Test |
| experiments/datasets/faq_sft/ |未找到|A3 prepare_data預計輸出位置；本機不存在已準備好的train/validation/test資料，不能編造 |
| experiments/a3_faq_sft/common.py、prepare_data.py |程式|snapshot明確從兩CSV和原manifest產生42/9/9原始Q/A；不是A2 Positive |
| experiments/a5_rag_finetuned_llm/config.py、pipeline.py、README.md |程式|A5使用A3 Adapter加文件RAG，不是另有可確認的獨立訓練題庫 |
| experiments/evaluation/common.py |程式|dataset()呼叫A3 snapshot，強制60原始FAQ；非直接讀fixed_questions檔 |
| experiments/evaluation/config.json |設定|指canonical60、FAQ60索引、A2模型／候選、A3 Adapter、原manifest |
| experiments/algorithms/semantic_cache/scenarios.jsonl |8行|功能驗證；normalization另4行 |
| 同目錄／threshold_functional/calibration/evaluation.jsonl |12/8/8行|SC1-3功能與分離候選；不改寫或搬進新正式Gold |
| experiments/algorithms/speculative_decoding/scenarios.jsonl |7行|既有SD與Prompt Lookup共用功能輸入 |
| experiments/algorithms/topic_identification/calibration/evaluation.jsonl |8/10行|原TI腳本；不是本版20組對話，也非全八架構多輪QA能力證明 |
| experiments/algorithms/integrated_efficiency/scenarios.jsonl |10行|INT功能情境，原檔保留 |

## 八架構歷史Run的證據限制

使用者先前報告GPU完成八架構60題測試；但IDE頁籤所指
experiments/results/unified_faq/full_60_1d5d4ab3d1cc4e728a6148d87a38bcde 在本機未找到。
因此本輪歷史主檔依**現行評估程式資料流＋真實CSV**保留P/S60，不宣稱已逐筆對照該GPU Run。
凍結前必須核對GPU原run.json的dataset/question_ids與attempts；若那次實際用了不同題庫，
另記為不同歷史來源，不能默默把P/S IDs改成A/B IDs。

## 雜湊／canonical版本風險

兩原始CSV SHA256：

- 家長：85f8e782f08b01f4aba0681fe9d8975e7e4a0be79ba1250a79251740c4e4aaac
- 高中生：749a448ee28a7ddd388b4f246dace412acaaba9fabfc718daae81353cf06736f

目前 canonical_60_v2 的SHA256是60b4fad2ff93b16b94b5f980de6d883eaccd64ad1942e5d36dd2850303869a63；
faq_bge_m3_60_v2/version.json記錄7ff63fb1a27558345f33b034023243a8a35e774019f53397108861d40679f8ac。
lf_restored及0cf90副本的SHA與後者一致。這與先前換行問題相關，但本輪不正規化、不改receipt、
不假裝索引已重新驗證。新題庫採CSV原文，不依賴挑一個canonical副本猜答案。
目前canonical60的60組ID/Q/A與兩CSV讀取對照無差異，僅證明來源文字忠實度。

## A0文件與索引

Type1實體檔案42個，Type2檔案9個；inventory逐檔記錄SHA256。不是42/9筆知識或chunk數。
本輪逐段讀取的數字來源：Type1/畢業學分.txt 明列112學年度入學、81必修、47選修、總128、
外系至多16；Type1/課程資訊_專題實作專區.txt 有115年6月10日中午12點公告。
其餘大部分只作路徑／Hash盤點，不宣稱逐篇完成事實審核或現行有效性查核。

A0文件索引 experiments/vector_stores/type1_type2_bge_m3_v1/ 的 index.faiss/index.pkl 均存在，
沒有version.json的既有格式由A0Adapter允許；與FAQ匹配索引完全不同，不混用。
文件FAISS SHA256：cbc03a54befce8553cc1d0dc69c59aa119b0f179b260ed6b60b6d743fe0eac47。
本輪沒有載入FAISS或pickle，無法證明特定文件版本已被建入索引。建庫程式以Type1/2來源、
清理、chunk及BGE向量建立；實際chunk內容／建置來源receipt仍待執行端另做唯讀盤點。

## 訓練／選參數／測試：不能混同

- 原manifest Train P01～P21/S01～S21；Validation P22～P25/S22～S26；Test P26～P30/S27～S30。
- A2候選42行及A3資料準備規則是**可見程式／候選**；不直接證明每個checkpoint實際用過哪些文字。
- 本機models只看到A2兩個smoke目錄；未找到A3正式prepared dataset/adapter，不能捏造A3訓練receipt。
- 使用者先前回報的UnseenValidation、Hard Diagnostic及SC/SD/TI功能測試屬已曝光開發資訊；
  舊題庫不能重新命名為乾淨最終Test。TI舊Calibration和Threshold selection範本也不能當真實最佳參數。
- 新版Historical42題是Train原問題，18題是保留意圖原問題；新Evaluation不是unseen intent保證。
  新問法與既有positive、診斷／測試／功能情境的精確／近似重疊仍需新validation報告與人工檢查。

## 尚需確認

GPU歷史Run實際題目快照、A3/A5實際訓練來源、模型選參數紀錄、文件索引建置receipt、
官網現行規章與歷史文件適用範圍、多Gold及跨family語意交集、所有新候選人工品質。
本輪沒有以缺少來源為由自行填造60題、答案、索引或執行成績。
