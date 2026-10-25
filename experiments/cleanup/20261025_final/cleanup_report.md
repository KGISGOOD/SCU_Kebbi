# 專案清理與完整基礎測試準備

本輪只做靜態讀寫、逐檔封存、檔案SHA-256及PowerShell靜態語法解析。未執行Python、模型、Benchmark、訓練或Git指令；未安裝套件。`畢業專題_爬蟲/`及`.git/`未遞迴掃描、修改或清理。

## 實際動作

- 初始盤點293個檔案（不含上述保護目錄）；完整檔案／功能對照見`cleanup_manifest.csv`。
- 封存8個檔案，逐檔確認工作區範圍、一般檔案、目的不存在及修改前SHA；移動後重新計算SHA，8個均一致。
- 永久刪除檔案0個：沒有找到可刪除的pycache、pyc或明確暫存檔。不以「未被import」作為刪除依據。
- 移除1個空目錄：`experiments/results/unified_faq/`；刪除前確認沒有任何子項，未使用遞迴刪除。
- 本機results原本沒有檔案；沒有搬移或刪除任何實驗結果。GPU上的唯一結果、完整訓練模型及衍生訓練資料尚未同步，未聲稱檢查或封存成功。
- 本機兩個Smoke模型目錄保留，包括權重、Tokenizer、Pooling、Normalize、訓練設定、日誌、來源及完成／載入收據；新的完整測試明確拒絕使用這些Smoke模型。
- 最終清單：279個原檔逐位元Hash核對未變、6個必要程式／設定／文件修改、5個新入口／範本／共用模組、8個封存檔、1個已移除空目錄。現行檔案由293減為290（不計封存及本清理文件）；沒有永久刪除研究檔案。

## 封存清單與依賴處理

原路徑完整保留於`experiments/archive/cleanup_20261025_final/`下：

| 原檔案 | 處理原因／正式替代 |
|---|---|
| SD `diagnose_failed.ps1` | 固定指向舊Run的一次性除錯／Resume腳本 |
| SD `diagnose_logits.py` | 舊FP16分歧的逐Token診斷，非正式生成入口 |
| SD `diagnose_logits.ps1` | 舊Run診斷編排，已由新完整入口取代 |
| SD `v2.py` | 獨立失敗的替代驗證版本；保留原始碼追溯，不作現行Lossless實作 |
| SD `fp32_control.py` | 單題重播工具；正式需要的原樣數值／顯存函式抽到`numerics.py` |
| SD `fp32_control.ps1` | 依賴舊Run及單題輸出的臨時入口 |
| SD `acceptance.py` | 跨專案驗收整合至`research_tools/acceptance.py`，去除diagnose依賴 |
| FAQ matcher `MODIFICATIONS_SUMMARY.txt` | 無程式／設定引用的歷史修改與同步說明 |

逐檔原始路徑、大小、用途、理由及前後SHA見`archive_manifest.csv`。封存程式只作歷史參考；若重現，應在隔離副本依原路徑及Hash還原，不覆寫現行專案。

## 現行架構及必要模組

| 目錄 | 有效功能／入口 | 保留理由 |
|---|---|---|
| `baselines/rag_baseline/` | A0 Dense RAG；chat/build_index及既有手動檢索測試 | 現行A0及Adapter依賴 |
| `baselines/faq_semantic_matching/` | A1 FAQ Top-1；matcher/faq_loader/build_index | A1及60題canonical/index資格驗證 |
| `a2_faq_finetuning/` | A2-1/A2-2；訓練、資料驗證、FAQ推論 | 現行訓練及模型來源驗證仍使用 |
| `a3_faq_sft/` | LoRA SFT與模型載入、修訂資料契約 | A3/A5依賴；不重新訓練 |
| `baselines/hybrid_rag_reranker/` | A4-1/A4-2；BM25/RRF/Reranker | 使用原始BGE，不改用A2 |
| `a5_rag_finetuned_llm/` | A5文件RAG＋A3 LoRA | 完整QA架構之一 |
| `algorithms/semantic_cache/` | benchmark/threshold_benchmark；None/Exact/Semantic | SC及INT快取依賴；保留小型情境資料 |
| `algorithms/speculative_decoding/` | benchmark/engine/prompt_lookup/numerics/reporting | 真正HF assisted generation；FP16及顯式FP32皆保留 |
| `algorithms/topic_identification/` | benchmark/features/context/metrics | TI校準、固定門檻評估、獨立Context Reduction能力 |
| `algorithms/integrated_efficiency/` | benchmark/pipeline/reporting | INT0～INT3，獨立Cache及Initial Fill／Measured |
| `algorithms/shared/`、`shared/` | A0Adapter、Embedding、LLM、FAISS與資料載入 | 不另建一套RAG |
| `research_tools/` | suites/review/adapters/runtime/run/run_all/acceptance/collect_provisional | r4、Gold、QA Worker、11階段編排及報告 |
| `evaluation/` | wrappers/preflight/model_resources/prepare_a1/scorer | QA、模型、來源與索引完整性 |
| `configs/` | basic_tests及sd_fp32範本；GPU本機*.local.json | 不依賴舊results，且不覆寫本機設定 |
| `datasets/` | FAQ、固定60、原230、r3/r4、訓練候選與審核來源 | 原始資料／Manifest／Hash全部保留 |
| `models/`、`vector_stores/` | 模型及索引、來源和收據 | 不清理或重建權重與索引 |
| `robot/` | baseline_gradio及實際import的service/config/embeddings/vectorstores/retriever/ollama/prompts/orchestrator/relevance | 唯一歷史Ollama Baseline及可能的整合支援，不稱為HF正式A0 |
| `archive/`、`cleanup/` | 歷史原碼與本輪稽核 | 與正式可執行程式分離 |

QA原有test_baseline/test_faq/test_hybrid/test_retrieval是人工可啟動的現行架構診斷／評估CLI，不是證實無用的舊演算法，保留。訓練及A2困難題審核文件亦可能具有來源證據價值，不因年代或無import而刪除。

## 新完整入口

`research_tools/run_basic_tests.ps1`先建立不存在的GPU本機profile，然後呼叫既有`run_all`。範本明確指向已知完整訓練模型的QA設定，若不存在就停止，不以開發電腦的Smoke資料取代。

11階段：QA Evaluation；SC Calibration/Evaluation；SD Calibration/Evaluation；TI Calibration/Evaluation；INT Cold及Prefilled各自Calibration/Evaluation。SD各階段含六種SD模式與獨立Prompt Lookup子研究及各自SD0。QA／SC／TI／INT的核心不改。

SD FP32是一個顯式數值對照設定：相同原始Target/BGE/知識庫、seed、device及生成長度，僅允許SD的precision與其他演算法不同。使用`numerics.py`固定SDPA、停用TF32及檢查顯存，不回退精度。Calibration與Evaluation須一致傳遞此策略，且記入context、完整Run identity與模型identity。不得以FP32單題12/12推論完整Evaluation已通過，也不合併跨精度／跨演算法加速比。

每階段失敗仍保留raw；SD失敗的可讀測量透過現有`sd_projection`整理，但不補認成功。根輸出仍是summary.xlsx、report.md、raw/；驗收與回溯整理各使用新的獨立目錄。Formal與Gold Guard、訓練批准收據、Token完全一致性檢查不取消。

## 尚待確認及GPU同步

`unresolved_files.csv`共19項：12項本機恢復／備份／舊索引保留項、7項GPU未同步位置／結果家族。這些未搬移。

GPU應保留完整模型目錄及其訓練來源設定，勿以開發電腦models資料夾鏡像覆蓋。GPU歷史原Run、FP16失敗、v2失敗、FP32單題結果，須由使用者先確認實際位置及逐檔SHA，再完整封存；本輪沒有操作GPU。

同步新增與修改的現行程式、設定範本、README/.gitignore與本輪archive/cleanup。依archive_manifest逐檔處理GPU舊工具，不使用鏡像同步刪除GPU獨有資料；若GPU原碼SHA與開發清單不同，先保留為不同歷史版本。

## 驗證界線

靜態本地import路徑掃描未找到缺失。三個有效PowerShell腳本以Parser.ParseFile解析（不執行），錯誤0；兩份新JSON範本以唯讀JSON解析讀取成功。已檢查正式程式不再引用封存的SD工具或固定舊Run ID。沒有使用Python parser、匯入專案或執行任何測試。第三方API、顯存、模型載入、全部題數／品質與完整FP32 Lossless仍須GPU執行驗證。
