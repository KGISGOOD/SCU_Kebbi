# 資科萬事通 — SCU_Kebbi

東吳大學資料科學系畢業專題：以系務文件、FAQ、檢索與本機語言模型提供問答，研究在維持答案品質下減少回應時間。正式研究使用 Hugging Face Transformers、PyTorch、Llama-3.2-3B-Instruct、BGE-M3及FAISS，不使用Ollama。

整體流程是「使用者／凱比互動端 → 問答服務 → FAQ Matching或文件RAG與LLM → 回答」。研究CLI不等於已驗證的機器人硬體控制介面。`robot/baseline_gradio.py`是保留的歷史Gradio＋Ollama系統，與正式A0區分。

## 目錄與保護範圍

```text
SCU_Kebbi/
├── README.md
├── .gitignore / .gitattributes
├── experiments/
│   ├── requirements.txt
│   ├── baselines/{rag_baseline,faq_semantic_matching,hybrid_rag_reranker}/
│   ├── a2_faq_finetuning/
│   ├── a3_faq_sft/
│   ├── a5_rag_finetuned_llm/
│   ├── shared/
│   ├── algorithms/{shared,semantic_cache,speculative_decoding,topic_identification,integrated_efficiency}/
│   ├── evaluation/
│   ├── research_tools/
│   ├── datasets/{faq,faq_finetuning,unified_research_v1}/
│   ├── models/
│   ├── vector_stores/
│   └── results/                 # 新Run輸出，不存模型或權威題庫
├── robot/                      # 唯一歷史Ollama Baseline、支援模組與原始資料
└── 畢業專題_爬蟲/               # 完全保護，不修改或清理
```

本文件是主要架構指南。`experiments/datasets/unified_research_v1/README.md`受Dataset Manifest Hash保護，作為額外README例外；爬蟲內文件亦保留。必要的訓練、審核、復原技術文件留在原模組。

## 八種QA架構

| 架構 | 流程與核心位置 |
|---|---|
| A0 | BGE-M3 Dense Retrieval → Top-5文件 → Baseline Prompt → 原始Llama 3B；`baselines/rag_baseline/` |
| A1 | 原始BGE-M3 → FAQ Top-1 → 原有答案，不呼叫生成模型；`baselines/faq_semantic_matching/` |
| A2-1 | Positive Pair微調BGE-M3 → FAQ Top-1 → 原有答案；`a2_faq_finetuning/` |
| A2-2 | 同基礎Positive Pair加Hard Negative微調BGE-M3 → FAQ Top-1 → 原有答案 |
| A3 | Llama 3B＋FAQ LoRA SFT Adapter，無RAG；`a3_faq_sft/` |
| A4-1 | Dense＋BM25＋RRF混合檢索 → Llama；`baselines/hybrid_rag_reranker/` |
| A4-2 | A4混合檢索再加Reranker → Llama |
| A5 | A0文件RAG＋A3微調Adapter；`a5_rag_finetuned_llm/` |

`shared/`提供資料、Embedding、FAISS、檢索與LLM載入。`evaluation/wrappers.py`重用各架構，擷取Prompt、Token、設定與時間；新QA逐架構啟動獨立Worker。A1／A2的生成時間與Tokens/sec為`not_applicable`，不能把FAQ匹配當成LLM生成速度比較。

原FAQ在`畢業專題_爬蟲/type2/資料擴充 - 家長 FAQ 資料集（30 題）.csv`及同目錄`資料擴充 - 高中生 FAQ 資料集（30 題）.csv`。A1正式Canonical是`experiments/datasets/faq/faq_canonical_60_v2.jsonl`，索引是`experiments/vector_stores/faq_bge_m3_60_v2/`；舊3題版不能替代60題。

A0／A4／A5文件庫是`experiments/vector_stores/type1_type2_bge_m3_v1/`，保留`index.faiss`與`index.pkl`；既有文件庫無version.json時不補造，保留檔案Hash與結構驗證，不與FAQ索引混用。A2的`faq_finetuning/faq_training_candidates.jsonl`是42個Train意圖的訓練資料，`faq_retrieval_candidates.jsonl`是60題檢索候選，答案仍核對原CSV。

QA設定指向`experiments/models/a2_1_smoke_task8/final_model`與`a2_2_smoke_task8/final_model`。完整Tokenizer、Pooling、Normalize、權重、來源、完成及回載驗證紀錄都要保留。A3／A5使用GPU既有Base＋Adapter；預設`experiments/models/a3_smoke_v1/final_adapter`需在GPU確認存在，不自動替換模型。

A2操作與相容性見 [TRAINING.md](experiments/a2_faq_finetuning/TRAINING.md)、[TRAINING_COMPATIBILITY.md](experiments/a2_faq_finetuning/TRAINING_COMPATIBILITY.md)。A2／A3的needs_review預設阻擋正式訓練，allow-unreviewed只作provisional。A3使用原生Chat Template及assistant標籤邊界檢查；A5沿用模型載入器並保留完整Top-5 Prompt，不以截斷文件規避輸入限制。訓練參數與來源以實際設定／Checkpoint為準。

## 推論效率演算法

**Semantic Cache**：SC0 None不查Cache；SC1-1 Exact以正規化問題匹配；SC1-2 Semantic以BGE-M3正規化向量搜尋獨立FAISS Cache。Hit回原答案；Miss走完整A0 RAG與生成再寫入。穩定Cache ID與FAISS label分開管理，namespace保留模型、Prompt、知識庫、生成設定識別。模式、門檻、Repeat使用獨立Cache；第一版無LRU或持久化。Hit／Miss都記真實E2E與分項時間，相似度高不是安全重用證明。SC1-3預設比較0.75／0.80／0.85／0.90／0.95，各門檻實際重跑序列，不事後套用單一命中軌跡。

**Speculative Decoding**：SD0為3B Target-only；SD2-1是本機1B Draft＋3B Target的Transformers assisted generation；SD2-2比較候選長度2／4／6／8。SD2-4 Prompt Lookup從目前Prompt找候選，不需1B，但仍由Target驗證。不能以Draft單獨答案冒充，也不靜默退回SD0。同題僅準備一次A0 Top-5與Prompt，各模式用相同有效生成參數；Greedy輸出Token對照SD0，不一致即失敗。Warm-up、載入、Repeat、CUDA同步與顯存分開紀錄；Generation-only與共享準備的受控E2E分開。無可靠接受率就列unavailable。API檢查以現有Transformers 5.16.1為依據，其他版本另審查。

**Attention Topic Identification**：TI0為BGE-M3 Cosine；TI3-1取真實eager Attention與明確內容Token Span；TI3-2依模型層數比較前／中／後層；TI3-3比較完整歷史、Embedding與Attention判斷下的Context Reduction。Calibration選門檻，獨立Evaluation讀凍結threshold。Attention不是因果解釋，判斷本身也要計時。多輪Adapter不改原單輪A0；指代／追問採保守保留，至少保留最近完整對話，不讀Gold歷史標籤來裁剪輸入。上下文及答案品質仍需人審。

**INT0～INT3**：INT0一般RAG；INT1 RAG＋Semantic Cache；INT2 RAG＋Draft或Prompt Lookup；INT3 Cache＋所選解碼。每個Variant、Repeat、Warm-up使用獨立Cache，先於該Pipeline執行`initial_fill`，再沿用同一Cache處理`measured`，不共享他組答案。兩階段成本分開，fill不進measured分母，但也不能省略後宣稱整段加速。保留原獨立online E2E與Token配對公式；Hit不是lossless decoding證明。詳見 [QA_INT.md](experiments/research_tools/QA_INT.md)。

## 研究題庫、人審與總覽

權威資料位於`experiments/datasets/unified_research_v1/`：Historical60／Calibration60／Evaluation110，共230題；`questions_legacy_fixed60.jsonl`是另一份補充歷史60題，不加入230題。`conversations.jsonl`含20組80輪；`cache_workloads.jsonl`含10組200次measured，目前來源為Evaluation；`cache_initial_fill.jsonl`另有4次fill。`question_relations.jsonl`是關係候選，不是已核准重用關係。保留Schema、Reader、Validation、來源登錄與Manifest原始Hash。

原CSV FAQ60與`experiments/datasets/scu_fixed_questions_v1_1.jsonl`不是同一題庫。Question ID不是Gold FAQ ID；Train意圖曝光仍須記錄，新問法不能直接稱unseen。needs_review不是Gold，結構驗證不等於語意正確。Calibration可選設定，Evaluation凍結後只作獨立評估，不反覆調參。

`review.py export`匯出UTF-8 BOM問題／配對／多輪審核CSV；人審後`merge`成獨立View，不改原JSONL。答案需明確選keep_candidate／override／unavailable並記錄審核人與證據；同Family、同FAQ、高相似度都不自動核准reusable，配對相容也不代表生成答案已驗證。

唯一閱讀總覽的預設位置為`experiments/datasets/題庫總覽.csv`。本機只完成產生器，尚未執行Python產出CSV。於GPU或分析電腦執行：

```powershell
python -m experiments.research_tools.review overview --project-root .
```

匯出重用Reader、來源Hash與審核邏輯，確認230個唯一ID及60／60／110分布，按Split與原ID排序。欄位包括原題、題型中文、Family、原候選答案、預期行為、來源位置、Train曝光、原始審核、人審狀態與備註。無答案留空、無View則人審欄留空，不升格Gold。UTF-8 BOM寫出後重新讀回逐欄比對，並核對前後來源身分。

可加`--view <已合併View>`，覆寫答案另欄呈現，原candidate_answer不變。預設拒絕覆寫；更新時用`--output experiments/datasets/題庫總覽_新版.csv`另存，人工決定保留哪份閱讀檢視。CSV不作Benchmark輸入或第二權威題庫。

資料流：`原JSONL → 總覽CSV供閱讀`；`原JSONL → 人審View／Adapter → Benchmark`。

## 設定與完整性

從正確專案根目錄啟動模組，`--project-root`不會代替Python Import路徑。QA使用`experiments/evaluation/config.json`格式；演算法各有`config.example.json`，需另設GPU實際本機路徑。`algorithms/**/config*.local.json`已受`.gitignore`保護。模型全部離線，缺檔報錯，不下載、訓練或自動建索引。

`experiments/requirements.txt`、`robot/requirements.txt`是歷史清單，不是現有GPU環境鎖檔，不要據此重裝或降版。LF／CRLF會改變SHA，保留`.gitattributes`，不要直接修改version或receipt的Hash。復原見 [A1_CANONICAL_RECOVERY.md](experiments/evaluation/A1_CANONICAL_RECOVERY.md)、[RESOURCE_PREPARATION.md](experiments/evaluation/RESOURCE_PREPARATION.md)。

Functional加allow-unreviewed仍永遠provisional。正式Adapter需`--purpose calibration|evaluation`、已完成View及`--settings`；Evaluation另需同題庫／View／設定的`--calibration-export`和非provisional實測`--calibration-run`。目前完整Cache序列主要來自Evaluation，不能挪來校準。QA Worker模型只接問題文字，reference_answer／expected_behavior／scorer_only只給結果端；無獨立人審Gold映射則FAQ Accuracy列unavailable。未審核Hit保持unverified。

## 有效入口

以下模組皆以`python -m <模組>`從根目錄啟動，參數以`--help`或連結操作文件為準。本輪整理沒有執行這些命令。

| 用途 | 模組 |
|---|---|
| 舊CSV FAQ60八架構 | `experiments.evaluation.run`，子命令preflight／smoke／full／resume／report |
| 新QA／INT | `experiments.research_tools.qa_benchmark`／`int_benchmark` |
| Adapter／人審／總覽 | `experiments.research_tools.adapters`／`review` |
| SC／門檻比較 | `experiments.algorithms.semantic_cache.benchmark`／`threshold_benchmark` |
| SD／Prompt Lookup | `experiments.algorithms.speculative_decoding.benchmark` |
| TI | `experiments.algorithms.topic_identification.benchmark` |
| 舊INT功能情境 | `experiments.algorithms.integrated_efficiency.benchmark` |
| A2驗證／評估 | `experiments.a2_faq_finetuning.validate_a2_data`／`evaluate_faq_retrieval`／`evaluate_hard_diagnostic` |
| A2訓練／互動 | `experiments.a2_faq_finetuning.train_faq_embedding`／`chat` |
| A3準備／訓練／互動 | `experiments.a3_faq_sft.prepare_data`／`train`／`chat` |
| A0／A4互動 | `experiments.baselines.rag_baseline.chat`／`experiments.baselines.hybrid_rag_reranker.chat` |
| A5互動 | `experiments.a5_rag_finetuned_llm.chat` |

QA／INT三題Calibration Smoke的完整命令見 [QA_INT.md](experiments/research_tools/QA_INT.md)，輸出每次使用新GUID目錄。INT每Variant每Repeat為一次fill＋三次measured，另有Warm-up；省略initial-fill-ids可檢查冷Cache。Review流程是`review export --output-dir <新目錄>` → 人工改CSV → `review merge --bundle <匯出目錄> --output-dir <新View>`，皆可指定`--project-root .`。

### GPU資料與舊入口驗收

```powershell
$report = 'experiments/results/research_tools/validation_' + [guid]::NewGuid().ToString('N')
python -m experiments.datasets.unified_research_v1.validation --project-root . --report-dir $report
if ($LASTEXITCODE -ne 0) { throw 'Dataset validation failed' }
python -m experiments.research_tools.review overview --project-root .
if ($LASTEXITCODE -ne 0) { throw 'Overview export failed or CSV already exists' }
$run = 'experiments/results/unified_faq/smoke_' + [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run smoke --project-root . --run-dir $run --allow-unreviewed
```

### 演算法CLI參數

各命令前置`python -m`，並加`--project-root . --config <GPU本機設定> --run-dir <新目錄> --allow-unreviewed`；演算法輸出須在`experiments/results/algorithms/`下。

- SC `experiments.algorithms.semantic_cache.benchmark --mode all`；支援none／exact／semantic。
- SC1-3 `experiments.algorithms.semantic_cache.threshold_benchmark --thresholds 0.75,0.80,0.85,0.90,0.95`；Evaluation另需`--split evaluation --selection-file <凍結選擇檔>`。
- SD `experiments.algorithms.speculative_decoding.benchmark --mode all --repeats 3 --warmups 1`；指定`--mode sd2_4`跑Prompt Lookup。
- TI `experiments.algorithms.topic_identification.benchmark --phase calibrate --mode detection`；獨立Evaluation加`--phase evaluate --thresholds <完成Calibration的thresholds.json>`，`--mode all`另執行Context Reduction。
- 新題庫以Adapter `--target sd|prompt_lookup|sc|sc_threshold|ti --split <來源Split>`匯出，對接各Benchmark的scenarios／calibration／evaluation參數。多輪保留Conversation歷史，Cache保留題序、填充與配對。

設定格式各自不同，不直接把INT設定交給SC。A1 `evaluation.prepare_a1`、A2 `build_faq_candidates`仍保留；現有完整資源不需重建。清理舊結果後必須重新取得Calibration收據與threshold，工具程式Hash變更也可能要求重新匯Adapter，不關閉驗證來避錯。

## 歷史系統與REVIEW

`robot/baseline_gradio.py`與config、Embedding、檢索、Prompt、service、orchestrator、Ollama封裝保留。它以工作目錄相對的`vector_store_dept`搜尋索引，歷史啟動需在`robot/`或明確設定`PODCAST_VECTOR_DIR`；不直接讀`robot/data/`，但兩者都暫保留。

Canonical LF恢復副本、A1恢復設定、舊FAQ索引／候選庫、訓練來源備份仍保留，避免破壞既有引用與Hash。`robot/`七個舊FAQ99 LoRA、模型測試及建庫腳本已確認無正式依賴並刪除；另清除舊比較題目檔與`experiments/test_imports.py`臨時Import診斷器。`robot/faq.txt`僅保留原始FAQ99來源，不是正式訓練、研究題庫或Baseline執行輸入。

`experiments/baselines/`的test_baseline、test_faq、test_hybrid與test_retrieval仍是現有架構的專用固定題庫Benchmark／檢索檢查，不是舊robot演算法實驗，保留供必要診斷與歷史入口相容。`experiments/analysis/knowledge_inventory_v1.md`保留作來源盤點參考。前輪已刪114個檔案及25個空目錄，本輪再刪9個無依賴的舊檔；`experiments/results/`目前無歷史結果。既有 [CLEANUP_PLAN.md](experiments/CLEANUP_PLAN.md)只記錄前輪114項範圍，本輪不另建報告。靜態整理不代表GPU功能或答案品質重新通過驗證。
