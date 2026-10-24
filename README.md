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
│   ├── datasets/{faq,faq_finetuning,unified_research_v1,experiment_suites_v1}/
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

A3原始v1快照維持needs_review及逐欄來源驗證，用於歷史provisional Smoke。正式訓練改用以下v2衍生資料與`--training-review <人工收據.json>`；原始資料不變，程式不自動批准。`--allow-unreviewed`或`--max-steps`仍使模型provisional；題庫Gold簽認不能代替訓練資料批准。

正式衍生訓練資料流程（程式尚待GPU驗證）：

- A2：`python -m experiments.a2_faq_finetuning.prepare_review --project-root . --output-dir <新目錄>`，原樣複製42筆候選，保留126個Positive與12個空Hard Negative，不自動批准。輸出`train.jsonl`、`dataset_manifest.json`、`training_review.template.json`。使用同入口加`--check-only`驗證；加`--training-review <人審收據>`可一併驗證批准。
- A3：`python -m experiments.a3_faq_sft.prepare_data --project-root . --data-dir <新目錄> --answer-revisions experiments/datasets/faq_finetuning/a3_answer_revisions.candidate.jsonl`。v2只改42個Train的assistant答案，保留原問題、原答案、修訂理由、來源定位/Hash、適用期間與不確定性；Validation/Test維持原始快照。現有v1快照與Smoke入口保留。
- 42筆A3修訂均為needs_review：20筆含部分本機來源、9筆為不保證結果的一般建議、13筆事實證據不足；不表示20筆完整答案已由官方認證。若人審要改候選，另存修訂JSONL，再建新資料目錄，不能編輯已釘選的資料或填造Hash。
- A2與A3正式批准收據都須由人類實際審核後填寫：模板中的`approved_faq_ids`完整列出42個Train ID；`decisions`每筆須含`decision: approved`及適用該ID的`evidence`，另填`reviewer`、帶時區`reviewed_at`及總體`evidence`。保留模板，將簽認檔另存為`training_review.signed.json`；模板沒有預填身分、簽認時間或批准。來源不足不能只簽名當成事實已證實。
- A2正式訓練同時指定`--training <衍生/train.jsonl> --training-manifest <衍生/dataset_manifest.json> --training-review <簽認收據>`；A3正式訓練使用v2的`--data-dir`與`--training-review`。A3修訂答案即使使用allow-unreviewed也必須有有效批准；原v1的provisional Smoke維持可用。
- 啟動及checkpoint儲存前均檢查來源、實際資料與收據；正式參數不得加allow-unreviewed/max-steps。A2兩組沿用同一原始BGE-M3、資料與正配對排程；A2-2困難候選覆蓋不均，原始42筆中P13/P14兩筆被排除，並非各意圖等量增加負例。
- FAQ60包含42個已見Train意圖及18個保留意圖；A3已學習Train原問題對應的修訂答案。QA整體分數是混合曝光條件，不能稱全部未見泛化或未曝光答案測試。

## 推論效率演算法

**Semantic Cache**：SC0 None不查Cache；SC1-1 Exact以正規化問題匹配；SC1-2 Semantic以BGE-M3正規化向量搜尋獨立FAISS Cache。Hit回原答案；Miss走完整A0 RAG與生成再寫入。穩定Cache ID與FAISS label分開管理，namespace保留模型、Prompt、知識庫、生成設定識別。模式、門檻、Repeat使用獨立Cache；第一版無LRU或持久化。Hit／Miss都記真實E2E與分項時間，相似度高不是安全重用證明。SC1-3預設比較0.75／0.80／0.85／0.90／0.95，各門檻實際重跑序列，不事後套用單一命中軌跡。

**Speculative Decoding**：SD0為3B Target-only；SD2-1是本機1B Draft＋3B Target的Transformers assisted generation；SD2-2比較候選長度2／4／6／8。SD2-4 Prompt Lookup從目前Prompt找候選，不需1B，但仍由Target驗證。不能以Draft單獨答案冒充，也不靜默退回SD0。同題僅準備一次A0 Top-5與Prompt，各模式用相同有效生成參數；Greedy輸出Token對照SD0，不一致即失敗。Warm-up、載入、Repeat、CUDA同步與顯存分開紀錄；Generation-only與共享準備的受控E2E分開。無可靠接受率就列unavailable。API檢查以現有Transformers 5.16.1為依據，其他版本另審查。

**Attention Topic Identification**：TI0為BGE-M3 Cosine；TI3-1取真實eager Attention與明確內容Token Span；TI3-2依模型層數比較前／中／後層；TI3-3比較完整歷史、Embedding與Attention判斷下的Context Reduction。Calibration選門檻，獨立Evaluation讀凍結threshold。Attention不是因果解釋，判斷本身也要計時。多輪Adapter不改原單輪A0；指代／追問採保守保留，至少保留最近完整對話，不讀Gold歷史標籤來裁剪輸入。上下文及答案品質仍需人審。

**INT0～INT3**：INT0一般RAG；INT1 RAG＋Semantic Cache；INT2 RAG＋Draft或Prompt Lookup；INT3 Cache＋所選解碼。每個Variant、Repeat、Warm-up使用獨立Cache，先於該Pipeline執行`initial_fill`，再沿用同一Cache處理`measured`，不共享他組答案。兩階段成本分開，fill不進measured分母，但也不能省略後宣稱整段加速。保留原獨立online E2E與Token配對公式；Hit不是lossless decoding證明。詳見 [QA_INT.md](experiments/research_tools/QA_INT.md)。

## 研究題庫、人審與總覽

權威資料位於`experiments/datasets/unified_research_v1/`：Historical60／Calibration60／Evaluation110，共230題；`questions_legacy_fixed60.jsonl`是另一份補充歷史60題，不加入230題。`conversations.jsonl`含20組80輪；`cache_workloads.jsonl`含10組200次measured，目前來源為Evaluation；`cache_initial_fill.jsonl`另有4次fill。`question_relations.jsonl`是關係候選，不是已核准重用關係。保留Schema、Reader、Validation、來源登錄與Manifest原始Hash。

原CSV FAQ60與`experiments/datasets/scu_fixed_questions_v1_1.jsonl`不是同一題庫。Question ID不是Gold FAQ ID；Train意圖曝光仍須記錄，新問法不能直接稱unseen。needs_review不是Gold，結構驗證不等於語意正確。Calibration可選設定，Evaluation凍結後只作獨立評估，不反覆調參。

`review.py export`匯出UTF-8 BOM問題／配對／多輪審核CSV；人審後`merge`成獨立View，不改原JSONL。答案需明確選keep_candidate／override／unavailable並記錄審核人與證據；同Family、同FAQ、高相似度都不自動核准reusable，配對相容也不代表生成答案已驗證。

舊230題的完整技術總覽功能仍保留相容性，但本輪不產生另一份總覽。現在人工閱讀唯一使用`experiments/datasets/研究題目審核表.csv`的六欄Suite審核表，見下方研究Suite說明。需要舊技術檢視時，才另行使用原入口：

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

`experiments/baselines/`的test_baseline、test_faq、test_hybrid與test_retrieval仍是現有架構的專用固定題庫Benchmark／檢索檢查，不是舊robot演算法實驗，保留供必要診斷與歷史入口相容。歷史結果、暫時清理計畫與盤點報告已清除；`experiments/results/`供新Run使用。靜態整理不代表GPU功能或答案品質重新通過驗證。

## 研究Suite V1：固定用途與唯一人工審核表

`experiments/datasets/experiment_suites_v1/`不覆寫原230題。目前修訂為`spoken_candidate_r2`；`suites.json`保存固定選取、SC題組、SD引用、TI候選主題／歷史依賴與INT引用。`questions.jsonl`保存60個QA口語覆寫及90個SC來源連結紀錄，SD與INT直接引用SC，不另複製問題。QA只在明確指定新Suite時建立記憶體資料視圖；原CSV與舊入口不變。原始資料Hash、Suite檔案Hash及修訂識別共同區分新舊收據，不得混用舊Adapter或人審View。

| 研究 | Calibration | Evaluation | 資料方式 |
|---|---:|---:|---|
| QA | — | 60題 | P01–P30、S01–S30的獨立口語候選；沿用Historical曝光標記，研究用途是QA評估 |
| SC | 10組×4＝40次 | 20組×4＝80次 | 原需求、完全重複、可重用改寫候選、不同需求候選 |
| SD／Prompt Lookup | 20題 | 40題 | 引用SC的原需求與不同需求，共60個不同來源意圖 |
| TI | 8組×4＝32輪 | 12組×4＝48輪 | 沿用20組原對話與主題順序，口語覆寫另存；首輪不計切換分類，因此24／36個判斷 |
| INT | SC Calibration固定5組＝20次 | SC Evaluation固定10組＝40次 | 只存SC組ID，沒有另一套重複題目 |

SC／SD按不同資訊意圖組成30個候選家族，60個來源FAQ意圖不跨本研究Calibration／Evaluation。廣義職涯、數學、升學主題仍可能相關，不能因不同ID就宣稱語意獨立；P18／P19、P24／S11等邊界及其他風險列在Suite機器資料，需人工仲裁。SC是控制式診斷工作負載，命中率不能冒充真實流量命中率。

TI按完整對話家族重分8／12組，degree／math／fees留Calibration，career及其餘家族留Evaluation。原CV04／CV08曾在舊Calibration，現依career整體放Evaluation；既有資料曝光不會消失，不得稱為新盲測。每輪保留原問題、原Split、來源Turn、Topic候選、切換與所需歷史，均待人工確認。助手歷史沿用既有中性腳本，不是模型實測答案，不冒充真實多輪使用者紀錄。

唯一人工閱讀檔是`experiments/datasets/研究題目審核表.csv`：UTF-8 BOM，只有「研究項目、階段、題組或對話編號、題號、使用者會說的話、審核備註」六欄。共380列研究用途紀錄，包含INT／SD對SC的共用引用，不代表380個獨立問題。QA60使用新版口語候選，歷史原文保留於來源。CSV不是Gold、不是Adapter輸入；修改CSV備註不會自動修改資料或核准審核。完整來源、原題、新題、原因、曝光、切分及needs_review留在機器資料／Reader投影。TI-08薪資、TI-19畢業要求適用年度及TI-20專題規劃的細節變化不自動算主題切換；必要歷史與標籤均為候選。跨演算法會共用FAQ意圖，不能用QA或Evaluation輸出回頭挑選其他研究的參數。

本輪已直接建立這份CSV，沒有執行Python。未來重建可在GPU／分析電腦使用下面的`export-review`；已有檔案時拒絕覆寫，先自行保存審核內容或明確另存，再決定唯一閱讀版本。不要同時建立五份Excel。

### GPU最小功能驗證（非正式研究結果）

先從專案根目錄執行Suite檢查；這個命令不載入模型。開發電腦沒有執行過。

```powershell
python -m experiments.research_tools.suites check --project-root .
# 僅在審核表不存在時重建，保留同一個六欄契約：
# python -m experiments.research_tools.suites export-review --project-root .
```

所有Adapter加`--dataset experiment_suites_v1`，舊入口不加此參數仍使用原230題。QA用Historical；SC／SD／TI／INT用Calibration或Evaluation。SC／INT的功能前綴用`--groups 1`保留完整四問，TI用`--groups 1`保留完整四輪；QA／SD可用`--limit 3`。正式用途不接受這些縮減旗標。輸出使用新GUID目錄。

```powershell
$qaCfg = 'experiments/evaluation/config.json'
$qaData = 'experiments/results/research_tools/suite_qa_data_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --dataset experiment_suites_v1 --target qa --split historical --purpose functional --limit 3 --settings $qaCfg --output-dir $qaData
if ($LASTEXITCODE -ne 0) { throw 'QA export failed' }
$qaRun = 'experiments/results/research_tools/suite_qa_run_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.qa_benchmark --project-root . --adapter-dir $qaData --config $qaCfg --architectures all --run-dir $qaRun --allow-unreviewed

$scCfg = 'experiments/algorithms/semantic_cache/config.gpu.local.json'
$scData = 'experiments/results/research_tools/suite_sc_data_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --dataset experiment_suites_v1 --target sc --split calibration --purpose functional --groups 1 --settings $scCfg --output-dir $scData
if ($LASTEXITCODE -ne 0) { throw 'SC export failed' }
$scRun = 'experiments/results/algorithms/suite_sc_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.semantic_cache.benchmark --project-root . --config $scCfg --mode all --scenarios "$scData/scenarios.jsonl" --run-dir $scRun --allow-unreviewed

$sdCfg = 'experiments/algorithms/speculative_decoding/config.gpu.local.json'
$sdData = 'experiments/results/research_tools/suite_sd_data_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --dataset experiment_suites_v1 --target sd --split calibration --purpose functional --limit 3 --settings $sdCfg --output-dir $sdData
if ($LASTEXITCODE -ne 0) { throw 'SD export failed' }
$sdRun = 'experiments/results/algorithms/suite_sd_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config $sdCfg --mode all --scenarios "$sdData/scenarios.jsonl" --repeats 1 --warmups 1 --run-dir $sdRun --allow-unreviewed
# 同一個sdData/scenarios.jsonl可交給--mode sd2_4，需另一個新run-dir。

$tiCfg = 'experiments/algorithms/topic_identification/config.gpu.local.json'
$tiData = 'experiments/results/research_tools/suite_ti_data_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --dataset experiment_suites_v1 --target ti --split calibration --purpose functional --groups 1 --settings $tiCfg --output-dir $tiData
if ($LASTEXITCODE -ne 0) { throw 'TI export failed' }
$tiRun = 'experiments/results/algorithms/suite_ti_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.topic_identification.benchmark --project-root . --config $tiCfg --phase calibrate --mode detection --calibration "$tiData/calibration.jsonl" --run-dir $tiRun --allow-unreviewed

$intCfg = 'experiments/algorithms/integrated_efficiency/config.gpu.local.json'
$intData = 'experiments/results/research_tools/suite_int_data_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.adapters --project-root . --dataset experiment_suites_v1 --target int --split calibration --purpose functional --groups 1 --suite-cache-state prefilled --settings $intCfg --output-dir $intData
if ($LASTEXITCODE -ne 0) { throw 'INT export failed' }
$intRun = 'experiments/results/research_tools/suite_int_run_' + [guid]::NewGuid().ToString('N')
python -m experiments.research_tools.int_benchmark --project-root . --adapter-dir $intData --config $intCfg --mode all --run-dir $intRun --allow-unreviewed
```

INT預設cold；`--suite-cache-state prefilled`可先以每個所選組的Q1填充，再量完整四問。Initial Fill在獨立檔與獨立報表，不改20／40次measured題數。SC舊Benchmark不接受warm-fill，因此該旗標只開放給phase-aware INT，不能靜默把fill混進SC量測。

正式執行仍需完成問題／答案行為、配對與TI標籤審核，透過既有人審View及Calibration收據閘門。`review export/merge`新增同名`--dataset`選項供機器審核流程相容；本輪不另外產生技術CSV。所有新輸出目前是needs_review候選，沒有宣稱GPU成功、語意無洩漏或正式Gold已凍結。

## 共用研究執行與Excel報告

新增入口`python -m experiments.research_tools.run`，呼叫原Adapter與Benchmark，不改模型、題目、計時或解碼核心。支援`--target qa|sc|sd|ti|int --stage calibration|evaluation --config <本機設定>`。每次自動建立`experiments/results/algorithms/research/<項目_UTC_GUID>/`：

```text
summary.xlsx                 # Overview / Comparison / Per Question / Configuration / Validation / Groups
report.md                    # 證據支持的比較、樣本數與限制
raw/
  context.json / config.snapshot.json / commands.json
  adapter/                   # 原Adapter契約、逐題來源與scorer_only
  benchmark/                 # 原CLI完整輸出與完成收據；不搬動已建立的檔案
  prompt_lookup/             # SD另跑sd2_4，保留其獨立成對SD0
  worker_1.log               # stdout、stderr、Traceback；第二子實驗另有worker_2.log
  calibration.receipts.json  # Evaluation所依據的Calibration收據Hash（若適用）
  evidence.json / completed.json
```

不需額外Excel套件：使用Python標準函式庫寫OOXML。Excel儲存格以明確文字型態防止公式注入；超長內容截短顯示，完整回答留raw。原CLI必要JSON/CSV繼續留raw，不刪除重現證據。只有子Benchmark成功、來源驗證通過且Excel/Markdown都寫出後，才寫`raw/completed.json`；失敗保留`raw/failed.json`及階段，不能只看某個子Benchmark完成就判整體成功。

舊SD `--mode all`只含SD0/SD2-1/SD2-2，新入口自動再跑SD2-4的k=2/4/6/8。兩個子實驗各有SD0，並核對問題序列及Prompt/檢索身分；模型駐留可能不同，跨子實驗不合併SD0或直接計算加速比。Warmup不進Comparison平均；INT的Initial Fill獨立統計，不進Measured。SC是固定模式順序的單次受控序列，不將其延遲視為多次隨機實驗結論；intent proxy不當成錯誤重用Gold。

TI使用原Calibration凍結的thresholds；Evaluation必須指定本入口完成的`--calibration <整個Calibration輸出目錄>`。新報告補出Calibration逐題候選分類，該分數是擬合結果，不是獨立測試。新入口目前跑TI Detection；既有Context Reduction仍可用原CLI，不在這個Detection報告中冒充已量測生成效率。

正式執行不得加`--functional`，必須提供新版人審`--view`及人工簽認`--gold-certification`。人工簽認是獨立JSON，由審核者確認後保存，欄位如下；範例維持false，工具不會代填true：

```json
{
  "schema": "research_gold_certification_v1",
  "human_gold_certified": false,
  "dataset_identity": "從新版人審view_manifest.json取得dataset_identity",
  "review_view_sha256": "該view_manifest.json原始位元組SHA-256",
  "reviewer": "人工審核者",
  "reviewed_at": "人工簽認時間"
}
```

這份簽認不取代逐題/答案/Cache配對/TI標籤人審View。正式Guard仍逐項檢查；未認證只能用明確的`--functional`。SC正式配對需涵蓋所有可能的先前來源，包括同ID重訪。正式QA只有一份已曝光60題評估、沒有Calibration；特例僅限明確指定Suite的QA，仍需人審View與凍結設定，不把Historical曝光改成盲測。

GPU功能Smoke（先驗證新輸出流程，不是正式研究）：

```powershell
python -m experiments.research_tools.suites check --project-root .
if ($LASTEXITCODE -ne 0) { throw 'Suite check failed' }
python -m experiments.research_tools.run --project-root . --target sc --stage calibration --config experiments/algorithms/semantic_cache/config.gpu.local.json --functional --groups 1
```

正式執行（人工View及簽認完成後才可用；設定路徑請確認GPU本機）：

```powershell
$view = '<新版完成的人審View目錄>'
$gold = '<人工Gold簽認JSON>'
$review = @('--view', $view, '--gold-certification', $gold)
python -m experiments.research_tools.run --project-root . --target qa --stage evaluation --config experiments/evaluation/config.json @review
python -m experiments.research_tools.run --project-root . --target sc --stage calibration --config experiments/algorithms/semantic_cache/config.gpu.local.json @review
python -m experiments.research_tools.run --project-root . --target sd --stage calibration --config experiments/algorithms/speculative_decoding/config.gpu.local.json @review
python -m experiments.research_tools.run --project-root . --target ti --stage calibration --config experiments/algorithms/topic_identification/config.gpu.local.json @review
python -m experiments.research_tools.run --project-root . --target int --stage calibration --config experiments/algorithms/integrated_efficiency/config.gpu.local.json --cache-state prefilled @review

# 每個項目分別指定自己的完整Calibration輸出目錄；INT也須維持相同cache-state。
python -m experiments.research_tools.run --project-root . --target sc --stage evaluation --config experiments/algorithms/semantic_cache/config.gpu.local.json --calibration '<SC Calibration輸出目錄>' @review
python -m experiments.research_tools.run --project-root . --target sd --stage evaluation --config experiments/algorithms/speculative_decoding/config.gpu.local.json --calibration '<SD Calibration輸出目錄>' @review
python -m experiments.research_tools.run --project-root . --target ti --stage evaluation --config experiments/algorithms/topic_identification/config.gpu.local.json --calibration '<TI Calibration輸出目錄>' @review
python -m experiments.research_tools.run --project-root . --target int --stage evaluation --config experiments/algorithms/integrated_efficiency/config.gpu.local.json --cache-state prefilled --calibration '<INT Calibration輸出目錄>' @review
```

Evaluation嚴格要求同題庫、同人審View、同設定、同程式與Cache政策的成功Calibration收據，拒絕Provisional/Smoke收據。若要依探索結果改設定，先凍結選定設定並完成相符的正式Calibration，不能拿舊設定的收據直接跑Evaluation。本入口不修改任何GPU套件或模型；所有新程式僅經靜態審查，仍需GPU驗證CLI、Excel開啟與Guard失敗情境。

### r4三表整合候選（保留r3）

`experiments/datasets/experiment_suites_v1/revisions/spoken_candidate_r4/`保存新的questions.jsonl、suites.json及review.json。讀取/執行時明確加`--dataset experiment_suites_v1_r4`；未指定時維持舊入口，不自動切換。r4釘選r3原檔及三份審核輸入的原始位元組Hash，修改輸入需另行版本更新。r3的View、Adapter、Calibration收據不得混用。

本機整合閱讀檔為`experiments/datasets/研究題庫整合審核_r4.csv`（單一分區UTF-8 BOM）；摘要與完整GPU命令見`研究題庫整合審核_r4_摘要.md`。開發端沒有執行Python產生XLSX。GPU可執行`python -m experiments.research_tools.suites export-integrated --project-root . --revision spoken_candidate_r4`，重用既有XLSX Writer產生6工作表的`研究題庫整合審核_r4.xlsx`，已存在時拒絕覆寫。

版本檢查為`python -m experiments.research_tools.suites check --project-root . --revision spoken_candidate_r4`。r4仍needs_review、human_gold_certified=false；部分本機來源支持不能當完整官方答案認證，可能的範圍/時點差異不自動定為直接矛盾。正式結果仍沿用summary.xlsx、report.md與raw/證據，不建立第二套Benchmark或移除Guard。

## 一鍵執行完整r4研究

入口已實作於`experiments.research_tools.run_all`，開發端僅靜態檢查，尚待GPU驗證。不執行訓練、下載或索引重建。從專案根目錄執行：

```powershell
python -m experiments.research_tools.run_all --mode provisional
python -m experiments.research_tools.run_all --mode formal
```

預設QA優先讀`experiments/evaluation/config.gpu.local.json`，不存在則讀既有`config.json`；SC、SD、TI、INT分別讀其演算法目錄下的`config.gpu.local.json`。路徑不同時，參考`experiments/research_tools/run_all.example.json`，在GPU另存並編輯`run_all.local.json`，設定五個config路徑，以及正式使用的`view`與`gold_certification`。可用`--config <總控設定>`指定其他位置；CLI的`--view`、`--gold-certification`優先。所有本機設定及results已受gitignore保護，不自動加入Git。

Provisional會使用既有functional入口，但不傳limit/groups，執行完整題庫並標示非正式；Formal不加allow-unreviewed，先檢查Gold範圍、人工簽認、A2/A3批准來源、Checkpoint與資源Hash，不滿足即停止，不回退Provisional。模型及Tokenizer必須本機完整存在。演算法須共用Target/BGE/文件庫及device/precision/seed/max_new_tokens；總控不偷偷更改設定。A4維持原始BGE，A5沿用A3 Adapter。

順序共11個階段：QA八架構60題Evaluation；SC、SD、TI各Calibration→Evaluation；INT分別Cold與Prefilled各Calibration→Evaluation。SC40/80次、SD20/40題、TI24/36個非首輪判斷（保留32/48輪對話）、INT20/40次。SD另執行SD2-4並保留各自SD0。每個INT狀態使用自己的Calibration收據。TI目前是Detection，不將Context Reduction冒充已量測。

```text
experiments/results/algorithms/research_full/<UTC_GUID>/
  summary.xlsx                 # 九個中文工作表
  report.md                    # 狀態、實際樣本、比較與限制
  raw/
    identity.json / settings.json / state.json / active.lock
    stages/<階段>/attempt_<GUID>/  # 原Runner、Adapter、逐題證據與內部報告
    prior_reports/             # Resume前的報告，保留原證據
    receipt_<GUID>.json / failure_<GUID>.json
```

完成標記必須等全部11階段、最終來源驗證與整合報告都成功。單一階段失敗保留其他成果、阻擋其Evaluation，仍執行無相依的階段；預檢或身分變動則停止全Run。可續跑：

```powershell
python -m experiments.research_tools.run_all --mode provisional --resume <原research_full目錄>
```

Resume核對模型/設定/題庫/人審/程式/套件環境與已提交證據，成功階段不重跑；QA中斷時，只沿用同條件下具完整60題收據的架構，保留原答案及時間，未提交架構重跑。其他演算法以整個失敗階段為重跑單位，不接續不完整Cache或KV狀態。每次嘗試獨立目錄，不覆蓋歷史證據。

Excel包含研究總覽、QA架構比較、QA逐題比較、SC、SD、TI、INT、跨實驗同題比較、研究設定與限制。表頭凍結/篩選/配色及欄寬由既有OOXML Writer輸出，不新增Excel套件。QA另分42個已見與18個保留意圖；Warmup與Initial Fill不混入Measured平均，P95少於20筆不估計。跨實驗只對齊相同ID/版本/文字，不合併不同計時意義。Provisional品質未評定；正式資料Gold也不等於生成答案已正確，缺乏指標以unavailable呈現。
