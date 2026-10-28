# 804題固定測試入口

以本資料夾`input/`的JSONL與使用說明為唯一題目／分配版本。不修改原文、ID、順序或原分配。它不是校準集或訓練來源。新入口不引用舊Run、r4、舊FAQ CSV或60題預設。

## 執行與身分

`python -u -B -m experiments.benchmark_804.run_all --config experiments/benchmark_804/config.local.json`

預設為測試推論與效能證據蒐集，不表示Formal Gold已認證。未標準化的品質一律待評分。原有Formal／人審／模型收據驗證不變；此入口不簽署Gold。模型若仍provisional，須明確加`--allow-provisional`，不把旗標當批准。若任一必要QA架構不可用，仍執行其他可用項目但以partial/exit2結束。

先從`config.example.json`另存GPU的`config.local.json`，填本機3B、1B、BGE、Reranker、文件索引及已訓練A2/A3路徑。3個訓練路徑已依使用者提供的20261028訓練Run預填。全部模型離線，不下載、不訓練、不自動建文件索引。`run_gpu.ps1`提供一次填入4個本機基礎模型路徑的建立方式，既有本機設定不覆寫。程式與資料需完整同步，`.local.json`只留GPU。

`--check-only`只執行CPU資料／文字曝光稽核與報告，不載入模型或索引pickle；`--smoke-only`先在GPU執行獨立開發問題。預設完整入口每個Worker都先Smoke，通過才執行其完整工作；其他Worker不會被單一失敗抹除。Smoke取既有V2 Validation且與804文字不重複的2題，僅作功能檢查，不拿804選門檻或修Prompt。SD所有模式均跑開發Smoke並嚴格對照Token。

## 分配及可比較範圍

- 804題：單輪626；多輪178。其中561題同時標FAQ／RAG／LoRA，另65題標安全／不確定性。8種QA各用相同561題主比較及相同65題獨立安全cohort，不把安全題混入FAQ準確率。多輪不攤平成單輪QA。
- FAQ比對A1/A2三組共用V2 frozen的138個標準意圖與答案。這是事前既定的知識候選庫，沒有把804或Gold問題加入FAQ。使用原FAQMatcher的FAISS Top-1及原FAQAnswerer的正規化dot-product Top-1；移除60題資料契約僅發生在獨立Adapter，不改研究核心。僅建立RAM FAQ向量供推論，不寫出舊／新FAISS索引。
- SC用分配檔362題、原始順序，None／Exact／Semantic各自、每Repeat從空Cache開始。Cold完整序列後，保留該模式的答案Cache再跑同一序列作Warm；Warm是刻意的再訪實驗，不是獨立未見輸入。Cold成本另列，不能省略。題型不是配對Gold，同義／不同意圖標籤不自動代表可／不可重用。
- SD使用413個指定候選、相同Top-5／Prompt／Tokenizer，單題Prompt只準備一次；SD0、SD2-1(k4)、k2/k4/k6/k8。SD2-1與k4相同候選設定如實保留，不宣稱兩個獨立方法。Prompt Lookup另有配對SD0及k2/k4/k6/k8子研究，不受舊`--mode all`遺漏影響。SD0先建立基準，其他方法依題／Repeat輪替；Baseline-first是計時限制。預設FP32、SDPA、TF32關閉，缺VRAM即失敗，不降精度。各研究Draft駐留條件不同，不跨研究混算。
- INT使用同一362題順序；INT0標準、INT1 Cache、INT2 Draft/Lookup、INT3 Cache+Draft/Lookup，分兩個decoder子研究，均用自己的INT0。Cold與Warm分開，Warm的初始填充就是前一Cold pass；每個variant/repeat獨立Cache。沒有聲稱Attention與RAG/Cache/SD整合已完成。
- TI獨立功能Smoke會執行真實BGE及Attention。40組160輪目前是候選、18輪無分組，未人審前正式主題分數unavailable。需要獨立Review overlay和非804的完成Calibration收據，禁止在804 refit。完整8架構多輪QA目前也標unavailable。

## 稽核、模型資格及恢復

### 完整 Prompt 的 4096-token 配置

`max_input_tokens` 可設定至 4096。A0、A4、A5、SC、SD、Prompt Lookup、INT
均以完整生成 Prompt 為限制對象，不截斷文件；原始 Base 與 Draft 的
`config.json/max_position_embeddings` 必須至少容納輸入上限加 `max_new_tokens`，
執行時另檢查實際 Token 數與已載入模型容量。這不是 3090 顯存足夠的保證；
FP32 的既有顯存檢查與 OOM 失敗紀錄保留，不自動降低精度。
FAQ/Embedding 的 512-token 問題限制、A3 單輪問題限制及 TI Attention 的
2048-token 安全限制是不同用途，不隨生成 Prompt 預算改動。

修正前 QA A0/A4 的共用 LLMLoader 會截斷至 2048；因此舊 QA full 成功
不代表所有題目都使用完整檢索內容。現在只有明確選用完整 Prompt 協定的
Adapter 會開啟嚴格檢查，其他舊入口預設不變。

舊 Run 的設定、程式與模型 Hash 均釘選；修改程式或將 2048 改為 4096
後不可強行 Resume。請保留原 config.local.json，另存新的本機設定檔並
建立新 Run，以免修改舊 Run 仍釘選的原始設定。舊 A1 的 FAQ 測量可繼續
作為原 Run 的有效證據，但不冒充新的 4096 完整 Prompt 協定測量。
舊 Run 可使用 `--report-only <原Run目錄>` 重建衍生報告，不執行模型、不
改寫逐題紀錄；失敗 Worker 沒有 records/index.json 時會列入缺失證據與
失敗清單，其他已提交紀錄仍會進入報告。已存在的 index/逐題 Hash 仍嚴格驗證。

八種 QA 都在 plan 中。若 A2-1/A2-2/A3/A5 未啟動，先查看
`raw/resource_issues.json` 及 `raw/stages.json`：資料夾存在不等於通過資格。
Provisional 模型需明確 `--allow-provisional`，不等同正式研究批准；
Train Hash、完整 Epoch、Adapter/Base identity 等檢查不會因此略過。

稽核V2實際Train問題、FAQ標準問題與804的NFC／空白正規化完全相同及3-gram Jaccard>=.85，後者只是字面旗標。不能據零字串命中宣稱語意／知識未曝光。GPU另檢查trusted local index.pkl中的文件、source_group及原題完整出現情形；不把使用者上傳的檔案當pickle載入。若確有訓練／FAQ／知識文本原題命中，預設阻擋完整測量，保留所有題目。只有明確`--allow-overlap-for-measurement`可保留曝光分層繼續效能蒐證，不可聲稱獨立未見泛化。

Checkpoint先核對A2完整性、mode、>=3 Epoch、共同原始模型及Positive Schedule；Train Hash必須與本次稽核V2一致。A3核對Adapter完整性、>=3 Epoch、無smoke limit、V2 Train Hash及原始Base identity。GPU的A3/A5 Smoke會真正從磁碟重新載入Adapter並生成新Token，結果保留；開發端沒有冒充已載入。A4只用原始BGE。

`--resume <原Run目錄>`嚴格要求同設定、程式、資料、套件與模型位元組及相同provisional／overlap旗標。使用新Attempt檔與原子index，每題成功紀錄不覆寫。獨立QA/SD略過已測項；SD沿用Hash綁定的Prompt快照。SC/INT從已保存的Miss答案及當時原始embedding重建相同Cache ID／label／origin，不重新生成前綴答案，也不把恢復時間加到既有延遲。狀態序列失敗時停止該前綴，不在缺失Cache狀態下測Warm；其他模式繼續。原Token mismatch是已測失敗，不透過Resume自動洗掉。

SD保留逐Repeat原Token等價Guard、SD0重複穩定性與差異位置。失敗輸出的實測延遲仍列出，但排除Lossless-eligible Speedup。INT未命中時與同query/repeat/phase的INT0比Token及Prompt／Retrieval；Hit只屬答案重用，不能稱lossless。

## 人工標註的獨立overlay

可選`review_file`為JSON物件，需真實`reviewer`、帶時區`reviewed_at`、`dataset_sha256`，不能由程式自動批准：

- `questions`：以Qxxxx為key，每筆`status: verified`、`evidence`，可提供`gold_faq_id`（V2的Kxxx/Mxxx，不能填Qxxxx）、`gold_answer`、`gold_document_ids`、`source_ids`。有Gold才計標註子集指標；答案文字完全相同比較不等同語意正確。
- `cache_pairs`：每筆`origin_id`、`query_id`、`reusable`布林、`status: verified`及適用該pair的`evidence`。只評實際命中的來源，未標註誤用率unavailable。
- `conversations`：以原conversation_id為key，`status: verified`、`evidence`、原順序`question_ids`四個ID，後三輪的`shift_labels`布林。`history_scope: full_reviewed_messages`需提供`histories`（後三輪ID各自對應完整前文user/assistant，原問題不可改）。若採`previous_user_questions_only`，只做問題歷史特徵，且必須使用同協定的獨立Calibration，不能拿完整回答歷史的門檻套用。

`ti_thresholds`需為既有TI Benchmark的thresholds.json及同目錄completed.json／calibration.snapshot.json；來源Hash、模型／特徵身分與歷史協定必須一致，校準問題不可來自804。

## 輸出與規模

`experiments/results/benchmark_804/<UTC時間_UUID>/`：主要閱讀`summary.xlsx`、`report.md`；`raw/`保存題目／設定／標註快照、曝光稽核、模型來源、身分Hash、逐項狀態、逐題不可覆寫Attempts、Prompt快照與Worker stdout/stderr/traceback。

Excel含Overview、Comparison、Per Question、Question Coverage（完整804題）、By Type、Pending Quality、TI Metrics、Configuration、Validation、Failures。P95採nearest rank，Warmup排除，Cold/Warm分開；錯誤與未執行原因不隱藏。沒有第三方Excel套件需求。

`--report-only <Run>`只重整既有證據，不重跑模型；原始Attempts不變。完整完成標記只表示可用實驗測量完成且Token Guard沒有失敗，仍明列unavailable與待評分；不代表人審通過。

預設3 Repeats下：QA每架構1878筆；SC每模式2172筆；SD7434筆、Prompt Lookup6195筆；INT每個decoder研究8688筆。不是只推論804次，需預留長時間與磁碟。不要根據804輸出選最好的threshold或模型。
