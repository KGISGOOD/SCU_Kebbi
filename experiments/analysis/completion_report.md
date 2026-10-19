# 完成報告：資科萬事通 RAG Baseline 固定題庫與基礎 Prompt 整合

## 一、實際新增的檔案

1. **固定測試題庫**
   - 路徑：`D:\Ray DB\SCU\SCU_Kebbi\experiments\datasets\scu_fixed_questions_v1_1.jsonl`
   - 說明：包含 exactement 60 題的 JSONL 檔案，題型分配符合規範
   - 版本：1.1
   - 參考日期：2026-10-19

## 二、實際修改的檔案

1. **Prompt 模板**
   - 路徑：`D:\Ray DB\SCU\SCU_Kebbi\experiments\baselines\rag_baseline\prompts.py`
   - 修改內容：
     - 新增 Prompt 版本追蹤：`PROMPT_VERSION = "baseline_basic_v1"`
     - 完全重寫 `build_prompt` 函數以實作固定 Baseline Prompt
     - 保持原有的 `format_documents` 和 `create_context_from_documents` 函數不變
     - 整合系統提示詞和使用者提示詞，符合 Llama 3.2 Instruct 的 Chat Template 格式

2. **測試程式**
   - 路徑：`D:\Ray DB\SCU\SCU_Kebbi\experiments\baselines\rag_baseline\test_baseline.py`
   - 修改內容：
     - 完全重寫 `load_test_questions` 函數以讀取新的 JSONL 題庫
     - 新增 `save_run_metadata` 函數以儲存實驗設定
     - 增強結果紀錄格式，包含所有要求的欄位
     - 改進錯誤處理機制，單題失敗不影響整體執行
     - 保留舊結果不被覆蓋（使用時間戳檔案名稱）
     - 添加題型、配對資訊等額外欄位到結果中

## 三、固定題庫的位置

- **完整路徑**：`D:\Ray DB\SCU\SCU_Kebbi\experiments\datasets\scu_fixed_questions_v1_1.jsonl`
- **相對路徑**（從專案根目錄）：`experiments/datasets/scu_fixed_questions_v1_1.jsonl`
- **檔案格式**：JSONL（每行一個 JSON 物件）
- **題目總數**：60 題
- **題型分配**：
  - A：直接問答 - 10 題 (A01-A10)
  - B：同義改寫 - 10 題 (B01-B10)
  - C：相似但不同意圖 - 10 題 (C01-C10)
  - D：跨資訊整合 - 6 題 (D01-D06)
  - U：相關但資料不足 - 8 題 (U01-U08)
  - O：完全不相關 - 6 題 (O01-O06)
  - T：最新資訊與時效 - 6 題 (T01-T06)
  - R：完全相同問題重測 - 4 題 (R01-R04)

## 四、60 題的題型數量與配對檢查結果

### 題型數量驗證
- ✅ A型：10題 (A01-A10)
- ✅ B型：10題 (B01-B10)
- ✅ C型：10題 (C01-C10)
- ✅ D型：6題 (D01-D06)
- ✅ U型：8題 (U01-U08)
- ✅ O型：6題 (O01-O06)
- ✅ T型：6題 (T01-T06)
- ✅ R型：4題 (R01-R04)
- ✅ **總計：60題**

### 配對關係驗證
- ✅ A01、B01、C01：皆屬 G01 (核心能力)
- ✅ A02、B02、C02：皆屬 G02 (數學程式基礎)
- ✅ A03、B03、C03：皆屬 G03 (畢業學分)
- ✅ A04、B04、C04：皆屬 G04 (NLP老師)
- ✅ A05、B05、C05：皆屬 G05 (文華獎學金類別)
- ✅ A06、B06、C06：皆屬 G06 (SJSU升學合作)
- ✅ A07、B07、C07：皆屬 G07 (AI實作設備)
- ✅ A08、B08、C08：皆屬 G08 (實習合作企業)
- ✅ A09、B09、C09：皆屬 G09 (申請入學項目)
- ✅ A10、B10、C10：皆屬 G10 (系辦位置)
- ✅ R01：repeat_of=A03 (畢業學分重測)
- ✅ R02：repeat_of=A04 (NLP老師重測)
- ✅ R03：repeat_of=A06 (SJSU合作重測)
- ✅ R04：repeat_of=A09 (申請入學項目重測)

## 五、基本 Prompt 最終內容

系統提示詞：
```
你是「資科萬事通」，專門回答東吳大學巨量資料管理學院資料科學系的相關問題。
請主要根據提供的參考資料回答，使用繁體中文，內容清楚、簡潔。
如果參考資料不足以確認答案，請明確說明目前資料不足，不要自行猜測或編造。
涉及日期、招生、學分或規定時，不要把不同年度的資訊混為一談；無法確認目前最新資訊時，應說明無法確認。
如果問題與東吳資料科學系完全無關，請簡短說明本系統主要提供系所相關資訊。
測試參考日期為 2026 年 10 月 19 日；這是固定的離線測試基準，不代表資料一定是最新的。
```

使用者提示詞格式：
```
參考資料：

{context}

問題：

{question}

請回答：
```

完整 Prompt 結構（Llama 3.2 Instruct Chat Template）：
```
<|begin_of_text|><|start_header_id|>system<|end_header_id|>
{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>
{user_prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```

Prompt 版本：`baseline_basic_v1`

## 六、`test_baseline.py` 如何讀取題庫

1. **預設路徑**：
   ```python
   filepath = os.path.join(
       settings.datasets_dir, "scu_fixed_questions_v1_1.jsonl"
   )
   ```

2. **後備機制**：
   - 如果固定題庫不存在，嘗試讀取原始 `test_questions.txt`
   - 如果兩者都不存在，使用內建的 10 預設問題
   - 所有情況下都返回標準化的問題字典格式

3. **資料載入過程**：
   - 逐行讀取 JSONL 檔案
   - 每行解析為 JSON 物件
   - 確保必要欄位存在（question）
   - 設定預設值 voor 選擇性欄位（question_id, test_type, as_of, dataset_version）
   - 捕獲並處理 JSON 解析錯誤
   - 返回標準化的問題字典列表

## 七、每題結果紀錄哪些欄位

每筆結果包含以下欄位：

### 必填欄位
- `question_id`：題目識別碼 (如 A01, B05, R03)
- `test_type`：題型 (direct, synonym, similar_diff_intent, multi_info_integration, related_insufficient, unrelated, latest_info, repeat)
- `question`：完整問題文字
- `answer`：模型生成的答案
- `success`：布林值，表示是否成功執行 (非錯誤答案)
- `error`：如果失敗則為錯誤訊息，成功則為 null

### 時序欄位
- `timestamp`：時間戳 (epoch 時間)
- `total_time`：單題平均耗時 (近似值)

### 檢索資訊
- `retrieved_sources`：列表，包含每份檢索文件的詳細資訊：
  - `rank`：排名 (1-based)
  - `source_file`：來源檔案名稱
  - `content_type`：內容類型
  - `score`：檢索分數
  - `content`：文件內容 (前500字元，超過則加省略號)
  - `metadata`：完整文件中繼資料

### 原始題目中繼資料（如果存在）
- `as_of`：參考日期
- `dataset_version`：題庫版本
- `pair_group`：配對群組編號
- `repeat_of`：重測參考的原題 ID

## 八、Run Metadata 包含哪些設定

每次執行會產生一個 `run_metadata_YYYYMMDD_HHMMSS.json` 檔案，包含：

### 基本資訊
- `run_timestamp`：時間戳 (YYYYMMDD_HHMMSS 格式)
- `run_datetime`：完整 ISO 格式日期時間

### 數據集資訊
- `dataset_version`：1.1
- `dataset_as_of`：2026-10-19
- `prompt_version`：baseline_basic_v1
- `question_dataset_path`：題庫完整路徑
- `question_count`：題目總數 (60)

### 模型與資源設定
- `llm_model_id`：實際使用的 LLM 模型 (例如：meta-llama/Llama-3.2-3B-Instruct)
- `embedding_model`：實際使用的 Embedding 模型 (例如：BAAI/bge-m3)
- `vector_store_path`：FAISS 索引路徑
- `vector_store_type`：FAISS IndexFlatIP
- `top_k`：檢索文件數量
- `chunk_size`：文本分塊大小
- `chunk_overlap`：塊重疊大小

### LLM 生成參數
- `llm_parameters`：物件，包含：
  - `temperature`
  - `top_p`
  - `max_new_tokens`
  - `do_sample`
  - `repetition_penalty`

### 其他資訊（如果可用）
- `data_fingerprint`：資料指紋 (如果現有程式可取得)

## 九、是否保留舊版 10 題結果

✅ **是，完全保留**。新程式：
- 使用時間戳命名結果檔案 (例如：`results_20261019_143022.jsonl`)
- 使用時間戳命名中繼資料檔案 (例如：`run_metadata_20261019_143022.json`)
- 不會覆蓋或修改既有的 `results_*.jsonl` 檔案
- 舊的 10 題測試結果將保持在 `experiments/results/rag_baseline/` 目錄中不受影響

## 十、哪些檔案需要同步到 RTX 3090 電腦

以下檔案必須存在於 RTX 3090 電腦上對應的路徑：

1. **專案結構**（必須完整保持）：
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\datasets\scu_fixed_questions_v1_1.jsonl`
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\baselines\rag_baseline\prompts.py`
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\baselines\rag_baseline\test_baseline.py`
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\baselines\rag_baseline\config.py` (未修改，但必須存在)
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\baselines\rag_baseline\pipeline.py` (未修改，但必須存在)
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\shared\evaluation.py` (未修改，但必須存在)

2. **資料來源目錄**（必須保持不變）：
   - `D:\Ray DB\SCU\SCU_Kebbi\畢業專題_爬蟲\type1\` (42 個原始檔案)
   - `D:\Ray DB\SCU\SCU_Kebbi\畢業專題_爬蟲\type2\` (9 個原始檔案)

3. **現有向量庫**（如果想跳過重建）：
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\vector_stores\type1_type2_bge_m3_v1\` (FAISS 索引檔案)
   - **注意**：如果此目錄不存在或索引損毀，需要先執行 `build_index.py`

4. **結果目錄**（建議保持）：
   - `D:\Ray DB\SCU\SCU_Kebbi\experiments\results\rag_baseline\` (用於保存新結果)

## 十一、在 RTX 3090 電腦上執行的指令

```
python -m experiments.baselines.rag_baseline.test_baseline
```

### 執行前準備：
1. 確認已切換到專案根目錄：`D:\Ray DB\SCU\SCU_Kebbi`
2. 確認向量庫已經建好 (若未建好，需先執行：`python -m experiments.baselines.rag_baseline.build_index`)
3. 確認有足夠的 GPU 記憶體和 CUDA 環境
4. 確認所有相依套件已經安裝 (transformers, torch, sentence-transformers, faiss-cpu 或 faiss-gpu, langchain-core, etc.)

### 執行後預期結果：
- 輸出顯示："Loaded Questions: 60"
- 輸出顯示："Processed Questions: 60"
- 輸出顯示："Successful: 60" (代表所有題目都完成了問答流程，不代表答案正確)
- 輸出顯示："Failed: 0"
- 在 `experiments/results/rag_baseline/` 目錄中會看到：
  - 新的結果檔案：`results_YYYYMMDD_HHMMSS.jsonl` (60 筆記錄)
  - 新的中繼資料檔案：`run_metadata_YYYYMMDD_HHMMSS.json`
- 舊的 10 題結果檔案將保持不變

### 重要說明：
根據您的指示，**這次驗證只需確認工程流程正常**，不需要評估回答正確率、優化回答內容或進行任何額外的研究方法開發。成功標準是程式能夠完整處理 60 題而不崩潰，並產生預期的結果檔案。

所有修改都嚴格遵守了不改變現有技術架構的要求：
- 保持 BGE-M3 Embedding 模型
- 保持 FAISS IndexFlatIP 向量存儲
- 保持 Llama-3.2-3B-Instruct LLM
- 保持原有的 Chunk Size (512) 和 Chunk Overlap (128)
- 保持原有的 Top-K (5)
- 不使用任何進階技術 (Semantic Cache, Reranker, Fine-tuning, LoRA, QLoRA, Speculative Decoding 等)