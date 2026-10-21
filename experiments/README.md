# 資科萬事通：問答系統架構研究與實驗

## 一、專案簡介

本專案為「資科萬事通」的問答系統架構研究，主要針對東吳大學資料科學系相關資訊，實作並測試不同的資訊檢索與問答架構，作為後續系統整合與優化的基礎。

目前包含 A0、A1、A2 及 A4 四種實驗架構，其中 A2 分為一般對比學習與加入 Hard Negative 的兩種微調方式，A4 則分為是否使用 Reranker 的兩種版本。

### A0：原始 RAG + LLM（Baseline）

使用 BGE-M3 將文件轉換為向量，透過 FAISS 檢索最相關的 Top-5 文件，再交由 Llama-3.2-3B-Instruct 生成回答。

**流程：**

使用者問題 → BGE-M3 → FAISS Top-5 → LLM → 回答

### A1：FAQ Semantic Matching（常見問題語意匹配）

使用原始 BGE-M3 將 FAQ 問題轉換為向量，透過 FAISS 找出語意最相近的 FAQ，直接回傳對應的預設答案，不經過 LLM 生成。

目前建立 60 筆 FAQ 題庫，並提供 FAQ 專項 15 題測試及固定 60 題測試。

**流程：**

使用者問題 → 原始 BGE-M3 → FAISS FAQ Top-1 → 預設答案

### A2：FAQ Embedding Fine-tuning（FAQ 語意向量微調）

A2 延伸 A1 的 FAQ Semantic Matching 架構，針對 BGE-M3 進行領域微調，使模型學習使用者不同問法與 FAQ 標準問題之間的語意關係。

A2 沿用 A1 的 60 筆 FAQ 作為檢索候選庫，並將 FAQ 意圖區分為訓練、驗證及測試集合。

目前規劃兩種微調方式：

#### A2-1：Contrastive Fine-tuning（一般對比學習）

以使用者問題改寫與對應的 FAQ 標準問題建立 Positive Pair，透過對比學習調整 Embedding 模型，使語意相近的問題在向量空間中更接近。

訓練時使用批次內其他 FAQ 問題作為負例，並透過遮罩避免不適當的 False Negative 參與 Loss 計算。

**訓練流程：**

FAQ 訓練資料 → Positive Pair → BGE-M3 → Masked Contrastive Loss → A2-1 微調模型

**檢索流程：**

使用者問題 → A2-1 微調 BGE-M3 → FAISS FAQ Top-1 → 預設答案

#### A2-2：Contrastive Fine-tuning + Hard Negative

在 A2-1 的基礎上，額外加入與正確 FAQ 語意相近、但意圖不同的 Hard Negative，讓模型學習區分容易混淆的 FAQ。

A2-2 使用與 A2-1 相同的 Positive Pair 及訓練排程，主要差異為訓練 Loss 中額外加入 Hard Negative。

**訓練流程：**

FAQ 訓練資料 → Positive Pair + Hard Negative → BGE-M3 → Masked Contrastive Loss → A2-2 微調模型

**檢索流程：**

使用者問題 → A2-2 微調 BGE-M3 → FAISS FAQ Top-1 → 預設答案

#### A2 資料規劃

A2 使用 60 筆 FAQ，依 FAQ 意圖進行切分，避免同一 FAQ 意圖同時出現在微調訓練與保留的驗證、測試集合。

| 資料集合 | FAQ 意圖數 | 用途 |
|---|---:|---|
| Train | 42 | 建立模型微調所需的正負樣本 |
| Validation | 9 | 驗證未參與微調的 FAQ 意圖 |
| Test | 9 | 保留作為後續正式測試 |

另外建立以下資料：

- `faq_training_candidates.jsonl`：微調訓練候選資料。
- `faq_eval_queries.jsonl`：FAQ 語意檢索評估查詢。
- `faq_retrieval_candidates.jsonl`：完整 60 題 FAQ 檢索候選庫。
- `faq_hard_diagnostic.jsonl`：額外的困難問題診斷資料。
- `faq_split_manifest.json`：FAQ 意圖切分資訊。

其中，Unseen 代表該 FAQ 意圖未參與微調訓練，不代表該 FAQ 不存在於檢索候選庫。

訓練與評估資料具有人工審核狀態，未經核准的資料僅能透過明確指定的 provisional 模式進行測試。

### A4：Hybrid RAG + Reranker（混合檢索與重新排序）

結合 BM25 關鍵字檢索及 BGE-M3 語意檢索，分別取得 Top-10 文件，再透過 RRF（Reciprocal Rank Fusion）融合排名。

A4 分為兩種實驗版本：

#### A4-1：Hybrid RAG

使用 BM25 與 Dense Retrieval 進行混合檢索，經 RRF 排名融合後，選取 Top-5 文件交由 LLM 生成回答。

**流程：**

使用者問題 → BM25 Top-10 + BGE-M3 Dense Top-10 → RRF → Top-5 → LLM → 回答

#### A4-2：Hybrid RAG + Reranker

在 A4-1 的基礎上加入 BGE Reranker，針對 RRF 融合後的候選文件重新排序，再選取 Top-5 文件提供 LLM。

**流程：**

使用者問題 → BM25 Top-10 + BGE-M3 Dense Top-10 → RRF Top-10 → BGE Reranker → Top-5 → LLM → 回答

兩種版本使用相同的固定 60 題測試集，主要比較加入 Reranker 前後的回答與執行效率。

### 主要技術

| 項目 | 使用技術 |
|---|---|
| Embedding | BAAI/bge-m3 |
| Vector Database | FAISS IndexFlatIP |
| LLM | meta-llama/Llama-3.2-3B-Instruct |
| BM25 | 中文 jieba 分詞 |
| Rank Fusion | RRF（k = 60） |
| Reranker | BAAI/bge-reranker-v2-m3 |
| Fine-tuning | PyTorch、SentenceTransformers |
| Training Loss | Masked Contrastive Loss（InfoNCE） |
| Negative Sampling | In-batch Negative、Hard Negative |
| Chunk Size | 512 |
| Chunk Overlap | 128 |
| RAG 最終檢索文件數 | Top-5 |
| FAQ 最終匹配數 | Top-1 |
| GPU | NVIDIA RTX 3090 |

---

## 二、專案架構

以下列出各實驗的主要程式與資料夾。

```text
SCU_Kebbi/
├── 畢業專題_爬蟲/
│   ├── type1/
│   └── type2/
│
├── experiments/
│   ├── shared/
│   │   ├── data_loader.py
│   │   ├── embeddings.py
│   │   ├── vector_store.py
│   │   ├── retrieval.py
│   │   ├── llm_loader.py
│   │   └── evaluation.py
│   │
│   ├── baselines/
│   │   ├── rag_baseline/                  # A0
│   │   │   ├── config.py
│   │   │   ├── prompts.py
│   │   │   ├── pipeline.py
│   │   │   ├── build_index.py
│   │   │   ├── test_retrieval.py
│   │   │   ├── test_baseline.py
│   │   │   └── chat.py
│   │   │
│   │   ├── faq_semantic_matching/         # A1
│   │   │   ├── config.py
│   │   │   ├── faq_loader.py
│   │   │   ├── matcher.py
│   │   │   ├── build_index.py
│   │   │   ├── test_faq.py
│   │   │   └── chat.py
│   │   │
│   │   └── hybrid_rag_reranker/           # A4
│   │       ├── config.py
│   │       ├── bm25_retrieval.py
│   │       ├── fusion.py
│   │       ├── reranker.py
│   │       ├── pipeline.py
│   │       ├── prompts.py
│   │       ├── build_bm25_index.py
│   │       ├── test_retrieval.py
│   │       ├── test_hybrid.py
│   │       └── chat.py
│   │
│   ├── a2_faq_finetuning/                 # A2
│   │   ├── common.py
│   │   ├── training_config.json
│   │   ├── training_data.py
│   │   ├── training_loss.py
│   │   ├── training_compat.py
│   │   ├── train_faq_embedding.py
│   │   ├── validate_a2_data.py
│   │   ├── build_faq_candidates.py
│   │   ├── evaluate_faq_retrieval.py
│   │   ├── diagnostic_data.py
│   │   ├── evaluate_hard_diagnostic.py
│   │   ├── README.md
│   │   ├── TRAINING.md
│   │   ├── TRAINING_COMPATIBILITY.md
│   │   └── STATIC_REVIEW.md
│   │
│   ├── datasets/
│   │   ├── scu_fixed_questions_v1_1.jsonl
│   │   ├── scu_faq_matching_test_v1.jsonl
│   │   ├── faq/
│   │   │   └── faq_canonical_v1.jsonl
│   │   │
│   │   └── faq_finetuning/
│   │       ├── faq_split_manifest.json
│   │       ├── faq_training_candidates.jsonl
│   │       ├── faq_eval_queries.jsonl
│   │       ├── faq_retrieval_candidates.jsonl
│   │       └── faq_hard_diagnostic.jsonl
│   │
│   ├── vector_stores/
│   │   ├── type1_type2_bge_m3_v1/
│   │   ├── faq_bge_m3_v1/
│   │   └── hybrid_bm25_v1/
│   │
│   ├── models/
│   │   ├── a2_1_smoke_task8/
│   │   │   └── final_model/
│   │   └── a2_2_smoke_task8/
│   │       └── final_model/
│   │
│   ├── results/
│   │   ├── rag_baseline/
│   │   ├── faq_semantic_matching/
│   │   ├── hybrid_rag_reranker/
│   │   ├── a2_faq_retrieval/
│   │   │   ├── A1/
│   │   │   ├── A2-1/
│   │   │   └── A2-2/
│   │   └── a2_hard_diagnostic/
│   │       └── A1/
│   │
│   ├── requirements.txt
│   └── README.md
│
└── robot/
```

---

## 三、相關執行指令

所有指令皆在 `SCU_Kebbi` 專案根目錄執行。

### 1. 進入專案目錄

```powershell
cd "C:\Users\rsy\Desktop\SCU_Kebbi"
```

### 2. A0：原始 RAG

**建立向量庫**

```powershell
python -B -m experiments.baselines.rag_baseline.build_index
```

**測試檢索功能**

```powershell
python -B -m experiments.baselines.rag_baseline.test_retrieval
```

**執行固定 60 題**

```powershell
python -B -m experiments.baselines.rag_baseline.test_baseline
```

**互動式問答**

```powershell
python -B -m experiments.baselines.rag_baseline.chat
```

### 3. A1：FAQ Semantic Matching

**建立 FAQ 題庫**

```powershell
python -B -m experiments.baselines.faq_semantic_matching.faq_loader
```

**建立 FAQ 向量庫**

```powershell
python -B -m experiments.baselines.faq_semantic_matching.build_index
```

**執行 FAQ 專項 15 題**

```powershell
python -B -m experiments.baselines.faq_semantic_matching.test_faq --test-set faq
```

**執行固定 60 題**

```powershell
python -B -m experiments.baselines.faq_semantic_matching.test_faq
```

**互動式 FAQ 問答**

```powershell
python -B -m experiments.baselines.faq_semantic_matching.chat
```

### 4. A2：FAQ Embedding Fine-tuning

A2 使用獨立的訓練與評估程式，不需要修改 A1 原有的檢索架構。

以下訓練指令使用 GPU 電腦執行。

#### 4.1 驗證訓練資料

```powershell
python -B -m experiments.a2_faq_finetuning.validate_a2_data --project-root . --scope training
```

#### 4.2 建立完整 FAQ 檢索候選庫

```powershell
python -B -m experiments.a2_faq_finetuning.build_faq_candidates --project-root .
```

#### 4.3 驗證全部 A2 資料

```powershell
python -B -m experiments.a2_faq_finetuning.validate_a2_data --project-root . --scope all
```

#### 4.4 查看訓練程式參數

```powershell
python -B -m experiments.a2_faq_finetuning.train_faq_embedding --help
```

#### 4.5 A2-1：一般對比學習架構測試

使用 Positive Pair，執行一個 Epoch 的 Smoke Test。

```powershell
python -B -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_1 --model BAAI/bge-m3 --device cuda --epochs 1 --batch-size 3 --max-length 64 --learning-rate 2e-5 --seed 42 --allow-unreviewed --offline --gradient-checkpointing --verify-save-load --output-dir experiments/models/a2_1_smoke_task8
```

#### 4.6 A2-2：Hard Negative 對比學習架構測試

使用相同的 Positive Pair，額外加入 Hard Negative，執行一個 Epoch 的 Smoke Test。

```powershell
python -B -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_2 --model BAAI/bge-m3 --device cuda --epochs 1 --batch-size 3 --max-length 64 --learning-rate 2e-5 --seed 42 --allow-unreviewed --offline --gradient-checkpointing --verify-save-load --output-dir experiments/models/a2_2_smoke_task8
```

**說明：**

- `--epochs 1`：執行一個 Epoch，供架構測試。
- `--batch-size 3`：使用較小 Batch Size，降低顯存需求。
- `--max-length 64`：限制訓練輸入長度。
- `--gradient-checkpointing`：啟用 Gradient Checkpointing。
- `--verify-save-load`：儲存模型後重新載入驗證。
- `--allow-unreviewed`：允許使用尚未人工核准的資料進行 provisional 測試。
- `--offline`：使用已快取的模型，不從網路下載。

輸出目錄必須不存在，重新執行時需指定新的資料夾名稱。

#### 4.7 A2-1：微調模型檢索評估

```powershell
python -B -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model experiments/models/a2_1_smoke_task8/final_model --model-tag A2-1 --device cuda --split unseen_validation --allow-unreviewed --offline
```

#### 4.8 A2-2：微調模型檢索評估

```powershell
python -B -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model experiments/models/a2_2_smoke_task8/final_model --model-tag A2-2 --device cuda --split unseen_validation --allow-unreviewed --offline
```

#### 4.9 A1：原始模型檢索評估

使用相同的 A2 評估資料及候選庫，測試未微調的原始 BGE-M3。

```powershell
python -B -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model BAAI/bge-m3 --model-tag A1 --device cuda --split unseen_validation --allow-unreviewed --offline
```

#### 4.10 困難診斷資料結構檢查

```powershell
python -B -m experiments.a2_faq_finetuning.evaluate_hard_diagnostic --project-root . --check-only
```

#### 4.11 A1：困難診斷檢索

```powershell
python -B -m experiments.a2_faq_finetuning.evaluate_hard_diagnostic --project-root . --model BAAI/bge-m3 --model-tag A1 --device cuda --allow-unreviewed --offline
```

困難診斷屬於開發階段的額外測試，不與正式 Validation、Test 混合。

### 5. A4：Hybrid RAG + Reranker

**建立 BM25 索引**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.build_bm25_index
```

**測試 BM25、Dense 與 RRF 檢索功能**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.test_retrieval
```

**A4-1：執行固定 60 題（不使用 Reranker）**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.test_hybrid
```

**A4-2：執行固定 60 題（使用 Reranker）**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.test_hybrid --use-reranker
```

**A4-1：只測試前 5 題**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.test_hybrid --num-questions 5
```

**A4-2：只測試前 5 題**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.test_hybrid --use-reranker --num-questions 5
```

**顯示檢索來源**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.test_hybrid --use-reranker --num-questions 5 --show-sources
```

**互動式問答**

```powershell
python -B -m experiments.baselines.hybrid_rag_reranker.chat
```

### 6. 測試資料與輸出位置

**固定 60 題測試集（A0、A1、A4）**

`experiments/datasets/scu_fixed_questions_v1_1.jsonl`

**A1 FAQ 專項測試集**

`experiments/datasets/scu_faq_matching_test_v1.jsonl`

**A2 訓練資料**

`experiments/datasets/faq_finetuning/faq_training_candidates.jsonl`

**A2 評估查詢**

`experiments/datasets/faq_finetuning/faq_eval_queries.jsonl`

**A2 FAQ 檢索候選庫**

`experiments/datasets/faq_finetuning/faq_retrieval_candidates.jsonl`

**A2 困難診斷查詢**

`experiments/datasets/faq_finetuning/faq_hard_diagnostic.jsonl`

**A2 微調模型**

- A2-1：`experiments/models/a2_1_smoke_task8/final_model/`
- A2-2：`experiments/models/a2_2_smoke_task8/final_model/`

**實驗結果輸出位置**

- A0：`experiments/results/rag_baseline/`
- A1 原架構：`experiments/results/faq_semantic_matching/`
- A4：`experiments/results/hybrid_rag_reranker/`
- A2 FAQ 檢索：`experiments/results/a2_faq_retrieval/`
- A2 困難診斷：`experiments/results/a2_hard_diagnostic/`

A2 的訓練輸出另外包含模型權重、訓練紀錄、設定資訊及完整性 Manifest。評估輸出包含逐題檢索排名、評估摘要及執行 Metadata。

---

## 四、執行注意事項

### 1. 專案執行環境

所有指令應在 `SCU_Kebbi` 專案根目錄執行，並確保 Python 套件及必要模型已安裝或快取。

A2 訓練程式已針對目前 GPU 電腦所使用的 SentenceTransformers 6.1.0 進行相容性調整。

### 2. 索引與模型管理

A0、A1 及 A4 的索引若已建立，且資料與索引設定未更動，無須重複建立。

A4 共用 A0 的 BGE-M3 FAISS 向量庫。

A2 使用獨立的微調模型及 FAQ 檢索候選庫，不直接覆蓋 A1 原始模型或向量索引。

更換 Embedding 模型後，FAQ 候選資料須使用對應模型重新編碼，不能直接沿用其他模型產生的向量。

### 3. A2 訓練資料與測試資料

A2 使用 Train、Validation、Test 意圖切分。

模型微調只讀取訓練資料，驗證及測試查詢不應加入訓練流程。

目前部分資料標記為 `needs_review`。使用 `--allow-unreviewed` 執行的訓練及評估均屬 provisional 測試，不能直接視為正式實驗。

### 4. A2 Smoke Test 與正式訓練

README 中的 A2 訓練指令是架構 Smoke Test 設定，使用一個 Epoch、Batch Size 3 及 Max Length 64。

正式訓練應使用經核准的資料與正式設定，並將正式模型輸出至不同目錄，避免覆蓋 Smoke Test 模型。

### 5. A4 實驗參數

A4 使用 `--use-reranker` 區分 A4-1 與 A4-2。

可以透過 `--num-questions` 指定測試題數，或使用 `--show-sources` 查看檢索來源。

### 6. Python 執行方式

執行指令統一使用 `python -B`，避免建立新的 `.pyc` 快取檔案。

開發電腦主要負責程式碼與資料管理，模型訓練及實際實驗則在 GPU 電腦執行。

### 7. 實驗資料保存

建議分開保存以下內容：

- 原始資料及資料切分。
- 訓練設定與模型權重。
- 各次訓練與評估的執行紀錄。
- 模型與輸入資料的版本、Hash 資訊。

大型模型權重不建議直接透過一般 Git 提交，應另外備份或使用適合大型檔案的管理方式。