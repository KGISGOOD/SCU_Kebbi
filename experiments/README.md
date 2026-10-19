# 資科萬事通：問答系統架構研究與實驗

## 一、專案簡介

本專案為「資科萬事通」的問答系統架構研究，主要針對東吳大學資料科學系相關資訊，測試不同的資訊檢索與問答方法，並比較各架構的回答能力、匹配效果及執行效率，作為後續系統整合與優化的基礎。

目前已完成兩種架構的初步實作與功能驗證：

### A0：原始 RAG + LLM（Baseline）

使用 BGE-M3 建立文本向量，透過 FAISS 檢索相關資料，再由 Llama-3.2-3B-Instruct 根據檢索內容生成回答。

已建立固定 60 題測試集，紀錄各題回答、檢索結果及執行時間，作為後續研究與方法比較的基準。

### A1：FAQ Semantic Matching（常見問題語意匹配）

使用 BGE-M3 將常見問題轉換為向量，透過 FAISS 搜尋最相近的 FAQ 問題，直接回傳對應的預設答案，不經過 LLM 生成。

目前建立 60 筆 FAQ 題庫，並設計 15 題專項測試，驗證原始問題、同義改寫及不同意圖問題的匹配情況。

A1 已完成核心功能驗證，但相似度拒答門檻尚未校準，後續可作為 Adaptive Router 的快速 FAQ 回答路徑候選方案。

### 主要技術

- Embedding：BAAI/bge-m3
- Vector Database：FAISS IndexFlatIP
- LLM（A0）：meta-llama/Llama-3.2-3B-Instruct
- Retrieval（A0）：Top-K = 5
- Chunk Size（A0）：512
- Chunk Overlap（A0）：128
- FAQ Matching（A1）：BGE-M3 + FAISS，相似度 Top-1 匹配
- GPU：NVIDIA RTX 3090

---

## 二、專案架構

```text
SCU_Kebbi/
├── 畢業專題_爬蟲/
│   ├── type1/                          # 爬蟲資料
│   └── type2/                          # 爬蟲及 FAQ 資料
│
├── experiments/
│   ├── shared/                         # 共用核心模組
│   │   ├── data_loader.py              # 資料載入
│   │   ├── embeddings.py               # BGE-M3 Embedding
│   │   ├── vector_store.py             # FAISS 向量庫
│   │   ├── retrieval.py                # 資料檢索
│   │   ├── llm_loader.py               # LLM 載入
│   │   └── evaluation.py               # 測試與結果紀錄
│   │
│   ├── baselines/
│   │   ├── rag_baseline/               # A0：RAG + LLM
│   │   │   ├── config.py               # A0 設定
│   │   │   ├── prompts.py              # Prompt 模板
│   │   │   ├── pipeline.py             # RAG 問答流程
│   │   │   ├── build_index.py          # 建立 RAG 向量庫
│   │   │   ├── test_retrieval.py        # 檢索測試
│   │   │   ├── test_baseline.py         # 固定 60 題測試
│   │   │   └── chat.py                 # 互動式問答
│   │   │
│   │   └── faq_semantic_matching/      # A1：FAQ 語意匹配
│   │       ├── config.py               # A1 設定
│   │       ├── faq_loader.py           # FAQ 資料整理
│   │       ├── matcher.py              # FAQ 語意匹配核心
│   │       ├── build_index.py          # 建立 FAQ 向量庫
│   │       ├── test_faq.py             # FAQ 匹配測試
│   │       └── chat.py                 # 互動式 FAQ 問答
│   │
│   ├── datasets/
│   │   ├── scu_fixed_questions_v1_1.jsonl   # 固定 60 題測試集
│   │   ├── scu_faq_matching_test_v1.jsonl    # A1 專項 15 題測試集
│   │   └── faq/
│   │       └── faq_canonical_v1.jsonl        # FAQ 題庫（60 筆）
│   │
│   ├── vector_stores/
│   │   ├── type1_type2_bge_m3_v1/      # A0 FAISS 向量庫
│   │   └── faq_bge_m3_v1/              # A1 FAQ FAISS 向量庫
│   │
│   ├── results/
│   │   ├── rag_baseline/               # A0 測試結果
│   │   └── faq_semantic_matching/      # A1 測試結果
│   │
│   ├── requirements.txt
│   └── README.md
│
└── robot/                               # 機器人相關程式
```

---

## 三、相關執行指令

所有指令皆在 **SCU_Kebbi 專案根目錄** 執行。

### 1. 進入專案目錄

```powershell
cd "C:\Users\rsy\Desktop\SCU_Kebbi"
```

### 2. A0：建立 RAG 向量庫

```powershell
python -B -m experiments.baselines.rag_baseline.build_index
```

讀取 `type1`、`type2` 資料，使用 BGE-M3 建立 FAISS 向量索引。

**向量庫已建立完成，除非資料或索引設定更新，否則不需要重新執行。**

### 3. A0：測試檢索功能

```powershell
python -B -m experiments.baselines.rag_baseline.test_retrieval
```

測試 BGE-M3 + FAISS 是否能檢索相關資料，不執行 LLM 回答生成。

### 4. A0：執行固定 60 題測試

```powershell
python -B -m experiments.baselines.rag_baseline.test_baseline
```

執行固定 60 題 RAG 問答測試，流程如下：

題目 → BGE-M3 → FAISS Top-5 → Llama 3.2 3B → 回答 → 儲存結果

**測試題庫：**

`experiments/datasets/scu_fixed_questions_v1_1.jsonl`

**結果儲存位置：**

`experiments/results/rag_baseline/`

主要輸出檔案：

- `results_YYYYMMDD_HHMMSS.jsonl`：各題回答及檢索紀錄
- `run_metadata_YYYYMMDD_HHMMSS.json`：實驗設定與版本資訊

### 5. A0：啟動互動式問答

```powershell
python -B -m experiments.baselines.rag_baseline.chat
```

手動輸入問題，測試 RAG 系統的即時回答功能。

### 6. A1：建立 FAQ 題庫

```powershell
python -B -m experiments.baselines.faq_semantic_matching.faq_loader
```

讀取 `type2` 中的家長 FAQ 及高中生 FAQ 資料，各 30 筆，共整理 60 筆 FAQ。

**FAQ 儲存位置：**

`experiments/datasets/faq/faq_canonical_v1.jsonl`

**FAQ 資料已建立完成，除非來源資料更新，否則不需要重新執行。**

目前部分 FAQ 答案來自 AI 生成資料，內容尚需進一步確認，不應直接視為學校官方認證資訊。

### 7. A1：建立 FAQ 向量庫

```powershell
python -B -m experiments.baselines.faq_semantic_matching.build_index
```

使用 BGE-M3 將 FAQ 問題轉換為向量，透過 FAISS IndexFlatIP 建立專用索引，並保留問題與預設答案的對應關係。

**向量庫儲存位置：**

`experiments/vector_stores/faq_bge_m3_v1/`

**向量庫已建立完成，除非 FAQ 題庫、Embedding 模型或索引設定更新，否則不需要重新執行。**

### 8. A1：執行 FAQ 專項 15 題測試

```powershell
python -B -m experiments.baselines.faq_semantic_matching.test_faq --test-set faq
```

使用 15 題 FAQ 專用測試集，包含：

- 原始 FAQ：5 題
- 同義改寫：5 題
- 不同意圖：5 題

測試流程如下：

題目 → BGE-M3 → FAISS FAQ Matching → 選取最相近 FAQ → 回傳預設答案 → 儲存結果

**測試題庫：**

`experiments/datasets/scu_faq_matching_test_v1.jsonl`

**結果儲存位置：**

`experiments/results/faq_semantic_matching/`

已完成的功能驗證結果：

- 原始 FAQ：5/5 題成功取得對應答案
- 同義改寫：5/5 題成功取得對應答案
- 不同意圖：部分問題出現不適當匹配
- 平均處理時間：約 0.03 秒／題（不含模型初始化）
- 程式執行錯誤：0

**目前 `threshold=None`，尚未啟用經校準的相似度拒答門檻，因此系統可能將不適用的問題匹配至其他 FAQ。此測試主要用於確認架構功能，不代表正式回答準確率。**

最新版實驗輸出：

- `results_20261019_164838_faq.jsonl`：15 題 FAQ 匹配結果
- `run_metadata_20261019_164846.json`：實驗執行資訊

### 9. A1：執行原本固定 60 題測試

```powershell
python -B -m experiments.baselines.faq_semantic_matching.test_faq
```

使用與 A0 相同的固定 60 題測試集，觀察 A1 對一般校系問題、資訊整合問題、資訊不足及非相關問題的處理情況。

**測試題庫：**

`experiments/datasets/scu_fixed_questions_v1_1.jsonl`

結果儲存於：

`experiments/results/faq_semantic_matching/`

此測試用於觀察 FAQ 架構的適用範圍，不代表所有題目都應由 FAQ Matching 回答。

### 10. A1：啟動互動式 FAQ 問答

```powershell
python -B -m experiments.baselines.faq_semantic_matching.chat
```

手動輸入問題，測試 FAQ 語意檢索及直接回傳預設答案的功能。

---

**補充：** 所有指令使用 `python -B`，避免 Python 產生新的 `.pyc` 快取檔案。已存在的 `__pycache__` 可自行刪除。

目前 A0 已完成基準系統建置與測試，A1 已完成 FAQ 核心功能驗證。後續將持續測試其他問答架構，並評估不同方法在回答品質、檢索效果與執行效率上的差異。