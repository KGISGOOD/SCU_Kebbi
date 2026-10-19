# 資科萬事通 RAG + LLM Baseline

## 一、專案簡介

本專案為「資科萬事通」的 RAG（檢索增強生成）基準系統，主要針對東吳大學資料科學系相關資訊進行問答。

系統使用 BGE-M3 建立文本向量，透過 FAISS 檢索相關資料，再由 Llama-3.2-3B-Instruct 根據檢索內容生成回答。

目前已建立固定 60 題測試集，用於驗證 Baseline 問答流程，並保存各題的回答、檢索結果及實驗紀錄，作為後續研究與方法比較的基礎。

**主要技術：**
- Embedding：BAAI/bge-m3
- Vector Database：FAISS IndexFlatIP
- LLM：meta-llama/Llama-3.2-3B-Instruct
- Retrieval：Top-K = 5
- Chunk Size：512
- Chunk Overlap：128
- GPU：NVIDIA RTX 3090

## 二、專案架構

```text
SCU_Kebbi/
├── 畢業專題_爬蟲/
│   ├── type1/                      # 爬蟲資料
│   └── type2/                      # 爬蟲資料
│
├── experiments/
│   ├── shared/                     # 共用核心模組
│   │   ├── data_loader.py          # 資料載入
│   │   ├── embeddings.py           # BGE-M3 Embedding
│   │   ├── vector_store.py         # FAISS 向量庫
│   │   ├── retrieval.py            # 資料檢索
│   │   ├── llm_loader.py           # LLM 載入
│   │   └── evaluation.py           # 測試與結果紀錄
│   │
│   ├── baselines/
│   │   └── rag_baseline/
│   │       ├── config.py           # Baseline 設定
│   │       ├── prompts.py          # Prompt 模板
│   │       ├── pipeline.py         # RAG 問答流程
│   │       ├── build_index.py      # 建立 FAISS 向量庫
│   │       ├── test_retrieval.py   # 檢索測試
│   │       ├── test_baseline.py    # 固定 60 題測試
│   │       └── chat.py             # 互動式問答
│   │
│   ├── datasets/
│   │   └── scu_fixed_questions_v1_1.jsonl
│   │
│   ├── vector_stores/
│   │   └── type1_type2_bge_m3_v1/  # FAISS 向量庫
│   │
│   ├── results/
│   │   └── rag_baseline/           # 測試結果
│   │
│   ├── requirements.txt
│   └── README.md
│
└── robot/                           # 機器人相關程式
```

## 三、相關執行指令

所有指令皆在 **SCU_Kebbi 專案根目錄**執行。

### 1. 進入專案目錄

```powershell
cd "C:\Users\rsy\Desktop\SCU_Kebbi"
```

### 2. 建立 FAISS 向量庫

```powershell
python -B -m experiments.baselines.rag_baseline.build_index
```

讀取 `type1`、`type2` 資料，使用 BGE-M3 建立 FAISS 向量索引。

**向量庫已建立完成，除非資料或索引設定更新，否則不需要重新執行。**

### 3. 測試檢索功能

```powershell
python -B -m experiments.baselines.rag_baseline.test_retrieval
```

測試 BGE-M3 + FAISS 是否能正確檢索相關資料，不執行 LLM 回答生成。

### 4. 執行 Baseline（固定 60 題）

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

### 5. 啟動互動式問答

```powershell
python -B -m experiments.baselines.rag_baseline.chat
```

手動輸入問題，測試 RAG 系統的即時回答功能。

---

**補充：** 所有指令使用 `python -B`，避免 Python 產生新的 `.pyc` 快取檔案。已存在的 `__pycache__` 可自行刪除。