# 資科萬事通：問答系統架構研究與實驗

## 一、專案簡介

本專案為「資科萬事通」的問答系統架構研究，主要針對東吳大學資料科學系相關資訊，實作並測試不同的資訊檢索與問答架構，作為後續系統整合與優化的基礎。

目前包含以下三種架構：

### A0：原始 RAG + LLM（Baseline）

使用 BGE-M3 將文件轉換為向量，透過 FAISS 檢索最相關的 Top-5 文件，再交由 Llama-3.2-3B-Instruct 生成回答。

流程：

使用者問題 → BGE-M3 → FAISS Top-5 → LLM → 回答

### A1：FAQ Semantic Matching（常見問題語意匹配）

使用 BGE-M3 將 FAQ 問題轉換為向量，透過 FAISS 找出語意最相近的 FAQ，直接回傳對應的預設答案，不經過 LLM 生成。

目前建立 60 筆 FAQ 題庫，並提供專項 15 題及固定 60 題測試。

流程：

使用者問題 → BGE-M3 → FAISS FAQ Top-1 → 預設答案

### A4：Hybrid RAG + Reranker（混合檢索與重新排序）

結合 BM25 關鍵字檢索及 BGE-M3 語意檢索，分別取得 Top-10 文件，再透過 RRF（Reciprocal Rank Fusion）融合排名。

A4 分為兩種實驗版本：

**A4-1：Hybrid RAG**

BM25 Top-10 + Dense Top-10 → RRF → Top-5 → LLM → 回答

**A4-2：Hybrid RAG + Reranker**

BM25 Top-10 + Dense Top-10 → RRF Top-10 → BGE Reranker → Top-5 → LLM → 回答

兩個版本使用相同的固定 60 題測試集，主要比較加入 Reranker 前後的回答與執行效率。

### 主要技術

- Embedding：BAAI/bge-m3
- Vector Database：FAISS IndexFlatIP
- LLM：meta-llama/Llama-3.2-3B-Instruct
- BM25：中文 jieba 分詞
- Fusion：RRF（k = 60）
- Reranker：BAAI/bge-reranker-v2-m3
- Chunk Size：512
- Chunk Overlap：128
- 最終檢索文件數量：Top-5
- GPU：NVIDIA RTX 3090

---

## 二、專案架構

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
│   │   ├── rag_baseline/                 # A0
│   │   │   ├── config.py
│   │   │   ├── prompts.py
│   │   │   ├── pipeline.py
│   │   │   ├── build_index.py
│   │   │   ├── test_retrieval.py
│   │   │   ├── test_baseline.py
│   │   │   └── chat.py
│   │   │
│   │   ├── faq_semantic_matching/        # A1
│   │   │   ├── config.py
│   │   │   ├── faq_loader.py
│   │   │   ├── matcher.py
│   │   │   ├── build_index.py
│   │   │   ├── test_faq.py
│   │   │   └── chat.py
│   │   │
│   │   └── hybrid_rag_reranker/          # A4
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
│   ├── datasets/
│   │   ├── scu_fixed_questions_v1_1.jsonl
│   │   ├── scu_faq_matching_test_v1.jsonl
│   │   └── faq/
│   │       └── faq_canonical_v1.jsonl
│   │
│   ├── vector_stores/
│   │   ├── type1_type2_bge_m3_v1/
│   │   ├── faq_bge_m3_v1/
│   │   └── hybrid_bm25_v1/
│   │
│   ├── results/
│   │   ├── rag_baseline/
│   │   ├── faq_semantic_matching/
│   │   └── hybrid_rag_reranker/
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

### 4. A4：Hybrid RAG + Reranker

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

### 5. 測試資料與結果位置

**固定 60 題測試集**

`experiments/datasets/scu_fixed_questions_v1_1.jsonl`

**A1 FAQ 專項測試集**

`experiments/datasets/scu_faq_matching_test_v1.jsonl`

**實驗結果**

- A0：`experiments/results/rag_baseline/`
- A1：`experiments/results/faq_semantic_matching/`
- A4：`experiments/results/hybrid_rag_reranker/`

實驗輸出主要包含：

- `results_*.jsonl`：各題回答、檢索結果及執行時間
- `run_metadata_*.json`：實驗執行資訊

---

## 四、執行注意事項

- A0、A1 與 A4 的索引若已建立完成，且資料及索引設定未更動，無須重複建立。
- A4 共用 A0 的 BGE-M3 FAISS 向量庫。
- A4 使用 `--use-reranker` 區分 A4-1 與 A4-2。
- A4 可使用 `--num-questions` 指定測試題數。
- 所有指令使用 `python -B`，避免產生新的 `.pyc` 快取檔案。
- 正式測試使用相同固定 60 題，以便進行架構比較。