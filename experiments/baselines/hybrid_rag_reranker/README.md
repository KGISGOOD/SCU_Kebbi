# Hybrid RAG + Reranker (A4) 實作說明

這是專題「資科萬事通」的 A4 Hybrid RAG + Reranker 架構實作。

## 目錄結構
```
experiments/baselines/hybrid_rag_reranker/
├── config.py              # 配置檔案
├── bm25_retrieval.py      # BM25 檢索實作
├── fusion.py              # RRF Fusion 實作
├── reranker.py            # BGE Reranker 實作
├── pipeline.py            # 主要管道實作
├── prompts.py             # 提示詞模板（與 A0 相同）
├── build_bm25_index.py    # BM25 索引建立腳本
├── test_retrieval.py      # 檢索元件測試
├── test_hybrid.py         # 完整管道測試
├── chat.py                # 互動式聊天介面
└── README.md              # 本檔案
```

## 1. 新增及修改的檔案

所有檔案都是新增的，位於 `experiments/baselines/hybrid_rag_reranker/` 目錄下：
- `config.py` - 配置設定
- `bm25_retrieval.py` - BM25 檢索實作
- `fusion.py` - RRF Fusion 實作
- `reranker.py` - BGE Reranker 實作
- `pipeline.py` - 主要管道實作
- `prompts.py` - 提示詞模板
- `build_bm25_index.py` - BM25 索引建立腳本
- `test_retrieval.py` - 檢索測試
- `test_hybrid.py` - 完整管道測試
- `chat.py` - 互動式聊天介面
- `README.md` - 說明檔案

## 2. 是否完整沿用 A0 的 chunk、Embedding、LLM 和 Prompt

✅ **完全沿用 A0 的實作以確保公平比較**：
- **Chunking策略**: 完全沿用 A0 的 `chunk_documents()` 函式，使用相同的 `chunk_size=512` 和 `chunk_overlap=128`
- **Embedding模型**: 沿用 A0 的 BAAI/bge-m3，透過 `experiments.shared.embeddings.BGEM3Embedder`
- **LLM模型**: 沿用 A0 的 meta-llama/Llama-3.2-3B-Instruct，透過 `experiments.shared.llm_loader.LLMLoader`
- **Prompt template**: 完全沿用 A0 的提示詞，確保只測試檢索策略的影響
- **Dense Vector Store**: 使用與 A0 相同的 FAISS IndexFlatIP，位於 `experiments/vector_stores/type1_type2_bge_m3_v1/`

## 3. BM25 採用的中文分詞策略

實作了一個簡單的中文分詞函式 `_simple_chinese_tokenize()`：
- 先將文本轉換為小寫以保持一致性
- 使用正則表達式按標點符號和空白字元分割：`[\s\-\_\.\,\!\?\;\:\~\@\#\$\%\^\&\*\(\)\+\=\[\]\{\}\|\/\\\"\'\n\r\t]+`
- 過濾掉空標記和長度小於2的標記（避免單字符標記）
- 這是一個實用的基礎實作，避免引入外部依賴如 jieba，同時能夠處理中英混合文本

## 4. RRF、Reranker 的實際實作方法

### RRF Fusion (Reciprocal Rank Fusion)
- 位於 `fusion.py` 中的 `reciprocal_rank_fusion()` 函式
- 參數 k=60（預設值，可透過 config 調整）
- 實作步驟：
  1. 為 BM25 和 Dense 檢索結果分別建立排名映射（文件 ID → 排名）
  2. 計算每個文件的 RRF 分數：`score += 1/(k + rank)`，其中 rank 為該檢索器中的排名
  3. 如果文件只在一個檢索結果中出現，仍會獲得該檢索器的貢獻
  4. 按 RRF 分數降序排序文件
  5. 返回顯前 top_k 個文件及其分數

### Reranker (BGE Reranker)
- 位於 `reranker.py` 中的 `Reranker` 類別
- 使用模型：BAAI/bge-reranker-v2-m3
- 實作步驟：
  1. 將 query 和 document 組合成 pairs，格式為：`query [SEP] document`
  2. 使用 tokenizer 處理批次的 query-document pairs
  3. 透過模型獲得 logits 作為相關性分數
  4. 根據分數降序重新排序文件
  5. 返回重新排序後的顶-k 個文件及其分數
- 整合點：在 `pipeline.py` 中的 `HybridRAGRerankerPipeline` 類別中，當 `use_reranker=True` 時，會在 RRF fusion 後應用 reranker

## 5. 必須複製到 RTX 3090 電腦的檔案

以下目錄和檔案必須完整複製：
- `experiments/baselines/hybrid_rag_reranker/` (完整目錄)
- `experiments/shared/` (保持不變，提供共用元件)
- `experiments/datasets/scu_fixed_questions_v1_1.jsonl` (測試題目)
- `experiments/vector_stores/type1_type2_bge_m3_v1/` (A0 的 dense 向量存儲，必須已存在)
- `experiments/畢業專題_爬蟲/type1/` 和 `type2/` (原始資料目錄)

## 6. 建立 BM25 索引、檢索測試及完整 60 題測試的指令

### 建立 BM25 索引
```bash
python -m experiments.baselines.hybrid_rag_reranker.build_bm25_index
```
選項：
- `--force-rebuild`：強制重建即使有效索引存在
- `--chunk-size INT`：區塊大小 (預設: 512)
- `--chunk-overlap INT`：區塊重疊 (預設: 128)

### 測試檢索元件 (BM25、Dense、Hybrid)
```bash
python -m experiments.baselines.hybrid_rag_reranker.test_retrieval
```
選項：
- `--queries QUERY [QUERY ...]`：測試查詢
- `--top-k INT`：返回結果數量 (預設: 10)
- `--show-details`：顯示詳細元資料

### 測試完整管道 (A4-1: 無 reranker)
```bash
python -m experiments.baselines.hybrid_rag_reranker.test_hybrid
```

### 測試完整管道 (A4-2: 有 reranker)
```bash
python -m experiments.baselines.hybrid_rag_reranker.test_hybrid --use-reranker
```
通用選項：
- `--questions-file FILEPATH`：自訂題目檔案
- `--output-dir DIRPATH`：結果輸出目錄
- `--num-questions INT`：測試題目數量
- `--show-sources`：顯示檢索來源

### 互動式聊天
```bash
python -m experiments.baselines.hybrid_rag_reranker.chat
```
將提示選擇模式：
- 1：A4-1 (無 reranker)
- 2：A4-2 (有 reranker)

## 7. 尚待實際執行驗證的部分

以下部分尚未實際執行驗證（依照限制只能讀取、建立及修改檔案）：
- BM25 索引建立流程的實際執行與驗證
- 向量存儲載入與驗證（假設 A0 的索引已存在且有效）
- BGE-M3 embedding 模型載入與向量生成
- BGE Reranker 模型載入與預測
- Llama-3.2-3B-Instruct 模型載入與文字生成
- 完整的端到端測試 (test_hybrid.py)
- 互動式聊天功能 (chat.py)

您將在 RTX 3090 電腦上手動執行以上指令來驗證實作的正確性。實作已設計為與現有 A0 架構相容，只增添了混合檢索和 reranker 功能，而不修改任何現有元件。