# 資科萬事通 RAG + LLM Baseline

這是一個基於檢索增強生成（RAG）的實驗性基線系統，使用 BGE-M3 嵌入模型、FAISS 向量搜索和 Llama-3.2-3B-Instruct 語言模型來回答關於資訊管理學系的問題。

## 專案結構

```
experiments/
├── shared/                 # 共用核心模組
│   ├── data_loader.py      # 資料載入與解析 (type1/type2)
│   ├── embeddings.py       # BGE-M3 嵌入模型載入
│   ├── vector_store.py     # FAISS 向量庫管理
│   ├── retrieval.py        # 密集檢索
│   ├── llm_loader.py       # Llama-3.2-3B-Instruct 載入
│   └── evaluation.py       # 評估工具
│
├── baselines/rag_baseline/ # 基線專用模組
│   ├── config.py           # 配置管理
│   ├── prompts.py          # Prompt 模板
│   ├── pipeline.py         # RAG 流程串接
│   ├── build_index.py      # 建立向量庫
│   ├── test_retrieval.py   # 檢索測試
│   ├── test_baseline.py    # 完整基線測試
│   └── chat.py             # 互動式聊天介面
│
├── datasets/               # 測試資料集
├── vector_stores/          # 已建立的 FAISS 索引
├── results/                # 測試結果
├── requirements.txt        # Python 依賴
└── README.md               # 本檔案
```

## 快速開始

### 1. 環境準備

```bash
# 複製專案到目標機器
git clone <repository-url>
cd SCU_Kebbi

# 建立虛擬環境
python -m venv experiments/.venv
source experiments/.venv/bin/activate  # Linux/macOS
# 或
.\experiments\.venv\Scripts\activate   # Windows

# 安裝依賴
pip install -r experiments/requirements.txt
```

### 2. 下載 Llama-3.2-3B-Instruct 模型

由於授權限制，需要手動同意模型使用條款：

1. 前往 [Hugging Face模型頁面](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
2. 登入您的 Hugging Face 帳號
3. 點擊 "Agree to access repository" 並接受條款
4. 設定存取權杖：
   ```bash
   pip install huggingface_hub
   huggingface-cli login
   ```
   輸入您的存取權杖（可從個人設定頁取得）

### 3. 建立向量索引

```bash
python -m experiments.baselines.rag_baseline.build_index
```

這將：
- 從 `type1/` 和 `type2/` 目錄載入資料
- 使用 BGE-M3 模型生成嵌入向量
- 建立 FAISS 向量索引（使用 IndexFlatIP 確保可重現性）
- 保存索引和版本資訊到 `experiments/vector_stores/type1_type2_bge_m3_v1/`

### 4. 執行測試

#### 檢索測試
```bash
python -m experiments.baselines.rag_baseline.test_retrieval
```

#### 完整基線測試
```bash
python -m experiments.baselines.rag_baseline.test_baseline
```

#### 互動式聊天
```bash
python -m experiments.baselines.rag_baseline.chat
```

## 配置說明

所有配置位於 `experiments/baselines/rag_baseline/config.py` 中：

- **資料路徑**：指向 `type1/` 和 `type2/` 目錄
- **向量索引路徑**：`experiments/vector_stores/type1_type2_bge_m3_v1/`
- **嵌入模型**：BAAI/bge-m3 (1024 維度)
- **LLM 模型**：meta-llama/Llama-3.2-3B-Instruct
- **檢索參數**：Top-K = 5
- **文本切分**：chunk_size = 512, chunk_overlap = 128
- **Prompt 模板**：基於檢索內容的問題回答模板

## 數據處理說明

### 資料來源
- `type1/`：43 個檔案包含系統公告、課程資訊、訪談及社群貼文 (CSV/TXT)
- `type2/`：8 個檔案主要為 FAQ 類資料及畢業資訊 (CSV/TXT)

### 處理流程
1. **遞迴掃描**：處理兩個目錄下的所有 `.txt` 和 `.csv` 檔案
2. **編碼處理**：嘗試多種編碼 (UTF-8, UTF-8-SIG, CP950, BIG5, GBK)
3. **內容清理**：移除 HTML 標籤、標準化空白字元
4. **Metadata 提取**：
   - source_group: type1 或 type2
   - source_file: 檔名
   - source_path: 相對路徑
   - content_type: 根據檔名推斷 (FAQ, 公告, 訪談, 社群貼文等)
   - title, published_date, original_row 等
5. **去重**：基於完全相同的 `page_content` 移除重複文件
6. **文本切分**：使用 RecursiveCharacterTextSplitter
   - FAQ 內容保持完整 (不切分) 除非極長
   - 其他內容依段落、句號切分
   - chunk_size = 512, chunk_overlap = 128

### 注意事項
- FAQ 問題與答案被保留在相同 Document 中
- 不使用原本 `robot/data` 或舊 FAISS 索引
- 所有路徑相對於專案根目錄，確保可移植性

## 模型資訊

### 嵌入模型：BAAI/bge-m3
- 維度：1024
- 最大序列長度：512
- 特性：支援多向量檢索，但此實作僅使用稠密向量
- 正規化：L2 正規化使內積等於餘弦相似度

### 語言模型：Llama-3.2-3B-Instruct
- 參數：30 億
- 上下文長度：4096 tokens
- 版本：指令調校版本
- 授權：需要同意 Meta 的模型使用條款

## 成績輸出

測試結果將保存為 JSONL 格式於 `experiments/results/rag_baseline/` 目錄：
- 每行為一個 JSON 物件
- 包含問題、答案、檢索結果及時間統計
- 檔案名稱格式：`results_YYYYMMDD_HHMMSS.jsonl`

## 除錯提示

### 常見問題
1. **模型載入失敗**：
   - 確認已同意 Hugging Face 模型條款
   - 檢查網路連接和磁碟空間
   - 驗證 `huggingface_hub` 已正確安裝並登入

2. **向量庫為空**：
   - 確認已執行 `build_index.py`
   - 檢查 `type1/` 和 `type2/` 目錄路徑是否正確
   - 查看建立過程的日志訊息

3. **顯存不足 (GPU)**：
   - 使用 CPU 模式：將 `settings.llm_device` 設為 `"cpu"`
   - 或使用量化：在 `config.py` 中設定 `llm_load_in_8bit = True`

4. **中文亂碼**：
   - 確保終端機支援 UTF-8 編碼
   - 在 Windows 上使用 `chcp 65001` 設定代碼頁

## 未來改進方向

1. **進階檢索**：
   - 實作混合檢索 (稠密 + 稀疏)
   - 加入重新排序模型
   - 實作多向量檢索 (利用 BGE-M3 的 ColBERT 特性)

2. **LLM 優化**：
   - 實作請求批處理以提高吞吐量
   - 加入快取機制減少重複計算
   - 探索更小型或更專業的語言模型

3. **使用者介面**：
   - 開發網頁介面或 Slack/Discord 機器人
   - 加入對話歷史和上下文保持
   - 實作來源標註和引用功能

4. **評估與基準**：
   - 建立人工標註的測試集
   - 實作自動評估指標 (準確率、相關度等)
   - 加入 A/B 測試框架比較不同配置

## 聯絡與貢獻

此專案為實驗性基線，歡迎提出改進建議或貢獻程式碼。

授權：MIT