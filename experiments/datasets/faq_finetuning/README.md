# FAQ Embedding Fine-tuning 資料集

## 目錄結構

```
experiments/datasets/faq_finetuning/
├── faq_data_audit.md              # FAQ 資料品質檢查報告
├── faq_training_candidates.jsonl  # 訓練樣本候選檔案 (JSONL 格式)
├://faq_split_plan.md             # 資料集分離規劃
└── README.md                      # 本檔案
```

## 檔案說明

### 1. faq_data_audit.md
記錄 FAQ 原始題庫的品質檢查結果，包括：
- 原始 FAQ 題庫基本資訊
- FAQ 題庫詳細內容
- 品質問題檢查（重複問題、相同意圖不同文字、相似但意圖不同、同一問題不同答案、內容完整性）
- 重要限制與觀察
- 建議後續行動
- 結論

### 2. faq_training_candidates.jsonl
訓練樣本候選檔案，採用 JSONL 格式，每行代表一個 FAQ 的訓練候選：
```json
{
  "faq_id": "...",                    // FAQ 識別碼 (如 P01, S01)
  "question": "...",                  // 原始 FAQ 問題
  "positive": ["...", "...", "..."],  // 約 3 個同意圖改寫 (正樣本)
  "negative": ["...", "...", "..."],  // 明確不同意圖的負樣本 (一般負樣本)
  "hard_negative": ["...", "...", "..."], // 意圖相似但不同的負樣本 (難負樣本)
  "review_status": "needs_review"     // 需要人工審核的狀態
}
```

### 3. faq_split_plan.md
資料集分離規劃，詳細說明：
- 原始資料概況（60 筆 FAQ：30 筆家長 FAQ + 30 筆高中生 FAQ）
- 分離原則（訓練/驗證/測試集完全分離、避免改寫模板重複等）
- 分離比例（訓練集 70%、驗證集 15%、測試集 15%）
- 具體分離策略（按來源分層分配）
- 改寫樣本產出預估
- 後續處理流程

### 4. test.txt
測試檔案（可忽略）

## 資料處理流程

1. **FAQ 資料品質檢查**
   - 檢查原始 FAQ 資料的完整性和品質
   - 識別並記錄品質問題
   - 輸出 `faq_data_audit.md`

2. **訓練樣本生成**
   - 為每個 FAQ 生成約 3 個同意圖改寫作為正樣本
   - 選擇明確不同意圖的 FAQ 作為一般負樣本
   - 選擇意圖相似但不同的 FAQ 作為難負樣本
   - 標記為 `needs_review` 等待人工審核
   - 輸出 `faq_training_candidates.jsonl`

3. **人工審核**
   - 人工檢查訓練樣本的品質和正確性
   - 通過審核後將 `review_status` 更新為 `approved`
   - 不通過的樣本標記為 `rejected` 或修改後重新審核

4. **資料集分離**
   - 根據 `faq_split_plan.md` 中的規劃
   - 將通過審核的樣本按 70%/15%/15% 分離到訓練/驗證/測試集
   - 確保同一個改寫模板不會同時出現在訓練集與最終測試集

5. **格式轉換**
   - 將通過審核的訓練樣本轉換為 FlagEmbedding 官方支援的 JSONL 格式：
     ```json
     {"query":"問題","pos":["正樣本1","正樣本2"...],"neg":["負樣本1","負樣本2"...]}
     ```
   - 此步驟將在後續階段進行

## 使用說明

此資料集專為 BAAI/bge-m3 模型的 FAQ Embedding Fine-tuning 設計，目的是改進模型對同一意圖 vs 不同意圖 FAQ 問題的區分能力。

### 比較方案
- **A1**: 原始 BGE-M3 模型（基線）
- **A2-1**: BGE-M3 + 正向對比學習 + 一般負樣本
- **A2-2**: BGE-M3 + 正向對比學習 + 難負樣本

### 後續工作
1. 完成所有 60 筆 FAQ 的訓練樣本生成
2. 完成人工審核過程
3. 按照分離規劃劃分訓練/驗證/測試集
4. 轉換為 FlagEmbedding 格式
5. 進行模型微調實驗
6. 評估 A1、A2-1、A2-2 三種方案的效果

## 注意事項

- 所有新增內容僅限於 `experiments/datasets/faq_finetuning/` 目錄
- 不得修改既有的 A0/A1/A4、robot/、畢業專題_爬蟲/ 等目錄中的檔案
- 訓練過程僅針對 Dense Embedding 部分進行 fine-tuning
- 最終測試問句不得參與模型微調或超參數調整