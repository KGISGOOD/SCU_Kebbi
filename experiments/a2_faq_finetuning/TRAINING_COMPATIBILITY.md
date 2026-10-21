# A2 任務 8：ST 6.1.0 訓練相容性修復

## 狀態與查核依據

只完成程式編輯與靜態查核，沒有執行 Python、模型、測試、訓練、安裝或改動 GPU 套件。使用者回報環境為 Python 3.10.1、torch 2.13.0+cu126、sentence-transformers 6.1.0、transformers 5.16.1、numpy 1.26.4；先前 A1／Validation／Hard Diagnostic 可執行，不代表 backward、optimizer 或 checkpoint 已可執行。

查核指定版本官方原始碼，而非只放寬範圍：

- [ST v6.1.0 BaseModel](https://github.com/huggingface/sentence-transformers/blob/v6.1.0/sentence_transformers/base/model.py)：preprocess 是新輸入 API；tokenize 是仍存在的 deprecated alias。forward 仍依已載入模組順序執行，save_pretrained/save 保存 ST 格式，transformers_model 提供底層模型。
- [ST v6.1.0 SentenceTransformer](https://github.com/huggingface/sentence-transformers/blob/v6.1.0/sentence_transformers/sentence_transformer/model.py)：encode 使用 inference_mode，不能用來訓練。
- [ST v6.1.0 Pooling](https://github.com/huggingface/sentence-transformers/blob/v6.1.0/sentence_transformers/sentence_transformer/modules/pooling.py)：保留 sentence_embedding 輸出，支援舊 pooling 設定轉換；本程式不改為自行平均 token。
- [ST v6.1.0 套件設定](https://github.com/huggingface/sentence-transformers/blob/v6.1.0/pyproject.toml)：Python>=3.10，torch>=2.2，transformers>=5,<6，numpy>=1.24；A2 另保守限制 torch<3，並保留舊版 API profile。未枚舉全部轉移相依套件，沒有保證任意組合相容。
- [Transformers v5.16.1](https://github.com/huggingface/transformers/blob/v5.16.1/src/transformers/modeling_utils.py)：gradient_checkpointing_enable 接受 gradient_checkpointing_kwargs；新版分支明確指定 use_reentrant=False。

## 確認的問題與修改

原報錯發生在舊範圍 gate，在模型載入前便中止。這是**已確認的版本策略不符**；沒有證據說原 forward 或 save 已在 GPU 電腦失敗。auto_model 在新版仍有相容 alias，也不能宣稱已移除。

新增 training_compat.py，實作明確兩個 profile：

| profile | 輸入及 forward | 底層存取與儲存 |
|---|---|---|
| legacy_st_2_tokenize | 原範圍的 tokenize → model(features) | 舊 auto_model／save |
| st_6_1_0_preprocess | ST==6.1.0 的 preprocess(inputs=..., prompt=None) → model(features) | 公開 transformers_model／save_pretrained，safe_serialization=True |

新版直接建立 torch SentenceTransformer，offline 加 local_files_only=True；遞迴把 feature tensor 放到裝置，保留非 tensor 的處理資訊；檢查必要 API、torch backend、Pooling、可訓練參數及 unprompted 契約。若帶 default prompt 或不是預期的 dense ST 模組，就明確停止，不默默改 pooling／prompt／架構。保留 loaded Pooling 設定並將模組類型／pooling 模式寫入 provenance。使用者提供的版本符合新 profile 的數值條件，仍不是執行通過證明。

修改 train_faq_embedding.py 接上上述 adapter，沒有改 training_data.py、training_loss.py、training_config.json 或 A2 評估數學。沒有新 Trainer、datasets、accelerate、LoRA、混合精度或新套件需求；不修改 experiments/requirements.txt，避免影響其他實驗及現有 GPU 環境。

## Forward、loss 與梯度

model.train() → preprocess（舊版 tokenize）→ feature tensor 移到 device → model(features) → 原 checkpoint 的 Transformer／Pooling 等模組 → sentence_embedding。

- training forward 必須不在 inference_mode；以 enable_grad 執行並檢查輸出為 B×D tensor，requires_grad=true、grad_fn 非空。沒有 detach／NumPy／推論 encode。
- Query 與 Positive 各自 forward；A2-2 額外文件也走同一 forward，與 base 文件 cat，從原 Masked InfoNCE 計算 loss，再 loss.backward。
- 三路輸出 retain_grad；backward 後逐路檢查梯度存在且 finite，並檢查至少有模型參數梯度，之後才 gradient clip、AdamW.step。缺梯度不會當成功。norm 只在檢查後 detach 用於日誌，不影響 loss。
- 某些被 mask 的配對數學上應為零梯度，極端或對稱情況也可能產生零 norm；不能強迫每個 embedding 分量非零。當 hard 已在 base 欄位，梯度由 base 分支承接，不重複編碼以製造額外分母權重。
- embedding_branch_gradients 每步記錄 query、positive_documents 與實際存在的 extra_hard_documents。原 loss、step、seed、LR、document IDs、mask、負例曝光等日誌仍保留。

正例索引、cosine 正規化、temperature、遮罩分母及 CE 均不變。兩組繼續同一正配對、mode 無關 schedule、epoch、seed、optimizer steps、AdamW 與 LR。新增檢查不抽樣、不多做訓練更新、不讀評估 query。hard 改變分母與計算成本的原比較限制仍存在。

## 儲存、回載及失敗語意

final_model 仍為 ST modules.json＋模型／tokenizer／Pooling 設定，現有 A2 evaluator 的 SentenceTransformer(model_path) 可依其 API 載入，不需修改 A1。新版 export 包含新版模組參照，應在相同 ST 6.1.0 環境回載，**不保證能拿回舊 ST 2.x 載入**。

來源 hash、provenance、provisional、pending、completion manifest、exclusive output 及現有 gate 皆保留。output 不存在才新建，既有目錄不覆寫。對已建立的 run，錯誤或 KeyboardInterrupt 會寫 failure.json，不寫成功 training_summary；若還沒建立輸出就失敗，只回報錯誤，不補造成功檔。

新增可選 --verify-save-load：

1. 最終匯出後、寫 completion 之前，將匯出模型從本機 CPU 回載，不用第二份 GPU 模型。
2. 檢查 max_seq_length 已保存；兩個 Train 原題在原模型與回載模型均以 eval／推論模式編碼，正規化後以 rtol=atol=1e-4 比較。
3. 通過寫 a2_reload_verification.json，之後才建立完整性 manifest；summary 記 final_model_reload_verified=true。若不加旗標則 false，不假裝已做回載。
4. 失敗時不建立 completion，保留 pending／provenance／failure 供診斷，評估器不能將該半成品當成功模型。此檢查不使用 Validation 或 Test。

這段 inference 僅是訓練結束後的可選匯出檢查，不在 gradient forward 內。CPU 副本額外占用主記憶體；CPU/CUDA kernel 差異可能超過目前容差。回載檢查實作仍未執行，不代表已證實可載或權重有效更新，更不代表檢索準確率。

## GPU 電腦 Smoke Test 指令

在 GPU 電腦切到專案根目錄，將本次程式與文件更新過去；不要重裝套件。需已有原始 BGE-M3 快取，--offline 不會下載模型。兩個輸出路徑都必須不存在，曾失敗的舊路徑不要重用。

以下用 batch=3、max-length=64 降低 activation 用量，兩組完全相同；仍是 BGE-M3 全參數 FP32 AdamW，不保證特定 GPU 放得下。每 epoch 照樣用全部 126 對，預期 42 steps（42 個 FAQ÷3×3 輪）；這是推算，不是已執行值。batch=2 仍因診斷原題 masking 而被拒絕。

```text
python -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_1 --model BAAI/bge-m3 --device cuda --epochs 1 --batch-size 3 --max-length 64 --learning-rate 2e-5 --seed 42 --allow-unreviewed --offline --gradient-checkpointing --verify-save-load --output-dir experiments/models/a2_1_smoke_task8
python -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_2 --model BAAI/bge-m3 --device cuda --epochs 1 --batch-size 3 --max-length 64 --learning-rate 2e-5 --seed 42 --allow-unreviewed --offline --gradient-checkpointing --verify-save-load --output-dir experiments/models/a2_2_smoke_task8
```

這不是最佳設定或正式實驗；needs_review 明確 opt-in 所得模型繼續 provisional。正式比較應使用同一凍結本機原始 checkpoint 的檔案 hash，不能只靠可變 HF 名稱宣稱初始權重相同。若 host RAM 不足，可拿掉 --verify-save-load，但須另行回載確認，且不能稱該 run 已驗證回載。若 GPU 仍 OOM，不默默改其中一組的設定；FP32 AdamW 的固定參數／梯度／動量用量不因短序列消失。

確認兩個 run 的 loss／分支與模型梯度 finite、planned 與 actual steps 相符、pair schedule hash 相同、版本及 profile 正確；A2-2 實際有額外 hard 的步驟應出現 extra_hard_documents 梯度紀錄。再查看 final_model/a2_reload_verification.json、a2_checkpoint_complete.json 與成功 summary。模型檔案存在本身不是通過。

若既有 60 題 corpus 及查詢已備妥，可再在相同環境單獨確認 A2 評估器載入 final_model；只跑 Unseen Validation，不碰最終 Test：

```text
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model experiments/models/a2_1_smoke_task8/final_model --model-tag A2-1 --device cuda --batch-size 3 --max-length 64 --seed 42 --split unseen_validation --allow-unreviewed --offline
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model experiments/models/a2_2_smoke_task8/final_model --model-tag A2-2 --device cuda --batch-size 3 --max-length 64 --seed 42 --split unseen_validation --allow-unreviewed --offline
```

評估器會明確覆寫 max-length，因此這裡與 smoke 一律指定 64。不能拿這些短序列 smoke 分數當正式 512 設定比較結果。

## 仍待 GPU 電腦確認

- Python 語法／匯入、實際安裝套件與官方標籤一致性、未列出的相依套件、BGE-M3 本機快取完整性。
- preprocess→forward 的實際 features、載入的 pooling、CUDA backward、checkpointing、deterministic attention kernel 相容性與顯存。
- 梯度有傳遞和參數確實更新、全部 batch 覆蓋、Loss finite、save_pretrained 實際輸出及同版 A2 evaluator 回載。
- 長度設定保存與 CPU/CUDA embedding 容差、主記憶體。未實測不保證 1e-4 適用所有硬體，不遇錯就自動放寬。

未修改三份 JSONL、原始 CSV、原 A1／其他實驗或 GPU 環境；沒有任何本機執行成功或效果提升聲明。任務 8 到此停止。
