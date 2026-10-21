# A2 FAQ 資料準備與檢索評估

本目錄為獨立 A2 程式，不修改或呼叫既有 A1 的 loader/matcher。
狀態：程式碼與查詢候選已撰寫，尚未執行 Python、測試、parser、模型推論或訓練。本說明中的命令由使用者在另一台電腦執行。

## 檔案用途

| 檔案 | 用途 |
|---|---|
| common.py | 嚴格 UTF-8 CSV/JSONL、完整 ID 檢查、相對根目錄定位、SHA-256、拒絕覆寫輸出 |
| validate_a2_data.py | 訓練、manifest、60 題候選庫及 138 題評估檢查；分開 error 與 review |
| build_faq_candidates.py | 從兩份 CSV 建獨立的 60 題 question-only 候選庫，不使用舊 3 題 canonical |
| evaluate_faq_retrieval.py | 共用 BGEM3Embedder，完整 60 名 dense 檢索及分集合指標 |
| __init__.py | A2 模組入口識別，不載入模型 |
| train_faq_embedding.py | 任務 4：Train-only 檢查與 A2-1/A2-2 微調入口，未執行 |
| training_data.py、training_loss.py | 唯一意圖批次排程與可變難例的 masked MNRL |
| training_config.json、TRAINING.md | 共同訓練設定、操作與方法限制 |
| training_compat.py、TRAINING_COMPATIBILITY.md | 任務 8：ST 6.1.0 API 介接、分支梯度檢查、可選回載檢查與 Smoke Test 命令 |
| STATIC_REVIEW.md | 任務 5：資料流靜態審查、修復紀錄與另一台電腦的 Smoke Test |
| evaluate_hard_diagnostic.py、diagnostic_data.py | 任務 7：獨立 Train 意圖診斷入口／只讀驗證，不混入原 138 題結果 |
| HARD_DIAGNOSTIC_REVIEW.md | 30 題待審診斷、歧義排除、使用命令與限制 |
| ../datasets/faq_finetuning/faq_eval_queries.jsonl | 138 題待人工審核查詢，不是已凍結 Test |

## 環境與路徑

資料驗證及候選庫建置只使用 Python 標準函式庫，不會載入 torch 或模型。
評估沿用 experiments/requirements.txt 既有的 torch、numpy、sentence-transformers（其下使用 transformers、huggingface-hub）；不增加新第三方套件。不需 FAISS：60 題以 NumPy 正規化內積作精確排序，與 cosine 相同，並可確定 FAQ ID 同分順序。
重用 experiments/shared/embeddings.py 的 BGEM3Embedder；不重用 A1 會再加 P/S 前綴的 loader，也不重用會跳過壞 JSON 行的共用 IO 或尚未實作完整指標的 compute_metrics。

以下命令須先在另一台電腦切換到該台電腦的 SCU_Kebbi 專案根目錄。使用模組形式 python -m experiments.a2_faq_finetuning.<程式>；不要把檔案當獨立腳本直接執行。
--project-root . 可改為該台電腦的專案路徑。預設由程式檔位置推算根目錄，不寫死磁碟或使用者名稱。其他相對輸入／輸出參數都以 --project-root 為基準；絕對路徑亦可使用。
指定 project-root 不會替 Python 更改 import 搜尋路徑，因此模組命令仍須在正確專案根目錄執行。

實際 CSV 編碼採嚴格 UTF-8（容許 BOM），欄位為「編號」「問題內容」「ai答覆」；其他欄位可隨來源是「關切維度」或「問題分類」。問題不 strip、不改寫。CSV 本身的 P01/S01 直接作 ID，錯誤前綴、缺欄位、錯誤編碼、欄數不符或缺題均明確報錯，不猜其他編碼。

## 1. 先驗證訓練資料（無模型）

```text
python -m experiments.a2_faq_finetuning.validate_a2_data --project-root . --scope training --report experiments/results/a2_data_validation/train_review.json
```

報告包含合法 JSON／重複 JSON key／七欄型態、42 個唯一 Train ID、CSV 原題、split 交集、Positive 原題複製／重複、負樣本來源與跨欄位碰撞、診斷查詢重複、可疑文字及空 Hard Negative ID。
錯誤行不會被靜默略過後宣稱通過。error 會使結束碼為 1；輸出路徑等 IO 錯誤為 2。結束碼 0 只表示可確定的結構檢查無錯誤，不表示語意審核、難例排名、資料凍結或模型成功。
程式絕不改寫任何輸入。--report 僅寫新的檢查結果；若省略則將完整 JSON 報告印到 stdout。每次使用新的報告名稱，既有檔案不覆寫。

診斷原文出現在 positive/negative/hard_negative 為 error；出現在必須保留的 source question 則為 review，列出 QF ID。後者五筆已知存在，因此不能再把舊 15 題當獨立 Test。
review 包含近似字串、已知 FAQ 語意邊界、Japanese kana/已知污染 token 等人工複查；繁體中文及 Python、AI、GPU、AWS、GCP、Azure 等英文不會被一律判錯。掃描器只是啟發式，漏報和誤報仍可能存在。

## 2. 建立獨立完整候選庫（無模型）

```text
python -m experiments.a2_faq_finetuning.build_faq_candidates --project-root .
```

未來執行時預設新增：

- experiments/datasets/faq_finetuning/faq_retrieval_candidates.jsonl
- experiments/datasets/faq_finetuning/faq_retrieval_candidates.jsonl.metadata.json

每行為 faq_id、question、source_group（parents/highschool），P01–P30、S01–S30 各一筆，沒有 Positive 或答案嵌入。
metadata 記錄 CSV hash、候選庫 hash、題數與建置方式。
舊 experiments/datasets/faq/faq_canonical_v1.jsonl 保留不變。本次未執行建置，因此上述兩個輸出尚未產生。
若輸出已存在，程式會停止；需要重建時使用 --output 指定新的候選庫路徑，並在後續命令以 --candidates 明確選定。禁止覆寫來源或用舊 3 題庫替代。

## 3. 驗證完整資料與評估題庫（無模型）

先完成第 2 步，再執行：

```text
python -m experiments.a2_faq_finetuning.validate_a2_data --project-root . --scope all --report experiments/results/a2_data_validation/full_review.json
```

預設評估檔 experiments/datasets/faq_finetuning/faq_eval_queries.jsonl：

| scenario | split | ID 命名 | 題數 |
|---|---|---|---:|
| seen | seen_test | ST_P01_01 等 | 84（42 意圖各 2） |
| unseen | unseen_validation | UV_P22_01 等 | 27（9 意圖各 3） |
| unseen | unseen_test | UT_P26_01 等 | 27（9 意圖各 3） |

驗證器檢查 query_id、gold、來源集合、逐意圖題數、原題／訓練樣本／診斷文字 exact 或正規化複製、跨集合重複。對較高字面相似度另列人工複查，預設 --near-threshold 0.82；該值只決定警示，不證明語意洩漏或無洩漏。短句、單純改字與複合意圖的遺失仍須人工檢查。
review 報告的 known_semantic_boundary 會列實際原題、來源 split、受影響 query_id；人工審核需閱讀完整答案，不能只憑 FAQ ID 不同或程式沒有 error 就批准。
所有 138 題目前仍 needs_review，gold_faq_id 是建議單一標籤，並非已確認唯一答案。審核疑點亦記錄於 A2_DATA_REVIEW.md 第 10 節。
Seen Test 及 Unseen Test 不可拿來選超參數、挑難例或改訓練資料。

## 4. 原始 BGE-M3 的驗證集診斷執行

以下範例只供使用者之後在另一台電腦操作，Codex 未執行：

```text
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model BAAI/bge-m3 --model-tag A1 --device cpu --batch-size 16 --max-length 512 --seed 42 --split unseen_validation --allow-unreviewed
```

預設 split 就是 unseen_validation；只有顯式 --split all 才會評估三組。這避免挑訓練設定時順手看最終 Test。
若查詢還未核准，預設在載入模型前停止。--allow-unreviewed 僅供明確的草稿診斷，輸出 provisional=true，不能當正式結論。即使資料人工批准，仍須先確認語意風險及環境正確。
已有本機模型／快取且不希望連網時使用 --offline；也可 --model models/bge-m3 指定本機模型。模型 ID 模式在另一台電腦實際執行時可能需要既有快取或網路；本次沒有下載。

## 5. 資料與設定凍結後的三組公平比較

完成獨立人工審核並將評估資料由使用者核准後，再使用相同資料 hash、query、長度、seed、batch、精度與硬體執行：

```text
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model BAAI/bge-m3 --model-tag A1 --device cuda --batch-size 32 --max-length 512 --seed 42 --split all
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model models/a2_1/final_model --model-tag A2-1 --device cuda --batch-size 32 --max-length 512 --seed 42 --split all --offline
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model models/a2_2/final_model --model-tag A2-2 --device cuda --batch-size 32 --max-length 512 --seed 42 --split all --offline
```

CUDA 非必要；可將三組一律改成 cpu，或使用 --device auto 並核對實際 device。不存在的 CUDA 會明確報錯，不悄悄改成 CPU。
A2-1/A2-2 模型路徑只是預期的 run 下 final_model；任務 4 已新增訓練入口，但未執行或建立任何模型。訓練與評估整合見 [TRAINING.md](TRAINING.md)。
評估器會讀取模型內 a2_training_provenance.json；未審核資料訓練的模型不得在評估時去掉 provisional 標記。

每個模型重新編碼同一 60 題與所選 query，使用既有 BGE-M3 dense 包裝器、float32、L2 正規化、精確內積及 FAQ ID 同分排序。不重用 A1 向量索引搭配微調模型。
固定 seed、torch deterministic 演算法及關閉 TF32；跨硬體／套件仍可能有差異，不承諾 bitwise 一致。若環境不支援某 deterministic 運算，程式應報錯，不會靜默放寬設定。

## 輸出與指標

每次產生新的 experiments/results/a2_faq_retrieval/<model-tag>/<UTC時間_隨機識別>/。可 --output-dir 指定不存在的新資料夾，禁止覆寫既有 run。
沒有任何預先填入的示意分數：

- per_query.jsonl：query_id、query、gold、scenario/split、Top-5 FAQ ID／分數、hit_at_1、hit_at_5、gold_rank、gold_score、reciprocal_rank，以及完整 60 名 ID／分數。
- metrics.json：三組各自樣本數、意圖數、Top-1 Accuracy、Recall@5、完整排名 MRR、平均 Top-1／gold cosine，以及逐 FAQ／macro 平均。未選取的集合標示 evaluated=false、metrics=null，不假裝分數為零。
- run_metadata.json：模型 ID／本機權重檔 hash、可取得的 HF revision、資料與程式 hash、套件版本、實際 CPU/CUDA、seed、batch、長度、float32／正規化、排序規則、查詢審核狀態及 provisional 標記。

單一 gold 下，Top-1 是 gold_rank==1 的比例，Recall@5 是 gold_rank<=5 的比例，MRR 是完整 60 名排名倒數的平均（不是 MRR@5）。模型成功載入不等於檢索命中。沒有跨情境混算的總準確率。
遠端模型 revision 若無法從包裝器取得，metadata 會標示 remote_revision_unresolved；正式比較宜使用已凍結的本機快照，不能只靠可變模型名稱。載入本機模型時會讀取權重檔 hash，會增加 IO 時間。
評估前驗證所有輸入，結束前再次核對 hash；若輸入或本機權重於執行中變動，停止保存結果。

## 本次仍待另一台電腦驗證

所有語法、模組匯入、CSV/JSON parser、資料檢查結果、套件相容、模型載入、CPU/CUDA、指標計算及檔案輸出均尚未實際執行確認。未新增／執行程式測試。
建議順序：training 驗證 → corpus 建置 → all 驗證與人工語意覆核 → 驗證集診斷 → 凍結後正式比較。
不得把第一次驗證集跑完視為已完成 A2 實驗。任務 4 的微調程式及研究限制另見 [TRAINING.md](TRAINING.md)，同樣尚未實際執行驗證。
