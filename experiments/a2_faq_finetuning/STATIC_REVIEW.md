# A2 任務 5：靜態審查紀錄

後續版本註記（任務 8）：本文保留任務 5 當時的審查歷史。訓練現已新增 ST 6.1.0 的 preprocess／公開 Transformer／save_pretrained 介接及分支梯度檢查，見 [TRAINING_COMPATIBILITY.md](TRAINING_COMPATIBILITY.md)。原 Masked InfoNCE、排程、資料 gate 與 manifest 保留；不以此歷史紀錄代表新版已執行驗證。

本次僅閱讀程式、CSV、設定及檔案 hash，直接修正 A2 程式。未執行 Python、parser、測試、模型、訓練或安裝。下列程式行為是靜態追蹤結論，數值推算不是執行結果。訓練及評估 JSONL 均未修改。

## 實際缺陷與修正

| 位置 | 修改前 | 修改後 |
|---|---|---|
| validate_a2_data / common | 只在驗證結尾計算 hash，可能把讀取後的新內容當成已驗證版本 | 先 snapshot 全部輸入，讀取後比對；訓練與評估重新讀取資料後再次比對 |
| train_faq_embedding | config 讀取後到記錄 hash 之間可被改寫，記錄內容可能與實際設定不符 | 讀取設定時保留 digest 與內容重讀檢查；後續使用同一 digest |
| checkpoint export / load | 先存權重再存 provenance；中斷時，評估器可能將無 provenance 的 A2 目錄視為原始模型 | 建目錄後先寫 pending 標記；成功匯出與 provenance 後，最後寫完整檔案 SHA-256 manifest；回載前核對 manifest 與 provenance |
| evaluate_faq_retrieval | model-tag 只是輸出標籤，可把 A2-2 標成 A2-1，甚至以原模型標成 A2 | A2 標籤必須有完整 A2 export；與 provenance.mode 一致，A2 模型也不能標成 A1 |
| train_faq_embedding | 初始本機模型只在最後檢查是否改變，中間 checkpoint 可能先存出 | 模型載入後及每次 checkpoint 匯出前也檢查初始 identity |
| training_loss | float target 被直接轉 long，可能靜默截斷；負索引先通過 Python 索引才由 CE 報錯；直接使用 loss 時 scale 無檢查 | 拒絕非整數／越界 target、非 boolean mask、不合法 scale、空矩陣或維度不符，提供明確錯誤 |

補上的 hash 檢查是偵測一般工作期間的檔案變動，不是檔案鎖或防惡意竄改機制。完整性檢查不能代替 SentenceTransformer 實際回載。舊版 A2 export 若沒有 completion manifest，現在會拒絕載入，需另行確認完整性；本次不補造任何舊模型標記。

## Loss 與資料流

實際路徑：main → collect_validation(include_eval=False) → 人工審核 gate → prepare_training → make_schedule → batch_documents → tokenize/model.forward → loss.backward → AdamW.step → save_model。

- query 是 Positive 改寫，正文件是同 FAQ 原始 question。文件前 B 欄依樣本順序擺放原題，targets=0..B-1，對應正確。
- L_i = -s(q_i,d_i)/T + logsumexp(s(q_i,d_j)/T, j 屬於該列允許集合)。兩側先 L2 正規化，scale=20 等價 T=0.05；正例包含於分母，沒有錯誤排除對角線。
- bool mask 將不允許 logits 設為 -inf；每列必須保留正例與至少一個負例，避免全遮罩與零訓練訊號。最後 cross_entropy 對 query 平均。
- 正規化、矩陣乘法與 CE 都在 forward 反向圖中；沒有以 encode/no_grad/detach 訓練。被遮罩的 query-document logit 梯度為零，但同一文件仍可因其他有效配對得到梯度；共享模型參數也不會因某一格被遮罩就停止更新。
- A2-1 只用批內其他意圖原題。A2-2 額外開放該 query 自己的有效 hard_negative，來源文件依 FAQ ID 去重。若已是有效批內負例，不新增欄、不加權。不同 owner 可以共用同一文件欄，但各自的 mask 獨立。
- 沒有 Hard Negative 的 FAQ 仍保留基礎 loss。JSONL 的一般 negative 陣列在兩組都只驗證、不投入 loss；不得誤述為使用了每題的兩個顯式一般負例。
- 每輪每意圖只出現一次，每 epoch 三輪，三句改寫各用一次。尾批不 drop、不複製；必要時整輪重排，失敗明確停止。排程不依 mode 或 hard 數量改變。

## 假負例排除表的實際意義

prepare_training 將設定轉成雙向 frozenset。ordinary_allowed 同時用於排程合法性、批內 mask 與 hard 篩選；設定不是未使用的文件欄位。非法或非 Train ID 會停止。

| 配對 | 實際原題重點 | 判斷 |
|---|---|---|
| P18 / P19 | 四年學雜費與獎助學金 / 優秀入學獎學金標準及免學費 | 廣義與特定獎學金交集，合理保守排除；不是兩題完全等義 |
| P21 / S13 | AWS、GCP、Azure 額度補助 / 深度學習或 AI 伺服器資源 | 詢問資源種類不同，但答案可能交集；維持風險排除 |
| S12 / S13 | 筆電規格及高階 GPU 需求 / 系上 AI 伺服器資源 | 不同詢問，硬體替代關係可能造成假負例；保守排除 |
| P13 / S04 | 跟不上進度及不及格的預警輔導 / 上機考緊張寫不出的補救 | 廣義學習輔導與特定考試補救交集 |
| P13 / P14 | 學習預警輔導 / 專屬導師的生活與課業諮詢 | 意圖不完全相同，服務可能交集；雙向難例不投入 |
| P16 / P17 | 業界面試是否偏好傳統資工 / 相比國立大學的優勢競爭力 | 不同意圖，僅競爭力主題相關；目前屬較保守排除，須人工決定是否過度遮罩 |

本次未擅改排除表。不能由來源 ID 不同推論語意必不同，也不能由此表宣稱已完成完整語意審核。依目前資料記載與程式規則，42 個難例候選預期排除 P13↔P14 兩筆，保留 40 筆；這是待執行核對的推算。難例尚未經原始 BGE-M3 排名驗證。

## 公平性與評估

在資料／config／CLI 參數相同的條件下，兩組共用正配對、mode 無關排程、seed、epochs、AdamW、LR、scale、最大長度及每批一次 optimizer step。預設 batch=8，42 題每輪切成 8/8/8/8/8/2，三輪為 18 steps/epoch，三 epoch 預期 54 steps、378 次正配對曝光；不是已執行數字。

每步重設相同 torch RNG，先 query forward、base forward，再額外 hard forward，以保持共同部分的隨機流。參數更新後兩組表示本來就會分歧；額外分母、梯度、前向計算量、activation 記憶體及額外文件 padding 無法保持完全相同。沒有等數量一般負例對照，不能把差異全歸因於「困難度」。尾批較小的每步平均權重也存在，但兩組排程一致。

程式沒有跨 run 自動強制參數相等。須比較 run_config.arguments、config_contents、code_sha256、input_sha256、initial_local_identity 的相對檔案 hash、positive_pair_schedule_sha256、optimizer 設定，以及實際 summary/log 步數。HF 可變模型名稱本身不能保證相同初始權重；正式比較應使用同一凍結本機原始 checkpoint。跨平台比較檔案內容而非含絕對路徑的整個 identity 字典。

評估三組走同一 BGEM3Embedder → FP32/eval → encode → L2 normalize → 全 60 題內積排序。FAQ ID 作同分次序，完整排名算 MRR、保存 Top-5；不是 MRR@5。不同模型必須指定相同 corpus/query hash、split、max-length、batch 等參數。沒有針對 A2 使用另一種 pooling 或分數公式；ST modules.json 回載仍須實測。

資料未 approved 時，正式訓練在模型載入前阻擋。--allow-unreviewed 只略過審核狀態、不略過結構錯誤；provisional 寫進每個 checkpoint，評估會繼承且要求 opt-in。查詢是否可用仍有獨立 gate。保存的是模型快照，不含 optimizer/RNG resume 狀態；不能宣稱可等價續訓。

## 另一台電腦的最小 Smoke Test（本次未執行）

1. 先驗證無模型的資料流程：`python -m experiments.a2_faq_finetuning.validate_a2_data --project-root . --scope training`。確認 error 與 review 分開，結構通過不代表人工批准。以相同設定產生兩模式排程，確認每 epoch 126 個配對、每意圖三句各一次、無重複意圖批次、schedule hash 相同；batch=8 每 epoch 18 步。檢查空 hard、hard 已在 base、共享 hard 來源及 P13/P14 排除案例。
2. 用小型人工 tensor（不載 BGE）檢查 loss：等分數且一正一負時應為 log(2)；手算允許集合的 CE 與輸出一致；改動完全被遮罩文件不影響 loss，其獨立文件 embedding 梯度為零；query/正例/有效負例得到有限梯度。負 target、浮點 target、全遮罩、正例被遮罩、scale<=0 應明確報錯。
3. needs_review 狀態不加 opt-in 時，訓練須在模型載入前停止。另用明確 provisional smoke run，不改 JSONL，從同一凍結本機 checkpoint、相同參數分別執行一 epoch：

```text
python -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_1 --model models/bge-m3-original --device cuda --epochs 1 --batch-size 8 --max-length 64 --seed 42 --allow-unreviewed --offline --output-dir experiments/models/a2_1_smoke_task5
python -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_2 --model models/bge-m3-original --device cuda --epochs 1 --batch-size 8 --max-length 64 --seed 42 --allow-unreviewed --offline --output-dir experiments/models/a2_2_smoke_task5
```

上述模型路徑是示例，必須先有原始本機 checkpoint；輸出目錄須不存在。縮短至 64 只供流程驗證，仍是全參數訓練，不保證顯存可容納。若需減 batch、長度或啟用 gradient checkpointing，兩組同步調整。CPU 用 --device cpu，仍可能很慢。

4. 確認兩組 schedule hash、初始檔案 hash 與步數一致，loss/gradient 有限，權重有更新，40 候選及 hard_new/額外文件等實際統計符合預期。檢查 export 的 completion manifest。於另一個程序回載 final_model，比較同一組文字在儲存前 eval 模式與回載後的 embedding，應在合理浮點容差內一致；不要拿 train 模式 dropout 輸出比較。
5. 先依 README 建立 60 題 corpus 並驗證全部資料。原模型及兩個 smoke 模型只評估 unseen_validation，A2 使用對應 --model-tag，三組統一 --batch-size 8 --max-length 64 --seed 42 --allow-unreviewed --offline 與相同查詢。核對 27 題、60 候選、Top-5 及全排名 MRR，不能只看程式完成。
6. 在另一台的臨時模型副本測試：缺 completion、缺 provenance、改動權重、錯 model-tag、把 run 上層當模型路徑均須拒絕；provisional 模型不得輸出正式結果。不得用 Seen Test 或 Unseen Test 做上述流程／設定選擇。

仍待實測：Python 語法／匯入、舊 ST 與 Hub/Transformers 的實際相依相容性、CUDA deterministic 運算、梯度與 checkpointing、顯存、save/load round trip、數值及指標正確性。靜態審查未提供任何模型效果證據。
