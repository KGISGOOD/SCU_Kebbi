# A2 BGE-M3 微調操作與方法契約（任務 8 相容性修訂）

任務 8：新增明確的 ST 6.1.0 API 路徑，未修改 GPU 電腦套件。可直接複製的 Smoke Test 與來源查核見 [TRAINING_COMPATIBILITY.md](TRAINING_COMPATIBILITY.md)。本機只編輯／靜態審查，沒有執行成功紀錄。

任務 5 補充：見 [STATIC_REVIEW.md](STATIC_REVIEW.md)。A2 模型匯出現在包含 pending 標記與最後寫入的 a2_checkpoint_complete.json（檔案 hash manifest）；缺完整性標記、內容不符或 model-tag 不符都會阻止評估。pending 檔是匯出嘗試紀錄，只有 completion manifest 完整有效時才能使用。舊 A2 匯出若缺 manifest 不再默認接受，本次不替舊模型補造完成紀錄。原始 BGE-M3 的 A1 載入及 cosine 排名流程保留。

狀態：只完成程式撰寫及靜態閱讀，沒有執行 Python、parser、測試、模型載入、下載或訓練。所有命令由使用者在另一台電腦執行。
原始 CSV、正式訓練 JSONL、評估查詢、split、A1 與舊 worktree 均不修改。資料仍 needs_review 時，預設不會開始模型載入或正式訓練。

## 1. 版本與採用的 API

experiments/requirements.txt 保留原內容，舊設定範圍為：

- sentence-transformers >=2.2.2,<2.5.0。
- torch >=2.0.0,<2.4.0。
- transformers >=4.43.0,<4.48.0。
- NumPy 沿用現有專案；packaging 為既有 Transformers 相關相依套件，用於版本比較。

任務 8 改由 training_compat.py 選擇 API profile：保留舊範圍的 tokenize/save 路徑；另外支援靜態查核過的 ST==6.1.0 路徑，檢查 Python>=3.10、torch>=2.2,<3、transformers>=5,<6、numpy>=1.24。這不是整個版本組合已測試的保證，不自動安裝、升級或改用 Trainer。

ST 6.1.0 直接載入 SentenceTransformer（torch backend、offline 時 local_files_only=True），使用 preprocess(inputs=texts, prompt=None) → model(features)["sentence_embedding"]，並以 save_pretrained(..., safe_serialization=True) 匯出。Pooling 沿用 checkpoint，不重建。tokenize 在該版是 deprecated alias，並非已移除；原 save/forward 也仍存在，不能把本次報錯虛稱為它們已實際執行失敗。

訓練不經 BGEM3Embedder 的推論入口；評估仍沿用已可執行的 BGEM3Embedder，其內部同樣以 SentenceTransformer 載入模型。訓練沒有 encode、no_grad 或 NumPy 轉換；只在可選的訓練後回載檢查使用 encode 作推論。每路 embedding 檢查 requires_grad/grad_fn，backward 後檢查 retained gradient，逐步寫入日誌。Loss 數學及資料排程不變。

方法依據為 [SentenceTransformers 2.2.2 官方 MNRL 原始碼](https://github.com/UKPLab/sentence-transformers/blob/v2.2.2/sentence_transformers/losses/MultipleNegativesRankingLoss.py)：
標準 MNRL 將 query-positive 配對及其他文件的 cosine 分數經 scale-softmax／cross entropy 計算；硬負例也能加入，但會供整批 query 使用。
本資料有無難例、數量 0–3、診斷文字及假負樣本遮罩需求，不能假裝把不同長度二元／三元列直接交給內建 collator 就可正確工作。
另參考 [2.2.2 SentenceTransformer tokenize／forward／save 介面](https://github.com/UKPLab/sentence-transformers/blob/v2.2.2/sentence_transformers/SentenceTransformer.py)。本次使用既有底層 API 及自訂遮罩 loss，不宣稱曾執行驗證這些介面。

## 2. A2-1 與 A2-2 的實際資料與 loss

兩組皆使用 MaskedMultipleNegativesRankingLoss，這是 MNRL-compatible 的 cosine InfoNCE，並非直接呼叫套件的原生 loss。
對 query q_i、正確原題 d_i、該列允許的文件集合 A_i：

L_i = -log[ exp(scale*cos(q_i,d_i)) / Σ(d ∈ A_i) exp(scale*cos(q_i,d)) ]。

先 L2 normalize embedding，再乘 scale（預設 20，溫度等價 0.05），遮罩無效列為 -inf，最後對全部 query 取 cross entropy 平均。
同一 objective 用於兩組，差異只有 A2-2 多開啟其自身經允許的難負候選；沒有新的 loss 權重、不同 optimizer 或額外訓練輪次。

每題的三個 positive 改寫各作一次 query，原始 question 作 positive document，形成每 epoch 126 組：
(query=同意圖改寫, positive=同 FAQ 原題, faq_id=來源 ID)。
不能把原題與改寫方向只在其中一組反轉。

| 模式 | 基礎負樣本 | 額外難負樣本 |
|---|---|---|
| a2_1 | 批次中其他不同 FAQ 的原題，經同意圖／診斷／語意排除遮罩 | 無 |
| a2_2 | 與 a2_1 完全相同的 in-batch 基礎 | 該 query 所屬 FAQ 的所有可用 hard_negative 原題（0–3） |

本次明確選擇 MNRL 的 in-batch 作一般對比基礎，兩組均**不把 JSONL 的 negative 陣列當作額外顯式負樣本**。negative 欄位仍經資料驗證且保留不改。
這取代早期規劃中「每題再加入兩個顯式一般 Negative」的建議，避免把未實作的資料使用方式寫成已完成。兩組仍使用同樣的一般負樣本基礎，實際政策會寫入 run_config 與模型 provenance。

### 變長 Hard Negative 的處理

1. 每批 positive documents 先按批次順序列出，一個 FAQ ID 一個文件。
2. A2-2 把該批用到的有效 hard_negative 來源 ID 加入文件池；同一文件只編碼一次，不因重複來源增加權重。
3. 每個 query 有自己的 boolean mask，額外文件只對指定它作難負例的 query 開放，不會自動變成其他 query 的負例。
4. 若難例已是該批可用的 in-batch negative，重用原欄位，不重複放入 softmax；該列此時不會增加新的分母項。
5. 沒有難例的 FAQ 保留原配對與基礎 loss，不填空字串、不複製 positive、不丟掉該題，也不拿保留意圖補數。

forward 分開編碼 query、原始 positive 文件及額外難例，之後串接文件 embedding。不建立不規則的 InputExample 二元／三元混合 batch。對新增文件作額外 forward，計算量與記憶體因此增加。

## 3. False negative 與資料審核

訓練前呼叫 collect_validation(include_eval=False)；任何可確定的格式、ID、來源、診斷樣本文字或跨切分錯誤都會停止。這不是讀取全評估題庫的捷徑，入口沒有 eval query 參數。
只從驗證後的 42 個 Train ID 選取參與梯度的文字；CSV 的保留原題僅在來源一致性驗證時閱讀，不進模型 forward。
一般資料驗證通過不等於語意批准。所有訓練列須 review_status=approved，包含該列正樣本與難負候選的人工審核；程式不會把 needs_review 改成 approved。

每批 FAQ ID 不重複。三句改寫分三輪處理，避免同 FAQ 的正配對互當 in-batch negative。
P01、P02、S01、S02、S05 的原題因也是診斷查詢，只能作自身 positive，對其他 query 的負文件位置遮罩。診斷題不作訓練 query。
training_config.json 預設排除以下雙向配對，不論是 in-batch 或 hard：P18/P19、P21/S13、S12/S13、P13/S04、P13/P14、P16/P17。
這是保守的來源 ID 排除表，不是模型判定的同義結果。其餘不同 ID 是否實際同意圖，仍依賴完整人工審核；不宣稱程式能自動保證語意沒有假負例。

P13/P14 在資料報告仍有導師與預警答案交集，所以本次不投入該雙向難例。正式 JSONL 不改；被遮罩的候選原文與原因會記入 excluded_hard_candidates。
依目前檔案閱讀，42 個人工難負候選中這兩筆被排除，預期留下 40 筆／28 題、14 題有效難例為空；這是靜態資料推算，實際數量由訓練電腦執行後記錄。
所有難例仍未經原始 BGE-M3 排名驗證，metadata 固定標示 hard_negatives_model_rank_verified=false，不能聲稱是已驗證的模型困難樣本。

--allow-unreviewed 是明確的研究／debug opt-in；只略過審核狀態 gate，不略過結構或來源錯誤，不停用假負遮罩。即使使用者之後批准查詢，這類模型仍帶 provisional=true，評估器會要求同樣 opt-in 並保留 provisional 標記。
使用這個開關的結果不是正式實驗，不能因為檔案成功儲存就改稱正式模型。

## 4. 批次、更新次數與公平限制

訓練設定預設 batch=8、epochs=3、LR=2e-5、AdamW、FP32、seed=42、max length=512、scale=20、weight decay=0.01、gradient clip=1。
AdamW betas=(0.9,0.999)、eps=1e-8，對全部可訓練參數使用同一 weight decay；學習率固定，不另設 schedule 或 early stopping。
這些是待驗證的共同起點，不是經 Validation 挑出的最佳參數。

採與 mode 無關的種子排程，每 epoch 三輪，每輪各 42 個 FAQ：
- 每題的三句改寫各用一次；每 epoch 共 126 個配對。
- 各輪隨機打亂，但批內 ID 唯一，不能形成完全沒有有效負樣本的 query。
- 必要時以相同 RNG 重排整輪以找到有效組合；找不到明確報錯，不靜默跳過。
- 最後只剩 1 筆時，從前一批挪 1 筆形成兩個有效尾批；不 drop_last、不複製資料。
- 每個 batch 恰一個 optimizer step，無 gradient accumulation。actual batch size、optimizer steps 及每次的 pair IDs 都記錄。
- 預設 42 題分為 8/8/8/8/8/2，三輪為每 epoch 18 步、3 epoch 預期 54 步；這是排程算式，未執行確認。

positive_pair_schedule.jsonl 及其 SHA-256 可比較兩組配對／順序／更新數是否相同。相同 config、seed、epochs、batch 和資料應產生同一排程；不得以資料或排除表不同的兩個 run 冒充只有 hard 的差異。
每步重設相同 torch seed，並先分別做 query/base 文件 forward，再處理額外 hard，以盡量維持共同 dropout 流；梯度、額外 forward 及硬體運算仍會因模式而不同。
使用 deterministic algorithms 並關閉 TF32；不承諾跨版本、硬體 bitwise 一致，也不會在 deterministic 失敗時偷偷改設定。

每步記錄 base_negative_exposures、hard_visible_exposures、hard_new_exposures、effective_negative_exposures、extra_document_count：
- hard_visible 是此列被允許的難例，其中可能已在基礎分母。
- hard_new 是實際新增的 softmax 負例數。
- effective 是每列分母中除了 positive 的總數。
- extra_document_count 是這一步額外編碼的去重文件數，與 query-negative 配對曝光數不同。

新增難例改變分母、梯度、單步算力及可能的 padding；不能將差異全歸因於語意難度。若要隔離數量效應，須另設等數量一般負例對照，本次不新增第四組。
不能只為 A2-2 縮 batch 或長度。若記憶體不足，兩組需使用相同新設定重做；減少 batch 也會減少 in-batch 負例，不等價於只調吞吐量。
沒有自動從 A2-1 接續 A2-2；若初始路徑帶本程式產生的 A2 provenance，入口會拒絕，兩組須從同一原始 checkpoint 開始。
目前五個診斷原題不可當別人的負例，所以 batch=2 無法讓包含它們的批次中每列都有有效負例；程式會在載入模型前明確停止，需使用至少 3。不是靜默丟掉這五題，亦不以 A2-2 的 hard 偷補 A2-1 缺少的訊號。

## 5. 另一台電腦的操作

先切換該台電腦的專案根目錄。路徑皆相對 --project-root，沒有寫死磁碟或使用者名稱。下列命令未在 Codex 執行。

先做既有訓練資料檢查：

```text
python -m experiments.a2_faq_finetuning.validate_a2_data --project-root . --scope training
```

人工批准資料後，使用相同凍結原始本機 checkpoint（以下 models/bge-m3-original 為預期路徑，不是本次已存在的模型），在另一台電腦分別執行：

```text
python -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_1 --model models/bge-m3-original --device cuda --batch-size 8 --epochs 3 --learning-rate 2e-5 --seed 42 --max-length 512 --output-dir experiments/models/a2_1_run --offline
python -m experiments.a2_faq_finetuning.train_faq_embedding --project-root . --mode a2_2 --model models/bge-m3-original --device cuda --batch-size 8 --epochs 3 --learning-rate 2e-5 --seed 42 --max-length 512 --output-dir experiments/models/a2_2_run --offline
```

可 --model BAAI/bge-m3 使用已快取或可連線的模型 ID，但可變模型名稱不保證跨 run 同一 checkpoint。正式比較優先用同一凍結本機模型 hash；HF revision 能取得時也記錄，無法解析則不能靠名稱宣稱一致。
--device cpu 或 auto 亦可；不強制 CUDA。CPU 全參數訓練仍可能耗費很多記憶體與時間，沒有實測。
--config 可指定另一份相同 schema 的設定，CLI 參數覆寫 config；實際兩者與 hash 都會寫入 run_config。

研究 debug 若確需使用未審資料，須在上述命令顯式加 --allow-unreviewed。文件不把這個開關放進正式範例，入口預設會列出未批准的 FAQ ID 並停止。
程式不作自動升降套件或模型下載預備工作；--offline 要求本機 checkpoint／快取可用。

不提供續訓／恢復 optimizer 的命令。本階段 checkpoint 是可供評估的模型權重快照，不是可位元等價恢復的完整訓練狀態。中斷後保留 failure.json 與已完成 checkpoint，重跑須新輸出目錄，不能冒充接續同一 run。

## 6. 儲存格式與評估整合

預設輸出 experiments/models/a2_faq_finetuning/<mode>/<UTC時間_識別>/；也可使用不存在的 --output-dir。既有目錄即使空的也不覆蓋。

輸出：
- run_config.json：完整實際參數、套件／程式版本與 hash、資料來源 hash、初始 checkpoint identity、資料統計、已排除難例、可訓練參數量及預期步數。
- data_validation.json：此次 Train-only 驗證與人工 review 提示。
- positive_pair_schedule.jsonl：所有 epoch 的配對與批次，兩組公平比較可直接比對 hash。
- training_log.jsonl：逐 optimizer step 的 loss、實際 batch、梯度 norm、負樣本曝光及完整 mask，step 後 flush。
- checkpoints/epoch_001 等：依 --save-every-epochs 保存中間 epoch；最後一輪改存 final_model，避免再存相同末輪兩次。
- final_model/：SentenceTransformer.save 匯出的 modules.json、模型／tokenizer／pooling 等檔案，以及 a2_training_provenance.json。
- training_summary.json：只有全部完成才產生，記錄實際步數、樣本與負例曝光、平均訓練 loss、耗時、可取得的 CUDA peak allocation。不是檢索準確率。
- failure.json：若輸出已建立後發生錯誤，記錄已完成步數及失敗原因；不當作已完成結果。

模型匯出 API 與現有評估器的 SentenceTransformer(model_path) 相同，可讀取 final_model 或中間 epoch 目錄。靜態已對齊格式，但尚未實際做 save/load round trip。
任務 8 可加 --verify-save-load：final_model 匯出後先在 CPU 回載，以兩題 Train 原題比對正規化 embedding，檢查長度設定保存；通過才寫 completion manifest／成功 summary。失敗保留 pending／failure，不寫完成標記。這是未執行的檢查實作，不是已取得的驗證結果；CPU 回載額外需要主記憶體，且 CPU/CUDA 容差仍待實測。
不要將 run 的上層目錄當模型路徑；應指定其中 final_model。

可用 Unseen Validation 比較已保存 epoch 以選設定，但本程式不在訓練中讀 query、不自動挑 checkpoint，不作 Test early stopping。固定 epochs 的 final_model 是最後一輪，不是「最佳模型」。

```text
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model experiments/models/a2_1_run/final_model --model-tag A2-1 --device cuda --split unseen_validation --offline
python -m experiments.a2_faq_finetuning.evaluate_faq_retrieval --project-root . --model experiments/models/a2_2_run/final_model --model-tag A2-2 --device cuda --split unseen_validation --offline
```

需先在另一台電腦建置同一 60 題 corpus、驗證評估題庫並完成其人工審核。若模型或查詢仍 provisional，評估也須顯式 --allow-unreviewed，輸出仍 provisional。
Seen Test 與 Unseen Test 不用於設定／checkpoint 選擇；全部凍結後才用 --split all。三組使用同一 60 題／查詢 hash 和同樣 dense cosine 設定。
A1 的原始模型仍由評估器原路徑載入，不訓練也不修改 A1。A2 評估器本次僅增加模型 provenance 檢查與 provisional 傳遞，不更動 baseline 排名或指標。

## 7. 記憶體與仍未驗證事項

不能可靠聲稱需要或能塞進多少 GB GPU 記憶體；本機禁止執行，沒有 BGE-M3 實測。
FP32 AdamW 的參數、梯度、兩份動量粗略下限為 16 × 可訓練參數量 bytes，還不含 activation、attention、CUDA workspace、暫存張量與框架成本。此值只在另一台實際載入模型後記入 run_config，不是完整 GPU 估算。
本實作不是 LoRA，不使用 AMP／量化／多 GPU，全部可訓練參數更新。A2-2 的額外文件 forward 仍需保留反向圖，預設 batch=8、length=512 不保證能放入特定 GPU。
可兩組同步減小 batch／max-length，或加 --gradient-checkpointing（若底層 Transformer 不支援會明確停止）。縮短長度會改截斷資訊，必須兩組一致並重新記錄。CPU 也不是零成本的替代品。

尚待使用者在另一台電腦驗證：Python 語法與匯入、ST／Torch／Transformers／Hub 版本組合、tokenize/forward/save、loss 梯度與 mask、排程覆蓋、尾批及去重、CPU/CUDA determinism、實際記憶體、checkpoint 回載及評估指標。
沒有執行程式或測試，不能聲稱已成功訓練、通過 parser、套件相容或取得任何模型效果。
