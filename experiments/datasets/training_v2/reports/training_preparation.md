# V2.0訓練準備

本輪只執行原生資料驗證、Python AST語法檢查與標準函式庫資料載入／排程檢查，沒有載入大型模型、訓練或執行Benchmark。

## 定稿與衍生資料

9筆答案（K097、K117、K132各3題）只刪除題目未詢問的即時狀態附註，逐筆前後內容見`final_answer_cleanup_report.md`。母資料仍為336題、134個基本知識點、138個意圖群組、198組正例、54筆有向困難負例候選。未取得人工批准。

`frozen/`保存定稿母資料、三欄Excel、驗證報告、來源登錄快照及資料Manifest。Manifest的SHA-256以實際位元組計算；來源登錄快照以`source_manifest_sha256`額外綁定。訓練載入器只讀取凍結資料並重新推導轉換結果，不回查舊FAQ。`prepare_training.py`拒絕覆寫既有frozen/splits/converted目錄；`--check-only`只驗證，`--report`僅能建立新報告。資料目錄的`.gitattributes`禁止Checkout換行轉換。

## 共用切分

固定seed=42。以22個既有leakage family為起點，合併多資訊整合的知識點及已記錄的語意重疊排除關係，形成20個不可拆分的連通家族。用確定性的子集合動態規劃優先接近20%題數、其次接近20%意圖數，再偏好更多驗證主題；不是逐題隨機切分，也不是依模型成績選擇切分。

|項目|Train|Validation|
|---|---:|---:|
|題目|269|67|
|意圖|110|28|
|原始知識家族|14|8|
|A2正例|159|39|
|有向Hard Negative候選|52|2|
|A3問答|269|67|

約80.06%／19.94%。正例、Hard Negative、知識點、同義群組及重疊排除關係均未跨集合；獨立檢查也未發現跨集合重疊的來源證據區間。本次恰無跨集合Hard Negative需要移除。這是較嚴格的未見知識家族驗證，不是只留出同一家族中的未見意圖。部分主題只出現在其中一側；不能聲稱各主題分層均衡。Validation僅2筆有向困難負例，無法充分評估困難負例泛化。沒有建立Test，沒有使用804題或任何正式測試題庫。

## A2實際使用方式

既有`train_faq_embedding`新增`--v2-data-dir`分支，保留舊入口。V2正例仍是Query-to-Query：改寫問題作Query，標準問題作Positive，不是Query-to-Passage。每組改寫數可變，單獨的無改寫意圖不虛構Positive；可在既有有效Hard Negative關係中作文件端。整合題若沒有正例配對，僅提供A3監督。

兩模式共用同一159組Train正例、相同seed、batch排程、optimizer設定及原始BGE-M3。3 Epoch均規劃60次更新、477次正例曝光。兩者排程SHA：`6f2e92fd06fbc4b80b508db223fb74428a1e821817e5d9e4e84cb55315facc3d`。A2-1只有合法batch內負例；A2-2再加入相應意圖的顯式負例欄位，沿用既有Masked cosine InfoNCE及False Negative Masking，沒有將顯式負例寫入後忽略。

既有Hard Negative有些以改寫題作anchor；轉換保留原始有向Pair ID與端點證據，訓練時依共同意圖套用到該意圖的正例查詢，文件端使用對方意圖的標準問題。不是新增新的來源負例。52筆Train候選不是52組經BGE排序驗證的困難負例，也不等於每epoch固定52次負例曝光。排程／mask輕量檢查中，A2-2三epoch有276次顯式負例可見曝光；實際有效新增負例與梯度由訓練log記錄。沒有Hard Negative的意圖仍保留；覆蓋不均必須揭露。

SentenceTransformers仍採現有明確API profile；6.1.0使用preprocess及可微分forward，保留原Pooling，Query／Positive／額外負例均參與梯度。開發環境ST3.2.1不在現有受審查profile內，沒有在開發機放寬限制或實測模型相容性。GPU現有6.1.0／Transformers5.x仍須實際驗證。

每epoch額外記錄固定Validation排程上的query-weighted masked InfoNCE，無梯度、無optimizer；它不是Top-1準確率。A2-2的Validation亦包含該集合自己的候選負例，不能把它與A2-1不同負例分母的Loss直接當準確率比較。

## A3實際使用方式

既有`train --data-dir .../converted`識別V2契約，保留舊Smoke／修訂資料入口。每筆是system／user／assistant三則messages；可見訊息沒有來源路徑、評分欄位或內部證據。`encode_sample`、原生Chat Template邊界、assistant-only labels與按位置mask的padding保持原樣，拒絕截斷。訓練只使用Train；Validation每epoch計算assistant supervised-token-weighted Loss，不反向傳播。

原生Tokenizer與Chat Template必須在GPU電腦以實際本機Llama Tokenizer執行`--check-only`；本機只有Python3.8、Torch2.4.1、Transformers4.46.3、ST3.2.1且未安裝PEFT，沒有完成模型或Tokenizer執行驗證。不能把messages結構通過等同tokenization通過。

LoRA仍為r=8、alpha=16、dropout=.05、q_proj/v_proj，使用既有save_pretrained、Tokenizer及a3_provenance/a3_complete格式，因此保持A5載入契約；Adapter實際存回與A5回載仍待GPU確認。

## 初始設定與限制

- A2：LR=2e-5、batch=8、max_length=512、3 Epoch、seed42、scale20、weight_decay=.01，兩模式相同，啟用gradient checkpointing。全參數BGE-M3與Adam狀態、查詢及文件分支會占用大量VRAM，不能保證3090不OOM；若需減batch，兩模式應一起調整並用全新Run。
- A3：LR=1e-4、batch=1、gradient accumulation=4、3 Epoch、max_length512、原LoRA配置，BF16或明確指定FP16；gradient checkpointing沿用原設定。LR較舊2e-4保守，避免少量資料過快擬合；這不是經Validation調參後的最佳值。
- 現有最小迴圈新增Validation Loss紀錄，但沒有自動Early Stopping、best-checkpoint選擇或自動重訓。Final仍為最後完成epoch；先看Validation走勢，若需早停／選模應另作明確設定，不得用Test選參數。
- 未經真實批准仍須`--allow-unreviewed`，輸出provisional。正式訓練使用既有training_review Schema，綁定實際Train與converted Manifest SHA、確切Train ID、逐ID批准、真實reviewer／reviewed_at／evidence。A2的ID為110個Train意圖；A3為269個Train題目。沒有生成假批准收據。

## GPU執行

先同步training_v2完整目錄與修改的A2/A3程式。先執行資料檢查，再使用本機Tokenizer檢查；`train_gpu.ps1`將依序完成上述檢查及三種訓練，任何階段失敗立即停止，不覆寫既有模型。輸出為`experiments/models/training_v2/<新時間與UUID>/`。沒有預設Hub下載或替換模型。

正式命令應傳入真實`-A2Review`和`-A3Review`；只有明確選擇`-AllowUnreviewed`時才允許完整預備訓練。模型完成後仍需檢視Validation、checkpoint完整性及實際載入，不可因資料轉換成功宣稱正式研究資格完成。
