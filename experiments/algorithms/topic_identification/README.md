# Topic Identification：輔助研究與上下文效率

此版已撰寫並靜態審查，沒有在開發機執行 Python、測試、模型或訓練。
沒有修改八種 QA 架構、Semantic Cache、Speculative Decoding 的程式或資料。
分類好不代表回答品質不變；本研究需同時看額外辨識成本、必要上下文及人工答案品質。

## 已實作版本

| 版本 | 實作 |
|---|---|
| TI0 | 最近一則歷史 user 問題作主題錨點；本機 BGE-M3 與目前 query 的 normalized cosine |
| TI3-1 | 本機 LlamaModel prefill 的真實 eager Attention，全部層／全部 query heads |
| TI3-2 | 依實際層數分為前／中／後三段，計算相同特徵並各自校準門檻 |
| TI3-3 | 全歷史、BGE 判斷後保守縮減、Attention 判斷後保守縮減，三組多輪 RAG 生成 |

`--mode detection`（預設）只執行 TI0／TI3-1／TI3-2；`--mode all` 另執行 TI3-3。
`--phase calibrate` 不讀取 Evaluation，不生成回答；即使 mode=all 也只完成校準。
`--phase evaluate` 必須指定已完成校準 Run 的 thresholds.json；不重新選門檻。
`--phase all` 先校準並凍結，再讀 Evaluation；同次執行仍分開記錄，不混算成績。

## Attention 方法：自主設計，不是標準因果解釋

API 靜態審查範圍是 Transformers **5.16.1**；不同版本明確報錯，不以假資料替代。
既有 A0Adapter 只負責本機資源載入，本程序把自己載入的 Target 切換為 eager；
不寫回模型／設定。TI3-3 三組生成也使用同一 eager backend，沒有跨組偷偷換加速器。

使用 `model.model(..., output_attentions=True, use_cache=False, return_dict=True)`，
直接取 LlamaModel 的 prefill 結果，不計算多餘的詞彙 logits。若 attentions 缺失，
層數或 `(batch=1, query_heads, S, S)` 不符，就報錯。

內容範圍以 native chat template 的確定性 marker 渲染取得字元區間，替換後須逐字等於
正常 template 輸出，才使用 fast tokenizer offset_mapping 對應 Token；沒有尾端猜測。
此版不 padding，並依 attention_mask、特殊 Token ID、非零且完整落在內容區間的 offset
排除 Padding、角色標籤、BOS/EOT 與跨邊界 Token。保留每則訊息字元及 Token 範圍供查核。
Prompt 使用系統說明、完整 scripted user/assistant 歷史、最後 user query，不加生成提示。

令 Q 為目前問題的內容 Token，H 為所有歷史 user/assistant 內容 Token。
每層、每個 head、每個 q，先計算指向 H 的 Attention 總質量 h 與指向 Q 的質量 c：

```text
ratio(layer, head, q) = h / (h + c)
layer_score = mean over all query heads and current-query content tokens
group_score = mean of layer_score over the selected layer group
```

不將 batch 當 head，也不平均整個上三角的 causal 零值作為 token-pair 密度。
Q 的未來位置仍是因果遮罩的零質量；System 與格式 Token 不進分母。
這個總質量比例仍受歷史長度、位置、head／layer 與文字內容影響，沒有消除所有混淆。
不是舊程式區塊元素平均的同一公式，版本為 content_mass_ratio_layer_head_query_mean_v1。

前／中／後層由 N 個 decoder layers 的 `[0,N//3)`、`[N//3,2N//3)`、
`[2N//3,N)` 決定，記錄零起算層索引，沒有指定固定第幾層或在 Evaluation 選最佳層。
四個 Attention 特徵來自同一次真實 forward，逐層聚合，不額外 stack 大張量；
TI3-2 的各組延遲引用共享 forward，不能相加或宣稱是分別執行的組別速度。

Attention 不能直接當作模型理解或輸出原因的證明。方法的有效性必須與 TI0、標註資料及
人工品質核對一起檢驗；目前不控制 Cache 或 Speculative 策略。
TI0 的錨點只取最近一則 user，Attention 則看到完整對話，這是簡單基準與完整歷史特徵的
比較，尚未控制所有輸入資訊；不能把結果差異全歸因於 Attention 這個訊號。

官方 API 來源：

- https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/models/llama/modeling_llama.py
- https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/modeling_utils.py

## 校準與 Evaluation

- calibration.jsonl：8 個 scripted 對話，4 Same Topic、4 Topic Shift。
- evaluation.jsonl：10 個獨立 scripted 對話，5 Same Topic、5 Topic Shift。
- 涵蓋連續追問、明確換題、相似不同意圖、指代、省略、換題但仍需先前條件。
- 對話為人工設計候選，assistant 內容不是模型實測輸出，不計作模型歷史回答品質證據。
  不編造系所新事實；檢索是否足以回答也仍待人工確認。
- 所有標籤 needs_review；必須 --allow-unreviewed 才可開發執行，結果 provisional。
- Topic 定義是對話主題，不是 FAQ ID：學費與獎學金可同屬財務主題，仍需審查這個邊界。

每個特徵使用相同規則：score < threshold 預測換題，換題是正類。
門檻候選是 calibration 分數間中點及兩端；優先最大化正類 F1，再 Accuracy，再較小
門檻。每層組各自按同一準則校準，數值可以不同。Evaluation 絕不重調門檻。
Calibration 成績放在 thresholds，不標為測試結果；Evaluation 輸出 Accuracy、
Precision、Recall、F1 與 TP/TN/FP/FN，零分母採 0。
Classification 延遲加總實測特徵時間與另外量測的 CPU 門檻比較時間；原始特徵紀錄另有
純 forward 時間。TI3-3 則直接量測整個辨識呼叫，不借用這些離線報表的時間。

thresholds.json 包含 payload hash、模型／程式／方法識別、calibration 資料 hash、
scenario/family/query/context 指紋。Evaluate 要求原校準目錄有成功 completed.json，
不能使用失敗或中途中斷的校準產物。模型、程式、dtype 或特徵規則不同須重新獨立校準。
跨集合檢查 ID、family、完整對話及 query 正規化文字重複，但不能因此宣稱無語意洩漏。
這是小型開發題庫，不是未見 FAQ 泛化測試，不應依 Evaluation 結果改題後仍當同一固定集。

## TI3-3 的獨立多輪 Adapter

A0 本身仍是單輪，這裡不修改它。新 adapter 保留 A0 原 System、Top-5 context、目前
問題格式，於目前 user 回合之前插入具 user/assistant 角色的完整歷史。三組使用同一
Target、dtype、greedy 設定、seed+repeat、相同檢索文件與排序、eager backend。
檢索 query 仍是目前問題，不暗中加入 query rewrite。指代型問題可能因此檢索不足，
即使生成看得到歷史也不保證回答正確；這是三組共同限制。

縮減流程：

1. 無預測換題：保留全部歷史。
2. 預測換題但有指代／延續線索（這、那、前面、如果、比較、呢等）：保留全部。
3. 其餘預測換題：只移除較舊的完整回合，永遠保留最近一組 user+assistant。

規則較保守，可能保留不相關歷史、使速度更慢。它不是完整共指解析器，也無法涵蓋
所有隱含歷史依賴。`required_history_ids` 僅在策略執行及回答後評分，不參與決策。
`annotated_required_context_retained` 只反映待審核標註是否仍在 Prompt，不是語意安全保證。
`context_safety_guaranteed=false`、人工品質欄位初始 null，不填假分數。

三組各自重算辨識，不重用先前 Evaluation 的特徵以省略決策成本。分類用一次共享
Attention forward 比較 Layer；TI3-3 又獨立計算，兩種研究用途與延遲分開。
每題一次檢索，三組／重複共用，保存檢索快照；end_to_end = 該共享檢索耗時＋本組
獨立執行耗時（含辨識、保守規則、完整／縮減 Prompt 計數、生成與 decode）。
這是受控合成時間，並非每組各自實測一遍檢索；不要加總重複引用的 retrieval_seconds。
Full 的辨識時間為 0，不代表整體零延遲。除 generation latency 外，同時看判斷成本。

全歷史及縮減 Prompt 都必須符合 A0 2048 input tokens 與模型 context window，否則
整次報錯，不默默截斷。Prompt 計數的監測成本也納入本組執行時間，不能稱為零成本。
每個情境／策略有 warmup、正式 repeats，旋轉三組順序；暖機保存但排除摘要。
人工應填 answer_correct、context_sufficient、equivalent_quality_to_full、reviewer、notes。
沒有完成品質審查前，不能宣稱「維持品質下的加速」，也不能只依 Token 減少下結論。

## GPU 未來指令

此階段不需現在執行。使用既有環境，不下載模型或套件。需本機 Llama 3.2 3B Instruct、
BGE-M3、A0 文件索引（index.faiss＋index.pkl；不要求 version.json）。不要指向 FAQ60 索引。

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
if (Test-Path experiments/algorithms/topic_identification/config.gpu.local.json) { throw 'Edit existing local config; do not overwrite.' }
Copy-Item experiments/algorithms/topic_identification/config.example.json experiments/algorithms/topic_identification/config.gpu.local.json
notepad experiments/algorithms/topic_identification/config.gpu.local.json
```

填入實際本機模型目錄，已存在的 ignore 規則涵蓋 config*.local.json；不修改 Git 狀態。
CUDA 使用相同 fp16 或 bf16；CPU 須 fp32。Attention 會儲存多層 S×S 矩陣，
max_attention_tokens 預設 512，超長就報錯，記憶體不足不自動換精度或刪歷史。

先只校準：

```powershell
$tiCal = 'experiments/results/algorithms/ti_cal_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.topic_identification.benchmark --project-root . --config experiments/algorithms/topic_identification/config.gpu.local.json --phase calibrate --mode detection --run-dir $tiCal --allow-unreviewed
```

獨立評估 TI0／TI3-1／TI3-2，必須使用已完成的門檻：

```powershell
$tiEval = 'experiments/results/algorithms/ti_eval_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.topic_identification.benchmark --project-root . --config experiments/algorithms/topic_identification/config.gpu.local.json --phase evaluate --mode detection --thresholds "$tiCal/thresholds.json" --run-dir $tiEval --allow-unreviewed
```

同樣凍結門檻、另開新 Run 執行 TI3-3：

```powershell
$tiContext = 'experiments/results/algorithms/ti_context_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.topic_identification.benchmark --project-root . --config experiments/algorithms/topic_identification/config.gpu.local.json --phase evaluate --mode all --thresholds "$tiCal/thresholds.json" --run-dir $tiContext --allow-unreviewed
```

每次新 Run，沒有 resume 或覆寫。需要一次跑完可用 --phase all --mode all，但仍先凍結
Calibration，再開啟 Evaluation；不能因此將校準／暖機成本混入正式逐題指標。

## 輸出及驗證狀態

- run.json、identity.json：模式、來源、模型／套件／程式、特徵、Layer、硬體、載入時間。
- thresholds.json 或 thresholds.used.json：校準產物或使用的凍結門檻。
- calibration/evaluation.snapshot.json：完整對話與標註。
- calibration/evaluation_features.jsonl：真實 per-layer 分數、GPU forward／完整辨識時間、
  Token spans 與實際 Prompt；大矩陣不寫入檔案。
- classification.jsonl／classification.csv：Evaluation 逐題 score、門檻、預測、gold、延遲。
- context_retrieval_*.json：TI3-3 共用文件快照。
- context_results.jsonl／context_comparison.csv：暖機／逐題回答、縮減、時間、人工品質欄位。
- summary.json：分開顯示校準資訊、Evaluation 分類及三組縮減統計，不將資料混算。
- completed.json：只有流程完成、來源驗證通過才建立；仍不表示方法有效或答案品質通過。
- failed.json：失敗階段、scenario、error 與 traceback；既有原始紀錄保留。

尚待 GPU：API 真正回傳 Attention、offset/template 一致性、VRAM、數值穩定性、延遲、
分類效果、依賴標註與答案品質。本版是可執行流程的靜態實作，不是已驗證有效的研究結果。
