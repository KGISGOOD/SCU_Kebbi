# INT0～INT3：獨立線上 RAG 效率比較（Milestone B）

狀態：完整流程已撰寫並靜態審查，本輪沒有執行 Python、測試或 GPU。沿用已存在的 A0、SC、SD，
不變動八架構、Semantic Cache、原始資料、索引或結果；不使用 Ollama、不下載模型。

| 模式 | Cache | Miss／每次無 Cache 請求 | Hit |
|---|---|---|---|
| int0 | none | 完整 A0 Top-5 → 原 Prompt → 3B greedy | 不適用 |
| int1 | 自有 Semantic | 完整 A0 → 3B greedy → 寫入自己的 Cache | 儲存答案 |
| int2 | none | 完整 A0 → 選定 Decoder | 不適用 |
| int3 | 自有 Semantic | 完整 A0 → 選定 Decoder → 寫入自己的 Cache | 儲存答案 |

## 依賴與處理流程

`benchmark → IntegratedPipeline → CachedA0 → DecoderAdapter → SDEngine.prepare/generate`。
命中在 CachedA0 內直接返回，不呼叫 DecoderAdapter.answer，因此沒有 RAG、Prompt 或 Target/Draft
生成。Miss 才呼叫既有 A0Adapter 的 RAGPipeline.retrieve、原 build_prompt 與 Target tokenizer，
保留完整 Top-5、不截斷；SD Engine 執行真正 Target-only／Target 驗證解碼。

解碼器：`standard`／`draft_1b`／`prompt_lookup`。前兩種沿用原 API 與成功的 Draft KV 修正；
Prompt Lookup 重用 SD2-4。INT0、INT1 固定 standard；INT2、INT3 才使用選定 Decoder。
預設 draft_1b、候選 4、固定排程，只代表既有功能設定，不代表最快。Prompt Lookup 預設 k4、
匹配 n-gram 2，可獨立指定。所有模式使用同一有效 Target GenerationConfig、seed+repeat、精度、裝置。

選 prompt_lookup 時不載入 1B。單跑 int0/int1 即使設定 decoder=draft_1b 也不要求 1B 存在。
all + draft_1b 時整個 Run 都常駐 1B（INT0/1 不呼叫它），identity 明確記錄；不同常駐條件的
不同 Run 不能直接解讀為只有演算法差異。OOM 明確失敗，不自動換精度／offload／fallback。

每個 variant/repeat/warmup 都新建獨立 AnswerCache／FAISS IndexIDMap2／origin table。
命名空間含 variant、decoder、repeat、warmup、threshold、完整模型／Prompt／索引／程式來源識別。
INT1 的答案不會填入 INT3；只有各自 Miss 的實際輸出才能入庫。沿用 Cache ID／FAISS label
完整性檢查，另核對命中來源與儲存答案。沒有 LRU、持久化或多輪答案重用。
含 history/messages/context/session_id 的情境會被拒絕；不得拿此版處理需要對話上下文的回答。

各 variant 取得相同查詢順序，題目之間交錯旋轉 variant 的執行順序；同一 variant 仍按序存取 Cache。
暖機跑獨立完整序列，保留紀錄但不列入正式統計；每個正式 repeat 再從空 Cache 開始。

## 計時、比較與安全

- 真實 online E2E：CachedA0 的 CUDA 同步 wall clock，包含正規化／identity、Cache embedding/lookup、
  Miss 的獨立檢索／Prompt／生成、Cache insert。Cache／Pipeline 建構不計入單題；屬初始化成本。
- 每個 Miss 都獨立檢索，**不是 SD Benchmark 的共享前處理受控合成時間**。
  整合報表刪除巢狀 SD synthetic E2E 欄位，避免誤用；只引用實際 generation-only 等分項。
- 模型載入、來源 Hash、暖機、寫報表、人工標籤評估分開；整次 wall time 在 completed 保存。
- Hit 的本次 input/generated tokens=0，輸出 Token IDs/hash 來自 answer origin，明確標記 scope。
  Hit 的 prompt_identity=null；另存 answer_origin_prompt_identity。不是新生成的 Prompt/Token。
- all 模式：各 Miss 的實際 Prompt／文件 identity 必須與相同 scenario/repeat 的 INT0 相同，
  再逐 Token 比較。若不同，保存失敗報表、不建立 completed。另檢查 INT0 跨 repeat 的穩定性。
- 單跑 int1/int2/int3：保留真實路徑、輸出與時間，但 Token 對照及 INT0 speedup 標 unavailable/null。
  應跑 all 才有同 Run 的配對比較，不以不同 Run 的答案偷偷補入 Cache。
- Hit 不強制與該問題 INT0 逐 Token 相同（改寫本來可能生成不同文字），也不當作 lossless 通過。
  回收實際 origin 的 pair label；needs_review 時一律 unverified，不推論答案安全。
- 加速比：相同題序／正式 repeat 的 INT0 online 總時間 ÷ variant online 總時間。
  未具備配對或出現失敗時該版加速比 null。平均／中位數、hitrate、分項總時間都保留。
  比率只是描述性速度資料，**不是已保證答案品質的加速**；品質人工欄位預設 null。

10 筆功能情境含 3 筆完全重複／重訪、同義改寫、升學與就業近題、數學跨主題、實習年級條件
與即時天氣系務外問題。所有題與 pair label 都 needs_review；不編造系規／答案。這不是凍結
的研究 Test，也不拿它反覆選門檻後宣稱泛化。正式研究另建獨立 calibration/evaluation 工作負載。
若實際命中未標註的來源 pair，仍 unverified；即使 pair approved/allowed，也不等同生成答案事實正確。

## GPU 本機設定

沿用 Transformers **5.16.1**、torch、sentence-transformers、numpy、FAISS 及現有 A0 依賴。
沒有新增套件。只接受已下載的本機模型目錄及原文件 FAISS（index.faiss/index.pkl），不需要 version.json。
GPU 根目錄操作；若設定已存在，編輯它而非覆寫：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
$intConfig = 'experiments/algorithms/integrated_efficiency/config.gpu.local.json'
if (-not (Test-Path -LiteralPath $intConfig)) {
    Copy-Item -LiteralPath experiments/algorithms/integrated_efficiency/config.example.json -Destination $intConfig
}
notepad $intConfig
```

填入 Target/BGE/Draft 本機目錄，knowledge_index 維持文件庫，不是 FAQ60 索引。
`config*.local.json` 已被根目錄既有 ignore 規則涵蓋，本次未修改 .gitignore 或 Git 狀態。

## 先完成 Milestone A，再驗證 INT

先按 [SD2-4 說明](../speculative_decoding/SD2_4_README.md) 獨立驗證，再驗證真正 INT0：

```powershell
$intRun = 'experiments/results/algorithms/int0_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.integrated_efficiency.benchmark --project-root . --config $intConfig --mode int0 --repeats 1 --warmups 1 --run-dir $intRun --allow-unreviewed
```

Prompt Lookup 的完整整合比較（不載入 1B）：

```powershell
$intRun = 'experiments/results/algorithms/int_lookup_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.integrated_efficiency.benchmark --project-root . --config $intConfig --mode all --decoder prompt_lookup --prompt-lookup-tokens 4 --max-matching-ngram-size 2 --threshold 0.75 --repeats 1 --warmups 1 --run-dir $intRun --allow-unreviewed
```

另一個全新 Run 比較原成功 Draft 設定：

```powershell
$intRun = 'experiments/results/algorithms/int_draft_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.integrated_efficiency.benchmark --project-root . --config $intConfig --mode all --decoder draft_1b --threshold 0.75 --repeats 1 --warmups 1 --run-dir $intRun --allow-unreviewed
```

可單跑 `--mode int1`、`int2`、`int3`；`--decoder standard` 可只測整合基礎流程。
功能驗證後，用全新 Run 改 `--repeats 3` 或更多。沒有 resume、覆寫或 ZIP。

## 產物與驗收

run.json、identity.json、scenarios.snapshot.json、measurements.raw.jsonl、results.jsonl、
summary.json、comparison.csv、completed.json／failed.json。每列包含完整答案與 Token IDs/hash，
Miss 另含完整 Top-5/Prompt snapshot；包含模型／資料／Prompt／程式／索引來源、記憶體／生成診斷。
逐題原始紀錄立即寫入，生成中斷保留已存列，failed 包含 stage/request/traceback/cache_failure。
來源檢查、輸出比較與報表全完成才建立 completed；完成不等於人工答案品質通過。

GPU 應檢查：INT1/3 每 repeat 首題 Miss、namespace 不同；Hit 的 retrieval/generation=0、
generated tokens=0、reused_from_scenario 指向正確來源；Miss 有 Top-5/實際生成；all 的 Miss
token_comparison_status=passed；沒有 failed.json。不要預設 paraphrase 一定 Hit 或近題一定 Miss。

仍待實測：HF 候選觀察介面、RTX3090 顯存、Token 一致性／重複性、所有時間與實際命中、
中文答案品質及 false reuse。未增加安全重用判別模型；不能宣稱單一 cosine 門檻解決意圖混淆。
