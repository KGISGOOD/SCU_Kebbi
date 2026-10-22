# A0–A5 統一 FAQ 問答測試（第一輪）

> A1 後續已找到 LF／CRLF 的精確 Hash 證據。新增唯讀 byte/docstore 診斷、只寫全新檔案的 LF 恢復與清楚的 Preflight 跳過訊息，見 [A1_CANONICAL_RECOVERY.md](A1_CANONICAL_RECOVERY.md)。原 canonical、FAISS 與 version.json 均不改寫。下節「當時尚未定位」是先前盤點狀態，現以本恢復文件為準。

## A1 completion/hash mismatch 的唯讀定位

若 A1 顯示 `A1 new index completion/hash/canonical mismatch`，先不要重建、修改 version.json 或重新保存 canonical。修正版讓 prepare_a1 建置後自查、Preflight 與 audit-index 共用同一個 `index_completion_report`；所有 hash 都用 A3 `sha()` 以 `rb` 讀取原始 bytes、計算小寫 SHA-256，沒有字串序列化、換行正規化或忽略差異。Preflight 失敗會列出每個不一致欄位的 expected／actual，以及實際解析的 canonical/index/docstore/version 絕對路徑，仍然阻止載入。

本次開發機已收到一份 GPU A1 資料副本，唯讀 PowerShell 檢查結果如下（不是模型／Python 測試）：

| 欄位 | version.json 記載與本機實際 SHA-256（兩者一致） |
|---|---|
| canonical_sha256 | `7ff63fb1a27558345f33b034023243a8a35e774019f53397108861d40679f8ac` |
| index_sha256 | `f96399a2e96b663f10b4fc9c36f01378762491e894e677554d7a9e7c24207ffa` |
| docstore_sha256 | `6c92de766e1f6544ca5bf7250845d9ea1a83be6ffb53de6665a4464cf7683498` |

receipt 為 schema `a1_60_preparation_v1`、status `completed`、faq_count 60。預設 config 及 Wrapper 均指向 `faq_canonical_60_v2.jsonl`／`faq_bge_m3_60_v2`；Wrapper 只 load 與檢查 docstore，沒有 build/save 呼叫。建置端則是 build/save → load → 檢查 Q/A/mapping → hash → 寫入 version.json。靜態資料流未發現兩端 Hash 算法不同或讀取後自動改寫的程式缺陷。

**本機副本無法重建失敗當時 GPU 的檔案狀態。** 目前只能確認舊訊息把 status/count/三個 hash 合併，無法定位；不能斷言是換行、傳輸、重建或設定變動。八個 loader 曾成功與後續 Smoke 失敗，也不能證明上述檔案在兩次執行間未改變。

錯誤階段另有一項可由程式確定的意義：Worker 在 architecture_preflight 前先執行 verify_inputs，後者會重新盤點並要求 resources 與 run.json 完全相等。因此此錯誤表示該 Smoke Run 建立時的 a1_canonical 盤點就已同樣失敗；若 Worker 才看到新的資源差異，應先在 verify_inputs 拋 InputChanged。且此訊息位於 60 題 ID／原始 Q/A 檢查之後，表示該次讀到的 canonical 已先通過那些檢查，並非只載入了舊三題而產生這個特定錯誤。

同步本次 `prepare_a1.py`、`preflight.py`、README（不複製資料覆蓋 GPU），在 GPU 專案根目錄先執行：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'

# 以目前 config 檢查；不載入模型、FAISS 或 pickle，不寫任何檔案
python -m experiments.evaluation.prepare_a1 --project-root . audit-index

# 把下列目錄替換為這次 A1 失敗的 Smoke Run；讀取其凍結設定，不修改舊 Run
python -m experiments.evaluation.prepare_a1 --project-root . audit-index --run-dir 'experiments/results/unified_faq/實際失敗的Smoke目錄'
```

audit-index 輸出全部預期／實際 hash 與完整路徑、CSV 60 題 Q/A 核對結果。指定 --run-dir 時也保留舊 Run 的 a1_index／a1_canonical 資源紀錄，並比較索引檔案與 version.json 是否自該 Run 建立後改變；退出碼 2 表示有差異或驗證失敗。這個命令不驗證 BGE 權重、FAISS 二進位結構或語意匹配結果；完整 loader 驗證仍由原 Preflight／Wrapper 執行。

判讀方式：

- 目前 config 通過、失敗 Run 不通過：先看兩份路徑、舊資源錯誤及 changes_since_run_creation；不要直接改舊 Run。
- canonical hash 不符但 CSV Q/A 核對通過：只證明內容語意來源一致，不代表 byte receipt 相同；查明是否有 BOM、換行、欄位順序或其他重寫。禁止直接更新 receipt 讓它通過；優先找回與 receipt 匹配的原始版本。
- index/docstore hash 不符：停止，保留現有檔案與 receipt，核對備份及檔案同步紀錄，不自動重新保存或重建。
- status/count/schema 不符：檢查是否選到其他建置版本或不完整輸出，不把它改成 completed。
- 全部通過：此副本沒有 completion/hash 缺陷，不需要重建。用同一 BGE snapshot 做全新 A1 loader 與 Smoke；若再失敗，新的 Log 會指出當下確切欄位。

沿用先前成功的原始 BGE snapshot 絕對路徑（以下第一份是先前 receipt 記載；自訂快取請改實際位置）：

```powershell
$bgeSnapshot = Join-Path $env:USERPROFILE '.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181'
$loadTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run preflight --project-root . --architectures A1 --run-dir "experiments/results/unified_faq/a1_receipt_load_$loadTag" --bge-snapshot "$bgeSnapshot" --allow-unreviewed --load-models
```

確認 A1 `loaded_no_inference` 後，再使用全新 Run，僅驗證 A1 的三題：

```powershell
$smokeTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run smoke --project-root . --architectures A1 --run-dir "experiments/results/unified_faq/a1_receipt_smoke_$smokeTag" --bge-snapshot "$bgeSnapshot" --allow-unreviewed
```

不需重跑其他七架構、不覆寫 A1 索引或舊結果。若真的證實現有索引無法恢復，才另以 prepare_a1 index 的 --output-dir 建立全新 v3，搭配新的 config 與新 Run；本次沒有證據需要這樣做，也未執行重建。

> 第一輪 Smoke 的資源修正與 GPU 指令見 [RESOURCE_PREPARATION.md](RESOURCE_PREPARATION.md)：離線固定 BGE snapshot、建立全新 A1 60 題 canonical／索引、重用 A2 候選庫建置程式。新預設不再指向舊 A1 三題示範檔；所有資料與模型操作均由使用者在 GPU 電腦明確執行。

> Windows Fingerprint 修正版：GPU 已確認唯一差異為 `environment.platform` 的 build 26200／26100，處理方式見下節。先前 [WORKER_ENCODING_AUDIT.md](WORKER_ENCODING_AUDIT.md) 為當時的歷史盤點，其「欄位未知」結論已由本次證據更新。所有舊 Run 完整保留；本次仍未執行 Python、測試或模型。

## Windows 父子行程 Fingerprint 修正

GPU 父行程回報 `Windows-10-10.0.26200-SP0`，Worker 回報 `Windows-10-10.0.26100-SP0`。原先 `environment()` 把 `platform.platform()` 放入嚴格相等比較，因而八架構在載入模型前全部被 `verify_inputs` 拒絕。這是已確認的直接原因，並非模型失敗。

Windows 版本 API 可能受 executable manifest、相容性模式或版本回報來源影響。[Microsoft GetVersionEx 文件](https://learn.microsoft.com/en-us/windows/win32/api/sysinfoapi/nf-sysinfoapi-getversionexw) 明列相容性模式可能回報不同於實際安裝的 OS 版本；[Python 的相關版本回報 issue](https://github.com/python/cpython/issues/91025) 亦說明 Windows 版本識別的限制。這些是可能原因，不能僅憑兩個 build 字串斷言本機存在某個特定 shim／manifest。診斷值是各 API 的觀察結果，不宣稱某一個必為真實 OS build。

父行程、Worker、resume、check-inputs 與 Wrapper runtime contract 現在都使用同一個 `environment()`：

| 嚴格比對欄位 | 內容 |
|---|---|
| schema_version | `stable_environment_v2`，不同規則不默默相容 |
| python / python_implementation | Python 版本與實作 |
| python_executable / python_prefix | 解析後的執行檔、Python 環境位置 |
| os_name / sys_platform | OS 家族，不含 Windows build 或 marketing release |
| machine / pointer_bits | CPU 架構與 Python 指標位元數 |
| host | 主機名稱 |
| packages | 原本 14 個重要套件的完整版本字串（含 local version suffix） |

`environment_diagnostics` 另外保存 `platform.platform()`、release、version；Windows 額外保存 `win32_ver()`、`sys.getwindowsversion()`（含 platform_version）、`__COMPAT_LAYER`。父行程寫入 run.json 與 preflight；Worker 寫入自己的 job，並將差異輸出到 log 的 `diagnostic_differences_non_blocking`。只有這些 OS 版本診斷的差異不阻擋；穩定 identity 的差異仍完整列出原值／現值並失敗。

沒有放寬 `verify_inputs` 的其他比較：Python／套件、原碼 SHA-256、FAQ 與 split、模型／索引／候選庫來源、run.json 凍結設定及實際 runtime contract 都保留。diagnostic Metadata 本身仍納入 run.json 的內容 hash，人工修改 frozen run 仍被拒絕；只是不拿目前 Worker 的易變 OS 字串要求與父行程相等。此規則不保證 OS 更新前後具有完全相同數值結果，跨 OS 更新的研究比較仍需人工參考診斷資訊。

**舊 Run 不轉換、不覆寫、不 resume。** 舊 environment schema 與新規則不同，應使用全新 Run。請同步 common.py、run.py、worker.py 與本文件到 GPU 電腦，先確認 60 題中文與八個 Worker：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
python -m experiments.evaluation.run check-data --project-root .

# check-data 成功後執行；GUID 避免碰撞任何既有 Run
$runTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run preflight --project-root . --run-dir "experiments/results/unified_faq/worker_verify_stable_$runTag" --allow-unreviewed --verify-workers
```

八份 job 應為 `verified_no_model_load`；這不代表模型已載入。Windows build 不同可留下非阻擋診斷，但不應再單獨產生 integrity violation。若其他欄位不同，仍必須先處理；資源盤點失敗仍會返回退出碼 2。

檢查通過且資源正常後，再執行全新三題 Smoke（P01/P22/P26，八架構共最多 24 筆）：

```powershell
$smokeTag = [guid]::NewGuid().ToString('N')
python -m experiments.evaluation.run smoke --project-root . --run-dir "experiments/results/unified_faq/smoke_stable_$smokeTag" --allow-unreviewed
```

確認八架構 jobs／attempts 狀態與非空答案，不將啟動成功等同回答正確。本次僅完成靜態程式審查，父子行程一致性、模型啟動與 Smoke 均待 GPU 實際驗證。

本目錄提供八種架構的獨立、離線批次測試與人工評分報表。Codex僅撰寫程式、讀取原始碼與靜態審查，**沒有執行Python/Node、測試、推論、訓練、下載或安裝**。下列指令由使用者在GPU電腦執行，不能把本文件視為已跑通或已有實驗成績。

## 架構盤點與Wrapper

| 架構 | 實際重用介面 | 資源與差異 |
|---|---|---|
| A0 | RAGPipeline.retrieve / generate_answer、BGEM3Embedder、FAISSVectorStore、LLMLoader | 原始BGE-M3、A0 Type1/2 Dense索引、Llama3.2-3B；Dense Top5，A0原始手寫prompt，2048-token截斷保持不變。 |
| A1 | FAQMatcher.match(return_faq=True)、load_index | 既有canonical與既有FAQ FAISS，嚴格要求60題且Q/A與CSV一致；保留Top1匹配、內部Top2與既有threshold。不呼叫build_index，不用CSV查gold代替檢索。 |
| A2-1 / A2-2 | FAQAnswerer.answer(query) | 各自final_model、完整60題a2_candidates、CSV答案mapping；保留精確正規化內積/FAQ ID同分排序、checkpoint驗證與provisional gate。兩組max_length預設64，對應既有smoke，不偷偷改模型。 |
| A3 | load_adapter_for_chat；最小單輪生成相容層 | 同一A3 Base/Adapter/保存system與原生template；沒有檢索、不讀標準答案建立prompt；沿用512輸入上限。 |
| A4-1 | HybridRAGRerankerPipeline.retrieve / rerank_results / generate_answer | 原A4 loader/chunker、BM25+Dense Top10、RRF(k60)，LLM使用前5筆。原CLI即在記憶體建BM25，不要求磁碟hybrid_bm25_v1才能運作。 |
| A4-2 | 同上，use_reranker=True | 保留現有rerank Top10，再交前5筆至LLM；不是擅自改成只rerank5筆。BGE reranker使用原設定。 |
| A5 | A5Pipeline.answer(query)、A3 Adapter loader | 與A0同一dense_index/原始BGE、Top5，原生template、不截斷完整RAG prompt；模型EOS設定沿用A3。 |

A4使用既有資料載入/切分函式，不保存任何新索引；Wrapper另核對目前BM25文件的ID/內容與既有Dense文件一致，不一致則明確失敗，不混用不同資料或自動重建。

預設模型路徑：

- A2-1：`experiments/models/a2_1_smoke_task8/final_model`
- A2-2：`experiments/models/a2_2_smoke_task8/final_model`
- A3/A5：`experiments/models/a3_smoke_v1/final_adapter`
- Base：`meta-llama/Llama-3.2-3B-Instruct`
- Dense：`experiments/vector_stores/type1_type2_bge_m3_v1`
- A1：`experiments/datasets/faq/faq_canonical_60_v2.jsonl`與`experiments/vector_stores/faq_bge_m3_60_v2`（先依資源準備文件建立，保留舊 v1）

本次本機實際閱讀發現：A1 canonical僅3筆；A2正式60題`faq_retrieval_candidates.jsonl`不在本機；A3 Adapter未同步；A0索引有index.faiss/index.pkl但沒有version.json。**這些不是已修正的問題，也不能據此假設GPU電腦相同。** GPU Preflight會依實際資源判定，各架構獨立失敗；工具不替換候選庫、不建立新模型或索引。

## 檔案

| 檔案 | 用途 |
|---|---|
| config.json | 可複製修改的路徑/裝置/精度/長度/seed設定 |
| common.py | 原始60題、固定split、嚴格IO、原子不可覆寫紀錄、OS鎖 |
| preflight.py | 離線快取與檔案盤點、來源hash、各架構資源需求 |
| model_resources.py | 本機 cache lookup／明確 snapshot，必要權重、tokenizer、ST modules 與 hash 驗證 |
| prepare_a1.py | 全新 A1 60 題 canonical／索引準備，以及 A1/A2 原始 Q/A 一致性檢查 |
| inspect_bge_snapshots.ps1 | GPU 上唯讀盤點兩個 BGE snapshot，列出必要檔與 hash |
| wrappers.py | 原演算法最小相容層、生成telemetry與同步計時 |
| worker.py | 一個架構一個子行程；逐題保存、失敗隔離、執行環境契約 |
| run.py | 單一CLI：preflight/smoke/full/resume/report/import-review |
| report.py | 自動統計、人工評分匯入、CSV與本機HTML快照 |
| __init__.py | Package識別 |

未修改A0–A5原始程式、權重、CSV、訓練資料或FAISS；不開發A6。

## 題庫、reference與切分

唯一測試來源是兩份既有CSV：

- `畢業專題_爬蟲/type2/資料擴充 - 家長 FAQ 資料集（30 題）.csv`
- `畢業專題_爬蟲/type2/資料擴充 - 高中生 FAQ 資料集（30 題）.csv`

重用A3的嚴格source snapshot，讀取P01–P30、S01–S30、「問題內容」「ai答覆」原文，不做改寫/清洗答案。source欄保留CSV相對路徑與行號，所有來源記錄SHA-256。空Q/A、欄位錯誤、ID缺漏/重複、重複原題或非固定split皆停止，沒有另一套FAQ來源。

split讀取原A2/A3 manifest，必須符合：Train P01–21/S01–21=42；Validation P22–25/S22–26=9；Test P26–30/S27–30=9。完整題序固定P01…P30、S01…S30；Smoke固定P01/P22/P26，各split一題，八架構順序與題序一致。

run.json保存faq_id、question、reference_answer、split、source。Worker只呼叫`wrapper.answer(question)`；介面沒有reference_answer/gold ID參數。reference只在紀錄/報表與後續評分使用。A1/A2的標準FAQ候選與答案屬架構本身；RAG正常檢索到的文件也可能包含FAQ Q/A，這兩者不是測試程式直接把gold塞進prompt。

## Preflight兩個層級

預設`preflight`是資源盤點，**不載入模型、不生成**：檢查資料/manifest、套件metadata、本機/既有HF cache權重與tokenizer檔案、索引、A1完整Q/A、A2 tag與checkpoint hash、A3完成狀態/訓練設定/來源紀錄、Base/Adapter來源一致性、A4原始文件。

HF 使用 `try_to_load_from_cache` 的 config.json 定位或明確 `--bge-snapshot`，不下載、不要求整個 Hub repo 完整；必要推論檔與全部權重 shards 另行驗證及記錄 hash。盤點成功標為`resources_ready_not_loaded`，不能稱為模型可執行。壞/缺檔標failed；缺version標warning/provenance未完整驗證，不補造資訊。索引二進位/語意、權重內容與runtime套件相容性仍須loader確認。

`preflight --load-models`會依序在獨立子行程初始化實際Wrapper，不問測試題，不產生回答，輸出`loaded_no_inference`或failed。這才檢查FAISS映射/維度、實際模型載入與A4 BM25資料一致性，仍不等於答案正確。報告位於run/preflight/，載入詳細資訊在run/jobs/及contracts/。

生成模型共用同一Base資源；若A3有效Hub provenance有commit，A0/A4的Base快取也以該commit盤點，A3/A5沿用原loader的commit檢查。本機Base則核對完整檔案hash；不接受用另一來源冒充同一Base。

## 設定、可比性與生成監測

預設seed42、greedy(do_sample=False)、max_new_tokens256、單輪、無人工提示修正。每題重設Python/NumPy/torch/CUDA seed，子行程設定PYTHONHASHSEED，以便subset/resume不依賴先前題目順序；不保證跨套件/硬體bitwise一致。

所有生成架構使用原模型/原prompt。小型GenerationCapture包住實例的`model.generate`，只統一獲授權的greedy/max_new_tokens設定，移除舊LLMLoader在greedy下仍傳入的temperature/top_p/top_k，並在GenerationConfig副本將這三項還原為greedy可接受的中性預設；不改原模型config、權重、prompt，不使用judge。原始移除值與實際kwargs/config均記錄。這是測試解碼設定，不是修改演算法提高分數。

捕捉實際input_ids、有效prompt（含特殊token）、新增output_ids、完整原始decode與乾淨回答、token數及生成耗時。停止原因由最後token/EOS/長度推定，`stop_reason_source`明確標示，不能冒充模型提供了完整停止追蹤；達到上限與同位置EOS可以同時成立。

差異如實保留在contracts：

- A0/A4手寫prompt、2048截斷及原tokenizer特殊token行為不偷偷修正；effective_prompt可用來檢查問題是否被截掉。
- A3保留保存system與原生template，輸入上限512；A5使用RAG規則與原生template、預設2048且超長報錯。不可把這些差異全部歸因於LoRA。
- A0/A4保留tokenizer EOS覆寫；A3/A5保留原模型EOS列表。pad與實際kwargs保留在原始紀錄。
- A1原BGE max_length512；A2兩組smoke預設64，FP32/正規化一致；不得假稱所有Embedding長度已相同。
- 生成式precision預設FP16，可在新run改BF16；A1/A2維持原Embedding FP32設計，reranker保留既有FP32。
- A4保留Dense/BM25 Top10、RRF及reranker策略，實際提供LLM前5篇；保存BM25/Dense/Hybrid/Final及LLM文件不同階段。
- A0/A5共用同一索引路徑/hash與Top5；report另核對兩者成功配對題的實際文件ID、順序與全文，列出不一致，未配對不能宣稱通過。

Runtime contracts保存GPU型號/總VRAM、CUDA/套件版本、參數、prompt版本、模型來源與provisional；run.json保存完整資源hash與原碼hash。沒有為了分數調閾值、提示或檢索設定。

## 執行與RTX3090記憶體

Parent不載入torch模型；依A0→A1→A2-1→A2-2→A3→A4-1→A4-2→A5順序啟動子行程。每個子行程同架構只初始化一次，完成/失敗後退出，再載入下一架構，避免八模型同時駐留。初始化失敗仍記錄該架構所選題的錯誤，其他架構繼續。

無法可靠預估24GB峰值；A4-2同時載入LLM/BGE/reranker，或長context可能OOM。遇錯先記錄，不自動改精度/模型/長度。要更改設定請建立新run。CPU需`--device cpu --precision fp32`，不承諾RAM或速度足夠。現有A1 constructor會自動選裝置，Wrapper再明確移到統一裝置，因此CPU模式啟動可能短暫使用CUDA。

Wrapper匯入既有A0/A1/A4模組時，其Settings可能建立原本宣告的空資料夾；不呼叫任何索引build/save或舊測試結果輸出。已追蹤檔案與索引內容不修改。HF快取、Python自身cache行為依環境；模型解析全程強制離線。

## 保存、續跑與完整性

每個run使用新目錄及獨立run_id，既有run只能透過resume/report/import-review使用。不能覆寫run.json、已完成答案或先前報表。

每題立即寫完整暫存JSON、flush/fsync後，以同檔案系統exclusive hard link原子發布attempts/<arch>/<FAQ_ID_嘗試序號_uuid>.json；只有成功發布才算提交。重試順序使用遞增序號，不依賴可能校時的系統時間。需要支援hard link的檔案系統（Windows建議NTFS），不支援就報錯，不降級成不安全追加。

沒有脆弱的主JSONL追加流程，因此不存在截斷末行的恢復猜測。中斷留下的.tmp不算結果，不讀取、不合併；已提交JSON含內容hash，意外修改會被拒絕。歷史失敗attempt保留；同一題成功後不得再生成第二筆成功紀錄。報表選最新有效attempt，all_results.jsonl每架構/題一筆，歷史全量保留在attempts。

OS鎖（Windows msvcrt / POSIX flock）在程序結束自動釋放；鎖檔保留，不必刪除stale lock。Parent整個run加鎖，Worker另有架構鎖，避免兩個runner同時寫同一run。不要直接手動呼叫私有worker CLI。

resume核對題庫、來源、所有模型/索引/corpus、原始碼、參數、套件/host；run.json本身亦有內容hash，Worker還核對實際硬體contract。成功題跳過；失敗題預設保留，只有`--retry-failed`才新增attempt。資源原本缺失後補齊、程式/模型/設定改動或換電腦，必須開新run，不將不同版本混在同一結果。retry只適合身份未變的暫時失敗。

若發現執行中的輸入完整性改變，另留integrity_violations，HTML會紅字標示統計不可用於比較。程序硬崩潰且沒有committed answer時，Parent記錄worker error；失敗不會當成空白正確答案。

## 輸出與評估

```text
<run>/run.json                 # 完整凍結設定/60題/來源模型原碼版本
      preflight/              # 資源盤點、可選載入報告
      contracts/              # 各架構runtime/hardware契約
      logs/                   # 每次子行程完整stdout/stderr
      jobs/                   # 初始化時間、狀態、錯誤
      attempts/               # 每題每次嘗試，不可覆寫
      reviews/                # 不可覆寫的人工作業匯入事件
      reports/<snapshot>/     # 每次報表新快照
          all_results.jsonl
          comparison.csv
          summary.csv
          review.csv
          report.html
          retrieval_comparability.json
          report_complete.json
```

報表命令自動合併，無需手動拼檔。所有CSV使用UTF-8 BOM（utf-8-sig）以便Excel開啟繁體中文。HTML為完整本機檔案，不需server、部署或外部網路。

- all_results：60×8個鍵位的最新狀態；未選/未跑明確not_selected/not_run，不能把Smoke的其餘題視為已測。成功、錯誤、模型與完整來源內容都保留。
- comparison：60題為列、八架構答案為欄；失敗顯示狀態及錯誤。
- summary：各架構train/validation/test/full四組；有母體42/9/9/60、所選requested、成功/失敗/未完成、完成率、成功覆蓋率、FAQ Accuracy、人工分數與時間。
- review：原題/標準答案/實際回答/attempt與hash，以及空白分數/旗標/備註欄。
- HTML：分組摘要、按FAQ展開八種答案、參考答案、錯配/錯誤、人工評分、設定與provisional。失敗未知的provisional不宣稱False。

### 自動評估

僅A1/A2計算FAQ Top1 Accuracy = 成功題中predicted_faq_id==faq_id的數量 / 成功題數。沒有成功題則空白，不填0。另報success/population（Full為/60）與success/requested，以及(success+failed)/requested完成率，避免只看成功樣本。

不對生成模型自動算答案正確率，不使用文字相似度、LLM judge或BLEU冒充正確。延遲統計成功題的mean/median/P95；P95使用排序後(n-1)*.95線性插值。另保留失敗題實際耗時供排錯，初始化失敗為None。無法取得的token/stage資料為None，不捏造。

計時在CUDA前後synchronize，總耗時包含query encoding、retrieval/rerank、prompt/tokenization、generation與telemetry，不含模型啟動、前置hash檢查或結果fsync。生成時間只包model.generate；A0/A4另記generation_stage含prompt處理。沒有預暖機，第一題包含首次kernel等成本。因而小型Smoke不適合穩健效能結論。

### 人工評分

在新報表的review.csv填score：

| 分數 | 定義 |
|---|---|
| 3 | 正確且完整 |
| 2 | 主要正確，但略有不足 |
| 1 | 部分正確，有重要遺漏 |
| 0 | 明顯錯誤、答非所問或關鍵資訊不正確 |
| N/A | 標準答案不足、過時或無法判斷 |
| 空白 | 尚未評分 |

旗標欄為「事實錯誤、答案不完整、答非所問、過度生成、未依據檢索資料、標準答案本身有問題」，可填1/0/true/false或空白；notes寫原因。不要改問題、標準答案、回答、attempt/hash等識別欄。

匯入會核對run、成功狀態、最新attempt、回答hash與原文；舊失敗重試後的評分不會套到新答案。所有列先驗證成功才提交匯入事件，錯誤不部分匯入。重複評分的新事件取最新；空白不清除舊分數。只有成功回答可評分，失敗由執行統計處理。人工平均只計0–3，N/A另計，空白不當0/3；未評分HTML明確標示。

## GPU PowerShell操作

先同步evaluation目錄及所有既有架構程式。若需調整資源路徑，複製config.json成自己的設定檔並修改，再在新run使用`--config <路徑>`；不要為此覆寫模型或索引。命令必須從GPU專案根目錄執行。

### 1. Preflight

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'

# 只盤點，不載入模型
python -m experiments.evaluation.run preflight --project-root . --run-dir experiments/results/unified_faq/preflight_v1 --allow-unreviewed

# 需要實際確認所有可用架構能初始化時：使用另一個新目錄
python -m experiments.evaluation.run preflight --project-root . --run-dir experiments/results/unified_faq/preflight_load_v1 --allow-unreviewed --load-models
```

查看印出的Report目錄與run/preflight/*.json。任何資源失敗先修正同步問題；若修改程式/套件/資料，後續使用新run。不要將resources_ready_not_loaded看成GPU成功。

### 2. 固定3題Smoke（最多24筆回答）

```powershell
python -m experiments.evaluation.run smoke --project-root . --run-dir experiments/results/unified_faq/smoke_v2 --allow-unreviewed
```

### 3. Full 60題（最多480個最新回答鍵位）

```powershell
python -m experiments.evaluation.run full --project-root . --run-dir experiments/results/unified_faq/full_v1 --allow-unreviewed
```

只有全部問題/資源/設定凍結且使用者決定曝光Test後才執行Full。某架構缺項仍會留下錯誤結果，不能稱八架構已全數成功。

### 4. Resume與單獨重跑

```powershell
# 中斷後，只處理尚無紀錄的題目
python -m experiments.evaluation.run resume --run-dir experiments/results/unified_faq/full_v1

# 只重試A5失敗題；成功題仍跳過，舊attempt保留
python -m experiments.evaluation.run resume --run-dir experiments/results/unified_faq/full_v1 --architectures A5 --retry-failed

# 要重新生成整個架構：使用新run，不覆寫舊成功答案
python -m experiments.evaluation.run full --project-root . --run-dir experiments/results/unified_faq/a5_repeat_v1 --architectures A5 --allow-unreviewed
```

不要刪除鎖檔、手動修JSONL末行或改run.json來繞過版本比對。

### 5. 報表與人工評分匯入

```powershell
python -m experiments.evaluation.run report --run-dir experiments/results/unified_faq/full_v1
```

命令會印出新reports/<snapshot>/。直接在檔案總管雙擊report.html；同目錄review.csv可另存成自己的評分副本。填完後把下例路徑換成實際檔案：

```powershell
python -m experiments.evaluation.run import-review --run-dir experiments/results/unified_faq/full_v1 --csv 'C:\Users\rsy\Desktop\faq_review_filled.csv' --reviewer reviewer_1
```

匯入成功會自動產生含評分的新報表，先前快照完整保留。report/import-review不用模型，可在資源未同步的電腦查看既有run；resume則嚴格要求原環境/路徑/資源不變。

退出碼：0表示該工具流程完成（Preflight預設僅資源盤點）；2表示資源/架構/題目/IO等失敗；130表示中斷。所有結論仍以逐架構Preflight、jobs及逐題status為準。

## 研究邊界與待實測

這輪原題匹配會與canonical問題逐字相同，A1/A2成功率可能偏高；不是未見改寫泛化。候選庫本來包含保留FAQ，RAG Type1/2也可能含其答案，不可宣稱Test知識完全未接觸。A2/A3/A5 smoke更新少，不能代表充分訓練後上限。原CSV ai答覆本身也可能過時或錯誤，人工應可標N/A。

本次Full依使用者要求會公開原先Test的輸出；觀察後不應再把同一組結果當成完全未使用的最終保留測試，或據此微調又宣稱同一Test獨立。A6不在此階段。

尚待GPU實測：Python語法/匯入、HF離線快取完整性、PEFT/Transformers/FAISS版本、生成攔截器與EOS、CUDA同步、A4資料一致性、原子hard-link/Windows鎖、子行程崩潰/中斷/resume、評分CSV Excel round-trip與HTML。未新增或執行自動測試，未填入任何示意實驗成績。

若錯誤請提供完整命令、run/preflight報告、該架構logs與jobs、必要的attempt狀態；不要為排錯自行覆寫原始FAQ、索引或模型。

API設計參考：[HF快取解析與local_files_only](https://huggingface.co/docs/huggingface_hub/package_reference/file_download)、[Python3.10 Windows檔案鎖](https://docs.python.org/3.10/library/msvcrt.html)、[os.link](https://docs.python.org/3.10/library/os.html#os.link)。文件支持介面設計，不取代本機檔案系統或GPU環境實測。子行程log明確使用UTF-8，避免Windows預設編碼造成中文錯誤紀錄亂碼。
