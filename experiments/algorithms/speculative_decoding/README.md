# A0 Speculative Decoding：SD0／SD2-1／SD2-2

狀態：程式已撰寫並靜態審查，未執行 Python、模型、測試或安裝。本文件的預期行為不是
GPU 通過紀錄。既有八架構、Semantic Cache、索引與結果均不修改。不使用 Ollama。

## 範圍與流程

1. 驗證本機路徑、指定套件 API、情境與全新輸出目錄。
2. 重用既有 `A0Adapter` 的本機 BGE、文件 FAISS、Target 模型；不啟用答案 Cache。
3. assisted 模式另外載入本機 1B，明確驗證 Tokenizer 與模型 Token 空間。
4. 每題只執行一次 `RAGPipeline.retrieve()`，保存完整 Top-5 與排序。
5. 使用既有 A0 `build_prompt/create_context_from_documents`，只 tokenize 一次，
   各模式取得同一份 input IDs／attention mask 的副本。保存 Prompt 與輸入 Token。
6. 每題各模式暖機後，多次重複、交錯生成，保存原始紀錄。
7. 每個 assisted 結果逐 Token 對照相同情境／repeat／warmup 的 SD0，再彙整。
8. 確認來源沒有改變、沒有 Token mismatch，才建立 completed.json。

原 A0 的 Prompt 與 tokenizer 特殊 Token 處理保持原樣。沒有偷偷改 chat template 或修正
原模板的特殊 Token 政策。本機 Target 與 Draft config 也需符合標準 Llama 3.2 3B
（hidden_size=3072、28 layers）／1B（2048、16 layers）；不同架構須另做相容性審查。

## 實驗版本

| CLI | 實際量測 |
|---|---|
| `--mode sd0` | SD0；不載入 Draft，設定檔的 Draft 佔位路徑可以不存在 |
| `--mode sd2_1` | SD0 ＋ SD2-1，長度取 standard_draft_tokens，範本為 4 |
| `--mode sd2_2` | SD0 ＋ SD2-2-k2/k4/k6/k8 |
| `--mode all` | 上述六個版本；SD2-1 與 sweep k4 刻意保留為不同量測 |
| `--mode sd2_3` | 預留介面，明確報未實作；不會暗中使用固定策略 |

SD0 的輸出只來自 Target。為了交錯比較，all／sd2_1／sd2_2 中 Draft 會持續留在顯存，
包括 SD0 量測時；SD0 不會呼叫 Draft。記錄 residency，不能把這些數字稱為真正僅載入
Target 的記憶體基準。若要量測純 Target 常駐條件，另跑獨立 `--mode sd0` Run；不同
residency 的結果不可當成只差解碼演算法的嚴格比較。BGE 也常駐，且有記錄。

## API 與相容性

此版僅接受先前 GPU 回報的 Transformers **5.16.1**。這是官方原始碼靜態審查的範圍，
不是已執行驗證的套件認證；其他版本直接報錯，不能只放寬版本字串。

使用原生 `target.generate(..., assistant_model=draft)`；Target 的 GenerationConfig
必須選到 assisted_generation。記錄 Target／Draft 的實際 forward 次數，assisted
未呼叫任一模型即失敗。不單獨輸出 Draft 答案，也不自製接受／拒絕邏輯。

候選長度在 **Draft generation_config** 設定：

```text
num_assistant_tokens = k
num_assistant_tokens_schedule = "constant"
assistant_confidence_threshold = 0.0
```

每次生成前從原始 Draft 設定複製，清除前次策略狀態；生成後記錄並檢查這三個值。
0.0 用來停用 confidence 提前停止與自適應；k 是每輪候選上限，不是實際接受量。
EOS、剩餘輸出預算仍可能縮短候選。接受率／接受 Token 數不做推估，標 unavailable。
forward 次數只是路徑診斷，不是接受率。

Tokenizer 必須通過 fast backend、詞彙→ID、added/special token、特殊 ID 及模型
embedding/head 尺寸檢查。兩個 chat_template 都記錄，但只使用 Target 的 A0 Prompt。
不支援不同 Tokenizer 轉換；不相容直接失敗。即使通過這些檢查，也必須再看實際逐 Token
對照，不能宣稱所有輸入都 lossless。

參考的官方版本原始碼：

- https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/generation/utils.py
- https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/generation/candidate_generator.py
- https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/generation/configuration_utils.py

## 共同生成設定

- greedy、num_beams=1、相同 max_new_tokens、repetition_penalty=1.0。
- EOS／PAD 與現有 A0 生成呼叫一樣採 Target tokenizer 的值。
- 每次重新複製相同 Target GenerationConfig，保留來源模型其他設定並保存完整有效設定。
- 所有版本 use_cache=True、dynamic cache；不跨請求重用 KV。
- 所有版本停用自動 compile，讓 forward 路徑與計數可觀察；模型不是量化或 offload。
- 依 repeat 取 seed+repeat，各版本相同；seed 不等於數值確定性保證。
- 完整輸入超過設定上限或模型 context window 就失敗；不截斷 Top-5。此階段上限
  不得超過 A0 既有 2048 input-token 政策，輸入＋輸出亦不得超過 Target／Draft 上限。

這是沿用 A0 的受控 greedy 比較，不等於以 A0 預設 sampling=True 做比較。中性
temperature/top_p/top_k 不作抽樣使用，並明確記錄。若未來變更生成政策，必須另開 Run。

## 計時與顯存

- Generation-only：同步 → Target.generate（含 Draft 草擬／Target 驗證）→ 同步。
  不含 tokenization、decode、寫檔與模型載入；含輕量 forward 計數 hook 的成本。
- mode_execution：本次模式的設定、輸入 clone、記憶體觀測、生成、decode 等處理。
- end_to_end：**同題共享前處理實測值＋本次 mode_execution**。這是受控合成時間，
  不是每個模式／repeat 都獨立重做 RAG 的 wall-clock 實測。
- retrieval_seconds 只實際測一次。不要把各列重複引用的這個欄位加總成總檢索成本。
  summary/completed 另存 unique_shared_preparation_seconds；completed 也有整次 wall time。
- CUDA 前後 synchronize，每次 reset peak stats，保存 allocated/reserved peak、
  生成前 allocated、GPU free/total。CPU 的 CUDA 欄位為 null。
- A0 載入（含來源 hashing）與 Draft 模型／Tokenizer 載入時間獨立記錄。
  Draft 來源 hashing 不在 Draft 模型載入時間內，但包含在整次 wall time。
- 從 seed 決定初始模式順序，按 scenario/repeat 旋轉；可降低固定排序偏差，有限次數
  不保證完美平衡。每個模式／情境皆有 warmup，暖機結果保留但排除效能統計。

生成加速比 = 同題同 repeat SD0 generation_seconds / 本模式 generation_seconds。
端到端加速比 = 配對 SD0 受控 end_to_end / 本模式受控 end_to_end，不混用分母。
Token mismatch 或 SD0 自身不具重複性時，該列失敗且加速比為 null。時間分布仍保存供
診斷，不能把失敗模式的速度當有效研究成果。Tokens/sec 計算包含實際輸出的特殊 Token。
Summary 的 aggregate TPS 為總生成 tokens／總生成時間，不是每題 TPS 的平均。

RTX 3090 必須同時容納 Target、BGE、Draft 與暫存 KV。精度由同一設定控制；不保證
一定放得下。OOM 會保存錯誤並停止，沒有自動改量化／精度／offload。調整輸入／輸出
預算或 dtype 後應另開 Run，且每個模式保持一致。不得只縮短某個模式的內容。

## GPU 未來操作

開發機不要執行以下指令。本輪不需要現在開始 GPU 實驗。

GPU 專案根目錄 `C:\Users\rsy\Desktop\SCU_Kebbi`：

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
if (Test-Path experiments/algorithms/speculative_decoding/config.gpu.local.json) { throw 'Edit existing local config; do not overwrite.' }
Copy-Item experiments/algorithms/speculative_decoding/config.example.json experiments/algorithms/speculative_decoding/config.gpu.local.json
notepad experiments/algorithms/speculative_decoding/config.gpu.local.json
```

填入現有 Target、BGE 與 Draft 本機目錄；不假設 1B 已下載。目錄／權重缺失直接報錯，
不能讓工具下載。knowledge_index 指向 `type1_type2_bge_m3_v1`，不是 FAQ60 索引。
現有 ignore 規則已涵蓋 config*.local.json；本工具不寫入該檔。

先驗證 SD0（不需要 Draft 已存在）：

```powershell
$sdRun = 'experiments/results/algorithms/sd0_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config experiments/algorithms/speculative_decoding/config.gpu.local.json --mode sd0 --repeats 1 --warmups 1 --run-dir $sdRun --allow-unreviewed
```

再做完整比較（7 題、6 版本、每版 1 暖機＋3 正式重複，共 168 筆生成）：

```powershell
$sdRun = 'experiments/results/algorithms/sd_all_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config experiments/algorithms/speculative_decoding/config.gpu.local.json --mode all --repeats 3 --warmups 1 --run-dir $sdRun --allow-unreviewed
```

後續可增加 --repeats 5 或更多；增加重複不能替代更多代表性 Prompt 或輸出正確性審查。
每次使用新目錄，程式不 resume／覆寫。不建立 ZIP。

## 情境與輸出

7 筆 needs_review：短答、長答、NVIDIA DLI 專有名詞、條列回答、P07 高檢索依賴、
較長口語輸入、系務外天氣問題。這些是人工檢查候選；不預設回答長度，也不預設每個
問題一定在知識庫可回答。prompt_*.json 保存實際輸入長度，應據實分層分析。

- run.json：有效設定、題序來源 hash、模式及用途。
- identity.json：模型／Tokenizer／Prompt／套件／原始碼／索引／硬體與載入紀錄。
- scenarios.snapshot.json、prompt_*.json：原情境、一次取得的 Top-5、完整 Prompt、Token IDs。
- measurements.raw.jsonl：每次真正生成後立即寫入，狀態為待比較；保留執行順序及暖機。
- results.jsonl：Token 比對後的逐題狀態、時間、輸出、ID、記憶體與加速比。
- summary.json：非暖機統計、失敗數、相容性邊界與時間定義。
- comparison.csv：Excel 可讀的 UTF-8 BOM 表格；完整 Token IDs 在 JSONL。
- completed.json：所有比較與來源驗證通過才建立，不表示答案事實正確或所有輸入 lossless。
- failed.json：失敗階段、當前量測、exception、traceback；包含 OOM／中斷。

若生成中途失敗，保留 raw，不捏造其餘紀錄。若已生成但 Token 不一致，先輸出可分析
的 results／summary／CSV，再以錯誤結束，不建立 completed。若結束時來源變更，先以
failed.json 結束，raw 仍保留，不產生成功摘要；是否完成以 completed.json 為準。

## 後續邊界

SD2-3 只預留明確拒絕執行的介面；未加入 Adaptive 邏輯。未定義 SD2-4 的研究方法，
不自行把 Medusa／EAGLE 等技術冠上該版本。接受統計、Adaptive、其他 Draft 或資源配置
比較都需要另訂研究問題、版本 API 審查與題庫；本階段不開發 Attention Topic。
