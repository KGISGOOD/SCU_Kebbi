# SD2-4 Prompt Lookup（Milestone A）

本輪完成程式與靜態資料流檢查，尚未執行 Python／GPU 驗證。不需要整合版 INT 才能執行。
舊 all 仍只執行原六版本，sd0／sd2_1／sd2_2／sd2_3 介面不變。

## 真實解碼路徑

`--mode sd2_4` 執行 SD0 加 SD2-4-k2/k4/k6/k8，沿用原七題。每題只做一次原 A0 Top-5
檢索與 Prompt/tokenization，各模式取得相同 input IDs；相同 Target、greedy、生成長度、
裝置、精度、seed。各模式暖機、多次重複與交錯順序沿用既有 benchmark。

依 Transformers **5.16.1** 官方版本原始碼：

- [GenerationConfig](https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/generation/configuration_utils.py)：
  `prompt_lookup_num_tokens=k` 選擇 assisted_generation；`max_matching_ngram_size=n` 獨立控制匹配長度。
- [候選生成器](https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/generation/candidate_generator.py)：
  `PromptLookupCandidateGenerator` 從目前輸入前綴（含已生成 Token）尋找相同 n-gram 的後續片段。
- [Target 驗證](https://raw.githubusercontent.com/huggingface/transformers/refs/tags/v5.16.1/src/transformers/generation/utils.py)：
  `_get_candidate_generator` 選擇上述類別；`_assisted_decoding` 用 Target forward 驗證候選。

不載入、不要求 1B 目錄存在，也不傳 assistant_model。現有 SD 設定仍保留 draft_model 欄位，
在 sd2_4 中只當未使用佔位字串。無候選是合法結果；不是靜默切回另一條生成路徑。

`prompt_lookup.py` 僅在單次請求暫時觀察模型實例的候選工廠與該次生成器；檢查真實類別、k/n，
記錄實際候選呼叫次數、提案 Token 總數與空提案次數。沒有重寫提案或 Target 接受／拒絕規則。
若未呼叫真正生成器或 Target forward，失敗。這些輕量觀測的成本包含在生成計時。
提案數不是接受數：接受率與接受 Token 數仍 unavailable/null。

KV：use_cache=True；呼叫設定及暫時 Target 預設的 selector/cache_config 清空，HF 管理 DynamicCache。
不傳入前次請求的 past，finally 還原 Target 原設定和觀察方法。舊 Draft KV 政策不變。
此方法只供目前順序實驗，不可併發操作同一 Target 實例。

## GPU 操作

使用既有 SD 的 `config.gpu.local.json`，確認 model／bge 為已下載的本機路徑，knowledge_index
是文件索引 `type1_type2_bge_m3_v1`。不修改或覆寫設定；沒有本機設定時才從 config.example.json
複製並手動填入路徑。可加 `max_matching_ngram_size`，也可用 CLI 指定，預設 2。

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
$sdLookupRun = 'experiments/results/algorithms/sd24_' + [guid]::NewGuid().ToString('N')
python -m experiments.algorithms.speculative_decoding.benchmark --project-root . --config experiments/algorithms/speculative_decoding/config.gpu.local.json --mode sd2_4 --max-matching-ngram-size 2 --repeats 1 --warmups 1 --run-dir $sdLookupRun --allow-unreviewed
```

七題 × 五版本 ×（1 暖機＋1 正式）=70 筆生成。確認功能後以全新 Run、`--repeats 3`
量測則為 140 筆。不可把功能驗證當正式速度結論。

驗收看 identity 的 draft=null、residency=target+bge；各 lookup 列 generation_mode=assisted_generation、
prompt_lookup_statistics.generator_created=true/candidate_calls>0、Target forward>0，並檢查 k/n。
所有輸出逐 Token 對照同題同 repeat 的 SD0。Token mismatch／來源變更／生成失敗不產生 completed。

輸出維持 run.json、identity.json、Prompt snapshots、measurements.raw.jsonl、results.jsonl、
summary.json、comparison.csv、completed.json／failed.json。CSV 新增兩個可空欄位：
prompt_lookup_candidate_limit 與 max_matching_ngram_size；原 Draft 列不受影響。

Generation-only 是同步的 generate 時間；SD E2E 仍是「共享前處理＋該模式執行」的受控合成時間，
不能和 INT 真實獨立請求的 E2E 混用。RTX3090 顯存、HF 私有觀察介面、Token 一致性、實際速度
都待 GPU 驗證，不保證 Prompt Lookup 必然加速。缺少片段時如實記錄；沒有下載、fallback 或套件修改。
