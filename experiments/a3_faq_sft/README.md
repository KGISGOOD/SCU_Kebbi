# A3：FAQ Llama LoRA SFT

本目錄建立獨立的領域微調架構實驗：以 FAQ 問題與標準答案監督小型 LLM，讓它直接生成回答。A1/A2 是問題向量檢索後回傳既存答案；A3 不使用 FAISS、BM25、Dense Retrieval、FAQ matching 或外部文件。只有 42 筆原題訓練樣本，不預設模型能學會足夠的領域知識，也不預設它能回答未見事實。

本次只撰寫程式與靜態審查。**沒有執行 Python、測試、模型載入、訓練或生成；尚未經 GPU 功能驗證。** 不包含正式評估、最佳 checkpoint 選擇、A5 或機器人整合。

## 原本 robot/ LoRA 實作盤點

以下依原始碼判讀，沒有執行舊程式，也沒有修改它們。

| 原檔案 | 實際實作與限制 |
|---|---|
| `robot/train_lora.py` | `meta-llama/Llama-3.2-3B-Instruct`，FP16 base、PEFT LoRA，r=8/alpha=16/dropout=.05，目標 q_proj/v_proj；不是 QLoRA。TRL SFTTrainer/SFTConfig，3 epochs、batch=2、累積=4、LR=2e-4、長度512。未設定 assistant_only_loss，程式明確保持標準 loss，不能宣稱只監督答案。每 epoch 保存 checkpoint；沒有明確 final adapter 匯出與完整性標記。 |
| `robot/prepare_faq_dataset.py` | 讀 script-relative `faq.txt`，逐行 question<TAB>answer，建立 user/assistant messages；隨機保留10筆 validation，seed42，沒有 FAQ ID 或 A2 intent split。輸出 `faq_dataset/`，可能重寫。未加入 system message。 |
| `robot/make_dataset.py` | 工作目錄相對的 faq.txt；錯誤行直接跳過，shuffle 90/10；輸出 instruction/input/output JSONL。這不是 train_lora 實際讀取的 Dataset 格式與路徑。 |
| `robot/test_lora.py` | 固定 checkpoint-36，載入 base+PeftModel；原生 template 轉文字後 tokenizer 未明確設 add_special_tokens=False，有重複特殊 token 風險；三題固定生成，不是互動入口。 |
| `robot/check_lora_effect.py` | 比較 base、enabled、disabled、merged logits；但 disable_adapter() 直接呼叫，沒有使用其 context manager，disabled 區段不能據此證明 adapter 已關閉。也固定 checkpoint-36。 |
| `robot/faq.txt` | 舊 TSV 問答，沒有穩定 P/S ID；與本次兩份 CSV 不同，不能直接沿用其隨機切分或視為 A3 正式來源。可見原文也有錯字或未完成文字。 |
| `robot/requirements.txt` | 固定 transformers 4.42.3，沒有宣告 torch/peft/trl/datasets 的完整 LoRA 版本組合；與原碼的較新 processing_class、SFTConfig API 不可直接假定相容。`experiments/requirements.txt` 亦是舊實驗範圍，不能當目前 GPU 環境鎖檔。 |

沿用的是模型選擇、LoRA q/v 層策略及原生 Chat Template 的概念；不呼叫舊腳本，不沿用舊隨機切分或輸出目錄。

## 架構與檔案

```text
訓練：原始 CSV Q/A + 既有 intent manifest
   → 嚴格來源/ID/格式驗證 → 42/9/9 對話資料
   → 僅 Train 42 筆套原生 Chat Template → Assistant-only labels
   → 凍結 base + PEFT LoRA → 反向傳播/AdamW
   → 新 run/Adapter + tokenizer + 設定/來源 hash/狀態

問答：使用者問題 → 原生 Chat Template
   → 同一 base + A3 Adapter → generate → 僅解碼新 token
```

| 檔案 | 用途 |
|---|---|
| `common.py` | 標準函式庫嚴格 CSV/JSON/JSONL、固定 ID/split、來源 hash、獨占寫入 |
| `prepare_data.py` | 建立獨立 SFT 資料；`--check-only` 僅做資料檢查 |
| `training_data.py` | 原生模板回答邊界檢查、答案 labels、右側 padding collator |
| `training_config.json` | 可覆寫的共同預設 |
| `train.py` | 單裝置 LoRA、token 加權梯度累積、loss/梯度檢查、獨立 checkpoint/status |
| `model_loader.py` | 共用 base/tokenizer/Adapter 載入、版本紀錄、hash/來源驗證；保留未來 A5 重用可能 |
| `chat.py` | 無檢索、單輪互動生成 |
| `__init__.py` | 無載入副作用的 package 識別 |

所有相對資料/模型/輸出路徑以 `--project-root` 為準。模組命令仍須從專案根目錄執行，`--project-root` 不會自動更改 Python import path。程式內沒有使用者 D/C 槽絕對路徑。

## FAQ 來源與切分

來源：

- `畢業專題_爬蟲/type2/資料擴充 - 家長 FAQ 資料集（30 題）.csv`
- `畢業專題_爬蟲/type2/資料擴充 - 高中生 FAQ 資料集（30 題）.csv`

嚴格 UTF-8-sig，欄位為「編號」「問題內容」「ai答覆」。保留原文，不清洗掉會改變問題/答案的文字，不增加 P/S 前綴。兩份各須有30個唯一且完整 ID，不接受空問題/答案、重複標頭、錯誤欄數或損壞 JSON。

既有 `experiments/datasets/faq_finetuning/faq_split_manifest.json` 是唯一意圖切分來源；A3 程式同時檢查其內容符合目前協議，變動須另行研究審查，不會自動重新抽樣。

| split | 意圖 | 筆數 |
|---|---|---:|
| Train | P01–P21、S01–S21 | 42 |
| Validation | P22–P25、S22–S26 | 9 |
| Test | P26–P30、S27–S30 | 9 |

預設由使用者執行 prepare 後，才會新增 `experiments/datasets/faq_sft/`：train.jsonl、validation.jsonl、test.jsonl、dataset_manifest.json。本次沒有執行資料準備，因此沒有預先產生上述資料。每筆含 faq_id、split、source_group、source_file、source_row、review_status 及 messages：

```text
system：你是東吳大學資料科學系的智慧問答助理。請使用繁體中文回答；不知道時請明確說明，不要編造資訊。
user：CSV 問題內容原文
assistant：相同 FAQ ID 的 ai答覆原文
```

目前只使用原題各一筆；不使用 A2 Positive、負樣本、138題評估或30題診斷資料。訓練前重新逐筆與 CSV/manifest 比對；拒絕將其他 JSONL 當作 Train。Validation/Test 只在來源完整性檢查時讀取，不會送入訓練 forward、loss 或 optimizer，也不執行驗證生成或最佳 checkpoint 選擇。

資料輸出全部 `needs_review`；預設阻擋訓練。`--allow-unreviewed` 只是明確允許 provisional smoke，永遠不把資料改成 approved。目前驗證器要求輸出與原始準備格式完全一致，尚未實作正式核准流程；不能直接手改 review_status 後假裝完成審核。正式資料凍結/核准流程屬後續工作。

ID 分離不等於語意/事實完全隔離。P18/P19、P21/S13 等 Train FAQ 有重疊；P06（讀研必要性）與 S22（畢業後讀研比例）、P07（推甄/考試管道）與 S23（推甄去向）等跨切分題目主題接近，答案亦可能共享背景。不得宣稱保留意圖所有知識都是模型未見。基礎 LLM 的預訓練內容也不可控；SFT 不含某答案時，不能預期模型必然學會該事實。本次不變更 split。

CSV 的 ai答覆是資料來源原文，不表示已完成事實查核；例如 S03 原答覆在「會學 SQL」結束，S04 含「專案時作報告」原文，程式會保留，交由後續資料審核。

## Template、Loss 與梯度

SFT 用正確答案作 teacher forcing，以 causal next-token cross entropy 更新 LoRA 權重。LoRA 凍結 base，只訓練目標線性層的低秩增量；不是更新全部基礎模型，也不是檢索答案。

不手寫 Llama 控制 token，也不依賴原生模板是否提供 `{% generation %}` mask。程式先生成 system/user + assistant generation header 的 prefix，再生成完整對話。必須同時通過完整文字 prefix 與 token prefix 的精確比對；不相容便停止，不猜測邊界。模板字串二次 tokenization 明確 `add_special_tokens=False`，避免重複 BOS。

Prefix（system、user、assistant header）labels 全設 `-100`；答案內容及原生模板的回合結束 token 保留 supervision。Causal LM 內部位移 labels，正好讓 assistant header 最後位置預測回答第一個 token。Collator 使用右 padding、attention_mask=0、padding labels=-100；即使 pad=EOS，也依位置遮罩，不能把真正答案結尾一起遮掉。

不 silently truncate：若任何 Train 样本超過 max_length，報出 FAQ ID 與 token 數。每個 optimizer window 以實際 supervised token 數加權 microbatch mean loss；最後不足累積步數的尾批仍正確更新，不漏樣本。沒有用 eval loss 更新模型。

模型保持 train 模式，forward→loss→backward→gradient clip→optimizer。訓練沒有 no_grad/inference_mode；detach 只用在 log scalar，不影響 backward。檢查可訓練參數必須為 LoRA、梯度存在且有限、總 norm 非零、更新後參數有限。FP16 使用 GradScaler；BF16 不縮放；trainable adapter 以 FP32 保存/更新，base 使用所選 dtype。沒有 AMP overflow 靜默跳步後仍宣稱成功的設計，非有限梯度直接中止。

## 預設訓練設定與 GPU

| 參數 | 預設 |
|---|---|
| Base | meta-llama/Llama-3.2-3B-Instruct |
| LoRA | r8、alpha16、dropout0.05、q_proj/v_proj、bias none |
| Device / Precision | cuda / bf16 |
| Epoch / batch / accumulation | 3 / 1 / 4 |
| LR / optimizer | 2e-4 / AdamW、weight_decay0、constant LR |
| 長度 / seed / gradient norm | 512 / 42 / 1.0 |
| Gradient checkpointing | 開啟，use_reentrant=False |

42筆下，預設每 epoch 有42個 microbatch、11次 optimizer step；完整3 epochs 應33次更新。這是排程推算，不是執行結果。seed 固定 Python/torch/CUDA、shuffle 與 dropout 隨機來源；SDPA/套件/硬體仍可能造成差異，不承諾 bitwise reproducibility。

RTX 3090 24GB 採 BF16 base + LoRA + batch1 + checkpointing，避免全參數 FP32/Adam 狀態。3B 級 base 僅16-bit權重約6GB量級，但實際 VRAM 還有 activation、長詞彙 logits、workspace、LoRA/optimizer 與 allocator；未執行，無法可靠給出總峰值或保證24GB一定足夠。CPU 載入亦需要足夠 RAM。

如 OOM：維持 batch1、減少 accumulation 只減少有效 batch 不會顯著減少單次 activation；優先確認其他 GPU 程序、實際 token 長度及執行環境。本程式不截斷答案，因此降低 max-length 前必須先通過 token 檢查。必要時另行設計 QLoRA（BitsAndBytesConfig、prepare_model_for_kbit_training 等），但這次沒有實作或假設 Windows/CUDA bitsandbytes 已可用，不可把本程式標示為 QLoRA。FP16 是顯式替代 precision，顯存通常與 BF16相近。

## 套件與 API 證據

資料準備/結構驗證僅標準函式庫。訓練需要 torch、transformers、peft、packaging；tokenizer 尚需該 checkpoint 使用的 tokenizer/Jinja 等既有依賴。TRL/datasets/accelerate/bitsandbytes 會記錄是否存在，但本實作不直接使用 Trainer、TRL 或 datasets，也不要求為了 A3 重裝它們。

現有 A2 GPU run_config 記錄 Python3.10.1、torch2.13.0+cu126、transformers5.16.1；沒有 PEFT 版本，因此不能宣稱該環境已具備相容 PEFT。執行時記錄所有上述套件版本；缺少套件/API 時清楚失敗，不自動安裝、降版或變更環境。

採用公開的 AutoTokenizer.apply_chat_template、AutoModelForCausalLM.from_pretrained、PEFT LoraConfig/get_peft_model/PeftModel.from_pretrained/save_pretrained、torch.amp.GradScaler/autocast、AdamW。Transformers5選 dtype，4.x選 torch_dtype；這個分支處理載入參數命名，**不等於所有4.x/5.x與PEFT組合已驗證**。舊4.42.3環境及Llama3.2、SDPA、non-reentrant checkpointing、現有5.16.1與未知PEFT組合均須實測，不為繞過錯誤直接放寬相容性聲明。

靜態參考官方文件：

- [Transformers Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)：原生模板、generation prompt、避免重複特殊 token。
- [PEFT LoRA](https://huggingface.co/docs/peft/main/en/package_reference/lora)：低秩設定及目標模組。
- [PEFT PeftModel v0.21.0](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/peft_model)：Adapter 保存/載入；disable_adapter 為 context manager。

這些文件只支持 API 設計；main 文件不等於 GPU 已裝版本，v0.21.0也不是本次指定安裝版本。特定5.16.1文件頁未能透過查詢取得，沒有據此聲稱精確版本跑通。

## GPU 電腦操作（由使用者執行）

請同步整個 `experiments/a3_faq_sft/`，先確認現有環境、完整 Llama 模型快取（含授權可讀權限）、兩份 CSV 及 A2 intent manifest。`--offline` 同時傳 local_files_only 與設定離線環境變數；若快取不完整就停止，不下載替代模型。

### 1. 資料準備

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
python -m experiments.a3_faq_sft.prepare_data --project-root . --data-dir experiments/datasets/faq_sft
```

資料夾必須不存在。若中途中斷或已有資料，不覆寫、不自動刪除；用新的 `--data-dir`，後續命令同步指定。可用 `--manifest` 選擇專案內的 manifest，但內容仍須符合既有42/9/9協議。

### 2. 結構與 Token 檢查

```powershell
# 標準函式庫：不載入 tokenizer / 模型
python -m experiments.a3_faq_sft.prepare_data --project-root . --data-dir experiments/datasets/faq_sft --check-only

# 載入 tokenizer，驗證 Train 的模板邊界、assistant labels 及長度；不載入模型權重
python -m experiments.a3_faq_sft.train --project-root . --data-dir experiments/datasets/faq_sft --base-model meta-llama/Llama-3.2-3B-Instruct --max-length 512 --offline --check-only
```

結構檢查成功不表示語意或事實正確。token檢查不驗證 LoRA、optimizer、GPU顯存或 Adapter 回載。

### 3. 一次最小 Smoke Training

```powershell
python -m experiments.a3_faq_sft.train --project-root . --data-dir experiments/datasets/faq_sft --base-model meta-llama/Llama-3.2-3B-Instruct --device cuda --precision bf16 --epochs 1 --batch-size 1 --gradient-accumulation 4 --learning-rate 0.0002 --max-length 512 --seed 42 --lora-r 8 --lora-alpha 16 --lora-dropout 0.05 --target-modules q_proj v_proj --max-steps 2 --output-dir experiments/models/a3_smoke_v1 --offline --allow-unreviewed
```

此命令最多2次 optimizer 更新（預期使用8筆Train），不是完整 epoch、正式訓練或效果評估，輸出必為 provisional。若輸出已存在，改新的目錄名，不自動續跑或覆蓋；chat 的 adapter 路徑也需同步改。

若 GPU 不支援所選 BF16，顯式改 `--precision fp16`；CPU 僅提供 `--device cpu --precision fp32`，不保證資源或速度適合。`--base-model` 可改為專案相對或絕對完整本機 base 路徑；不存在的本機路徑不會當成其他Hub模型。`--training-data` 若指定，只允許對應已驗證的 data-dir/train.jsonl。

### 4. 載入 Adapter 互動問答

```powershell
python -m experiments.a3_faq_sft.chat --project-root . --base-model meta-llama/Llama-3.2-3B-Instruct --adapter experiments/models/a3_smoke_v1/final_adapter --device cuda --precision bf16 --max-input-tokens 512 --max-new-tokens 256 --seed 42 --offline --allow-unreviewed
```

手動輸入一兩題，確認能生成文字後輸入 exit/quit；也可 Ctrl+C/EOF 結束。預設greedy，若要抽樣加 `--do-sample --temperature 0.7 --top-p 0.9`。這些是生成設定，不是最佳化結論；不輸出準確率或平均延遲。

推論只讀 base、Adapter、tokenizer 與不含FAQ答案的模型 provenance，不開啟CSV/準備好的SFT/檢索候選。每次建立新的 system/user單輪對話；只decode新生成token，沒有多輪記憶或標準答案查表。保存的system prompt與訓練相同。超長輸入拒絕，不默默截斷。

如需回報問題：附完整命令、完整ERROR訊息、該run的training_config.json/started.json/failure.json（如存在）；生成問題另附實際輸入與模型/Adapter路徑。不要將載入成功視為答案正確。

## 輸出、完整性與回載

未指定output時使用新的 `experiments/models/a3_faq_sft/<UTC_timestamp_uuid>/`；指定時目錄必須不存在。

```text
<run>/
  started.json                 # 只表示已開始，不是成功
  training_config.json         # 實際參數、預設、套件/程式/config hash
  dataset_manifest.json        # 固定split、來源與資料版本
  tokenization_summary.json    # Train token數、mask策略、待審ID
  training_log.jsonl           # optimizer steps、loss、FAQ ID、token數、梯度norm
  checkpoints/epoch_001/...    # 完整epoch後的adapter快照（如適用）
  final_adapter/
    adapter_model.safetensors
    adapter_config.json
    tokenizer檔案/原生template
    a3_provenance.json
    a3_complete.json           # 檔案hash清單；最後寫入，沒有則拒載
  completed.json               # 所有保存步驟完成後才寫；明確smoke/full狀態
  failure.json                 # 失敗時盡力記錄，不標成成功
```

Adapter不包含完整 base。Hub base 保存解析出的 commit，chat 使用相同commit；完整本機 base 記錄檔案hash，換電腦可改路徑，但內容必須一致（hash讀取含權重，有IO成本）。Adapter載入前核對完成標記與全部檔案hash；未完成或被修改則拒載。訓練中只存 Adapter/tokenizer，不合併或覆寫原base。checkpoint不含 optimizer/RNG resume state，目前沒有resume功能。

匯出後只做檔案/hash驗證，不偷偷啟動第二次模型推論；completed.json明確標示reload_inference_verified=false。重新載入與生成由上述chat命令確認。這個成功標記僅表示執行流程結束，非模型品質證明。

## 尚待確認與後續界線

- GPU實際PEFT版本及其與Transformers5.16.1相容性，模型授權/完整離線快取、tokenizer模板與precision支援。
- Python語法/匯入/CSV parser、assistant邊界、label shift與padding、尾批梯度累積均未執行驗證。
- 24GB顯存峰值、LoRA梯度、checkpoint保存、Adapter回載及EOS停止行為未實測。
- 原始答案品質、跨切分事實/語意重疊與42筆資料不足仍需人工審核；不宣稱語意無洩漏。
- 尚無正式核准/資料凍結、生成評分、超參數選擇或安全拒答驗證。
- 未來A5可重用model_loader中的同一base、Adapter與生成設定，再單獨加入外部context；本次不實作A5。

完成本次Smoke後先回報錯誤或狀態，再決定後續工作；不要直接開始正式模型比較或60題評估。
