# A5：RAG + Fine-tuned LLM

A5 結合 A0 的外部文件 Dense Retrieval 與 A3 的 FAQ LoRA Adapter，建立後續可研究「領域微調 + 外部知識」的問答架構。本次沒有訓練模型、執行問答或做成效比較，也沒有接機器人硬體。

**本次沒有執行 Python、Node、模型、測試或安裝套件；所有功能僅經靜態程式碼審查，GPU 整合仍待使用者確認。**

## 流程與技術關係

```text
使用者問題
  → 原始 BAAI/bge-m3（L2 normalized query embedding）
  → A0 已存在的 FAISS IndexFlatIP：Dense Top-5
  → 按檢索排名格式化完整文件與來源
  → A0 的回答規則 + 參考資料/問題 User 區塊
  → A3 保存 tokenizer 的原生 Llama Chat Template（只套一次）
  → Llama-3.2-3B-Instruct + 同一 A3 LoRA Adapter
  → generate → 只解碼新生成 token → 顯示答案
```

BGE-M3 將問題轉成語意向量；FAISS 對既存文件向量進行精確內積檢索。RAG 將檢索文字提供給 LLM 作為外部參考；LoRA 以基礎模型之外的低秩權重增量改變模型生成行為。A5 不重新訓練這些元件，也不是將檢索結果直接當答案。

A0 = Dense RAG + 未微調 LLM；A3 = 微調 LLM、沒有檢索；A5 = 同一 Dense RAG + A3 Adapter。A5 不使用 A1/A2 FAQ ID matching，不使用 A2 微調 Embedding、不使用 BM25、Reranker、Router 或答案查表。

## 原始碼盤點與重用

| 既有元件 | 實際行為與 A5 處理 |
|---|---|
| A0 `config.py` | Type1/Type2、chunk512/overlap128（字元）、BGE-M3 normalize=True、IndexFlatIP、Top5。全域 Settings 的初始化會 mkdir 索引/結果/資料目錄，A5 不匯入它。 |
| A0 `build_index.py` | loader 讀 Type1/2 TXT/CSV，清理後 RecursiveCharacterTextSplitter 切512/128，再encode、build/save FAISS與version。既有索引無效時可能重建寫入，A5不呼叫此檔。 |
| `shared/data_loader.py` | TXT/CSV解碼、格式化、內容去重與來源metadata；家長FAQ包含Q/A，其他CSV依欄位串接。A5重用已建索引的內容，不重新讀取/清理/切分來源。 |
| `shared/embeddings.py` | BGEM3Embedder→SentenceTransformer，max_seq_length512，normalize_embeddings=True。A5直接重用原始BGE-M3 Query encoding。 |
| `shared/vector_store.py` | FAISSVectorStore.load讀index.faiss與本專案dict格式index.pkl；similarity_search會再正規化Query。constructor有mkdir(exist_ok=True)，A5先確認目錄與兩檔已存在才建物件。A5不呼叫build/save/_save_version_info。 |
| `shared/retrieval.py` | Retriever以embed_query產生Query向量，轉NumPy，再呼叫store.similarity_search。A5直接重用，固定Top5、不改排序或重新排名。 |
| A0 `pipeline.py` | retrieve→create_context_from_documents→build_prompt→LLMLoader.generate。factory即使缺索引仍會繼續載入LLM；A5遇缺索引先停止，不延用這個空庫流程。 |
| A0 `prompts.py` | system/user指令與文件來源格式；build_prompt手寫Llama特殊token，而且import會帶入上述Settings副作用。A5僅保留小型文字相容層，不套用其完整serialized prompt。 |
| `shared/llm_loader.py` | AutoModelForCausalLM原始base載入，不支援Adapter；CUDA預設FP16，輸入上限2048且默默截斷。量化flag只檢查衝突，未實際傳進模型loader。A5不使用此loader，也不假稱它支援LoRA。 |
| A3 `training_config.json/train.py/training_data.py` | 預設Llama3.2-3B、q_proj/v_proj LoRA；原生template、Assistant-only labels，保存adapter/tokenizer/provenance/hash完成標記。A5不執行訓練，也不讀Train/Validation/Test題庫。 |
| A3 `model_loader.py` | verify_adapter + load_adapter_for_chat，使用PeftModel.from_pretrained；比對base來源、Hub commit或本機hash，保留完整性與provisional gate。A5直接重用，不複製整套loader。 |
| A3 `chat.py` | 原生template一次，tokenizer add_special_tokens=False，generate後僅decode新增token；沿用base generation_config的EOS/回合結束設定。A5採相同生成輸入方式但加入RAG context。 |

## 本目錄檔案

| 檔案 | 用途 |
|---|---|
| `__init__.py` | Package識別，沒有啟動副作用 |
| `config.py` | 無檔案IO的A0對齊常數與生成設定 |
| `prompts.py` | A0 baseline_basic_v1指令/文件格式的最小相容層，輸出原生roles |
| `pipeline.py` | 索引檢查、共用Retriever、A3 loader、完整Top5 Prompt與生成 |
| `chat.py` | 持續單輪CLI、參數、來源摘要及清楚錯誤 |
| `README.md` | 盤點、使用方式、可比性及未驗證限制 |

只新增本目錄檔案；未修改 A0/A1/A2/A3/A4、原始資料、索引、模型或既有結果。Pipeline/CLI 不輸出結果檔，不新增評估指標或自動測試。

## 索引來源與完整性

預設只讀：

```text
experiments/vector_stores/type1_type2_bge_m3_v1/
  index.faiss
  index.pkl
  version.json  # 若既有版本有保存
```

這是A0原始BGE-M3索引，不是FAQ 60題候選庫。`--index-dir`可指向另一份相同A0索引快照，不能拿A1/A2索引替代。A5不呼叫chunker，不建立新向量知識庫、不重新編碼文件。

啟動時先檢查兩個索引檔存在並記錄記憶體中的SHA-256，再load；檢查IndexFlatIP/維度、至少5筆、完整唯一的index→docstore映射、文件非空且source_group包含Type1及Type2。載入後再比對hash，避免載入中被換檔。index.pkl只應使用本專案可信的既有檔案，格式與LangChain標準FAISS pickle不同。

若version.json存在，嚴格檢查BGE模型名、維度、512/128、L2、IndexFlatIP、chunk數、Type1/2來源名稱及載入文件fingerprint。不比較寫死的原電腦絕對路徑，以容許同一索引跨電腦同步。

**本次本機只找到index.faiss/index.pkl，沒有version.json。** 此時程式會顯示warning並標記provisional；只能確認索引結構與文件metadata，不能證明歷史Embedding權重/正規化/512與128來源設定或原始資料仍完全一致。即使version存在，也不重新切分目前原始資料來宣稱資料版本吻合；必須保留原始建置紀錄供後續比較。A5不會自動補寫version或將設定常數當作建置證據。

缺索引時會在模型載入前報錯，優先從GPU專案/備份恢復同一索引。若確定從未建立A0索引，可另行閱讀A0建置說明，由使用者決定在GPU電腦使用原有入口：

```powershell
# 僅供「確實沒有任何既有A0索引」時另行操作；不是A5啟動步驟。
# 此程式會寫入A0索引；有現成索引時不要為了A5執行它。
python -m experiments.baselines.rag_baseline.build_index
```

不要使用force-rebuild來處理同步問題。A5遇到索引不相容只報錯，不修改它。

## Base與Adapter載入

預設Base：`meta-llama/Llama-3.2-3B-Instruct`。

預設Adapter：`experiments/models/a3_smoke_v1/final_adapter`。依使用者提供資訊，此為兩步LoRA smoke；**本機目前沒有該Adapter目錄，無法讀取確認其實際內容**，不重建或重新訓練它。

請從已成功訓練的GPU目錄使用同一Adapter；需要完整保存：adapter_model.safetensors、adapter_config.json、tokenizer/template、a3_provenance.json、a3_complete.json及完成清單中的其他檔案。只拷貝權重不足以通過A3 loader。

load_adapter_for_chat會保留：

- SHA-256完整性完成標記；不接受中斷保存或被改過的Adapter。
- Base來源一致性：Hub使用訓練時解析commit；本機完整base要求檔案hash一致，可以更換路徑。
- vocabulary大小/tokenizer大小檢查。
- provisional模型要求明確`--allow-unreviewed`，不會將其改成正式模型。
- 實際PeftModel載入；沒有Adapter時不改用Base-only，也不merge/覆寫base。

A5對同一Adapter先作前置檢查，再由原loader檢查並載入。所有模型只在程序啟動時載入一次，連續提問不重載。

## Prompt與Chat Template

A5 `prompts.py`保留A0的system指令、User的「參考資料／問題／請回答」文字及文件來源分隔格式；順序為Retriever回傳排名，不另排序。固定參考日期也沿用2026-10-19，這是實驗prompt文字，不是目前日期或資料新鮮度保證。

其必要差異是：不重用A0手寫特殊token的build_prompt，而先建立system/user roles，交由保存tokenizer.apply_chat_template產生一次格式，再tokenize(add_special_tokens=False)。沒有把完整Chat Template字串再包進另一份template，不自行新增BOS/EOT控制token。此方式沿用A3原生模板介面，但A5的system規則為A0 RAG規則，和A3 SFT較短的system不完全相同，需視為比較限制。

所有完整Top5文件page_content都放入User context；沒有根據FAQ ID或人工gold選答案。若Type1/2本來包含FAQ Q/A，它們可被正常檢索進context，這是外部知識，不代表Adapter曾訓練這些意圖。後續A3/A5比較必須區分「微調未見」與「RAG可讀取」，不能把A5使用保留FAQ的外部文件稱為純參數泛化。

RAG規則要求資料不足時說明無法確認，但不是已驗證的拒答保證。若完全無有效Top5則清楚報錯，不退回無context生成；正常低分Top5沒有自訂閾值，模型仍須依資料判斷，分數不是答案正確機率。

## 生成設定與可比性

| 項目 | A0實際程式 | A5 |
|---|---|---|
| Dense embedding | 原始BGE-M3、normalize=True | 同一wrapper/原始模型 |
| 文件 | A0 FAISS Top5 | 同一索引/共用Retriever，維持排名 |
| 輸入長度 | tokenize max_length2048，truncation=True | 預設2048，超長報錯，不截掉問題或任一文件 |
| max_new_tokens | 512 | 512，可調 |
| do_sample | True | True；可--no-do-sample |
| temperature / top_p | .7 / .9 | .7 / .9，可調 |
| repetition_penalty | 1.0 | 1.0，可調 |
| LLM precision | CUDA預設FP16 | 預設FP16，可bf16/fp32 |
| pad | tokenizer.pad，缺少時EOS | 保存的A3 tokenizer.pad |
| eos | 顯式tokenizer.eos_token_id | 預設沿用A3/base generation_config；可--eos-policy tokenizer改成A0式覆寫 |
| seed | config宣告42，但A0 chat/pipeline未看到套用seed | A5明確設Python/torch/CUDA seed42 |

`--eos-policy model`保留模型原有EOS/回合結束停止列表，啟動畫面顯示實際ID；`tokenizer`只使用tokenizer EOS，可能與模型回合結束列表不同。沒有手寫特殊ID。這是A0/A3不能假定完全一致的技術差異；正式比較需統一並記錄，不在本次改A0。

當完整Top5超過輸入上限時，可在模型容量範圍內明確增大`--max-input-tokens`（例如4096），但這也改變與A0截斷行為的可比性；程式不會自動採用另一個長度、不會只用前幾篇假稱完整Top5。過長檢索Query仍受A0 BGE wrapper原有512-token限制。

同樣的模型名稱不代表不同電腦BGE快取revision完全相同；缺version尤其無法證明索引與目前Query encoder權重一致。seed固定也不保證跨套件/硬體bitwise一致。尚未進行任何公平比較驗證。

## GPU啟動（僅供使用者之後執行）

同步本A5目錄及原A3程式即可；不要為A5重訓Adapter或重新建索引。保留目前GPU上有效的模型、完整Adapter及索引。

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'

# 唯讀確認：前四項應為True；version.json若不存在會警告並標provisional
Test-Path 'experiments/vector_stores/type1_type2_bge_m3_v1/index.faiss'
Test-Path 'experiments/vector_stores/type1_type2_bge_m3_v1/index.pkl'
Test-Path 'experiments/models/a3_smoke_v1/final_adapter/a3_complete.json'
Test-Path 'experiments/models/a3_smoke_v1/final_adapter/adapter_model.safetensors'
Test-Path 'experiments/vector_stores/type1_type2_bge_m3_v1/version.json'

python -m experiments.a5_rag_finetuned_llm.chat --project-root . --base-model meta-llama/Llama-3.2-3B-Instruct --adapter experiments/models/a3_smoke_v1/final_adapter --index-dir experiments/vector_stores/type1_type2_bge_m3_v1 --device cuda --precision fp16 --offline --allow-unreviewed --show-sources
```

手動問一兩題即可，例如：

- 東吳資料科學系主要培養學生哪些能力？
- 東吳資料科學系有哪些人工智慧相關課程？

查看啟動時Base/Adapter/FAISS/provisional/生成設定，以及回答後的Top5來源、document_id、分數及200字摘要。摘要只影響終端機顯示，Prompt使用完整文件。顯示來源代表retrieval有回傳文件，不證明模型確實依據context或答案正確。輸入exit/quit、Ctrl+C或EOF結束；空白提示重輸；不保存歷史，每題單輪。

需要greedy時加`--no-do-sample`；需要對齊A0 EOS覆寫時加`--eos-policy tokenizer`。RTX3090的實際VRAM尚未測量；Llama+Adapter與BGE同時駐留，若顯存不足可明確加`--embedding-device cpu`，只搬Query encoder、不改檢索內容。CPU LLM需要`--device cpu --precision fp32`，並預留足夠RAM。不得將載入錯誤改成Base-only。

需要本機完整Base時，`--base-model`指定該路徑，但必須符合A3訓練保存的來源身份；不能把Hub來源Adapter擅自換成未核對的本機模型。所有相對路徑依project-root，模組命令仍須在專案根目錄執行。

## 依賴、錯誤與待確認

重用既有torch、transformers、peft、sentence-transformers、numpy、faiss、langchain-core、packaging等依賴；不新增需安裝的第三方功能。離線模式在匯入模型套件前設定HF離線變數，A3載入亦傳local_files_only；BGE依共用wrapper使用既有完整快取。缺檔/快取不完整就報錯，不自動改模型。實際PEFT/Transformers/FAISS相容性仍待GPU確認。

尚未驗證：

- Python語法/模組匯入及實際套件API。
- 本機缺少的A3 smoke Adapter實際內容/完整性/回載。
- FAISS二進位格式、mapping、維度、metadata與BGE權重是否吻合。
- 缺version索引的建置參數、歷史來源與正規化，不能由設定常數補證。
- 原生template、EOS停止、完整Top5 token長度、3090記憶體與生成行為。
- 模型是否遵循外部context或正確處理資料不足，需之後人工確認。

未加入LoRA訓練、正式評估、分數統計、平均延遲、最佳checkpoint、FAQ matching、BM25、reranker、cache、router或機器人。

錯誤請回傳完整命令、完整錯誤訊息、啟動時的Base/Adapter/index路徑與provisional；若是提問問題，另附問題及--show-sources輸出。程式沒有自動建立錯誤/結果檔案，不要為排除錯誤刪除模型或索引。

## 文獻與本實作範圍

1. Lewis et al., NeurIPS2020：[Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)。提供結合參數知識與外部檢索的概念；本實作是固定Dense檢索後prompt注入，不重現其joint training或RAG-token/sequence機制。
2. Hu et al., ICLR2022：[LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)。Adapter的低秩適應依據；A5只載入A3現有權重，不在本次訓練。
3. Wei et al., ICLR2022：[Finetuned Language Models Are Zero-Shot Learners](https://arxiv.org/abs/2109.01652)。Instruction tuning概念參考；其多任務大規模設定不等於本專案42筆FAQ的效果保證。
4. Chen et al., Findings of ACL2024：[M3-Embedding](https://aclanthology.org/2024.findings-acl.137/)。BGE-M3背景；本專案只使用dense embedding，沒有稀疏或multi-vector組合。

API參考：[Transformers Chat Templates](https://huggingface.co/docs/transformers/main/en/chat_templating)、[PEFT PeftModel](https://huggingface.co/docs/peft/v0.21.0/en/package_reference/peft_model)。官方文件支持介面設計，不能代替使用者環境的實際驗證。
