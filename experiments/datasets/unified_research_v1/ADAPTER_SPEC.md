# 共用資料與既有 Benchmark 的對接契約

本輪只新增資料層、reader 和規格，不修改八架構、SC、SD、TI、INT 的主程式。
不能把新 JSONL 直接塞給原本硬性要求 FAQ60 的 Worker，宣稱已整合完成。

## 共通規則

- question_id 是評估事件 ID；intent_id 是單一 FAQ 意圖或 null，不能互相冒用。
- 所有架構只接收 question／必要的真實歷史。reference_answer、expected_behavior、gold、
  題型標籤、expected_reuse 都留在 scorer 端，不寫進檢索查詢或生成 Prompt。
- 多意圖 D、模糊 G、拒答／時效題，未經人工定義單一合理 gold 時不做強迫的 FAQ Top-1 正確率。
  A1/A2 仍可執行 Top-1 檢索；錯誤不應由評估 ID 直接查表取答案掩蓋。
- FAQ Matching 的 generated tokens／tokens-per-second／generation latency 是 N/A，不是 0。
  Cache Hit 本次生成 tokens=0，和架構本身不生成 LLM Token 的 N/A 不同。
- A2/A3/A5 依 supporting_intent_ids、原 manifest、實際 checkpoint receipt 分層報告訓練重疊；
  候選知識庫含 FAQ 不等於查詢被訓練；新問法也不等於新意圖。
- 每次正式 Run 記錄題庫檔案／split／已凍結版本 Hash、模型、索引、Prompt 與生成設定，
  不沿用舊60題的 Run 身分，不覆寫既有紀錄。

## 八架構 QA

目前 experiments/evaluation/common.py 的 dataset() 從 A3 snapshot 讀原 CSV，強制60題。
未來新增明確的 research dataset adapter：以 as_qa_request() 產生 question_id/question，
Worker 結果主鍵改用 evaluation_question_id，另存 gold_faq_id（可 null）及 gold_set（多答案需人工審核）。
保留原 FAQ60 模式。先改 ID／scorer 與 integrity manifest，再逐架構 Smoke；不能只改資料路徑。
標準答案與來源需另供人工／scorer，而不是在 wrapper 中當模型輸入。

QA 評估至少區分來源支持、正確、完整、拒答／釐清、時效與注入防護。QA 品質和延遲獨立統計；
取最佳速度前先訂品質非劣性準則。多意圖題對 A1/A2 的單答案限制是架構能力差異，不能宣稱完全公平。

## SD／Prompt Lookup

reader.as_sd_scenario() 提供 scenario_id/query/category/review_status 與保留 research_split。
reviewed 才能映射成既有 CLI 的 approved；needs_review 不升級。未來先用只讀 exporter 在**新目錄**
輸出 SD scenario 檔與來源 manifest，再用既有 --scenarios。不是本輪自動執行／生成成果。
同題只建一次完整 Top-5 Prompt，保持 Target tokenizer／輸入 IDs／生成條件完全一致；逐 Token
對照 SD0。題型及實際 input/output tokens 分層，不能以預期答案長度代替實際 Token 數。
不要把共享前處理的 controlled E2E 和 INT 獨立 online E2E 混算。

## SC／INT 查詢序列

使用 cache_workloads 的 query_event_id 當 scenario_id，question_text→question；保留 question_id
作資料關聯。把 reference_question_id 解成**同 workload 先前的事件／實際 cache origin**，
不能僅靠 FAQ 相同就當可重用。reference label 都待審核；僅在人審後才映射 allowed/forbidden。
不同的實際 Hit origin 若無 pair 標註，一律 unverified。記錄候選 label、cache ID、actual origin、
exact/semantic observed hit 和答案品質，不能把 intended_repeat_type 當真實命中。

原 threshold/INT loader 現在要求 functional/calibration/evaluation 及其既有 schema。
未來需新增專用 workload adapter，維持 source_split，不可把 Evaluation 改名 functional 來繞過隔離。
每 variant／workload／repeat 各自從空 Cache 開始；100% 條件再執行 cache_initial_fill 的兩次
真實請求，暖填成本單列，正式20次全部是已指定問題的重訪。不從別的系統或 Calibration 填答案。
Semantic 暖填第二題可能 Hit，不能強制補入，仍須如實記錄。實際 Hit Rate 由正常流程決定。

## TI

reader.as_ti_case() 跳過每組 initial，留下20×3=60個待審核分類決策；Calibration/Evaluation
各10組，各30個決策。required_history_turn_ids 指整個 user/assistant pair，映射時展開成兩個
message IDs，不能只保留代名詞而刪掉其指涉。第一輪只有建立脈絡，不算 Topic Shift 判斷。

history 中 assistant 是中性人工腳本候選，**不是模型已生成答案或事實 Gold**。現有 TI 可對此
做特徵／context reduction 研究，不表示所有八架構已支援完整多輪 QA。
真正多輪 QA 需各系統自行生成歷史，另外保留條件、指涉正確性與答案品質；不能挪用別架構答案。

## 下一階段驗收

先在執行電腦用 validation.py 解析與檢查，不載入任何模型；完成人審／仲裁並凍結新版本後，
才新增 adapters。先對每個目標做少量 Smoke，確認標籤未進入 Prompt、ID 正確、單／多輪界線、
時間定義和來源完整性，再依獨立 Evaluation 清單正式量測。不能拿 Evaluation 反覆選 SC 門檻。
