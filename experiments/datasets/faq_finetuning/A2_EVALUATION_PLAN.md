# A2 評估題庫與公平比較規劃（任務 3 修訂）

日期：2026-10-21。狀態：138 題評估查詢候選及程式碼已撰寫，全部查詢 needs_review，尚未經 parser、執行、人工正式驗收或模型評估。
此版依任務 3 取代任務 2B 原先 264 題／A-dev 方案，不再採用 A-dev 與 B-validation 平均選模。不得將本次最終測試題倒過來作開發集。

## 正式資料與切分

專案根目錄由 --project-root 指定或由 A2 程式位置推算。
正式資料在 experiments/datasets/faq_finetuning/；舊 worktree 不再編輯。
唯一切分依據 faq_split_manifest.json；舊 faq_split_plan.md 的 42/10/8 詳細表不適用。

| scenario | split | 意圖 | 每意圖題數 | 查詢數 | 用途 |
|---|---|---|---:|---:|---|
| seen | seen_test | P01–P21、S01–S21 | 2 | 84 | 已見 FAQ 意圖新問法，最終評估 |
| unseen | unseen_validation | P22–P25、S22–S26 | 3 | 27 | 保留 FAQ 驗證，可選訓練設定 |
| unseen | unseen_test | P26–P30、S27–S30 | 3 | 27 | 保留 FAQ 最終評估 |

總計 138 題，位於 faq_eval_queries.jsonl。query_id 命名 ST_<FAQ ID>_<兩位序號>、UV_<FAQ ID>_<兩位序號>、UT_<FAQ ID>_<兩位序號>。每列含 query_id、query、gold_faq_id、scenario、split、review_status。
gold 是依原始 CSV 完整問句提出的單一建議標籤，不代表已證明其他 FAQ 不能合理回答。每題需人工覆核完整子意圖與所有可能 gold，特別是 A2_DATA_REVIEW.md 的邊界清單。
本次查詢由同一助手協助撰寫，不宣稱獨立作者盲寫或已完成雙人標註。它們在用途上與訓練隔離；仍需另一位審核者確認文字／模板獨立性與語意正確性。

## 撰寫及洩漏檢查

原始 question 保留於 60 題檢索庫，但不直接當作評估 query。新 query 不取自訓練 Positive、Negative、Hard Negative、備份或 15 題診斷文字，亦不只做標點或少數詞交換。
每句需保留原題的全部詢問，例如五年一貫與名額、畢業三至五年薪資與成長、課表／空堂／電腦教室三部分；不得新增身分、時點、政策、金額或門檻。

validate_a2_data.py 對 source question、Train 三類樣本及診斷文字作 exact／正規化查重；對評估題彼此作跨集合查重，對字面相似度較高者提出 review。字串檢查不能證明沒有語意洩漏。
P01/QF01、P02/QF04、S01/QF07、S02/QF10、S05/QF13 仍是訓練 anchor 與歷史診斷相同；不能將舊 15 題認定為乾淨 Test。備份與審核文件也不可被 loader 掃入訓練語料。
先固定資料，再人工覆核近重複模板及同義改寫家族；遇到真正僅極小差異的 query，須在任何正式模型評分前修訂並重驗，不得看過分數才刪題。
發現多 gold 或跨 split 同意圖時，保留爭議並停止凍結，另提出標註／切分決策，不在本任務擅改原始 FAQ、split 或訓練候選。

## 60 題共同候選庫

以兩份 CSV 建立獨立 faq_retrieval_candidates.jsonl，不使用舊 3 題 faq_canonical_v1.jsonl，不加入任何 Positive。
build_faq_candidates.py 嚴格讀取實際 UTF-8 欄位、保留既有 P/S ID 及原文，題數、來源、metadata/hash 有明確紀錄。不重加前綴形成 PP01/SS01。
本次未執行建置程式，因此 60 題輸出需由使用者在另一台電腦產生並驗證。

三組模型共用同一份 60 題 corpus 與同一份查詢 hash，但各自用自身模型同時編碼 question 與 query。只使用 dense、float32、L2 正規化、cosine／normalized inner product；無 sparse、reranker、LLM、拒答閾值。
程式以 NumPy 精確內積對 60 題完整排序，不依賴近似索引；同分以 FAQ ID 排序。保留完整排名計算 MRR，另存 Top-5 供分析。
Validation/Test 原題可在評估 corpus 中出現，但不能參與微調、in-batch negatives、hard negative 採礦或訓練生成。

## 指標與結果

三個 split 分別計算並輸出，禁止跨情境合併準確率。
單一 gold 的每題排名為 r：
Top-1 Accuracy = mean(r==1)；Recall@5 = mean(r<=5)，等同單一 gold 的 Hit@5；MRR = mean(1/r)，使用全 60 名，不是 MRR@5。
另記每組查詢數／意圖數、平均 Top-1/gold cosine、逐 FAQ 及 macro 指標。程式成功或模型載入成功不是檢索正確。
缺資料、壞 JSON、錯誤 ID、空 query 或推論失敗不得被靜默略過後改分母。
有多個可接受 gold 的題目須先人工處理，再決定是否修訂標註契約；本次程式不事後放寬匹配條件。

## 選模與 Test 使用邊界

只用 unseen_validation 選擇超參數、early stopping/checkpoint 與共同設定；需在訓練前書面固定選模指標（建議 validation MRR）、預算及同分規則（較早 checkpoint）。
seen_test 與 unseen_test 僅於資料、設定及模型凍結後作最終評估，不作難負樣本挑選或訓練資料修復依據。
評估程式預設 unseen_validation；--split all 才同時計算三組。資料未批准時預設停止載入模型；明確 --allow-unreviewed 只產生 provisional 診斷結果，不能稱正式實驗。
B-validation 被用來選模後不是無偏最終泛化證據，最終保留 unseen_test。

## A1／A2-1／A2-2 的公平比較

A1 使用原始 BGE-M3；A2-1、A2-2 在未來從相同原始 checkpoint 初始化，使用同一 Train 正配對與一般負樣本基礎。
A2-2 額外加入經確認的 Hard Negative，不替換 A2-1 原本的一般負樣本。42 題中目前 30 題有人工候選、12 題為空，不應只在 A2-2 丟掉空難例意圖。
配對方向、loss/collator/mask、optimizer、學習率、步數、batch 順序、種子及調參預算皆待後續訓練任務確定；本次不建訓練流程。
同 FAQ 的改寫不可互當 in-batch negative；診斷原文與已確認假負樣本配對也須遮罩。
只有人工語意相近的候選，不能稱原始模型已驗證的難負樣本；後續採礦僅在 Train 可用來源上進行，絕不使用本次最終 Test 查詢。
新增難負樣本也增加負樣本數量，三組比較不能單獨排除「數量」的影響；如另需等數量一般負樣本對照，須另行授權。

評估三組應固定同樣的 query/corpus hash、長度、batch、精度與硬體設定，記錄模型本機權重 hash／HF revision、程式 hash 及套件版本。不同模型不共用向量結果。
可預先約定種子 42、3407、2026 作重複實驗，報各 seed 及平均／波動；本次未執行。
每個保留集只有 9 個意圖、27 題，不能把同意圖改寫視為完全獨立樣本。後續不確定性分析宜按 FAQ ID 分群，不能在無結果時編造信賴區間或顯著性。

## 尚待確認的研究限制

P18/P19 的廣義獎助與新生獎勵子題、P21/S13 的算力來源交集，及 P06/S22、P04/S26、P08–P09/S27、P16/S28–S30 等跨切分主題，均需完整答案人工判斷。
不同 FAQ ID 不等於不同語意群組；現階段情境 B 的嚴格表述是「保留 FAQ ID 泛化」，不是已確認「整個未見主題泛化」。
實際問題及受影響 query_id 列於 A2_DATA_REVIEW.md 任務 3 補充；驗證器亦固定輸出已知邊界與原文，並不自動裁定同義。

程式執行說明見 experiments/a2_faq_finetuning/README.md。所有 parser、資料檢查、CPU/CUDA、模型載入、排名及指標正確性仍待另一台電腦實際驗證。本次沒有正式凍結資料、模型分數或訓練成果。
