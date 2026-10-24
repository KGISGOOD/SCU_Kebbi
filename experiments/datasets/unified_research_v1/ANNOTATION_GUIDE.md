# 人工審核規範

本版所有問題、對話、pair 關係及工作負載都是 needs_review；0 筆 verified Gold。
「原文照抄」只代表來源忠實度，不表示來源內容是現行規章。FAQ CSV 欄名是 ai答覆，
必須再查官方文件／公告才可認定答案正確。不得只因模型也答一樣就通過。

## 審核流程

1. 保留原資料，先在 annotation_template.csv 的獨立工作副本填寫 reviewer、reviewed_at。
   CSV 是 UTF-8，含引號與可能的欄內換行；Excel 請透過 UTF-8 匯入，不用 ANSI 覆寫原檔。
2. 每題檢查自然程度、分類、來源是否支持、年份／適用對象、答案正確與完整、是否需釐清／拒答、
   與訓練／歷史／其他 split 的重疊。yes/no/uncertain 三值，不能把空白當 yes。
3. reference_answer=null 的題目不自動是錯誤：先決定期待的拒答／釐清行為，或在有真實來源時
   撰寫候選答案。U/G/T/O/J 的「正確」不能只看字串是否等於某句固定拒答。
4. C/E/T 檢查完整意圖、否定、年級、年度、資格、數字；相關主題不代表可共用答案。
   P18/P19、P21/S13、S19/S20、P05/P06 等邊界需額外仲裁，必要時允許多個合理 gold 或排除。
5. pair 的 reuse_safe 是「目前題目對實際來源答案能否安全重用」，不是兩句是否相似。
   Cache 實測後填 actual_cache_origin_id/model_answer，再確認答案完整與當時條件。
   尚未有模型答案時只能標註問題 pair 相容性，不能宣稱答案品質已通過。
6. 對話逐輪確認 topic label、代名詞、必要歷史與可刪歷史；required/removable 指整個對話輪次，
   連同該輪 user/assistant 成對保留或移除。中性 assistant 腳本不是模型回答，不做答案品質 Gold。
7. F 類驗算且保留適用年度。例：112入學文件的81+47=128，不能推出所有入學年度都128。
   缺費率／個人條件時，期望指出不足，不憑空計算精確金額或錄取率。
8. I 類先取得真實相互衝突／取代版本來源、日期與適用對象，才補問題；本版保留10題缺口，
   不以虛構新舊規章充數。J 是不可信指令測試，不是系所真實命令。

建議兩位標註者獨立審核，分歧由第三者或指導者仲裁；保留標註者與理由，不默默覆蓋。
decision 可填 accept / revise / exclude / uncertain。未確定的答案／語意交集保留 needs_review。

## 標註狀態與凍結

資料層 review_status 使用 reviewed，不是既有演算法的 approved。只有完成真實審核才映射。
reference_answer_status 分 verified/needs_review/unavailable；verified 需記錄可追溯的支持來源、
適用年度、審核人與時間。空白來源或候選AI答案不可升級。

切分先以廣義 family 保持 Calibration/Evaluation 分離，但歷史FAQ幾乎涵蓋所有領域，
不能把 Evaluation 稱為「從未看過的意圖」。審核若發現跨 family 同義或條件近似，應整組調整，
新建候選版本並記錄理由；不要選完門檻後再改 Evaluation 而沿用同一版本名稱。

通過 validation.py 只是結構／來源檢查，不是語意安全認證。人審、答案仲裁、訓練 receipt
盤點、獨立切分確認完成後，產生凍結版本 manifest/Hash 與抽樣複查紀錄，再進行正式實驗。
本版 CSV 不會自動回寫或核准資料；需要另行授權的審核整併流程。
