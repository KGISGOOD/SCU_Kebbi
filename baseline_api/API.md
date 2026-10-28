# 資科萬事通 A0 API V1

單輪、非串流 HTTP 介面，不保存聊天歷史、不使用 Ollama。
尚未在 GPU 啟動或驗證。

## 與 Benchmark 隔離

本次只新增根目錄 `baseline_api/`，沒有修改 Benchmark、共用模型程式、
Prompt、資料、索引或模型權重；此目錄不在現有 Benchmark 的 code_hashes
掃描清單內。不要在正在執行 Benchmark 的 GPU 環境安裝依賴、啟動 API、
同步其他程式或改設定：安裝會改套件身份，API 會占用顯存與運算資源。
以下 GPU 指令請在 Benchmark 完成後執行。

## 重用流程與限制

`experiments.algorithms.shared.a0_adapter.A0Adapter` 在啟動時只建構一次：

1. 使用明確的本機 Llama 3.2 3B、BGE-M3 路徑；離線載入，沒有下載 fallback。
2. 載入既有 `index.faiss`、`index.pkl`，檢查維度、度量及文件映射；不建索引，
   不要求 `version.json`。只可指向受信任的專案索引，pickle 不接受外部上傳。
3. 每次 `answer(question)` 使用原 `RAGPipeline.retrieve()` 取得 Top-5 文件。
4. 原 `create_context_from_documents()`、`build_prompt()` 建構 A0 Prompt。
5. 完整 Prompt 不截斷，檢查輸入預算及模型 Context Window，再執行原
   `RAGPipeline.generate_answer()` / `LLMLoader.generate()`，採 Greedy。
6. 原生成器只解碼新 Token，移除特殊 Token，再 `strip()`；API 直接回傳答案。

不另加答案改寫、繁簡轉換或拒答規則。沿用 `baseline_basic_v1` 的固定
離線參考日期，不代表資料即時有效；API 不提供最新規定的保證。
BGE 問題上限仍為 512 tokens；生成輸入預算預設 4096、最多生成 256 tokens，
兩者合計不得超過實際模型容量。輸入過長明確報錯，不悄悄截斷。

## GPU 設定、啟動與停止

請先啟用原本能執行 A0 的 GPU Python 環境。不要重新安裝舊版
`experiments/requirements.txt`，避免替換已驗證的 Torch/Transformers。
以下只安裝 HTTP 套件；不安裝任何模型。

```powershell
Set-Location 'C:\Users\rsy\Desktop\SCU_Kebbi'
python -m pip install -r baseline_api/requirements.txt
if ($LASTEXITCODE -ne 0) { throw 'API 依賴安裝失敗' }

# 只需做一次；已存在的本機設定不要覆寫。
$config = 'experiments/configs/a0_api.local.json'
New-Item -ItemType Directory -Path 'experiments/configs' -Force | Out-Null
if (!(Test-Path -LiteralPath $config)) {
    Copy-Item -LiteralPath 'baseline_api/config.example.json' -Destination $config
}
notepad $config
```

將 `model`、`bge` 換成 GPU 實際本機路徑，可參照現有 Benchmark 設定的
`models.base`、`models.bge`；不要改 Benchmark 設定。
`knowledge_index` 指向文件 Dense Index，不能用 FAQ Matching 索引替代。
相對路徑一律相對 SCU_Kebbi 根目錄；本機設定位於現有忽略規則覆蓋範圍。
啟動時會先計算資源 Hash，模型載入可能需要一段時間。

```powershell
# 本機驗證；明確單一 process，不使用 reload/multiple workers。
python -m baseline_api.server --config experiments/configs/a0_api.local.json --host 127.0.0.1 --port 8000

# 給同一受信任區網的機器人／組員連線時，改用這個啟動方式：
# python -m baseline_api.server --config experiments/configs/a0_api.local.json --host 0.0.0.0 --port 8000
```

不要同時執行兩個啟動命令。組員連線網址使用 GPU 電腦的區網 IP，
例如 `http://192.168.1.20:8000`，不能用 `0.0.0.0` 當客戶端網址。
若 Windows 防火牆阻擋，只開放需要的私人區網 TCP 8000；V1 沒有登入驗證，
不要直接公開到網際網路。互動文件為 `/docs`。

停止：在 API 的 PowerShell 視窗按 `Ctrl+C`。進行中的 CUDA 工作不能安全地
由 Python 執行緒強制中止，程序可能等它結束。若 GPU 工作卡死，依啟動紀錄
的 **API pid** 執行 `Stop-Process -Id <API的PID>`；不要終止其他 Python 或
Benchmark 程序。重啟 API 後會重新載入模型一次。

## HTTP 規格

### GET /health

```json
{"service":"a0_baseline","status":"ready","models_ready":true,"busy":false,"accepting_requests":true}
```

模型與索引全部載入後回 HTTP 200；載入中 `status=loading` 或失敗
`status=failed` 回 503。模型就緒但忙碌時仍回 200，`busy=true`、
`accepting_requests=false`。載入失敗的詳細原因保留在伺服器終端，修正後重啟。

### POST /chat

Header：`Content-Type: application/json`，Body 只接收 `question`：

```json
{"question":"資料科學系有哪些學習資源？"}
```

成功 HTTP 200：

```json
{"answer":"實際 A0 生成的回答","success":true,"error":null,"request_id":"每次請求的識別碼"}
```

失敗也包含 `answer`（空字串）、`success=false`、`error`、`request_id`。

| HTTP | error | 機器人處理方式 |
|---|---|---|
| 422 | invalid_request / empty_question | 檢查 JSON；問題須為非空字串，最多 4000 字元，不傳 history |
| 422 | input_limit_exceeded | 問題或完整 RAG Prompt 超限，請縮短問題；不自動截斷文件 |
| 503 | model_not_ready | 尚未載入或後端失敗；查看 health，請勿持續立即重試 |
| 503 | busy | 前一筆 GPU 工作尚未完成；稍後再試，沒有排隊 |
| 504 | inference_timeout | 超過伺服器等待時間；結束機器人等待狀態 |
| 500 | inference_failed | 後端錯誤；查看 request_id 對應 traceback，修正後重啟 |

每次最多一個推論。HTTP 逾時預設 120 秒，可由設定調整至最多 600 秒；
這是回覆等待上限，不代表能硬切斷 CUDA。逾時後仍保持 busy，直到原工作
真正結束；不自動重送、不堆積工作、不回傳逾時工作的延遲答案。
端點以外的模型初始化、網路傳輸不計入此推論等待上限。
HTTP 客戶端也必須設定 timeout；預設範例使用 130 秒。
記錄僅含 request_id、耗時、Token 數與錯誤 traceback，不記錄問題全文及答案。

## 驗證與組員呼叫範例

等模型 ready 後，在另一個 PowerShell 視窗執行：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health -TimeoutSec 5
$payload = @{question='資料科學系有哪些學習資源？'} | ConvertTo-Json -Compress
Invoke-RestMethod http://127.0.0.1:8000/chat -Method Post `
  -ContentType 'application/json; charset=utf-8' `
  -Body ([Text.Encoding]::UTF8.GetBytes($payload)) -TimeoutSec 130
```

組員的 Python 不需要 Torch、Transformers 或 requests，直接複製
`baseline_api/client_example.py` 的 `ask()`：

```python
from baseline_api.client_example import ask

result = ask("資料科學系有哪些學習資源？", "http://192.168.1.20:8000", timeout=130)
if result["success"]:
    answer_to_speak = result["answer"]
else:
    answer_to_speak = "目前暫時無法回答，請稍後再試。"
# 將 answer_to_speak 交給既有凱比語音介面，不修改機器人 SDK。
```

也可執行 `python baseline_api/client_example.py --url http://127.0.0.1:8000`。
驗收時確認 health、正常回答、空白問題 422、忙碌 503 與逾時 504；
這些目前均尚待 GPU／HTTP 實際驗證，沒有宣稱已通過。
