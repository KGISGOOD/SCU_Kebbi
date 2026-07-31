# --------------------------------------------------------------
# test_pld.py  –  Prompt Lookup Decoding (PLD) 測試腳本
# --------------------------------------------------------------
# 此腳本會：
#   1️⃣ 載入您現有的 RAG 元件（向量庫、嵌入、Retriever）
#   2️⃣ 讀取測試問題列表（預設使用 robot/speed_test_questions.txt）
#   3️⃣ 依照 orchestrator 的做法產生 Prompt（問題＋檢索結果）
#   4️⃣ 執行基線 Ollama LLM（不加 PLD 參數）
#   5️⃣ 執行 PLD‑啟用的 HuggingFace LLM（在 model.generate 中加入
#        prompt_lookup_num_tokens 與 max_ngram_size）
#   6️⃣ 比較兩種方式的延遲與輸出相似度
#   7️⃣ 印出平均延遲、加速比與平均相似度作為測試結論
#
# 注意：執行前請確認：
#   - 已安裝所需套件（torch, transformers, sentence-transformers, langchain 等）
#   - Ollama 服務正在運行（http://127.0.0.1:11434/api/generate）
#   - 向量庫位於 robot/vector_store_dept/ （含 index.faiss 與 index.pkl）
#   - 您想要測試的 HuggingFace 模型名稱（例如 "gpt2"、"deepseek-ai/deepseek-llm-14b-chat" 等）
# --------------------------------------------------------------

import os
import sys
import time
import difflib
from pathlib import Path
from typing import List, Tuple

# ────────────────────────────────────────
# 0️⃣  基本路徑設定（讓腳本能夠匯入 robot/ 目錄的模組）
# ────────────────────────────────────────
ROOT = Path(__file__).resolve().parent          # D:\Ray DB\SCU\SCU_Kebbi\robot
sys.path.append(str(ROOT.parent))               # 讓 Python 能找到 robot 作為套件

# ────────────────────────────────────────
# 1️⃣  載入現有的 RAG 元件（與您目前服務完全相同）
# ────────────────────────────────────────
from robot.config import AppSettings
cfg = AppSettings()                             # 讀取環境變數：model_name、top_k、fetch_k、use_cpu 等

from robot.embeddings import HFEmbeddingsProvider
embeddings_provider = HFEmbeddingsProvider(
    model_name=cfg.model_name,                  # 例如 "BAAI/bge-m3"
    use_cpu=cfg.use_cpu
)

from robot.vectorstores import VectorStoreLoader
vector_loader = VectorStoreLoader(embeddings_provider)

# 向量庫放在 robot/vector_store_dept/
vector_store_dir = ROOT / "vector_store_dept"
vectorstores = vector_loader.load_all_from_dir(str(vector_store_dir))
if not vectorstores:
    raise RuntimeError(
        f"找不到任何 FAISS 向量庫：請確認 {vector_store_dir} 內有 index.faiss 與 index.pkl"
    )

from robot.retriever import MultiStoreRetriever
retriever = MultiStoreRetriever(
    vectorstores=vectorstores,
    top_k=cfg.top_k,          # 例如 5
    fetch_k=cfg.fetch_k       # 例如 100
)

# ────────────────────────────────────────
# 2️⃣  讀取測試問題列表
# ────────────────────────────────────────
def load_questions() -> List[str]:
    """
    優先使用 robot/speed_test_questions.txt（每行一個問題），
    若檔案不存在則回傳一個小樣本列表供快速驗證。
    """
    q_path = ROOT / "speed_test_questions.txt"
    if q_path.is_file():
        with q_path.open(encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
        return lines
    # 備用樣本問題（您可自行修改或擴充）
    return [
        "什麼是大數定律？",
        "請說明梯度下降法的基本步驟。",
        "Transformer 模型中自注意力機制的核心 idea 是什麼？",
        "如何評估一個分類模型的好壞？",
        "資料科學中常見的特徵工程技術有哪些？",
    ]

questions: List[str] = load_questions()

# ────────────────────────────────────────
# 3️⃣  產生 Prompt（問題＋檢索結果） – 複製 orchestrator 的做法
# ────────────────────────────────────────
def build_prompt(query: str) -> Tuple[str, List[Tuple[str, str]]]:
    """
    回傳 (prompt_text, unique_sources)
    - prompt_text:  "問題：{query}\n\n檢索資証：{combined_context}"
    - unique_sources: 去重複後的 [(episode_name, podcast_name), ...]
    """
    docs: List = retriever.retrieve(query)   # List[langchain.schema.Document]

    # 取前 k 筆（同上 orchestrator）
    ctx_lines: List[str] = []
    uniq_sources_set: set = set()
    for d in docs[: cfg.top_k]:
        page = d.page_content.strip()
        ep = d.metadata.get("episode_name", "Unknown Episode")
        pod = d.metadata.get("Podcast_name", "Unknown Podcast")
        ctx_lines.append(f"內容：{page}\n來源：{ep}, {pod}")
        uniq_sources_set.add((ep, pod))

    combined_context = "\n\n".join(ctx_lines)
    prompt = f"問題：{query}\n\n檢索資証：{combined_context}"
    return prompt, list(uniq_sources_set)

# ────────────────────────────────────────
# 4️⃣  基線 LLM：Ollama（使用您現有的 ChatOllamaLLM）
# ────────────────────────────────────────
from robot.llm.ollama import ChatOllamaLLM

ollama_llm = ChatOllamaLLM(
    model_name=cfg.ollama_model,          # 例如 "deepseek-r1:14b"
    url=cfg.ollama_url,                   # 例如 "http://127.0.0.1:11434/api/generate"
    stream=cfg.ollama_stream,             # 預設 False
    timeout_sec=cfg.ollama_timeout_sec,   # 例如 900
)

def call_ollama(prompt: str) -> Tuple[str, float]:
    """回傳 (answer_text, latency_seconds)"""
    t0 = time.perf_counter()
    answer = ollama_llm._call(prompt)     # 直接呼叫 Ollama 的 _call 方法
    t1 = time.perf_counter()
    return answer.strip(), t1 - t0

# ────────────────────────────────────────
# 5️⃣  PLD 啟用的 HuggingFace LLM 工廠類
# ────────────────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class HFPLDLLM:
    """
    極簡的 HuggingFace LLM 包裝，僅實作 _call 方法，
    內部使用 model.generate(...) 並加入 PLD 參數。
    """
    def __init__(
        self,
        model_name: str,
        device: str = None,
        prompt_lookup_num_tokens: int = 5,
        max_ngram_size: int = 3,
        max_new_tokens: int = 128,
        do_sample: bool = False,          # 為了可比較，預設貪婪解碼
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device.startswith("cuda") else torch.float32,
            low_cpu_mem_usage=True,
        ).to(self.device)
        self.model.eval()

        self.prompt_lookup_num_tokens = prompt_lookup_num_tokens
        self.max_ngram_size = max_ngram_size
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p

    def _call(self, prompt: str) -> str:
        """產生文字回覆（只回傳新生成的部分）"""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                prompt_lookup_num_tokens=self.prompt_lookup_num_tokens,
                max_ngram_size=self.max_ngram_size,
                do_sample=self.do_sample,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        # 只取新生成的 token（去掉輸入長度）
        input_len = inputs["input_ids"].shape[-1]
        generated_text = self.tokenizer.decode(
            generated_ids[0][input_len:], skip_special_tokens=True
        )
        return generated_text.strip()

def call_hf_pld(prompt: str, hf_llm: HFPLDLLM) -> Tuple[str, float]:
    """回傳 (answer_text, latency_seconds)"""
    t0 = time.perf_counter()
    answer = hf_llm._call(prompt)
    t1 = time.perf_counter()
    return answer.strip(), t1 - t0

# ────────────────────────────────────────
# 6️⃣  設定要測試的 HuggingFace 模型與 PLD 參數
# ────────────────────────────────────────
# 請依照您的硬體與需求自行修改以下兩行
HF_MODEL_NAME = "gpt2"                     # ← 例如 "gpt2"、"distilgpt2"、
                                            #   "facebook/opt-125m"，
                                            #   或如果顯存夠大可換成與 Ollama 相近的：
                                            #   "deepseek-ai/deepseek-llm-14b-chat"

# PLD 參數（可依實驗調整）
PLD_NUM_TOKENS = 5
PLD_MAX_NGRAM = 3

# 建立 HIF LLM 實例
hf_pld_llm = HFPLDLLM(
    model_name=HF_MODEL_NAME,
    prompt_lookup_num_tokens=PLD_NUM_TOKENS,
    max_ngram_size=PLD_MAX_NGRAM,
    max_new_tokens=128,          # 與 Ollama 的 num_predict 大致對齊（自行調整）
    do_sample=False,             # 貪婪解碼，以便直接比較
    temperature=0.0,
    top_p=1.0,
)

# ────────────────────────────────────────
# 7️⃣  執行測試、比較結果並印出摘要
# ────────────────────────────────────────
def similarity(a: str, b: str) -> float:
    """使用 SequenceMatcher 計算兩段文字的相似度（0~1）"""
    return difflib.SequenceMatcher(None, a, b).ratio()

ollama_latencies: List[float] = []
pld_latencies: List[float] = []
similarities: List[float] = []

print(f"=== 開始測試 {len(questions)} 個問題 ===")
for idx, q in enumerate(questions, start=1):
    prompt, _ = build_prompt(q)

    # ---- 基線 Ollama ----
    ans_ollama, lat_ollama = call_ollama(prompt)
    ollama_latencies.append(lat_ollama)

    # ---- PLD 加速的 HuggingFace ----
    ans_pld, lat_pld = call_hf_pld(prompt, hf_pld_llm)
    pld_latencies.append(lat_pld)

    # ---- 相似度 ----
    sim = similarity(ans_ollama, ans_pld)
    similarities.append(sim)

    print(
        f"[{idx:02d}] 問題: {q[:30]}... | "
        f"Ollama: {lat_ollama:.2f}s | PLD: {lat_pld:.2f}s | "
        f"相似度: {sim:.3f}"
    )

# ---- 統計 ----
avg_ollama_lat = sum(ollama_latencies) / len(ollama_latencies)
avg_pld_lat = sum(pld_latencies) / len(pld_latencies)
speedup = avg_ollama_lat / avg_pld_lat if avg_pld_lat > 0 else float("inf")
avg_sim = sum(similarities) / len(similarities)

print("\n=== 測試結果摘要 ===")
print(f"平均 Ollama 延遲: {avg_ollama_lat:.3f} 秒")
print(f"平均 PLD-HF 延遲: {avg_pld_lat:.3f} 秒")
print(f"平均加速比 (Ollama / PLD-HF): {speedup:.2f}×")
print(f"平均輸出相似度: {avg_sim:.3f} (1.0 表示完全相同)")

# 結語：若速度顯著提升且相似度接近 1.0，則 PLD 在您的資料上是有效的。