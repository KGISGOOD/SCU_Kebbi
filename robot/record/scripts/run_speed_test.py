import sys
sys.path.append(r"./")
from service import ChatService
from config import AppSettings
from embeddings import HFEmbeddingsProvider
from vectorstores import VectorStoreLoader
from retriever import MultiStoreRetriever
from llm.ollama import ChatOllamaLLM
from prompts import PromptFactory
from orchestrator import QAOrchestrator
from relevance import RelevancePolicy

settings = AppSettings()
emb = HFEmbeddingsProvider(settings.model_name, settings.use_cpu)
loader = VectorStoreLoader(emb)
stores = loader.load_all_from_dir(settings.parent_vector_dir)
retriever = MultiStoreRetriever(stores, top_k=settings.top_k, fetch_k=settings.fetch_k)
llm = ChatOllamaLLM(
    model_name=settings.ollama_model,
    url=settings.ollama_url,
    stream=settings.ollama_stream,
    timeout_sec=settings.ollama_timeout_sec,
)
prompts = PromptFactory()
orch = QAOrchestrator(retriever, llm, prompts)
relevance = RelevancePolicy()
service = ChatService(orch, relevance, retrieve_only=False)

with open("speed_test_questions.txt", encoding="utf-8") as f:
    questions = [line.strip() for line in f if line.strip()]

total_times = []
for q in questions:
    ctx, uniq_sources, vec_time = orch.retrieve_only(q)
    ans, llm_time = orch.ask(q, [])
    total = vec_time + llm_time
    total_times.append(total)
    print(f"問題: {q}")
    print(f"  答案: {ans}")
    print(f"  向檢時間: {vec_time:.3f}秒, LLM時間: {llm_time:.3f}秒, 總計: {total:.3f}秒\n")

avg_total = sum(total_times) / len(total_times) if total_times else 0
print("=== 結果 ===")
print(f"測試題數: {len(questions)}")
print(f"平均總延遲: {avg_total:.3f}秒")