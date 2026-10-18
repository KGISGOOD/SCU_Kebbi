import sys
import traceback

modules = [
    ("shared.data_loader", "shared.data_loader"),
    ("shared.embeddings", "shared.embeddings"),
    ("shared.vector_store", "shared.vector_store"),
    ("shared.retrieval", "shared.retrieval"),
    ("shared.llm_loader", "shared.llm_loader"),
    ("shared.evaluation", "shared.evaluation"),
    ("baselines.rag_baseline.config", "baselines.rag_baseline.config"),
    ("baselines.rag_baseline.prompts", "baselines.rag_baseline.prompts"),
    ("baselines.rag_baseline.pipeline", "baselines.rag_baseline.pipeline"),
    ("baselines.rag_baseline.build_index", "baselines.rag_baseline.build_index"),
    ("baselines.rag_baseline.test_retrieval", "baselines.rag_baseline.test_retrieval"),
    ("baselines.rag_baseline.test_baseline", "baselines.rag_baseline.test_baseline"),
    ("baselines.rag_baseline.chat", "baselines.rag_baseline.chat"),
]

for mod_path, mod_name in modules:
    try:
        __import__(mod_path)
        print(f"✓ Imported {mod_name}")
    except Exception as e:
        print(f"✗ Failed to import {mod_name}: {e}")
        # traceback.print_exc()
