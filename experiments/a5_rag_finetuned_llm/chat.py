"""Interactive A5 integration only; no evaluation, model training or result files."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from experiments.a3_faq_sft.common import ROOT, resolve
from .config import ADAPTER, DEFAULT_BASE, EMBEDDING_MODEL, INDEX, TOP_K, GenerationSettings
from .pipeline import A5Pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--index-dir", default=INDEX)
    parser.add_argument("--base-model", default=DEFAULT_BASE)
    parser.add_argument("--adapter", default=ADAPTER)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--embedding-device", choices=("cuda", "cpu"), help="default: same as LLM device")
    parser.add_argument("--precision", choices=("fp16", "bf16", "fp32"), default="fp16")
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--show-sources", action="store_true")
    parser.add_argument("--max-input-tokens", type=int, default=2048)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--eos-policy", choices=("model", "tokenizer"), default="model")
    parser.add_argument("--do-sample", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    try:
        generation = GenerationSettings(
            max_input_tokens=args.max_input_tokens, max_new_tokens=args.max_new_tokens,
            do_sample=args.do_sample, temperature=args.temperature, top_p=args.top_p,
            repetition_penalty=args.repetition_penalty, seed=args.seed, eos_policy=args.eos_policy)
        pipeline = A5Pipeline(root=root, index_dir=resolve(root, args.index_dir),
                              base_model=args.base_model, adapter=resolve(root, args.adapter),
                              device=args.device, precision=args.precision, embedding_device=args.embedding_device,
                              offline=args.offline, allow_unreviewed=args.allow_unreviewed, generation=generation)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        parser.exit(2, "啟動失敗 [%s]：%s\n" % (type(exc).__name__, exc))
    print("A5 RAG + Fine-tuned LLM\nBase：%s\nAdapter：%s\nFAISS：%s" %
          (args.base_model, pipeline.adapter, pipeline.index_dir))
    print("Embedding：%s (%s)；LLM：%s/%s；Dense Top-%d；索引文件數：%d" %
          (EMBEDDING_MODEL, pipeline.embedding_device, args.device, args.precision,
           TOP_K, pipeline.store.get_document_count()))
    print("provisional=%s；index_version_available=%s" %
          (pipeline.provisional, pipeline.index_version is not None))
    print("生成設定：%s" % json.dumps(asdict(generation), ensure_ascii=False))
    print("實際停止 token：%s；pad token：%s" % (
        pipeline.tokenizer.eos_token_id if args.eos_policy == "tokenizer" else
        pipeline.model.generation_config.eos_token_id, pipeline.tokenizer.pad_token_id))
    print("輸入 exit / quit 結束。此為單輪整合功能，生成回答仍需人工確認。")
    while True:
        try:
            query = input("\n問題：")
            if query.strip().lower() in ("exit", "quit"):
                return 0
            if not query.strip():
                print("請輸入非空白問題。")
                continue
            result = pipeline.answer(query)
            print("\n回答：\n%s" % (result["answer"] or "模型未產生可顯示的回答。"))
            if args.show_sources:
                print("\n實際送入 Prompt 的 Top-5 文件（依檢索順序；此處僅顯示摘要）：")
                for rank, (doc, score) in enumerate(zip(result["documents"], result["scores"]), 1):
                    print("%d. %s | %s | score=%.6f\n   %s" % (
                        rank, doc.metadata.get("source_file", "未知來源"),
                        doc.metadata.get("document_id", "無 document_id"), score,
                        doc.page_content[:200].replace("\n", " ")))
        except (KeyboardInterrupt, EOFError):
            print("\n結束問答。")
            return 0
        except Exception as exc:
            print("無法回答 [%s]：%s" % (type(exc).__name__, exc))


if __name__ == "__main__":
    raise SystemExit(main())
