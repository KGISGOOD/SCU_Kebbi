"""Interactive FAQ standard-answer lookup; no training or experimental metrics."""
from __future__ import annotations

import argparse
from pathlib import Path

from .common import DATA_DIR, default_root, resolve_path
from .faq_answering import FAQAnswerer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root())
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-tag", required=True, choices=("A1", "A2-1", "A2-2"))
    parser.add_argument("--candidates", default=str(DATA_DIR / "faq_retrieval_candidates.jsonl"))
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--allow-unreviewed", action="store_true",
                        help="explicitly use a provisional checkpoint for manual functional checks")
    parser.add_argument("--threshold", type=float, default=None,
                        help="optional, uncalibrated cosine cutoff; default: always return Top-1")
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    try:
        engine = FAQAnswerer(root=root, model=args.model, model_tag=args.model_tag,
                             candidates=resolve_path(root, args.candidates), device=args.device,
                             batch_size=args.batch_size, max_length=args.max_length,
                             offline=args.offline, allow_unreviewed=args.allow_unreviewed,
                             threshold=args.threshold)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        parser.exit(2, "啟動失敗 [%s]：%s\n" % (type(exc).__name__, exc))
    print("%s FAQ Semantic Matching\n模型：%s\n裝置：%s；FAQ：60 題；provisional=%s"
          % (engine.model_tag, engine.model_name, engine.device, engine.provisional))
    print("輸入 exit / quit 結束；相似度不是回答正確機率。")
    while True:
        try:
            query = input("\n使用者問題：\n")
        except (EOFError, KeyboardInterrupt):
            print("\n結束問答。")
            return 0
        if query.strip().lower() in ("exit", "quit"):
            return 0
        if not query.strip():
            print("請輸入非空白問題。")
            continue
        try:
            result = engine.answer(query)
        except KeyboardInterrupt:
            return 130
        except Exception as exc:
            print("無法回答 [%s]：%s" % (type(exc).__name__, exc))
            continue
        print("=" * 40)
        print("%s FAQ Semantic Matching\n模型：%s\nprovisional=%s" % (
            result["model_tag"], result["model"], result["provisional"]))
        print("使用者問題：\n%s\n\n匹配 FAQ：\nID：%s\n問題：%s\n相似度：%.6f" % (
            query, result["faq_id"], result["question"], result["similarity"]))
        print("\n回答：\n%s" % (result["answer"] if result["accepted"] else
                                "低於使用者指定門檻，未提供答案。"))
        print("=" * 40)


if __name__ == "__main__":
    raise SystemExit(main())
