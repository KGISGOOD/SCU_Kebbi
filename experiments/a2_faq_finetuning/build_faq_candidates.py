"""Build the independent 60-question A2 retrieval corpus, never A1's canonical."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import (CSV_SOURCES, DATA_DIR, DataError, default_root, load_source_faqs,
                     resolve_path, sha256_file, write_json_exclusive, write_jsonl_exclusive)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=default_root())
    parser.add_argument("--output", default=str(DATA_DIR / "faq_retrieval_candidates.jsonl"))
    args = parser.parse_args()
    root = args.project_root.expanduser().resolve()
    try:
        output = resolve_path(root, args.output)
        metadata_path = output.with_name(output.name + ".metadata.json")
        if output.exists() or metadata_path.exists():
            raise DataError("output already exists; choose a new output path: %s" % output)
        faqs = load_source_faqs(root)
        metadata = {
            "schema_version": "a2_faq_corpus_v1", "faq_count": len(faqs),
            "construction": "original CSV question only; no positive paraphrases",
            "csv_encoding": "utf-8-sig (strict decoding)",
            "sources": [{"path": str(relative), "sha256": sha256_file(root / relative)}
                        for _, _, relative in CSV_SOURCES],
        }
        write_jsonl_exclusive(output, faqs)
        metadata["corpus_sha256"] = sha256_file(output)
        write_json_exclusive(metadata_path, metadata)
        print(json.dumps({"output": str(output), **metadata}, ensure_ascii=False, indent=2))
        return 0
    except (DataError, OSError) as exc:
        parser.exit(2, "ERROR: %s\n" % exc)


if __name__ == "__main__":
    raise SystemExit(main())
