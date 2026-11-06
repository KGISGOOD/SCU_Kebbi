"""Read-only evidence checks and a seven-column view of existing measurements."""
import argparse
import csv
import hashlib
import json
from pathlib import Path

FIELDS = ["類別／架構", "題目", "實際回答", "總耗時（秒）", "FAQ耗時（秒）", "RAG耗時（秒）", "LLM耗時（秒）"]
TIMING_DEFINITION = {
    "units": "seconds; measured requests only, warmups excluded",
    "total": "answer call or topic detection; excludes model load, setup, evidence checks and report export",
    "faq": "entire measured FAQ matching call for A1/A2, including query embedding; no LLM generation",
    "rag": "measured document retrieval plus reranking when separately measured; excludes Prompt construction",
    "llm": "synchronized model.generate call only; excludes text decoding and Prompt construction",
    "unused_or_unmeasured": "blank, not a fabricated zero; Cache Hits do not inherit origin generation time",
    "sd_total": "controlled sum of once-measured shared preparation and mode execution; labelled in category",
    "ti": "total is detection cost, not an answer-generation latency; attention prefill is not generation",
    "order": "one CSV row per measured attempt, no averaging; IDs, repeats and failures remain in detailed JSONL",
}


def verify_evidence(root, cases):
    """Validate exact recorded evidence bytes/lines, without certifying semantics."""
    hashes = {}
    root = Path(root).resolve()
    for case in cases:
        for source in case.get("sources", []):
            relative = source["path"]
            if not relative.startswith(("畢業專題_爬蟲/type1/", "畢業專題_爬蟲/type2/")):
                raise ValueError("not a Type1/Type2 source: " + relative)
            p = (root / relative).resolve()
            if root not in p.parents:
                raise ValueError("source outside project")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            lines = p.read_text(encoding="utf-8-sig").splitlines()
            first, last = source["line_start"], source["line_end"]
            if not 1 <= first <= last <= len(lines):
                raise ValueError("invalid evidence location: " + relative)
            if digest != source["sha256"] or "\n".join(lines[first - 1:last]) != source["quote"]:
                raise ValueError("source evidence changed: " + relative)
            hashes[str(p)] = digest
    return hashes


def project(row, kind):
    if row.get("warmup"):
        return None
    out = row.get("output", row)
    timings = out.get("timings", {})
    hit = out.get("hit", out.get("cache_hit", False))
    faq = kind in {"A1", "A2-1", "A2-2"}
    total = row.get("end_to_end_seconds", timings.get("end_to_end_seconds", out.get("detection_seconds")))
    rag = out.get("retrieval_seconds", timings.get("retrieval_seconds"))
    if rag is not None and out.get("reranking_seconds") is not None:
        rag += out["reranking_seconds"]
    llm = out.get("generation_seconds", timings.get("generation_seconds"))
    if hit or faq:
        rag = llm = None
    label = kind
    if out.get("experiment_version"):
        label += "/" + out["experiment_version"]
    elif out.get("system_variant"):
        label += "/" + out["system_variant"] + "/" + out["decoder"]
    elif out.get("mode"):
        label += "/" + out["mode"]
    elif out.get("method"):
        label += "/" + out.get("split", "") + "/" + out["method"]
    if out.get("end_to_end_kind", "").startswith("controlled sum"):
        label += "（共用前處理成本）"
    answer = out.get("generated_answer", out.get("output_text", out.get("answer", "")))
    if "prediction_shift" in out and not answer:
        answer = "主題轉換" if out["prediction_shift"] else "主題延續"
    return dict(zip(FIELDS, [label, row.get("question", out.get("question", out.get("query", ""))),
                            answer, total, total if faq else None, rag, llm]))


def export(run, kind):
    """Never infer zero durations; retain detailed evidence separately in each run."""
    run = Path(run)
    if kind == "SC":
        files = sorted(run.glob("*/results.jsonl")) + sorted(run.glob("repeat_*/*/results.jsonl"))
    elif kind == "TI":
        files = [run / "calibration_classification.jsonl", run / "classification.jsonl"]
    else:
        preferred = run / "results.jsonl"
        files = [preferred if preferred.exists() else run / "measurements.raw.jsonl"]
    rows = []
    for p in files:
        if not p.exists():
            continue
        for line in p.read_text(encoding="utf-8-sig").split("\n"):
            if line.strip():
                value = project(json.loads(line), kind)
                if value is not None:
                    rows.append(value)
    target = run / "first_round_summary.csv"
    with target.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--runs", type=Path, nargs="+", required=True,
                        help="runs containing first_round_summary.csv; concatenated without deduplication")
    parser.add_argument("--output", type=Path, required=True, help="NEW UTF-8 BOM CSV")
    args = parser.parse_args()
    combined = []
    for run in args.runs:
        with (run / "first_round_summary.csv").open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != FIELDS:
                raise ValueError("summary schema differs: " + str(run))
            combined.extend(reader)
    with args.output.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(combined)


if __name__ == "__main__":
    main()
