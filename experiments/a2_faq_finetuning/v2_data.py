"""V2 frozen-data contract, shared by existing A2 and A3 loops (stdlib only).

This contract never reads legacy FAQ data or confers human approval.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

from .common import DataError

SYSTEM = "你是東吳大學資料科學系的系務問答助理。請依問題的年度與條件，用繁體中文清楚回答，不要編造未確認的資訊。"


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1048576), b""):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def read_rows(path):
    # JSON strings may contain U+2028 paragraph separators from source documents.
    # Only actual LF terminates a JSONL record; str.splitlines() would corrupt it.
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8-sig").split("\n") if s.strip()]


def inside(root, relative):
    p = (root / relative).resolve()
    if p != root and root not in p.parents:
        raise DataError("V2 path escapes project: " + str(relative))
    return p


def pairs(rows, key):
    result = {}
    for row in rows:
        for pair in row[key]:
            previous = result.setdefault(pair["pair_id"], pair)
            if previous != pair:
                raise DataError("inconsistent pair: " + pair["pair_id"])
    return [result[k] for k in sorted(result)]


def validate_partition(rows, split):
    ids = {r["question_id"] for r in rows}
    if len(ids) != len(rows) or not rows:
        raise DataError("duplicate/empty master")
    parts = {k: set(split["question_ids"][k]) for k in ("train", "validation")}
    if any(len(parts[k]) != len(split["question_ids"][k]) or not parts[k] for k in parts):
        raise DataError("empty/duplicate split IDs")
    if parts["train"] & parts["validation"] or parts["train"] | parts["validation"] != ids:
        raise DataError("split must cover frozen master once, without Test")
    owner = {i: k for k, values in parts.items() for i in values}
    for field in ("intent_group_id", "leakage_family_id"):
        groups = {}
        for r in rows:
            groups.setdefault(r[field], set()).add(owner[r["question_id"]])
        if any(len(v) != 1 for v in groups.values()):
            raise DataError("cross-split " + field)
    intent_owner = {r["intent_group_id"]: owner[r["question_id"]] for r in rows}
    for r in rows:
        for kid in r["knowledge_point_ids"] + r["negative_exclusion_intent_ids"]:
            if kid not in intent_owner or intent_owner[kid] != owner[r["question_id"]]:
                raise DataError("shared/overlapping knowledge crosses split: " + kid)
    for p in pairs(rows, "positive_pairs"):
        if owner[p["query_id"]] != owner[p["positive_id"]]:
            raise DataError("positive crosses split")
    return owner


def convert(rows, split):
    owner = validate_partition(rows, split)
    by_id = {r["question_id"]: r for r in rows}
    intents = {r["intent_group_id"]: r for r in rows if r["canonical_question_id"] == r["question_id"]}
    pp = pairs(rows, "positive_pairs")
    hn = pairs(rows, "hard_negative_pairs")
    files, dropped = {}, []
    for stage in ("train", "validation"):
        selected = [r for r in rows if owner[r["question_id"]] == stage]
        ids = {r["question_id"] for r in selected}
        positives = [p for p in pp if p["query_id"] in ids]
        hard = []
        for p in hn:
            if p["anchor_id"] not in ids:
                continue
            if p["negative_id"] not in ids:
                dropped.append({"pair_id": p["pair_id"], "reason": "cross_split_not_used"})
                continue
            a, n = by_id[p["anchor_id"]], by_id[p["negative_id"]]
            if (a["intent_group_id"] == n["intent_group_id"] or a["answer"] == n["answer"]
                    or n["intent_group_id"] in a["negative_exclusion_intent_ids"]):
                raise DataError("unsafe hard pair: " + p["pair_id"])
            hard.append(p)
        files["a2_" + stage + "_pairs.jsonl"] = positives
        files["a2_" + stage + "_hard.jsonl"] = hard
        files["a2_" + stage + "_questions.jsonl"] = selected
        a2 = []
        for iid, r in sorted(intents.items()):
            if r["question_id"] not in ids:
                continue
            group_pairs = [p for p in positives if p["intent_group_id"] == iid]
            a2.append({"faq_id": iid, "question_id": r["question_id"], "question": r["question"],
                       "positive": [by_id[p["query_id"]]["question"] for p in group_pairs],
                       "positive_pair_ids": [p["pair_id"] for p in group_pairs],
                       "hard_negative": sorted({intents[p["negative_intent_id"]]["question"] for p in hard if p["anchor_intent_id"] == iid}),
                       "negative": [], "split": stage, "review_status": "needs_review"})
        files["a2_" + stage + ".jsonl"] = a2
        files[stage + ".jsonl"] = [
            {"faq_id": r["question_id"], "intent_group_id": r["intent_group_id"],
             "leakage_family_id": r["leakage_family_id"], "split": stage,
             "review_status": "needs_review", "source_question_id": r["question_id"],
             "messages": [{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": r["question"]},
                          {"role": "assistant", "content": r["answer"]}]}
            for r in selected]
    return files, dropped


def load_bundle(root, directory):
    """Re-derive exact records from frozen evidence; reject edited derived files."""
    root, directory = Path(root).resolve(), Path(directory).resolve()
    m = read(directory / "dataset_manifest.json")
    if m.get("schema_version") != "training_v2_converted_v1":
        raise DataError("not a converted frozen V2 dataset")
    hashes = {str(directory / "dataset_manifest.json"): digest(directory / "dataset_manifest.json")}
    for relative, value in m["input_sha256"].items():
        p = inside(root, relative)
        if digest(p) != value:
            raise DataError("V2 input changed: " + relative)
        hashes[str(p)] = value
    frozen = inside(root, m["frozen_dir"])
    fm = read(frozen / "dataset_manifest.json")
    if fm.get("schema_version") != "training_v2_frozen_v1" or m["source_sha256"] != fm["source_sha256"]:
        raise DataError("frozen schema/source identity mismatch")
    source_snapshot = frozen / "source_manifest.json"
    if digest(source_snapshot) != fm["source_manifest_sha256"]:
        raise DataError("frozen source registry snapshot changed")
    hashes[str(source_snapshot)] = digest(source_snapshot)
    for name, value in fm["files_sha256"].items():
        p = inside(frozen, name)
        if digest(p) != value:
            raise DataError("frozen bytes changed: " + name)
        hashes[str(p)] = value
    for relative, value in fm["source_sha256"].items():
        if not relative.startswith(("畢業專題_爬蟲/type1/", "畢業專題_爬蟲/type2/")):
            raise DataError("disallowed fact source")
        p = inside(root, relative)
        if digest(p) != value:
            raise DataError("source bytes changed: " + relative)
        hashes[str(p)] = value
    rows = read_rows(frozen / "qa_master.jsonl")
    split = read(inside(root, m["split_manifest"]))
    expected, dropped = convert(rows, split)
    if set(m["files_sha256"]) != set(expected) or m["cross_split_hard_excluded"] != dropped:
        raise DataError("derived inventory mismatch")
    for name, records in expected.items():
        p = inside(directory, name)
        if digest(p) != m["files_sha256"][name] or read_rows(p) != records:
            raise DataError("converted data differs from frozen master: " + name)
        hashes[str(p)] = digest(p)
    if m["counts"] != {k: len(v) for k, v in expected.items()}:
        raise DataError("converted counts differ")
    return expected, m, hashes


def a2_inputs(root, directory, config):
    data, m, hashes = load_bundle(root, directory)
    rows = data["a2_train.jsonl"]
    master = data["a2_train_questions.jsonl"]
    ids = {r["faq_id"] for r in rows}
    exclusions = {tuple(sorted((r["intent_group_id"], other))) for r in master
                  for other in r["negative_exclusion_intent_ids"] if other in ids}
    effective = dict(config, false_negative_pairs=[list(p) for p in sorted(exclusions)])
    validation = {"structural_ok": True, "issues": [], "input_sha256": hashes,
                  "statistics": m["counts"], "v2_manifest": m,
                  "hard_negative_difficulty": "source-reviewed candidates; not BGE rank verified"}
    return rows, [], effective, validation


def make_schedule(by_id, batch_size, epochs, seed, diagnostic_ids, excluded):
    """Every stored positive once/epoch; variable positives, no repeated intent/batch."""
    from .training_data import ordinary_allowed
    pool = [{"pair_id": row["positive_pair_ids"][j], "faq_id": iid,
             "query": text, "positive": row["question"]}
            for iid, row in sorted(by_id.items()) for j, text in enumerate(row["positive"])]
    if len(pool) < 2 or batch_size < 2:
        raise DataError("not enough positive pairs for masked contrastive training")
    rng, schedule = random.Random(seed), []
    for epoch in range(1, epochs + 1):
        for attempt in range(10000):
            remaining = list(pool)
            rng.shuffle(remaining)
            batches = []
            while remaining:
                batch, tail = [], []
                for sample in remaining:
                    if len(batch) < batch_size and sample["faq_id"] not in {s["faq_id"] for s in batch}:
                        batch.append(sample)
                    else:
                        tail.append(sample)
                batches.append(batch)
                remaining = tail
            if len(batches[-1]) == 1 and len(batches) > 1:
                if len(batches[-2]) > 2:
                    candidate = next((s for s in batches[-2] if s["faq_id"] != batches[-1][0]["faq_id"]), None)
                    if candidate:
                        batches[-2].remove(candidate)
                        batches[-1].append(candidate)
            if all(len(b) >= 2 and all(any(ordinary_allowed(s["faq_id"], n["faq_id"], diagnostic_ids, excluded)
                                           for n in b) for s in b) for b in batches):
                break
        else:
            raise DataError("cannot schedule every V2 pair without false negatives; increase batch size")
        schedule.extend({"epoch": epoch, "round": 1, "samples": b} for b in batches)
    return schedule


def evaluate_a2(model, directory, root, config, args, profile, device, objective):
    """Same masked objective, deterministic held-out batches; no gradient/optimizer."""
    import torch
    from .training_data import prepare_training, batch_documents
    from .training_compat import _move_tensors
    data, manifest, _ = load_bundle(root, directory)
    rows = data["a2_validation.jsonl"]
    full = data["a2_validation_questions.jsonl"]
    ids = {r["faq_id"] for r in rows}
    exclusions = {tuple(sorted((r["intent_group_id"], other))) for r in full
                  for other in r["negative_exclusion_intent_ids"] if other in ids}
    effective = dict(config, false_negative_pairs=[list(p) for p in sorted(exclusions)])
    by_id, hard, diagnostic, excluded, _ = prepare_training(rows, [], effective)
    schedule = make_schedule(by_id, args.batch_size, 1, args.seed, diagnostic, excluded)
    model.eval()
    total, count = 0.0, 0
    with torch.inference_mode():
        for batch in schedule:
            samples = batch["samples"]
            docs = batch_documents(samples, by_id, hard, args.mode, diagnostic, excluded)
            def embed(texts):
                features = (model.preprocess(inputs=texts, prompt=None)
                            if profile == "st_6_1_0_preprocess" else model.tokenize(texts))
                return model(_move_tensors(features, device))["sentence_embedding"]
            loss = objective(embed([s["query"] for s in samples]), embed(docs["document_texts"]),
                             docs["allowed_mask"], docs["targets"])
            if not torch.isfinite(loss):
                raise DataError("nonfinite validation loss")
            total += float(loss.cpu()) * len(samples)
            count += len(samples)
    return {"validation_loss": total / count, "validation_positive_pairs": count,
            "definition": "query-weighted held-out masked InfoNCE, fixed batch schedule; not retrieval accuracy"}
