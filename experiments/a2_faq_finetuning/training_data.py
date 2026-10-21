"""Deterministic Train-only schedules and per-query document masks (stdlib only)."""
from __future__ import annotations

import hashlib
import json
import random

from .common import DataError


def excluded_pair(first, second, excluded):
    return frozenset((first, second)) in excluded


def ordinary_allowed(anchor_id, document_id, diagnostic_ids, excluded):
    return (anchor_id != document_id and document_id not in diagnostic_ids
            and not excluded_pair(anchor_id, document_id, excluded))


def prepare_training(rows, diagnostic_rows, config):
    """No evaluation file is accepted here. Stored 'negative' arrays are not used."""
    by_id = {row["faq_id"]: row for row in rows}
    questions = {row["question"]: row["faq_id"] for row in rows}
    diagnostic_texts = {row["question"] for row in diagnostic_rows}
    diagnostic_ids = {faq_id for faq_id, row in by_id.items()
                      if row["question"] in diagnostic_texts}
    excluded = set()
    for pair in config["false_negative_pairs"]:
        if (not isinstance(pair, list) or len(pair) != 2 or
                any(not isinstance(item, str) or item not in by_id for item in pair)
                or pair[0] == pair[1]):
            raise DataError("false_negative_pairs must contain two distinct Train IDs: %r" % pair)
        excluded.add(frozenset(pair))
    hard, excluded_hard = {}, []
    for faq_id, row in sorted(by_id.items()):
        hard[faq_id] = []
        for text in row["hard_negative"]:
            source_id = questions.get(text)
            if source_id is None or source_id == faq_id:
                raise DataError("%s: hard negative is not another Train original: %r" % (faq_id, text))
            if not ordinary_allowed(faq_id, source_id, diagnostic_ids, excluded):
                excluded_hard.append({"faq_id": faq_id, "source_id": source_id,
                                      "text": text, "reason": "diagnostic or configured semantic exclusion"})
            else:
                hard[faq_id].append(source_id)
    return by_id, hard, diagnostic_ids, excluded, excluded_hard


def make_schedule(by_id, batch_size, epochs, seed, diagnostic_ids, excluded):
    """Each FAQ appears once per round, three rounds/epoch; never drop a pair."""
    if not 2 <= batch_size <= len(by_id):
        raise DataError("batch size must be 2..%d" % len(by_id))
    if batch_size == 2 and diagnostic_ids:
        raise DataError("batch size 2 cannot give every row a valid in-batch negative while "
                        "diagnostic originals are masked; use at least 3. No pairs are dropped")
    rng = random.Random(seed)
    ids = sorted(by_id)
    schedule = []
    for epoch in range(1, epochs + 1):
        orders = {}
        for faq_id in ids:
            order = list(range(len(by_id[faq_id]["positive"])))
            if len(order) != 3:
                raise DataError("%s: exactly three Positive paraphrases required" % faq_id)
            rng.shuffle(order)
            orders[faq_id] = order
        for round_index in range(3):
            batches = None
            for _ in range(10000):
                shuffled = list(ids)
                rng.shuffle(shuffled)
                trial = [shuffled[i:i + batch_size] for i in range(0, len(ids), batch_size)]
                if len(trial[-1]) == 1:
                    trial[-1].insert(0, trial[-2].pop())
                if all(any(ordinary_allowed(faq_id, other, diagnostic_ids, excluded)
                           for other in batch) for batch in trial for faq_id in batch):
                    batches = trial
                    break
            if batches is None:
                raise DataError("cannot form batches with an eligible in-batch negative for every query; "
                                "increase batch size or review exclusions; no pairs were silently dropped")
            for batch in batches:
                samples = [{
                    "pair_id": "%s_p%02d" % (faq_id, orders[faq_id][round_index] + 1),
                    "faq_id": faq_id,
                    "query": by_id[faq_id]["positive"][orders[faq_id][round_index]],
                    "positive": by_id[faq_id]["question"],
                } for faq_id in batch]
                schedule.append({"epoch": epoch, "round": round_index + 1, "samples": samples})
    return schedule


def schedule_hash(schedule):
    return hashlib.sha256(json.dumps(schedule, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def batch_documents(samples, by_id, hard, mode, diagnostic_ids, excluded):
    """Deduplicate documents by source ID; explicit hard columns belong to their owner only."""
    base_ids = [sample["faq_id"] for sample in samples]
    if len(set(base_ids)) != len(base_ids):
        raise DataError("batch contains repeated intent")
    document_ids = list(base_ids)
    if mode == "a2_2":
        for sample in samples:
            for source_id in hard[sample["faq_id"]]:
                if source_id not in document_ids:
                    document_ids.append(source_id)
    mask = []
    base_negative_count = hard_visible = hard_new = 0
    for sample in samples:
        faq_id = sample["faq_id"]
        allowed = [source_id == faq_id or (
            source_id in base_ids and ordinary_allowed(faq_id, source_id, diagnostic_ids, excluded))
                   for source_id in document_ids]
        base_set = {source_id for source_id in base_ids
                    if ordinary_allowed(faq_id, source_id, diagnostic_ids, excluded)}
        base_negative_count += len(base_set)
        if mode == "a2_2":
            for source_id in hard[faq_id]:
                if not ordinary_allowed(faq_id, source_id, diagnostic_ids, excluded):
                    raise DataError("invalid hard negative escaped preprocessing")
                allowed[document_ids.index(source_id)] = True
                hard_visible += 1
                if source_id not in base_set:
                    hard_new += 1
        if sum(allowed) <= 1:
            raise DataError("%s has no effective negative; refusing zero-signal row" % faq_id)
        mask.append(allowed)
    return {
        "document_ids": document_ids,
        "document_texts": [by_id[faq_id]["question"] for faq_id in document_ids],
        "allowed_mask": mask,
        "targets": list(range(len(samples))),
        "base_negative_exposures": base_negative_count,
        "hard_visible_exposures": hard_visible,
        "hard_new_exposures": hard_new,
        "effective_negative_exposures": sum(sum(row) - 1 for row in mask),
        "extra_document_count": len(document_ids) - len(base_ids),
    }
