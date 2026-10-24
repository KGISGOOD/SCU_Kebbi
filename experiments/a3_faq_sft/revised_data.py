"""Train-only answer revisions; original rows and held-out splits remain immutable."""
from copy import deepcopy

from .common import DataError, read_rows, resolve, sha


def revisions(root, path, originals):
    original = {r["faq_id"]: r for r in originals if r["split"] == "train"}
    initial_hash = sha(path)
    proposals = read_rows(path)
    if len(proposals) != 42 or {p.get("faq_id") for p in proposals} != set(original):
        raise DataError("answer revisions must cover exactly the 42 Train IDs once")
    inputs = {path.relative_to(root).as_posix(): initial_hash}
    rows = {}
    for proposal in proposals:
        faq_id = proposal["faq_id"]
        for key in ("revised_answer", "reason", "applicable_period", "uncertainty", "evidence_status"):
            if not isinstance(proposal.get(key), str) or not proposal[key].strip():
                raise DataError(f"{faq_id}: missing revision field {key}")
        if proposal.get("review_status") != "needs_review" or not isinstance(proposal.get("sources"), list):
            raise DataError(f"{faq_id}: revision must remain a needs_review candidate with sources")
        sources = []
        for source in proposal["sources"]:
            source_path = resolve(root, source["path"])
            relative = source_path.relative_to(root).as_posix()
            locator = source.get("locator")
            if not isinstance(locator, str) or not locator.strip():
                raise DataError(f"{faq_id}: source locator is required")
            digest = sha(source_path)
            if locator not in source_path.read_text(encoding="utf-8-sig"):
                raise DataError(f"{faq_id}: source excerpt not found: {relative}")
            inputs[relative] = digest
            sources.append({**source, "path": relative, "sha256": inputs[relative]})
        row = deepcopy(original[faq_id])
        row["messages"][2]["content"] = proposal["revised_answer"]
        row["revision"] = {**proposal, "sources": sources,
                           "original_question": original[faq_id]["messages"][1]["content"],
                           "original_answer": original[faq_id]["messages"][2]["content"]}
        rows[faq_id] = row
    if any(sha(root / name) != digest for name, digest in inputs.items()):
        raise DataError("revision evidence changed during preparation")
    return rows, inputs
