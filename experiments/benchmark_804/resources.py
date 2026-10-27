"""Offline qualification; weights stay read-only. No CUDA work in this module."""
from __future__ import annotations

from .common import path, sha, read


def qualify(root, config, allow_provisional):
    from experiments.evaluation.model_resources import inference_identity
    from experiments.a2_faq_finetuning.common import read_checkpoint_provenance
    from experiments.a3_faq_sft.model_loader import verify_adapter, model_identity
    values, issues, hashes = {}, {}, {}
    for key, relative in config["models"].items():
        p = path(root, relative)
        try:
            if not p.is_dir():
                raise ValueError("LOCAL model directory missing; no Hub fallback: " + str(p))
            if key == "a3":
                provenance = verify_adapter(p)
                if provenance.get("smoke_step_limit") is not None or provenance.get("epoch", 0) < 3:
                    raise ValueError("A3 is not a full >=3 epoch adapter")
                train = path(root, config["training_bundle"]) / "train.jsonl"
                if provenance["train_sha256"] != sha(train):
                    raise ValueError("A3 does not bind the audited V2 Train bytes")
                info = {"path": str(p), "provenance": provenance}
            else:
                info = inference_identity(p, sentence_model=key in {"bge", "a2_1", "a2_2"})
                if key == "bge" and read_checkpoint_provenance(p) is not None:
                    raise ValueError("A1/A0/A4 original BGE path points to an A2-trained output")
                if key in {"a2_1", "a2_2"}:
                    provenance = read_checkpoint_provenance(p, require_a2=True)
                    if provenance["mode"] != key or provenance["completed_epoch"] < 3:
                        raise ValueError("wrong A2 mode or incomplete epochs")
                    expected = sha(path(root, config["training_bundle"]) / "a2_train.jsonl")
                    if not any(str(n).replace("\\", "/").endswith("/a2_train.jsonl") and h == expected
                               for n, h in provenance["input_sha256"].items()):
                        raise ValueError("A2 checkpoint does not bind audited V2 Train")
                    info["provenance"] = provenance
            if info.get("provenance", {}).get("provisional") and not allow_provisional:
                raise ValueError("provisional trained model requires explicit --allow-provisional; never Formal")
            # Preserve every existing model file in the identity, not only weights.
            for f in sorted(p.rglob("*")):
                if f.is_file():
                    hashes[str(f)] = sha(f)
            values[key] = {"value": info}
        except Exception as e:
            issues[key] = str(e)
    if "base" in values and "a3" in values:
        try:
            identity = model_identity(root, str(path(root, config["models"]["base"])))[1]
            if identity != values["a3"]["value"]["provenance"]["base_identity"]:
                raise ValueError("A3 original base identity differs (no replacement)")
        except Exception as e:
            issues["a3"] = str(e)
            values.pop("a3", None)
    for a, b in (("a2_1", "a2_2"),):
        if a in values and b in values:
            first, second = (values[k]["value"]["provenance"] for k in (a, b))
            if (first["positive_pair_schedule_sha256"] != second["positive_pair_schedule_sha256"]
                    or first["initial_local_identity"] != second["initial_local_identity"]):
                issues[a] = issues[b] = "A2 positive schedule or original base differs"
    if "bge" in values:
        original = values["bge"]["value"]["files_sha256"]
        for key in ("a2_1", "a2_2"):
            if key not in values:
                continue
            identity = values[key]["value"]["provenance"].get("initial_local_identity")
            files = {f["name"].replace("\\", "/"): f["sha256"] for f in (identity or {}).get("files", [])}
            if not files or any(files.get(name) != value for name, value in original.items()):
                issues[key] = "current original BGE inference files do not match A2 initial checkpoint receipt"
    return values, issues, hashes


def audit_index(root, config, data):
    """Trusted local project pickle only; never unpickle an uploaded test input."""
    import pickle
    from .data import normalize
    directory = path(root, config["dense_index"])
    hashes = {str(directory / n): sha(directory / n) for n in ("index.faiss", "index.pkl")}
    with (directory / "index.pkl").open("rb") as f:
        payload = pickle.load(f)
    docs = payload["docstore"]
    texts = [(i, normalize(d.page_content)) for i, d in enumerate(docs)]
    findings = []
    for r in data:
        q = normalize(r["question"])
        if len(q) >= 8:
            indices = [i for i, text in texts if q in text]
            if indices:
                findings.append({"question_id": r["id"], "document_positions": indices,
                                 "meaning": "full question present in knowledge text; not proof of deliberate injection"})
    groups = {d.metadata.get("source_group") for d in docs}
    if groups != {"type1", "type2"}:
        raise ValueError("dense index is not the existing Type1+Type2 document index")
    for d in docs:
        source = d.metadata.get("source_file", "")
        if "benchmark_804" in str(source) or "results" in str(source).replace("\\", "/").split("/"):
            raise ValueError("index source points at test/results data")
    return {"status": "inspected_local_docstore", "documents": len(docs),
            "full_test_question_matches": findings, "cache_initial_state": "empty_new_instance_per_mode_repeat",
            "source_groups": sorted(groups)}, hashes
