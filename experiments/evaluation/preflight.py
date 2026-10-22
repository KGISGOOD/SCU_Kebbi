"""Offline resource inventory, hashes and per-architecture readiness (no generation)."""
from pathlib import Path
import json
import os

from .common import DataError, digest, environment, read_json, read_rows, resolve, sha
from .model_resources import cached_model
from .prepare_a1 import index_completion_report

NEEDS = {
    "A0": ("bge", "base", "dense_index"),
    "A1": ("bge", "a1_index", "a1_canonical"),
    "A2-1": ("a2_candidates", "a2_1_model"),
    "A2-2": ("a2_candidates", "a2_2_model"),
    "A3": ("base", "a3_adapter"),
    "A4-1": ("bge", "base", "dense_index", "raw_documents"),
    "A4-2": ("bge", "base", "dense_index", "raw_documents", "reranker"),
    "A5": ("bge", "base", "dense_index", "a3_adapter"),
}


def directory_identity(path):
    if not path.is_dir():
        raise DataError("directory missing: %s" % path)
    files = {p.relative_to(path).as_posix(): sha(p) for p in sorted(path.rglob("*")) if p.is_file()}
    if not files:
        raise DataError("empty directory: %s" % path)
    return {"path": str(path), "files_sha256": files, "content_sha256": digest(files)}


def inspect_resources(root, config, questions):
    # Preflight never downloads or imports a model. Actual loader checks are optional
    # child processes (--load-models), and are reported separately from this inventory.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    resources = {}

    def check(name, function):
        try:
            resources[name] = {"status": "ready", "value": function(), "error": None}
        except Exception as exc:
            resources[name] = {"status": "failed", "value": None,
                               "error": "%s: %s" % (type(exc).__name__, exc)}

    def index(name):
        directory = resolve(root, config[name])
        files = {}
        for filename in ("index.faiss", "index.pkl"):
            path = directory / filename
            files[filename] = sha(path)  # missing is an error, never create
        version = directory / "version.json"
        if version.exists():
            files["version.json"] = sha(version)
            read_json(version)  # malformed version is not silently ignored
        return {"path": str(directory), "files_sha256": files,
                "provenance_verified": False,
                "warning": "version/shape/content need loader validation" if version.exists()
                           else "version.json missing; build provenance unverified"}

    def a1():
        path = resolve(root, config["a1_canonical"])
        rows = read_rows(path)
        expected = {row["faq_id"]: row for row in questions}
        ids = [row.get("faq_id") for row in rows]
        if any(not isinstance(i, str) for i in ids) or len(ids) != 60 or len(set(ids)) != 60 or set(ids) != set(expected):
            raise DataError("A1 canonical must contain all 60 P/S IDs, actual=%d; no partial FAQ test or PP/SS IDs"
                            % len(rows))
        for row in rows:
            source = expected[row["faq_id"]]
            if row.get("question") != source["question"] or row.get("answer") != source["reference_answer"]:
                raise DataError("A1 %s Q/A differs from original CSV" % row["faq_id"])
        version_path = resolve(root, config["a1_index"]) / "version.json"
        if version_path.is_file():
            version = read_json(version_path)
            if version.get("schema_version") == "a1_60_preparation_v1":
                index_dir = version_path.parent
                completion = index_completion_report(path, index_dir, version)
                if completion["status"] != "passed":
                    raise DataError("A1 new index completion/hash/canonical mismatch: "
                                    + json.dumps(completion, ensure_ascii=True))
                bge = resources["bge"]
                if (bge["status"] != "ready"
                        or version.get("embedding_model", {}).get("files_sha256") != bge["value"]["files_sha256"]):
                    raise DataError("A1 index was built with different BGE inference files; select its original snapshot")
        return {"path": str(path), "sha256": sha(path), "faq_count": 60}

    def a2_corpus():
        from experiments.a2_faq_finetuning.faq_answering import load_answering_data
        path = resolve(root, config["a2_candidates"])
        corpus, answers = load_answering_data(root, path)
        return {"path": str(path), "sha256": sha(path), "faq_count": len(corpus), "answer_count": len(answers)}

    def a2(name, mode):
        from experiments.a2_faq_finetuning.common import read_checkpoint_provenance
        path = resolve(root, config[name])
        provenance = read_checkpoint_provenance(path, require_a2=True)
        if provenance["mode"] != mode:
            raise DataError("%s checkpoint mode mismatch" % name)
        if provenance["provisional"] and not config["allow_unreviewed"]:
            raise DataError("%s provisional model requires --allow-unreviewed" % name)
        return {**directory_identity(path), "provenance": provenance}

    def a3():
        from experiments.a3_faq_sft.model_loader import verify_adapter
        path = resolve(root, config["a3_adapter"])
        provenance = verify_adapter(path)
        if provenance["provisional"] and not config["allow_unreviewed"]:
            raise DataError("A3 provisional adapter requires --allow-unreviewed")
        run_files = {}
        for filename in ("training_config.json", "dataset_manifest.json", "completed.json"):
            source = path.parent / filename
            run_files[filename] = sha(source)
            read_json(source)
        completed = read_json(path.parent / "completed.json")
        if (completed.get("status") not in ("completed_smoke", "completed_training")
                or completed.get("optimizer_steps") != provenance["optimizer_steps"]):
            raise DataError("A3 training completion state differs from adapter provenance")
        return {**directory_identity(path), "provenance": provenance, "training_run_sha256": run_files}

    def raw():
        hashes = {}
        for group in ("type1", "type2"):
            directory = root / "畢業專題_爬蟲" / group
            paths = [p for p in sorted(directory.rglob("*")) if p.is_file() and p.suffix.lower() in (".csv", ".txt")]
            if not paths:
                raise DataError("no BM25 source files: %s" % directory)
            hashes.update({p.relative_to(root).as_posix(): sha(p) for p in paths})
        return {"files_sha256": hashes, "construction": "existing A4 loader/chunker; BM25 built in memory; no save"}

    check("a3_adapter", a3)
    revision = None
    if resources["a3_adapter"]["status"] == "ready":
        provenance = resources["a3_adapter"]["value"]["provenance"]
        if provenance["base_identity"].get("kind") == "hub":
            if provenance["base_identity"].get("model_id") != config["base_model"]:
                # Do not let an incompatible Adapter change/fail A0's chosen base;
                # A3/A5 source mismatch is reported by architecture_preflight.
                revision = None
            else:
                revision = provenance.get("base_revision")
    check("base", lambda: cached_model(root, config["base_model"], revision))
    check("bge", lambda: cached_model(root, config["bge_model"], snapshot=config.get("bge_snapshot")))
    check("reranker", lambda: cached_model(root, config["reranker_model"]))
    check("dense_index", lambda: index("dense_index"))
    check("a1_index", lambda: index("a1_index"))
    check("a1_canonical", a1)
    check("a2_candidates", a2_corpus)
    check("a2_1_model", lambda: a2("a2_1_model", "a2_1"))
    check("a2_2_model", lambda: a2("a2_2_model", "a2_2"))
    check("raw_documents", raw)
    return resources


def architecture_preflight(resources, architectures):
    env = environment()["packages"]
    reports = {}
    for arch in architectures:
        errors = ["%s: %s" % (name, resources[name]["error"])
                  for name in NEEDS[arch] if resources[name]["status"] != "ready"]
        required = ["torch", "numpy", "transformers", "sentence-transformers"] if arch != "A3" else ["torch", "transformers", "peft"]
        if arch in ("A5",):
            required.append("peft")
        if arch in ("A0", "A1", "A4-1", "A4-2", "A5"):
            required.append("langchain-core")
            if env["faiss-cpu"] is None and env["faiss-gpu"] is None:
                errors.append("FAISS package metadata missing")
        for name in required:
            if env.get(name) is None:
                errors.append("package missing: " + name)
        if arch in ("A3", "A5") and all(resources[n]["status"] == "ready" for n in ("base", "a3_adapter")):
            provenance = resources["a3_adapter"]["value"]["provenance"]
            base = resources["base"]["value"]
            expected = provenance["base_identity"]
            actual = ({"kind": "local", "files_sha256": base["files_sha256"]}
                      if expected.get("kind") == "local" else
                      {"kind": "hub", "model_id": base["model_argument"]})
            if expected != actual or (expected.get("kind") == "hub" and not provenance.get("base_revision")):
                errors.append("A3/A5 base source identity does not match Adapter training provenance")
        warnings = [resources[name]["value"]["warning"] for name in NEEDS[arch]
                    if resources[name]["status"] == "ready" and resources[name]["value"].get("warning")]
        reports[arch] = {"status": "failed" if errors else "resources_ready_not_loaded",
                         "errors": errors, "warnings": warnings, "model_load_verified": False}
    return reports
