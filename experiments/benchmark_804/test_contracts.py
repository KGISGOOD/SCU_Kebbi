"""CPU/stdlb contract tests, not model or performance experiments."""
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from .common import ROOT, INPUT, Records, fingerprint, read, sha
from .data import load, select, review, jobs
from .report import derive, expected
from .run_all import dependencies, validate_config
from .worker import restore, sd_verdict


def generation(tokens, version="SD0", repeat=0):
    return {"measurement_id": version + str(repeat), "scenario_id": "fixture", "question_id": "fixture",
            "input_token_count": 10, "output_token_ids": tokens, "output_token_sha256": fingerprint(tokens),
            "prompt_identity": "same", "retrieval_identity": "same", "target_identity": "same", "seed": 42 + repeat,
            "generation_seconds": 2.0, "end_to_end_seconds": 3.0, "stop_reason": "eos",
            "job": "sd", "repeat": repeat, "phase": "measured", "experiment": version, "status": "success"}


class Contracts(unittest.TestCase):
    def test_actual_allocations_and_immutable_input(self):
        before = sha(ROOT / INPUT)
        data, audit = load(ROOT)
        self.assertEqual(len(data), 804)
        self.assertEqual(len(select(data, "FAQ比對")), 561)
        self.assertEqual(len(select(data, "語意快取")), 362)
        self.assertEqual(len(select(data, "推測解碼候選")), 413)
        self.assertEqual(audit["candidate_conversations"], 40)
        self.assertEqual(audit["ungrouped_turns"], 18)
        self.assertFalse(audit["groups_verified_by_source"])
        self.assertEqual(before, sha(ROOT / INPUT))

    def test_no_annotations_is_not_gold(self):
        data, _ = load(ROOT)
        config = read(ROOT / "experiments/benchmark_804/config.example.json")
        annotations = review(ROOT, config, data)
        ti = next(j for j in jobs(data, annotations, config) if j["id"] == "ti")
        self.assertFalse(ti["available"])
        self.assertEqual(annotations["questions"], {})

    def test_expected_and_dependencies(self):
        data, _ = load(ROOT)
        c = read(ROOT / "experiments/benchmark_804/config.example.json")
        validate_config(c)
        self.assertEqual(expected({"kind": "qa", "architecture": "A0"}, data, c), 1878)
        self.assertEqual(expected({"kind": "sd", "draft": True}, data, c), 7434)
        self.assertEqual(dependencies({"kind": "qa", "architecture": "A4-2"}), {"base", "bge", "index", "reranker"})

    def test_attempts_preserved_and_corruption_blocked(self):
        with TemporaryDirectory(dir=str(ROOT / "experiments/benchmark_804")) as d:
            self.assertIn(ROOT, Path(d).resolve().parents)
            store = Records(Path(d))
            store.put("q", {"status": "failed"})
            old = store.directory / store.index["q"]["file"]
            previous_hash = sha(old)
            store.put("q", {"status": "success"})
            self.assertEqual(sha(old), previous_hash)
            current = store.directory / store.index["q"]["file"]
            current.write_text("{}", encoding="utf-8")
            with self.assertRaises(ValueError):
                store.get("q")

    def test_strict_eos_mismatch_not_relaxed(self):
        baseline = generation([1] * 81 + [128009])
        mismatch = generation([1] * 81 + [26892, 128009], "SD2-2-k4")
        result = sd_verdict(mismatch, baseline)
        self.assertEqual(result["status"], "failed_token_mismatch")
        self.assertFalse(result["performance_eligible"])
        self.assertIsNone(result["generation_speedup_sd0"])
        self.assertEqual(result["token_difference"]["first_different_new_token_index"], 81)

    def test_paired_identity_mismatch_is_error(self):
        first, second = generation([1]), generation([1], "SD2-1")
        second["prompt_identity"] = "other"
        with self.assertRaises(ValueError):
            sd_verdict(second, first)

    def test_baseline_repeatability_blocks_all_eligible_claims(self):
        one, two = generation([1], repeat=0), generation([2], repeat=1)
        result = derive([one, two])
        self.assertTrue(all(r["status"] == "failed_sd0_not_repeatable" for r in result))
        self.assertTrue(all(not r["performance_eligible"] for r in result))

    def test_exact_cache_restore_does_not_generate(self):
        from experiments.algorithms.semantic_cache.cache import AnswerCache
        cache = AnswerCache("exact", {"frozen": True}, 1)
        cid = cache.insert("fixture question", "fixture answer")
        rebuilt = SimpleNamespace(cache=AnswerCache("exact", {"frozen": True}, 1))
        restore(rebuilt, {"cache_id": cid, "query": "fixture question", "answer": "fixture answer", "hit": False})
        self.assertEqual(rebuilt.cache.lookup("fixture question")[0].cache_id, cid)
        self.assertEqual(rebuilt.cache.lookup("fixture question")[0].answer, "fixture answer")


if __name__ == "__main__":
    unittest.main()
