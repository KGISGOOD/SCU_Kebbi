"""Strict FAQ/Review data checks and CPU FAISS initialization with a fake embedder."""
import csv
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

from baseline_api.faq_review import load_csv_candidates, read_reviews, FIELDS

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'experiments/configs/integrated_baseline_v1.experimental.speculative.json').read_text(encoding='utf-8'))


class FAQReviewTests(unittest.TestCase):
    def setUp(self):
        self.candidates, self.digest = load_csv_candidates(ROOT/CONFIG['faq']['faq_file'])
        with (ROOT/CONFIG['faq']['review_file']).open(encoding='utf-8-sig', newline='') as f:
            self.reviews = list(csv.DictReader(f))

    def load_modified(self, rows, candidates=None):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'review.csv'
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            return read_reviews(path, candidates or self.candidates)

    def test_current_dataset_has_exactly_138_corresponding_ids(self):
        decisions = read_reviews(ROOT/CONFIG['faq']['review_file'], self.candidates)
        self.assertEqual(len(decisions), 138)
        self.assertEqual(set(decisions), {r['question_id'] for r in self.candidates})
        self.assertTrue(all(r['decision'] == 'needs_review' for r in decisions.values()))

    def test_unknown_and_duplicate_are_distinct_errors(self):
        self.reviews[0]['faq_id'] = 'FAQ_old'
        with self.assertRaisesRegex(ValueError, 'unknown FAQ review ID'):
            self.load_modified(self.reviews)
        self.reviews[0]['faq_id'] = self.candidates[0]['question_id']
        with self.assertRaisesRegex(ValueError, 'duplicate FAQ review ID'):
            self.load_modified(self.reviews + [self.reviews[0]])

    def test_changed_content_or_dataset_identity_still_rejected(self):
        self.reviews[0]['answer'] += ' changed'
        with self.assertRaisesRegex(ValueError, 'candidate changed'):
            self.load_modified(self.reviews)
        self.setUp()
        changed = [dict(r, dataset_sha256='different') for r in self.candidates]
        with self.assertRaisesRegex(ValueError, 'candidate changed'):
            self.load_modified(self.reviews, changed)

    def test_missing_record_and_unsubstantiated_approval_rejected(self):
        with self.assertRaisesRegex(ValueError, 'every canonical ID'):
            self.load_modified(self.reviews[1:])
        self.reviews[0].update(decision='approved', reviewer='CPU fixture', reviewed_at='2026-11-18T00:00:00+00:00', evidence='fixture')
        with self.assertRaisesRegex(ValueError, 'insufficient evidence'):
            self.load_modified(self.reviews)

    def test_old_decisions_retained_in_reconciliation_record(self):
        report = json.loads((ROOT/'experiments/datasets/faq_review_reconciliation.json').read_text(encoding='utf-8'))
        self.assertEqual(len(report['old_review_records']), 138)
        self.assertEqual(report['faq_sha256'], self.digest)
        self.assertEqual(report['counts']['approved_transferred'], 0)
        self.assertEqual(report['counts']['duplicate_old_ids'], 0)

    def test_faqfallback_actual_csv_review_and_faiss_initialize(self):
        import numpy as np
        from baseline_api.faq import FAQFallback
        fake_dtype = object()
        class Embedder:
            embedding_dim = 138
            normalize_embeddings = True
            model = N(max_seq_length=512, eval=lambda: None,
                      parameters=lambda: [N(dtype=fake_dtype, is_floating_point=lambda: True)])
            def encode(self, texts):
                return np.eye(138, dtype=np.float32)
        adapter = N(identity={'prompt_version': CONFIG['prompt_version']}, paths={'bge': Path('CPU_FAKE_BGE')},
                    pipeline=N(prompt_version=CONFIG['prompt_version']), embedder=Embedder(), validate_query=lambda q: None)
        # No model loader is called. Keep CSV/Review checks, Matcher and FAISS real.
        with patch.dict('sys.modules', {'torch': N(float32=fake_dtype, cuda=N(is_available=lambda: False)),
                         'experiments.shared.embeddings': N(BGEM3Embedder=object)}), \
                patch('experiments.shared.model_resources.inference_identity',
                      return_value={'content_sha256': CONFIG['faq']['bge_content_sha256']}):
            faq = FAQFallback(ROOT, CONFIG, adapter)
            self.assertEqual(faq.matcher.vector_store.index.ntotal, 138)
            self.assertEqual(faq.identity['faq']['human_approved_candidates'], 0)
            vector = np.eye(138, dtype=np.float32)[0:1]
            answer, detail = faq.match_only(self.candidates[0]['question'], query_vector=vector)
            self.assertEqual(answer, self.candidates[0]['answer'])
            self.assertTrue(detail['faq_accepted'])
            self.assertFalse(detail['faq_human_verified'])


if __name__ == '__main__':
    unittest.main()
