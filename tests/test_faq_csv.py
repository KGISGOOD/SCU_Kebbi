"""Single-CSV FAQ tests; actual loader/Matcher/FAISS, synthetic BGE vectors."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

from baseline_api.faq_data import load_csv_candidates
from baseline_api.faq import FAQFallback, acceptance
from experiments.shared import model_resources

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT/'experiments/configs/integrated_baseline_v1.experimental.speculative.json').read_text(encoding='utf-8'))


class FAQCSVTests(unittest.TestCase):
    def setUp(self):
        self.rows, self.digest = load_csv_candidates(ROOT/CONFIG['faq']['faq_file'])

    def test_only_two_columns_138_rows_and_exact_answers(self):
        with (ROOT/CONFIG['faq']['faq_file']).open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ['問題', '答案'])
            original = list(reader)
        self.assertEqual(len(self.rows), 138)
        self.assertEqual([(r['question'], r['answer']) for r in self.rows],
                         [(r['問題'], r['答案']) for r in original])
        self.assertEqual(len({r['question_id'] for r in self.rows}), 138)
        self.assertTrue(all('review_status' not in r for r in self.rows))
        self.assertNotIn('review_file', CONFIG['faq'])

    def test_threshold_margin_conditions_and_empty_answer(self):
        candidate = dict(self.rows[0])
        q = candidate['question']
        self.assertEqual(acceptance(q, candidate, .89, .9, second_score=.7)[1], 'below_threshold')
        self.assertEqual(acceptance(q, candidate, .95, .9, second_score=.94)[1], 'insufficient_top2_margin')
        self.assertEqual(acceptance(q, dict(candidate, answer=''), 1., .9, second_score=0.)[1], 'empty_answer')
        self.assertEqual(acceptance('另一個意圖', candidate, 1., .9, second_score=0.)[1], 'intent_or_conditions_unverified')
        for status in ['approved', 'needs_review', 'rejected']:
            self.assertTrue(acceptance(q, dict(candidate, review_status=status), 1., .9, second_score=0.)[0])

    def test_wrong_columns_duplicate_questions_and_count_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'faq.csv'
            for header, data in [(['問題', '答案', 'extra'], [['q', 'a', 'x']]),
                                 (['問題', '答案'], [['q', 'a']]*138),
                                 (['問題', '答案'], [['q', 'a']])]:
                with path.open('w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f);writer.writerow(header);writer.writerows(data)
                with self.assertRaises(ValueError):
                    load_csv_candidates(path)

    def initialization(self, root, config):
        import numpy as np
        dtype = object()
        class Embedder:
            embedding_dim = 138
            normalize_embeddings = True
            model = N(max_seq_length=512, eval=lambda: None,
                      parameters=lambda: [N(dtype=dtype, is_floating_point=lambda: True)])
            def __init__(self):
                self.builds = []
            def encode(self, texts):
                self.builds.append(list(texts))
                return np.eye(138, dtype=np.float32)
        adapter = N(identity={'prompt_version': config['prompt_version']}, paths={'bge': Path('CPU_FAKE_BGE')},
                    pipeline=N(prompt_version=config['prompt_version']), embedder=Embedder(), validate_query=lambda q: None)
        with patch.dict('sys.modules', {'torch': N(float32=dtype, cuda=N(is_available=lambda: False)),
                         'experiments.shared.embeddings': N(BGEM3Embedder=object)}), \
                patch.object(model_resources, 'inference_identity',
                      return_value={'content_sha256': config['faq']['bge_content_sha256']}):
            faq = FAQFallback(root, config, adapter)
            vector = np.eye(138, dtype=np.float32)[0:1]
            q = faq.matcher.faqs[0]['question']
            answer, detail = faq.match_only(q, query_vector=vector)
            self.assertEqual(answer, faq.matcher.faqs[0]['answer'])
            self.assertTrue(detail['faq_accepted'])
            self.assertFalse(detail['faq_human_verified'])
        return faq, adapter

    def test_initialization_without_any_review_files_and_direct_hit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'only.csv'
            path.write_bytes((ROOT/CONFIG['faq']['faq_file']).read_bytes())
            config = dict(CONFIG, faq=dict(CONFIG['faq'], faq_file='only.csv'))
            before = path.read_bytes()
            faq, adapter = self.initialization(root, config)
            self.assertEqual(faq.matcher.vector_store.index.ntotal, 138)
            self.assertEqual(len(adapter.embedder.builds), 1)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ['only.csv'])
            self.assertEqual(path.read_bytes(), before)
            self.assertFalse(faq.identity['faq']['review_required'])

    def test_restart_reads_new_question_answer_and_rebuilds_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory);path = root/'only.csv'
            path.write_bytes((ROOT/CONFIG['faq']['faq_file']).read_bytes())
            config = dict(CONFIG, faq=dict(CONFIG['faq'], faq_file='only.csv'))
            first, _ = self.initialization(root, config)
            with path.open('w', encoding='utf-8-sig', newline='') as f:
                writer = csv.writer(f);writer.writerow(['問題', '答案'])
                writer.writerows([[r['question']+' 新版' if i==0 else r['question'],
                                   r['answer']+' 新版' if i==0 else r['answer']] for i,r in enumerate(self.rows)])
            second, adapter = self.initialization(root, config)
            self.assertNotEqual(first.identity['faq']['input_sha256'], second.identity['faq']['input_sha256'])
            self.assertEqual(adapter.embedder.builds[0][0], self.rows[0]['question']+' 新版')
            self.assertEqual(second.matcher.faqs[0]['answer'], self.rows[0]['answer']+' 新版')

    def test_faq_miss_enters_cache_before_rag(self):
        from baseline_api.integrated_chat import IntegratedQA
        engine = IntegratedQA.__new__(IntegratedQA)
        engine.adapter = N(config={}, embed=lambda q: 'CPU vector')
        engine.guard = N(verify=lambda: None)
        engine.faq = N(match_only=lambda q, **kw: (None, {'faq_accepted': False}))
        calls = []
        def answer(q, **kw):
            calls.append((q, kw))
            return 'Cache fixture', {'answer_source': 'cache', 'answer_verification': 'unverified', 'timings': {}}
        engine.cache = N(answer=answer, cache=N(entries={}))
        with patch('baseline_api.integrated_chat.measured', lambda fn,*a,**kw:(fn(*a,**kw),.01)):
            result, detail = engine.answer('q')
        self.assertEqual(detail['answer_source'], 'cache')
        self.assertEqual(calls, [('q', {'query_vector': 'CPU vector'})])


if __name__ == '__main__':
    unittest.main()
