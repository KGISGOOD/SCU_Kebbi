"""CPU tests: real FAISS/lifecycle/routing/HTTP/CSV; no model inference."""
import csv
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as N
import unittest
from unittest.mock import Mock, patch

import numpy as np

from baseline_api import cache as module
from baseline_api.integrated_chat import IntegratedQA
from baseline_api import server

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'experiments/configs/integrated_baseline_v1.experimental.speculative.json'


def vector(i=0):
    v = np.zeros((1, 256), dtype=np.float32)
    v[0, i] = 1
    return v


class Adapter:
    def __init__(self):
        version = json.loads(CONFIG.read_text(encoding='utf-8'))['prompt_version']
        self.identity = {'prompt_version': version}
        self.config = {'speculative': {}}
        self.pipeline = N(prompt_version=version)
        self.embedder = N(embedding_dim=256)
        self.calls = 0
        self.output = 'CPU fixture answer'

    def embed(self, q):
        return vector()

    def answer(self, q, **kwargs):
        self.calls += 1
        return self.output, {'documents': [{'content': 'CPU evidence', 'metadata': {'source': 'fixture'}}],
                             'truncated': False, 'stop_reason': 'eos', 'retrieval_seconds': .01,
                             'generation_seconds': .02, 'input_tokens': 2,
                             'speculative': {'used': True, 'generation_seconds': .01,
                                             'greedy_verification_seconds': .01, 'output_token_count': 2}}


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.
        self.clock = patch.object(module, 'time', N(monotonic=lambda: self.now, perf_counter=time.perf_counter))
        self.clock.start()
        self.addCleanup(self.clock.stop)
        # These are CPU lifecycle tests: avoid importing the GPU timing helper's
        # torch runtime beside the local FAISS OpenMP runtime.
        def cpu_measured(fn, *args, **kwargs):
            start = time.perf_counter()
            value = fn(*args, **kwargs)
            return value, time.perf_counter() - start
        for target in ['experiments.shared.a0_adapter.measured', 'baseline_api.integrated_chat.measured']:
            timer = patch(target, cpu_measured)
            timer.start()
            self.addCleanup(timer.stop)
        config = json.loads(CONFIG.read_text(encoding='utf-8'))
        self.options = config['cache']
        self.adapter = Adapter()
        self.guard = N(verify=Mock(), close=Mock())
        self.c = module.ReviewedCacheA0(self.adapter, self.options, resource_guard=self.guard)

    def save(self, q, i=0, answer='answer'):
        return self.c.save({'query': q, 'answer': answer, 'vector': vector(i)}, 'unverified')[0]

    def consistent(self):
        self.c.check_metadata()
        self.assertEqual(self.c.cache.index.ntotal, len(self.c.cache.entries))

    def test_defaults(self):
        self.assertEqual((self.options['threshold'], self.options['max_entries'], self.options['ttl_seconds']), (.9, 128, 3600))
        self.assertEqual(self.options['acceptance_mode'], 'experimental')
        self.assertTrue(self.options['auto_write'])
        self.assertFalse(self.options['persistent'])

    def test_capacity_129_evicts_oldest(self):
        keys = [self.save('q'+str(i), i) for i in range(129)]
        self.assertNotIn(keys[0], self.c.cache.entries)
        self.assertNotIn(keys[0], self.c.reviews)
        self.assertNotIn(keys[0], self.c.lru)
        self.assertEqual(len(self.c.cache.entries), 128)
        self.consistent()

    def test_hit_updates_lru_not_ttl(self):
        first = self.save('q0', 0)
        for i in range(1, 128):
            self.save('q'+str(i), i)
        stored = self.c.reviews[first]['monotonic']
        self.now += 10
        _, detail = self.c.answer('q0', query_vector=vector(0))
        self.assertEqual(detail['hit_type'], 'Exact')
        self.assertEqual(detail['answer_verification'], 'unverified')
        self.assertEqual(self.c.reviews[first]['monotonic'], stored)
        self.save('q128', 128)
        self.assertIn(first, self.c.cache.entries)
        self.assertNotIn('q1', self.c.cache.exact)
        self.assertEqual(self.adapter.calls, 0)

    def test_semantic_hit_moves_lru(self):
        key = self.save('112學年度畢業要多少學分？')
        self.save('other', 1)
        _, detail = self.c.answer('請問112學年度畢業要幾學分？', query_vector=vector())
        self.assertEqual(detail['hit_type'], 'Semantic')
        self.assertEqual(next(reversed(self.c.lru)), key)

    def test_miss_does_not_touch_existing_lru(self):
        self.save('112學年度畢業要多少學分？')
        self.save('other', 1)
        order = list(self.c.lru)
        self.adapter.output = ''  # Ineligible Miss must not add a new entry.
        for q in ['113學年度畢業要多少學分？', '轉學生112學年度畢業要多少學分？',
                  '112學年度畢業不用多少學分？', '系辦信箱是什麼？']:
            _, detail = self.c.answer(q, query_vector=vector())
            self.assertFalse(detail['cache_hit'])
            self.assertEqual(list(self.c.lru), order)

    def test_expiration_boundary_removes_all_metadata_and_vectors(self):
        key = self.save('q')
        self.now += 3600
        self.assertEqual(self.c.lookup('q', vector()), (None, None, None))
        self.assertNotIn(key, self.c.reviews)
        self.assertNotIn(key, self.c.lru)
        self.assertEqual(self.c.cache.labels, {})
        self.assertEqual(self.c.cache.exact, {})
        self.consistent()

    def test_expired_query_runs_rag_and_recreates_new_answer(self):
        key = self.save('q', answer='old')
        self.now += 3600
        answer, d = self.c.answer('q', query_vector=vector())
        self.assertFalse(d['cache_hit'])
        self.assertTrue(d['saved'])
        self.assertEqual(self.c.cache.entries[key].answer, answer)
        self.assertEqual(self.c.reviews[key]['monotonic'], self.now)
        self.consistent()

    def test_write_cleans_expired_before_capacity_eviction(self):
        old = self.save('old', 0)
        self.now += 1
        for i in range(1, 128):
            self.save('q'+str(i), i)
        self.now = 3700
        self.save('new', 128)
        self.assertNotIn(old, self.c.cache.entries)
        self.assertIn('q1', self.c.cache.exact)
        self.assertEqual(len(self.c.cache.entries), 128)
        self.consistent()

    def test_expired_top1_does_not_hide_valid_candidate(self):
        self.save('expired')
        self.now += 1
        self.save('valid')
        self.now = 3700
        _, detail = self.c.answer('valid', query_vector=vector())
        self.assertTrue(detail['cache_hit'])
        self.assertNotIn('expired', self.c.cache.exact)
        self.assertEqual(self.adapter.calls, 0)
        self.consistent()

    def test_conflict_and_identical_save_do_not_refresh_ttl(self):
        key = self.save('q')
        self.now += 10
        result, reason = self.c.save({'query': 'q', 'answer': 'different', 'vector': vector()}, 'unverified')
        self.assertIsNone(result)
        self.assertEqual(reason, 'append_only_conflict')
        self.save('q')
        self.assertEqual(self.c.reviews[key]['monotonic'], 100.)
        self.assertEqual(self.c.cache.entries[key].answer, 'answer')

    def test_repeated_evictions_never_reuse_live_faiss_labels(self):
        for i in range(400):
            self.save('q'+str(i), i % 256)
            self.consistent()
        self.assertEqual(len(self.c.cache.labels), 128)
        self.assertEqual(self.c.cache.next_label, 400)
        for i in range(272, 400):
            entry, score, candidate = self.c.lookup('q'+str(i), vector(i % 256))
            self.assertEqual(entry.query, 'q'+str(i))

    def test_index_remove_failure_fails_closed(self):
        key = self.save('q')
        with patch.object(self.c.cache.index, 'remove_ids', side_effect=RuntimeError('fixture failure')):
            with self.assertRaises(RuntimeError):
                self.c.remove(key)
        with self.assertRaises(RuntimeError):
            self.c.cache.lookup('q', vector())

    def test_invalid_insert_does_not_evict(self):
        for i in range(128):
            self.save('q'+str(i), i)
        before = list(self.c.lru)
        with self.assertRaises(ValueError):
            self.c.save({'query': 'bad', 'answer': 'valid', 'vector': np.zeros(256)}, 'unverified')
        self.assertEqual(before, list(self.c.lru))

    def integrated(self):
        engine = IntegratedQA.__new__(IntegratedQA)
        engine.adapter, engine.cache, engine.guard = self.adapter, self.c, self.guard
        engine.identity = {'fixture': True}
        engine.faq = N(match_only=lambda q, **kw: ('FAQ answer' if q == 'FAQ' else None,
                       {'faq_accepted': q == 'FAQ', 'faq_top1_score': 1., 'faq_top2_score': 0.,
                        'faq_acceptance_reason': 'fixture', 'faq_rejection_reason': None}))
        return engine

    def test_faq_bypasses_cache_and_generation(self):
        engine = self.integrated()
        with patch.object(self.c, 'answer', side_effect=AssertionError('FAQ touched Cache')):
            answer, d = engine.answer('FAQ')
        self.assertEqual(d['answer_source'], 'faq')
        self.assertEqual(len(self.c.cache.entries), 0)
        self.assertEqual(self.adapter.calls, 0)

    def test_resource_and_generation_failure_never_writes(self):
        self.guard.verify.side_effect = ValueError('changed')
        with self.assertRaises(ValueError):
            self.c.answer('q', query_vector=vector())
        self.guard.verify.side_effect = None
        with patch.object(self.adapter, 'answer', side_effect=ValueError('SD mismatch')):
            with self.assertRaises(ValueError):
                self.c.answer('q', query_vector=vector())
        self.assertFalse(self.c.cache.entries)

    def test_http_and_csv_contract_unchanged(self):
        from fastapi.testclient import TestClient
        engine = self.integrated()
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'ROOT', Path(folder)), \
                patch('baseline_api.integrated_chat.IntegratedQA', return_value=engine):
            with TestClient(server.create_app(CONFIG)) as client:
                for _ in range(300):
                    if client.get('/health').status_code == 200:
                        break
                    time.sleep(.01)
                responses = [client.post('/chat', json={'question': q}) for q in ['FAQ', 'q', 'q']]
                for response in responses:
                    self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual([r.json()['answer_source'] for r in responses], ['faq', 'rag', 'cache'])
                fields = (server.ChatResponse.model_fields if hasattr(server.ChatResponse, 'model_fields')
                          else server.ChatResponse.__fields__)
                self.assertEqual(set(responses[0].json()), set(fields))
                path = Path(folder) / 'experiments/results/api_logs/v6_chat_history.csv'
                self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))
                with path.open(encoding='utf-8-sig', newline='') as f:
                    rows = list(csv.reader(f))
                expected = ['問題', '答案', '回答來源', 'Embedding 耗時（秒）', 'FAQ 耗時（秒）',
                            'Cache 耗時（秒）', 'RAG 檢索耗時（秒）', 'SD 生成耗時（秒）',
                            'Greedy 驗證耗時（秒）', '資源驗證耗時（秒）', '總耗時（秒）', 'Output Tokens']
                self.assertEqual(rows[0], expected)
                self.assertEqual(len(rows), 4)
                self.assertEqual([r[-1] for r in rows[1:]], ['', '2', ''])
                for row in rows[1:]:
                    for value in row[3:11]:
                        if value:
                            self.assertRegex(value, r'^\d+\.\d{4}$')
                for response in responses:
                    self.assertRegex(response.text, r'"latency_seconds":\d+\.\d{4}[,}]')
                    self.assertIsInstance(response.json()['latency_seconds'], float)
                self.assertIsNone(responses[0].json()['stage_timings']['cache_lookup_seconds'])
                self.assertIsNone(responses[0].json()['stage_timings']['rag_llm_seconds'])
                self.assertIsNone(responses[2].json()['stage_timings']['rag_llm_seconds'])
                self.assertIsNone(responses[2].json()['stage_timings']['cache_write_seconds'])


if __name__ == '__main__':
    unittest.main()
