"""Mock completed SD/Greedy outputs; real routing, Cache, HTTP and CSV."""
import csv
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

import numpy as np

from baseline_api import speculative, server
from baseline_api.cache import ReviewedCacheA0
from baseline_api.integrated_chat import IntegratedQA
from experiments.baselines.rag_baseline.prompts import build_prompt
from test_prompt_encoding import Tokenizer

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'experiments/configs/integrated_baseline_v1.experimental.speculative.json'


def measured(fn, *args, **kwargs):
    start = time.perf_counter()
    value = fn(*args, **kwargs)
    return value, time.perf_counter()-start


class FallbackTests(unittest.TestCase):
    def setUp(self):
        for name in ['baseline_api.speculative.measured', 'experiments.shared.a0_adapter.measured',
                     'baseline_api.integrated_chat.measured']:
            patcher = patch(name, measured)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.config = json.loads(CONFIG.read_text(encoding='utf-8'))
        self.calls = []
        self.match = False
        self.fail_greedy = False
        self.greedy_truncated = False
        self.empty_greedy = False
        tokenizer = Tokenizer()
        self.wrapper = speculative.SpeculativeA0.__new__(speculative.SpeculativeA0)
        self.wrapper.config = dict(self.config, device='cpu')
        self.wrapper.options = self.config['speculative']
        self.wrapper.last = self.wrapper.question = None
        def generate(snapshot, inputs, version, *args):
            self.calls.append((version, inputs))
            if version == 'SD0' and self.fail_greedy:
                raise RuntimeError('Greedy failure fixture')
            greedy = version == 'SD0'
            same = not greedy or self.match
            ids = [4, 9] if same else [5, 6, 9]
            return {'generation_mode': version, 'forward_calls': {'target': 1, 'draft': int(not greedy)},
                    'generated_token_count': len(ids), 'output_token_ids': ids,
                    'output_text': '' if greedy and self.empty_greedy else 'SD answer' if same else 'Target answer',
                    'generation_seconds': .2 if greedy else .1, 'tokens_per_second': 20,
                    'stop_reason': 'token_limit' if greedy and self.greedy_truncated else 'eos',
                    'reached_token_limit': greedy and self.greedy_truncated}
        self.wrapper.engine = N(tokenizer=tokenizer, target=N(config=N(max_position_embeddings=20000)),
                                draft=N(config=N(max_position_embeddings=20000)), draft_path='CPU Draft', generate=generate)
        def answer(q, **kw):
            prompt = build_prompt('CPU evidence', q, prompt_version=self.config['prompt_version'], tokenizer=tokenizer)
            result = self.wrapper.generate(prompt)
            return result, {'documents': [{'content': 'CPU evidence', 'metadata': {'source': 'fixture'}}],
                            'retrieval_seconds': .01, 'generation_seconds': .3, 'input_tokens': 3}
        self.wrapper.base = N(paths={'model': 'CPU Target'}, answer=answer, embed=lambda q: np.array([[1., 0.]], dtype=np.float32),
                              embedder=N(embedding_dim=2), pipeline=N(prompt_version=self.config['prompt_version']),
                              identity={'prompt_version': self.config['prompt_version']})
        self.wrapper.identity = self.wrapper.base.identity
        self.guard = N(verify=lambda: None, close=lambda: None)
        self.cache = ReviewedCacheA0(self.wrapper, self.config['cache'], resource_guard=self.guard)

    def test_match_uses_sd_and_exactly_two_generations(self):
        self.match = True
        answer, detail = self.wrapper.answer('q')
        self.assertEqual(answer, 'SD answer')
        self.assertEqual(detail['speculative']['final_generation_source'], 'sd')
        self.assertFalse(detail['speculative']['token_mismatch'])
        self.assertEqual(detail['speculative']['lossless_status'], 'passed_this_request')
        self.assertTrue(detail['speculative']['sd_speedup_eligible'])
        self.assertEqual(len(self.calls), 2)
        self.assertIs(self.calls[0][1], self.calls[1][1])

    def test_mismatch_uses_completed_greedy_and_final_tokens(self):
        answer, detail = self.wrapper.answer('q')
        sd = detail['speculative']
        self.assertEqual(answer, 'Target answer')
        self.assertEqual(sd['final_generation_source'], 'target_greedy_fallback')
        self.assertTrue(sd['token_mismatch'])
        self.assertFalse(sd['exact_token_match_greedy'])
        self.assertEqual(sd['lossless_status'], 'failed')
        self.assertFalse(sd['sd_speedup_eligible'])
        self.assertEqual(sd['output_token_ids'], [5, 6, 9])
        self.assertEqual(sd['output_token_count'], 3)
        self.assertEqual(sd['raw_generation']['output_token_ids'], [4, 9])
        self.assertEqual((sd['generation_seconds'], sd['greedy_verification_seconds']), (.1, .2))
        self.assertIsNone(sd['tokens_per_second'])
        self.assertEqual(len(self.calls), 2)

    def test_cache_saves_only_final_answer_and_hit_does_not_generate(self):
        answer, detail = self.cache.answer('q')
        self.assertEqual(answer, 'Target answer')
        self.assertTrue(detail['saved'])
        self.assertEqual(next(iter(self.cache.cache.entries.values())).answer, answer)
        answer, detail = self.cache.answer('q')
        self.assertTrue(detail['cache_hit'])
        self.assertEqual(answer, 'Target answer')
        self.assertEqual(len(self.calls), 2)

    def test_greedy_truncation_controls_cache_eligibility(self):
        self.greedy_truncated = True
        answer, detail = self.cache.answer('q')
        self.assertEqual(answer, 'Target answer')
        self.assertFalse(detail['saved'])
        self.assertTrue(detail['rag_detail']['truncated'])

    def test_greedy_failure_or_empty_answer_never_served_or_cached(self):
        self.fail_greedy = True
        with self.assertRaises(RuntimeError):
            self.cache.answer('q')
        self.assertFalse(self.cache.cache.entries)
        self.fail_greedy = False
        self.empty_greedy = True
        with self.assertRaises(ValueError):
            self.cache.answer('q')
        self.assertFalse(self.cache.cache.entries)

    def test_http_remains_ready_and_csv_marks_fallback_without_schema_change(self):
        from fastapi.testclient import TestClient
        engine = IntegratedQA.__new__(IntegratedQA)
        engine.adapter, engine.cache, engine.guard = self.wrapper, self.cache, self.guard
        engine.identity = {'fixture': True}
        engine.close = lambda: None
        engine.faq = N(match_only=lambda q, **kw: (None, {'faq_accepted': False}))
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'ROOT', Path(folder)), \
                patch('baseline_api.integrated_chat.IntegratedQA', return_value=engine):
            with TestClient(server.create_app(CONFIG)) as client:
                for _ in range(300):
                    if client.get('/health').status_code == 200:
                        break
                    time.sleep(.01)
                response = client.post('/chat', json={'question': 'q'})
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()['answer'], 'Target answer')
                self.assertEqual(response.json()['answer_source'], 'rag')
                self.assertEqual(response.json()['speculative']['final_generation_source'], 'target_greedy_fallback')
                self.assertEqual(client.get('/health').json()['status'], 'ready')
                response = client.post('/chat', json={'question': 'q'})
                self.assertEqual(response.json()['answer_source'], 'cache')
                path = Path(folder)/'experiments/results/api_logs/v6_chat_history.csv'
                with path.open(encoding='utf-8-sig', newline='') as f:
                    rows = list(csv.reader(f))
                self.assertEqual(rows[0], server.V6_CSV_COLUMNS)
                self.assertEqual(len(rows[1]), 12)
                self.assertEqual(rows[1][2], 'RAG + Target Greedy Fallback')
                self.assertEqual(rows[1][-1], '3')
                self.assertEqual(rows[2][2], 'Cache')
                self.assertEqual(rows[2][-1], '')
                from test_run_v6_questions import runner
                runner.validate_record(rows[1])
        self.assertEqual(len(self.calls), 2)


if __name__ == '__main__':
    unittest.main()
