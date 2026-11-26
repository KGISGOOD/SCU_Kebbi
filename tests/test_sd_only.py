"""SD-only mode contracts with synthetic engine outputs; no GPU or model loading."""
from copy import deepcopy
import csv
import json
from pathlib import Path
import tempfile
import time
from types import SimpleNamespace as N
import unittest
from unittest.mock import patch

from baseline_api import server, speculative
from baseline_api.integrated_chat import IntegratedQA
import test_sd_fallback as fixture

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT/'experiments/configs/integrated_baseline_v1.experimental.speculative.sd_only.json'


class SDOnlyTests(unittest.TestCase):
    def setUp(self):
        fixture.FallbackTests.setUp(self)
        self.wrapper.options.update(mode='sd_only', verify_greedy=False)

    def integrated(self):
        engine = IntegratedQA.__new__(IntegratedQA)
        engine.adapter, engine.cache, engine.guard = self.wrapper, self.cache, self.guard
        engine.identity = {'fixture': True}
        engine.close = lambda: None
        engine.faq = N(match_only=lambda q, **kw:
                       ('FAQ answer', {'faq_accepted': True}) if q == 'FAQ'
                       else (None, {'faq_accepted': False}))
        return engine

    def test_explicit_configs_and_original_defaults_are_isolated(self):
        original = server.load_settings(fixture.CONFIG)
        new = server.load_settings(CONFIG)
        self.assertNotIn('mode', original['speculative'])
        self.assertTrue(original['speculative']['verify_greedy'])
        speculative.validate_speculative(original)
        self.assertEqual(new['speculative']['mode'], 'sd_only')
        self.assertFalse(new['speculative']['verify_greedy'])
        equivalent = deepcopy(new)
        equivalent['speculative'].pop('mode')
        equivalent['speculative']['verify_greedy'] = True
        self.assertEqual(equivalent, original)  # No changed thresholds, model parameters or Prompt.
        for mode, verify in [('verified', False), ('sd_only', True), ('unknown', False), ('sd_only', 0)]:
            invalid = deepcopy(new)
            invalid['speculative'].update(mode=mode, verify_greedy=verify)
            with self.subTest(mode=mode, verify=verify), self.assertRaises(ValueError):
                speculative.validate_speculative(invalid)
        invalid = deepcopy(original)
        invalid['speculative']['verify_greedy'] = False
        with self.assertRaises(ValueError):
            speculative.validate_speculative(invalid)

    def test_sd_only_generates_once_and_never_claims_greedy_passed(self):
        self.fail_greedy = True  # Calling SD0 would fail this request.
        answer, detail = self.wrapper.answer('q')
        sd = detail['speculative']
        self.assertEqual(answer, 'SD answer')
        self.assertEqual([call[0] for call in self.calls], ['SD2-1'])
        self.assertEqual(sd['final_generation_source'], 'sd')
        self.assertFalse(sd['greedy_verification_enabled'])
        self.assertIsNone(sd['greedy_verification_seconds'])
        self.assertIsNone(sd['exact_token_match_greedy'])
        self.assertIsNone(sd['token_mismatch'])
        self.assertIsNone(sd['paired_greedy'])
        self.assertIsNone(sd['sd_speedup_eligible'])
        self.assertEqual(sd['lossless_status'], 'not_checked_sd_only')
        self.assertEqual(sd['output_token_ids'], sd['raw_generation']['output_token_ids'])
        self.assertEqual(sd['output_token_count'], 2)
        self.assertEqual((sd['target_forward_calls'], sd['draft_forward_calls']), (1, 1))

    def test_faq_hit_rag_write_and_cache_hit_share_original_routing(self):
        engine = self.integrated()
        with patch.object(self.cache, 'answer', wraps=self.cache.answer) as cache_answer:
            answer, detail = engine.answer('FAQ')
            self.assertEqual(answer, 'FAQ answer')
            self.assertEqual(detail['answer_source'], 'faq')
            cache_answer.assert_not_called()
        self.assertEqual(self.calls, [])
        self.assertFalse(self.cache.cache.entries)
        answer, detail = engine.answer('q')
        self.assertEqual(detail['answer_source'], 'rag')
        self.assertTrue(detail['cache']['saved'])
        self.assertEqual(next(iter(self.cache.cache.entries.values())).answer, answer)
        self.assertEqual(detail['answer_verification'], 'unverified')
        answer2, detail = engine.answer('q')
        self.assertEqual((answer2, detail['answer_source']), (answer, 'cache'))
        self.assertFalse(detail['speculative']['used'])
        self.assertEqual([call[0] for call in self.calls], ['SD2-1'])

    def test_sd_failure_empty_or_truncated_output_not_cached(self):
        original = self.wrapper.engine.generate
        def empty(*args):
            result = original(*args)
            result['output_text'] = ''
            return result
        self.wrapper.engine.generate = empty
        with self.assertRaises(ValueError):
            self.cache.answer('q')
        self.assertFalse(self.cache.cache.entries)
        self.wrapper.engine.generate = lambda *args: (_ for _ in ()).throw(RuntimeError('SD failed'))
        with self.assertRaises(RuntimeError):
            self.cache.answer('q')
        self.assertFalse(self.cache.cache.entries)
        def truncated(*args):
            result = original(*args)
            result.update(reached_token_limit=True, stop_reason='token_limit')
            return result
        self.wrapper.engine.generate = truncated
        _, detail = self.cache.answer('q')
        self.assertFalse(detail['saved'])
        self.assertTrue(detail['rag_detail']['truncated'])

    def test_mode_is_part_of_resource_identity(self):
        self.wrapper.base.llm = N(generate=lambda prompt: None)
        self.wrapper.engine.identity = {'engine_fixture': True}
        with patch.object(speculative, 'A0Adapter', return_value=self.wrapper.base), \
                patch.object(speculative, 'SDEngine', return_value=self.wrapper.engine), \
                patch('experiments.algorithms.speculative_decoding.compatibility.check_api', return_value={}):
            verified = speculative.SpeculativeA0(ROOT, json.loads(fixture.CONFIG.read_text(encoding='utf-8')))
            only = speculative.SpeculativeA0(ROOT, json.loads(CONFIG.read_text(encoding='utf-8')))
        self.assertEqual(verified.identity['speculative_mode'], 'verified')
        self.assertEqual(only.identity['speculative_mode'], 'sd_only')
        self.assertNotEqual(verified.identity['greedy_verification_enabled'], only.identity['greedy_verification_enabled'])

    def test_http_health_csv_and_cross_request_cache(self):
        from fastapi.testclient import TestClient
        engine = self.integrated()
        with tempfile.TemporaryDirectory() as folder, patch.object(server, 'ROOT', Path(folder)), \
                patch('baseline_api.integrated_chat.IntegratedQA', return_value=engine):
            with TestClient(server.create_app(CONFIG, 'experiments/results/session')) as client:
                for _ in range(300):
                    health = client.get('/health')
                    if health.status_code == 200:
                        break
                    time.sleep(.01)
                self.assertEqual(health.status_code, 200, health.text)
                self.assertEqual(health.json()['api_mode'], 'integrated_speculative_sd_only')
                self.assertFalse(health.json()['greedy_verification_enabled'])
                responses = [client.post('/chat', json={'question': q}) for q in ('FAQ', 'q', 'q')]
                for response in responses:
                    self.assertEqual(response.status_code, 200, response.text)
                    self.assertTrue(response.json()['success'])
                    self.assertEqual(response.json()['api_mode'], 'integrated_speculative_sd_only')
                self.assertEqual([r.json()['answer_source'] for r in responses], ['faq', 'rag', 'cache'])
                sd = responses[1].json()['speculative']
                self.assertEqual(sd['final_generation_source'], 'sd')
                self.assertEqual(sd['lossless_status'], 'not_checked_sd_only')
                self.assertIsNone(sd['greedy_verification_seconds'])
                for response in (responses[0], responses[2]):
                    self.assertFalse(response.json()['speculative']['used'])
                path = Path(folder)/'experiments/results/api_logs/v6_sd_only_chat_history.csv'
                self.assertFalse((path.parent/'v6_chat_history.csv').exists())
                with path.open(encoding='utf-8-sig', newline='') as stream:
                    rows = list(csv.reader(stream))
                self.assertEqual(rows[0], server.V6_CSV_COLUMNS)
                self.assertTrue(all(len(row) == 12 for row in rows))
                self.assertEqual([row[8] for row in rows[1:]], ['', '', ''])
                self.assertEqual(rows[2][7], '0.1000')
                records = [json.loads(line) for line in (Path(folder)/'experiments/results/session/results.jsonl').read_text(encoding='utf-8').splitlines()]
                self.assertEqual(records[1]['timings']['speculative']['final_generation_source'], 'sd')
                self.assertIsNone(records[1]['timings']['speculative']['greedy_verification_seconds'])
                self.assertFalse((Path(folder)/'experiments/results/session/run.json').read_text(encoding='utf-8') == '')
        self.assertEqual([call[0] for call in self.calls], ['SD2-1'])


if __name__ == '__main__':
    unittest.main()
