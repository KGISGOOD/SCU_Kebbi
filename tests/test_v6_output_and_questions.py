"""Fixed FAQ snapshot and output-only formatting contracts; no GPU/API inference."""
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from baseline_api.timing_output import seconds_text, timing_json, executed_stage_timings
from experiments.baselines.rag_baseline import prompts
from test_run_v6_questions import runner, FakeAPI

ROOT = Path(__file__).resolve().parents[1]


class OutputAndQuestionTests(unittest.TestCase):
    def test_fixed_counts_bom_and_one_column(self):
        plan = runner.questions(ROOT/'tests/questions')
        self.assertEqual(len(plan), 200)
        for category, name, count in runner.FILES:
            path = ROOT/'tests/questions'/name
            self.assertTrue(path.read_bytes().startswith(b'\xef\xbb\xbf'))
            self.assertEqual(len(runner.questions(path.parent, category)), count)

    def test_a_is_exact_faq_and_b_preserves_intent_mapping(self):
        references = runner.expectations(ROOT/'tests/questions')
        with (ROOT/'experiments/datasets/faq_精簡問答_138筆.csv').open(encoding='utf-8-sig', newline='') as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames, ['問題', '答案'])
            faq = list(reader)
        self.assertEqual(len(faq), 138)
        a = [r for r in references if r['category'] == 'A']
        b = [r for r in references if r['category'] == 'B']
        self.assertEqual((len(a), len(b)), (35, 35))
        for standard in a:
            original = faq[standard['faq_row']-1]
            self.assertEqual(standard['question'], original['問題'])
            self.assertEqual(standard['reference_answer'], original['答案'])
        for paraphrase in b:
            if 'faq_row' in paraphrase:
                original = faq[paraphrase['faq_row']-1]
                self.assertEqual(paraphrase['reference_question'], original['問題'])
                self.assertEqual(paraphrase['reference_answer'], original['答案'])
            else:
                self.assertTrue(paraphrase['reference_review_required'])

    def test_changed_faq_snapshot_is_rejected_before_requests(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            manifest = json.loads((ROOT/'tests/questions/expectations.json').read_text(encoding='utf-8'))
            manifest['faq_sha256'] = '0'*64
            (path/'expectations.json').write_text(json.dumps(manifest), encoding='utf-8')
            with self.assertRaises(ValueError):
                runner.expectations(path)

    def test_seconds_are_formatted_only_on_output(self):
        original = {'stage_timings': {'embedding_seconds': .000123456, 'faq_seconds': None},
                    'latency_seconds': 3.4, 'faq_top1_score': .987654321, 'output_token_count': 4}
        encoded = timing_json(original)
        self.assertIn('"embedding_seconds":0.0001', encoded)
        self.assertIn('"latency_seconds":3.4000', encoded)
        self.assertIn('"faq_seconds":null', encoded)
        self.assertEqual(json.loads(encoded)['faq_top1_score'], .987654321)
        self.assertEqual(original['stage_timings']['embedding_seconds'], .000123456)
        self.assertEqual(seconds_text(None), '')
        self.assertEqual(seconds_text(''), '')
        self.assertEqual(seconds_text(0), '0.0000')
        for value in (-1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                seconds_text(value)

    def test_unexecuted_stages_null_without_changing_measurements(self):
        original = dict(answer_source='faq', cache=None, timings=dict(embedding_seconds=.123456,
                        cache_lookup_seconds=0., cache_write_seconds=0., rag_llm_seconds=0.))
        view = executed_stage_timings(original)
        self.assertIsNone(view['cache_lookup_seconds'])
        self.assertIsNone(view['rag_llm_seconds'])
        self.assertEqual(original['timings']['rag_llm_seconds'], 0.)

    def test_runner_preserves_history_and_saves_matching_diagnostics(self):
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder)
            history, output = directory/'history.csv', directory/'result.csv'
            with history.open('w', encoding='utf-8-sig', newline='') as f:
                csv.writer(f).writerow(runner.HISTORY_HEADER)
            fake = FakeAPI(history)
            def client(*args, **kwargs):
                result = fake(*args, **kwargs)
                if args[1] == 'POST':
                    result.update(faq_accepted=False, faq_top1_score=.95, faq_top2_score=.94,
                                  faq_score_margin=.01, faq_rejection_reason='insufficient_top2_margin')
                return result
            plan = [('A', '測試問題')]
            refs = [dict(category='A', question=plan[0][1], reference_answer='參考答案')]
            runner.run(plan, output, history, 'http://fixture', client=client, references=refs)
            before = history.read_bytes()
            result = runner.read_csv(output, runner.HEADER)
            self.assertEqual(len(result[0]), 13)
            self.assertEqual(result[0][-2], '0.0310')
            state = json.loads(output.with_suffix('.state.json').read_text(encoding='utf-8'))
            self.assertEqual(state['diagnostics']['0']['actual']['faq_rejection_reason'], 'insufficient_top2_margin')
            self.assertFalse(state['diagnostics']['0']['reference_answer_exact_match'])
            runner.run(plan, output, history, 'http://fixture', client=client, references=refs, resume=True)
            self.assertEqual(before, history.read_bytes())
            self.assertTrue(all(list(call[2]) == ['question'] for call in fake.calls if call[0] == 'POST'))

    def test_official_prompt_policy_without_version_branches(self):
        self.assertEqual(prompts.PROMPT_VERSION, 'baseline_senior')
        for rule in ('不得捏造任何人名', '無法確認', '必須禮貌拒答', '先簡短自然地澄清',
                     '活動總名額', '今年招生名額', '其他學校', '教師研究領域',
                     '資料互相矛盾', '不得將合作企業', '邱嘉豪教授實驗室'):
            self.assertIn(rule, prompts.SENIOR_SYSTEM_PROMPT)
        self.assertNotIn('一般知識問題可以直接回答', prompts.SENIOR_SYSTEM_PROMPT)


if __name__ == '__main__':
    unittest.main()
