"""No GPU or real API calls; all runner files are isolated in a temporary directory."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('runner', Path(__file__).with_name('run_v6_questions.py'))
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class FakeAPI:
    def __init__(self, path):
        self.path = path
        self.calls = []
        self.fail = False
        self.extra = False

    def __call__(self, url, method, endpoint, timeout, payload=None):
        self.calls.append((method, endpoint, payload))
        if method == 'GET':
            return dict(status='ready', api_mode='integrated_speculative', models_ready=True, accepting_requests=True)
        assert list(payload) == ['question']
        if self.fail:
            self.fail = False
            raise TimeoutError('simulated API timeout')
        answer = '回答, "引號"\n下一行'
        with self.path.open('a', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([payload['question'], answer, 'FAQ', '.01', '.02', '', '', '', '', '.001', '.031', ''])
            if self.extra:
                writer.writerow(['其他請求', answer, 'FAQ', '.01', '.02', '', '', '', '', '.001', '.031', ''])
        return dict(success=True, answer=answer, answer_source='faq', request_id=str(len(self.calls)))


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.history, self.output = self.root/'history.csv', self.root/'result.csv'
        with self.history.open('w', encoding='utf-8-sig', newline='') as f:
            csv.writer(f).writerow(runner.HISTORY_HEADER)
        self.api = FakeAPI(self.history)
        self.plan = [('A', '問題, "一"\n第二行'), ('B', '問題二')]
        self.quiet = patch('sys.stdout', new_callable=io.StringIO)
        self.stdout = self.quiet.start()
        self.addCleanup(self.quiet.stop)
        self.err = patch('sys.stderr', new_callable=io.StringIO)
        self.err.start()
        self.addCleanup(self.err.stop)

    def run_plan(self, **kwargs):
        return runner.run(self.plan, self.output, self.history, 'http://fixture', client=self.api, **kwargs)

    def test_actual_question_counts_and_order(self):
        directory = runner.ROOT/'tests/questions'
        plan = runner.questions(directory)
        self.assertEqual(len(plan), 200)
        expected = []
        for category, name, count in runner.FILES:
            rows = runner.read_csv(directory/name, ['問題'])
            self.assertEqual(len(rows), count)
            expected.extend((category, row[0]) for row in rows)
            self.assertEqual(len(runner.questions(directory, category)), count)
        self.assertEqual(plan, expected)

    def test_csv_format_errors_are_distinct(self):
        path = self.root/'invalid.csv'
        cases = [
            (b'question\nhello\n', '缺少 UTF-8 BOM'),
            ('\ufeff問題\n完整問題'.encode('utf-8'), '缺少結尾換行符號'),
            ('\ufeff問題\n"未結束的問題\n'.encode('utf-8'), '資料不完整或語法錯誤'),
            ('\ufeff問題\n問題,多餘欄位\n'.encode('utf-8'), '資料不完整或欄位數不符'),
            ('\ufeff錯誤欄位\n問題\n'.encode('utf-8'), '欄位名稱不相容'),
            (b'\xef\xbb\xbf\xff\n', '不是有效的 UTF-8'),
        ]
        for content, error in cases:
            with self.subTest(error=error):
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, error):
                    runner.read_csv(path, ['問題'])

    def test_complete_multiline_and_duplicate_questions_preserved(self):
        path = self.root/'valid.csv'
        rows = [['問題'], ['相同問題'], ['相同問題'], ['含逗號,引號"及\n換行的完整問題']]
        with path.open('w', encoding='utf-8-sig', newline='') as stream:
            csv.writer(stream).writerows(rows)
        before = path.read_bytes()
        self.assertEqual(runner.read_csv(path, ['問題']), rows[1:])
        self.assertEqual(path.read_bytes(), before)

    def test_200_question_expectations_preserve_b_duplicates_and_order(self):
        directory = runner.ROOT/'tests/questions'
        plan = runner.questions(directory)
        references = runner.expectations(directory)
        self.assertEqual([(r['category'], r['question']) for r in references], plan)
        b = [q for category, q in plan if category == 'B']
        self.assertEqual(len(b), 35)
        self.assertEqual(len(b)-len(set(b)), 8)

    def test_header_matches_real_api_and_13_columns(self):
        from baseline_api.server import V6_CSV_COLUMNS
        self.assertEqual(runner.HISTORY_HEADER, V6_CSV_COLUMNS)
        self.assertEqual(len(runner.HEADER), 13)

    def test_sync_success_progress_and_roundtrip(self):
        self.run_plan()
        rows = runner.read_csv(self.output, runner.HEADER)
        self.assertEqual([r[:2] for r in rows], [list(p) for p in self.plan])
        self.assertEqual(rows[0][2], '回答, "引號"\n下一行')
        self.assertEqual(rows[0][-2:], ['0.0310', ''])
        self.assertEqual(self.api.calls[0][:2], ('GET', '/health'))
        self.assertEqual([p['question'] for m, e, p in self.api.calls if m == 'POST'], [p[1] for p in self.plan])
        self.assertIn('[2/2]', self.stdout.getvalue())
        before = self.history.read_bytes()
        count = len(self.api.calls)
        self.run_plan(resume=True)
        self.assertEqual(len(self.api.calls), count)
        self.assertEqual(self.history.read_bytes(), before)

    def test_timeout_preserves_completed_rows_and_requires_explicit_retry(self):
        original = self.api
        def fail_second(*args, **kwargs):
            if args[1] == 'POST' and args[-1]['question'] == self.plan[1][1]:
                original.fail = True
            return original(*args, **kwargs)
        self.api = fail_second
        with self.assertRaises(TimeoutError):
            self.run_plan()
        self.assertEqual(len(runner.read_csv(self.output, runner.HEADER)), 1)
        self.api = original
        count = len(original.calls)
        with self.assertRaises(ValueError):
            self.run_plan(resume=True)
        self.assertEqual(len(original.calls), count)
        self.run_plan(resume=True, retry_uncertain=True)
        self.assertEqual(len(runner.read_csv(self.output, runner.HEADER)), 2)

    def test_durable_row_before_checkpoint_is_not_sent_again(self):
        self.run_plan()
        state = self.output.with_suffix('.state.json')
        value = json.loads(state.read_text(encoding='utf-8'));value['pending'] = 1
        state.write_text(json.dumps(value), encoding='utf-8')
        count = len(self.api.calls)
        self.run_plan(resume=True)
        self.assertEqual(len(self.api.calls), count)

    def test_resume_rejects_changed_question_or_category_order(self):
        self.run_plan()
        rows = runner.read_csv(self.output, runner.HEADER)
        rows[0][0] = 'F'
        with self.output.open('w', encoding='utf-8-sig', newline='') as f:
            writer = csv.writer(f);writer.writerow(runner.HEADER);writer.writerows(rows)
        with self.assertRaises(ValueError):
            self.run_plan(resume=True)

    def test_resume_rejects_changed_future_plan(self):
        self.run_plan()
        self.plan[1] = ('B', '更改後問題')
        with self.assertRaises(ValueError):
            self.run_plan(resume=True)

    def test_existing_output_requires_flag_and_restart_does_not_clear_history(self):
        self.run_plan()
        before = runner.read_csv(self.history, runner.HISTORY_HEADER)
        with self.assertRaises(ValueError):
            self.run_plan()
        self.run_plan(restart=True)
        self.assertEqual(runner.read_csv(self.history, runner.HISTORY_HEADER)[:len(before)], before)
        self.assertEqual(len(runner.read_csv(self.output, runner.HEADER)), 2)

    def test_concurrent_history_append_stops_without_wrong_pairing(self):
        self.api.extra = True
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual(runner.read_csv(self.output, runner.HEADER), [])

    def test_unsuccessful_answer_is_never_saved(self):
        self.api = lambda *a, **k: (dict(status='ready', api_mode='integrated_speculative', models_ready=True,
                                       accepting_requests=True) if a[1] == 'GET'
                                  else dict(success=False, answer='錯誤不是答案', error='fixture_error'))
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual(runner.read_csv(self.output, runner.HEADER), [])

    def test_target_fallback_source_is_preserved_without_extra_columns(self):
        original = self.api
        def fallback(*args, **kwargs):
            response = original(*args, **kwargs)
            if args[1] == 'POST':
                rows = runner.read_csv(self.history, runner.HISTORY_HEADER)
                rows[-1][2] = 'RAG + Target Greedy Fallback'
                rows[-1][-1] = '3'
                with self.history.open('w', encoding='utf-8-sig', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(runner.HISTORY_HEADER)
                    writer.writerows(rows)
                response.update(answer_source='rag', speculative={'final_generation_source': 'target_greedy_fallback'})
            return response
        self.api = fallback
        self.run_plan()
        rows = runner.read_csv(self.output, runner.HEADER)
        self.assertEqual(len(rows[0]), 13)
        self.assertEqual(rows[0][3], 'RAG + Target Greedy Fallback')
        self.assertEqual(rows[0][-1], '3')

    def test_unready_health_sends_no_question(self):
        calls = []
        def busy(*args):
            calls.append(args);return dict(status='loading')
        self.api = busy
        with self.assertRaises(ValueError):
            self.run_plan()
        self.assertEqual([c[1] for c in calls], ['GET'])

    def test_original_history_and_questions_not_modified_by_reading(self):
        files = [runner.ROOT/'tests/questions'/name for _, name, _ in runner.FILES]
        original = runner.ROOT/'experiments/results/api_logs/v6_chat_history.csv'
        files += [original] if original.exists() else []
        before = {p: p.read_bytes() for p in files}
        runner.questions(runner.ROOT/'tests/questions')
        self.run_plan()
        self.assertEqual(before, {p: p.read_bytes() for p in files})


if __name__ == '__main__':
    unittest.main()
