"""CPU prompt-contract tests; tokenizer/model stand-ins, no GPU or model files."""
import ast
import json
import tempfile
import time
from pathlib import Path
import subprocess
from types import SimpleNamespace as N
import unittest
from unittest.mock import Mock, patch

from experiments.baselines.rag_baseline import prompts
from experiments.shared.a0_adapter import A0Adapter
from baseline_api.speculative import SpeculativeA0

ROOT = Path(__file__).resolve().parents[1]


class Tensor:
    def __init__(self, ids):
        self.ids, self.shape = [ids], (1, len(ids))
    def tolist(self):
        return self.ids
    def __getitem__(self, item):
        return N(tolist=lambda: self.ids[item])
    def to(self, device):
        return self


class Tokenizer:
    bos_token_id = 1
    chat_template = 'CPU fixture template, NOT the real Llama template'
    def __init__(self):
        self.messages = None
        self.render_calls = 0
        self.encode_calls = []
    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False and add_generation_prompt is True
        self.messages = messages
        self.render_calls += 1
        return '<BOS>system\n\n'+messages[0]['content']+'<END>user\n\n'+messages[1]['content']+'<END>assistant\n\n'
    def __call__(self, text, add_special_tokens=True, truncation=True, **kwargs):
        self.encode_calls.append((text, add_special_tokens, truncation))
        ids = [1]*text.count('<BOS>') + [2, 3]
        if add_special_tokens:
            ids.insert(0, 1)
        if kwargs.get('return_tensors'):
            return {'input_ids': Tensor(ids), 'attention_mask': Tensor([1]*len(ids))}
        return {'input_ids': ids}


def measured(fn, *args, **kwargs):
    return fn(*args, **kwargs), .01


class PromptTests(unittest.TestCase):
    def test_system_text_matches_git_before_change(self):
        previous = subprocess.check_output(['git', 'show', 'HEAD:experiments/baselines/rag_baseline/prompts.py'], cwd=ROOT).decode('utf-8')
        node = next(n for n in ast.parse(previous).body if isinstance(n, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == 'SENIOR_SYSTEM_PROMPT' for t in n.targets))
        self.assertEqual(prompts.SENIOR_SYSTEM_PROMPT, ast.literal_eval(node.value))

    def test_roles_context_and_single_bos(self):
        tokenizer = Tokenizer()
        rendered = prompts.build_prompt('REFERENCE', 'QUESTION', prompt_version=prompts.SENIOR_PROMPT_VERSION, tokenizer=tokenizer)
        self.assertEqual(tokenizer.messages, prompts.build_messages('REFERENCE', 'QUESTION'))
        self.assertEqual(tokenizer.messages[0], {'role': 'system', 'content': prompts.SENIOR_SYSTEM_PROMPT})
        self.assertEqual(tokenizer.messages[1]['role'], 'user')
        self.assertEqual(tokenizer.messages[1]['content'], prompts.SENIOR_USER_TEMPLATE.format(context='REFERENCE', question='QUESTION'))
        ids = prompts.encode_prompt(tokenizer, rendered)['input_ids']
        self.assertEqual(ids, [1, 2, 3])
        self.assertEqual(tokenizer.encode_calls[-1][1:], (False, False))
        self.assertEqual(tokenizer.render_calls, 1)

    def test_missing_template_or_duplicate_bos_refused(self):
        with self.assertRaises(ValueError):
            prompts.build_prompt('c', 'q', prompt_version=prompts.SENIOR_PROMPT_VERSION)
        tokenizer = Tokenizer()
        tokenizer.chat_template = None
        with self.assertRaises(ValueError):
            prompts.build_prompt('c', 'q', prompt_version=prompts.SENIOR_PROMPT_VERSION, tokenizer=tokenizer)
        for text in ['<BOS><BOS>bad', 'no bos']:
            with self.assertRaises(ValueError):
                prompts.encode_prompt(tokenizer, text)

    def test_v2_v3_selection_preserves_roles_and_bos(self):
        for version, system in [(prompts.SENIOR_PROMPT_VERSION, prompts.SENIOR_SYSTEM_PROMPT),
                                (prompts.SENIOR_V3_PROMPT_VERSION, prompts.SENIOR_V3_SYSTEM_PROMPT)]:
            tokenizer = Tokenizer()
            rendered = prompts.build_prompt('CONTEXT', 'QUESTION', prompt_version=version, tokenizer=tokenizer)
            self.assertEqual(tokenizer.messages, prompts.build_messages('CONTEXT', 'QUESTION', prompt_version=version))
            self.assertEqual(tokenizer.messages[0], {'role': 'system', 'content': system})
            self.assertEqual(tokenizer.messages[1]['content'], prompts.SENIOR_USER_TEMPLATE.format(context='CONTEXT', question='QUESTION'))
            self.assertEqual(prompts.encode_prompt(tokenizer, rendered)['input_ids'], [1, 2, 3])
            self.assertEqual(tokenizer.render_calls, 1)
        with self.assertRaises(ValueError):
            prompts.validate_senior_prompt_version('baseline_basic_v1')
        with self.assertRaises(ValueError):
            prompts.validate_prompt_version('unknown')

    def test_api_health_displays_selected_and_loaded_v2_v3(self):
        from fastapi.testclient import TestClient
        from baseline_api import server
        base = json.loads((ROOT/'experiments/configs/integrated_baseline_v1.experimental.speculative.json').read_text(encoding='utf-8'))
        for version in prompts.SENIOR_PROMPT_VERSIONS:
            with tempfile.TemporaryDirectory() as directory:
                folder = Path(directory)
                config = dict(base, prompt_version=version)
                path = folder/'config.json'
                path.write_text(json.dumps(config), encoding='utf-8')
                adapter = N(identity={'prompt_version': version}, pipeline=N(prompt_version=version))
                engine = N(adapter=adapter, identity={}, close=lambda: None)
                with patch.object(server, 'ROOT', folder), patch('baseline_api.integrated_chat.IntegratedQA', return_value=engine):
                    with TestClient(server.create_app(path)) as client:
                        for _ in range(300):
                            health = client.get('/health')
                            if health.status_code == 200:
                                break
                            time.sleep(.01)
                        self.assertEqual(health.status_code, 200)
                        self.assertEqual(health.json()['prompt_version'], version)
                        self.assertEqual(health.json()['loaded_prompt_version'], version)

    def test_a0_renders_once_and_pipeline_does_not_retemplate(self):
        # Execute the actual Pipeline class without importing heavyweight model packages.
        path = ROOT/'experiments/baselines/rag_baseline/pipeline.py'
        node = next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n, ast.ClassDef))
        namespace = {'PROMPT_VERSION': prompts.PROMPT_VERSION, 'build_prompt': prompts.build_prompt,
                     'create_context_from_documents': prompts.create_context_from_documents}
        exec('from __future__ import annotations\n'+ast.get_source_segment(path.read_text(encoding='utf-8'), node), namespace)
        pipeline = namespace['RAGPipeline'].__new__(namespace['RAGPipeline'])
        tokenizer = Tokenizer()
        llm = N(tokenizer=tokenizer, strict_max_input_tokens=10000, max_new_tokens=256,
                model=N(config=N(max_position_embeddings=20000)), generate=Mock(return_value='answer'))
        pipeline.llm_loader, pipeline.prompt_version = llm, prompts.SENIOR_PROMPT_VERSION
        documents = [N(page_content='CONTEXT', metadata={'source_file': 'CPU fixture'})]
        pipeline.retrieve = Mock(return_value=(documents, [.9]))
        adapter = A0Adapter.__new__(A0Adapter)
        adapter.pipeline, adapter.llm = pipeline, llm
        adapter.validate_query = Mock()
        with patch('experiments.shared.a0_adapter.measured', measured):
            answer, detail = adapter.answer('QUESTION', query_vector='CPU vector')
        self.assertEqual(answer, 'answer')
        self.assertEqual(tokenizer.render_calls, 1)
        self.assertEqual(detail['input_tokens'], 3)
        self.assertEqual(llm.generate.call_args[0][0], tokenizer.encode_calls[0][0])

    def test_sd_and_extra_greedy_receive_identical_ids(self):
        tokenizer = Tokenizer()
        wrapper = SpeculativeA0.__new__(SpeculativeA0)
        wrapper.config = {'max_input_tokens': 4096, 'max_new_tokens': 256, 'device': 'cpu', 'precision': 'fp16', 'seed': 42}
        wrapper.options = {'draft_tokens': 4, 'verify_greedy': True}
        wrapper.question = 'QUESTION'
        wrapper.base = N(paths={'model': 'CPU Target'})
        seen = []
        def generate(snapshot, inputs, version, *args):
            seen.append((version, inputs, inputs['input_ids'].tolist()))
            return {'generation_mode': version, 'forward_calls': {'target': 1, 'draft': 1},
                    'generated_token_count': 2, 'output_token_ids': [4, 5], 'output_text': 'CPU answer',
                    'generation_seconds': .1, 'tokens_per_second': 20, 'stop_reason': 'eos', 'reached_token_limit': False}
        wrapper.engine = N(tokenizer=tokenizer, target=N(config=N(max_position_embeddings=10000)),
                           draft=N(config=N(max_position_embeddings=10000)), draft_path='CPU Draft', generate=generate)
        prompt = prompts.build_prompt('C', 'Q', prompt_version=prompts.SENIOR_PROMPT_VERSION, tokenizer=tokenizer)
        self.assertEqual(wrapper.generate(prompt), 'CPU answer')
        self.assertEqual([s[0] for s in seen], ['SD2-1', 'SD0'])
        self.assertIs(seen[0][1], seen[1][1])
        self.assertEqual(seen[0][2], [[1, 2, 3]])
        self.assertEqual(tokenizer.render_calls, 1)


if __name__ == '__main__':
    unittest.main()
