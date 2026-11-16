"""Opt-in A0 generation adapter; reuse the existing Target-verified SD engine."""
from copy import deepcopy
from pathlib import Path
import time
import uuid

from experiments.shared.a0_adapter import A0Adapter, measured, file_hash
from experiments.shared.identity import fingerprint
from experiments.algorithms.speculative_decoding.engine import SDEngine


def validate_speculative(config):
    options = config.get('speculative')
    if not isinstance(options, dict) or set(options) != {'enabled', 'draft_model', 'draft_tokens', 'verify_greedy'}:
        raise ValueError('invalid speculative configuration')
    if options['enabled'] is not True or options['verify_greedy'] is not True:
        raise ValueError('enabled and verify_greedy must be true; Token equivalence guard cannot be disabled')
    if not isinstance(options['draft_model'], str) or not options['draft_model'].strip():
        raise ValueError('local 1B draft_model path required')
    if type(options['draft_tokens']) is not int or options['draft_tokens'] < 1:
        raise ValueError('positive integer draft_tokens required')
    if config.get('device') != 'cuda' or config.get('precision') not in ('fp16', 'bf16', 'fp32'):
        raise ValueError('SD candidate requires explicit CUDA and floating precision; no fallback')
    if config.get('prompt_version') != 'baseline_senior_v2':
        raise ValueError('V5/V6 require baseline_senior_v2; sampling is not exposed')


def unused_statistics():
    return {'used': False, 'generation_mode': 'not_executed', 'draft_tokens_proposed': None,
            'accepted_tokens': None, 'acceptance_rate': None, 'target_forward_calls': None,
            'output_token_count': None, 'exact_token_match_greedy': None}


class SpeculativeA0:
    def __init__(self, root, config):
        validate_speculative(config)
        from experiments.algorithms.speculative_decoding.compatibility import check_api
        check_api()  # Fail before loading large models on an unreviewed runtime.
        self.config = deepcopy(config)
        self.options = self.config['speculative']
        if config['precision'] == 'fp32':
            from experiments.algorithms.speculative_decoding.numerics import configure
            configure(config)
        self.base = A0Adapter(root, config)
        sd_config = dict(config, draft_model=self.options['draft_model'],
                         standard_draft_tokens=self.options['draft_tokens'])
        self.engine = SDEngine(root, sd_config, with_draft=True, adapter=self.base)
        if config['precision'] == 'fp32':
            from experiments.algorithms.speculative_decoding.numerics import bind
            bind(self.engine)
        self.paths = dict(self.base.paths, draft_model=self.engine.draft_path)
        self.identity = dict(self.base.identity, speculative=self.engine.identity,
                             greedy_verification_enabled=self.options['verify_greedy'],
                             speculative_adapter_sha256=file_hash(Path(__file__)))
        self.original_generate = self.base.llm.generate
        self.last = None
        self.question = None
        self.base.llm.generate = self.generate

    def __getattr__(self, name):
        return getattr(self.base, name)

    def verify_sources(self):
        self.engine.verify_sources()

    def generate(self, prompt, **overrides):
        if overrides:
            raise ValueError('V5/V6 use fixed greedy settings; per-call overrides are not supported')
        start = time.perf_counter()
        inputs = self.engine.tokenizer(prompt, return_tensors='pt', truncation=False)
        count = inputs['input_ids'].shape[1]
        limit = min(self.engine.target.config.max_position_embeddings,
                    self.engine.draft.config.max_position_embeddings)
        if count > self.config['max_input_tokens'] or count + self.config['max_new_tokens'] > limit:
            raise ValueError('full prompt has too many tokens for Target/Draft; no truncation allowed')
        inputs = {k: v.to(self.config['device']) for k, v in inputs.items()}
        tokenization = time.perf_counter()-start
        if self.config['precision'] == 'fp32':
            from experiments.algorithms.speculative_decoding.numerics import ensure_memory
            ensure_memory(self.engine, count)
        snapshot = {'scenario_id': uuid.uuid4().hex, 'question': self.question,
                    'input_token_count': count, 'prompt_identity': fingerprint(inputs['input_ids'][0].tolist()),
                    'retrieval_identity': None, 'retrieval_seconds': None,
                    'prompt_and_tokenization_seconds': tokenization, 'shared_preparation_seconds': tokenization}
        sd = self.engine.generate(snapshot, inputs, 'SD2-1', self.options['draft_tokens'], 0, False, self.config['seed'])
        self.last = {'used': True, 'generation_mode': sd['generation_mode'],
                     'target_model': str(self.base.paths['model']), 'draft_model': str(self.engine.draft_path),
                     'draft_token_limit': self.options['draft_tokens'], 'draft_tokens_proposed': None,
                     'accepted_tokens': None, 'acceptance_rate': None,
                     'acceptance_statistics_status': 'unavailable; forward calls are not accepted-token counts',
                     'target_forward_calls': sd['forward_calls']['target'],
                     'draft_forward_calls': sd['forward_calls']['draft'], 'target_verification_rounds': None,
                     'output_token_count': sd['generated_token_count'], 'output_token_ids': sd['output_token_ids'],
                     'generation_seconds': sd['generation_seconds'], 'tokens_per_second': sd['tokens_per_second'],
                     'stop_reason': sd['stop_reason'], 'truncated': sd['reached_token_limit'],
                     'exact_token_match_greedy': None, 'greedy_verification_seconds': None,
                     'lossless_status': 'not_verified', 'raw_generation': sd}
        if self.options['verify_greedy']:
            baseline = self.engine.generate(snapshot, inputs, 'SD0', None, 0, False, self.config['seed'])
            match = baseline['output_token_ids'] == sd['output_token_ids']
            self.last.update(exact_token_match_greedy=match,
                             greedy_verification_seconds=baseline['generation_seconds'],
                             paired_greedy=baseline, lossless_status='passed_this_request' if match else 'failed')
            if not match:
                raise ValueError('SD token-equivalence failed against paired Target greedy; answer not served or cached')
        return sd['output_text']

    def answer(self, query, *, query_vector=None):
        self.last, self.question = None, query
        embedding = None
        if query_vector is None:
            query_vector, embedding = measured(self.base.embed, query)
        try:
            answer, detail = self.base.answer(query, query_vector=query_vector)
        except Exception as exc:
            exc.speculative_failure = self.last
            raise
        return answer, dict(detail, speculative=self.last, embedding_seconds=embedding,
                            truncated=self.last['truncated'], stop_reason=self.last['stop_reason'])

    def close(self):
        self.base.llm.generate = self.original_generate
