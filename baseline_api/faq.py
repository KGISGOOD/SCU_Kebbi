"""Conservative FAQ routing around the existing A0; no research runner changes."""
import math
import hashlib
import re
import unicodedata
from pathlib import Path


def validate_faq_config(config):
    if not isinstance(config, dict) or set(config) - {'acceptance_mode', 'min_margin'} != {'faq_file', 'threshold', 'threshold_status', 'bge_content_sha256', 'bge_revision'}:
        raise ValueError('invalid Baseline 2 FAQ configuration')
    if config.get('acceptance_mode', 'experimental') != 'experimental':
        raise ValueError('unknown FAQ acceptance_mode')
    margin = config.get('min_margin', 0.05)
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 <= margin <= 2:
        raise ValueError('invalid FAQ min_margin')
    value = config['threshold']
    if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('FAQ threshold must be explicit and finite, never null')
    if config['threshold_status'] != 'uncalibrated':
        raise ValueError('this candidate configuration must remain uncalibrated')
    for key in ('faq_file', 'bge_content_sha256', 'bge_revision'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError('missing FAQ ' + key)


def lexical_intent(text):
    """Bounded wording aliases, NOT a general semantic equivalence verifier.

    All unrecognized wording, negation, years, scopes and qualifications remain.
    """
    text = unicodedata.normalize('NFKC', text).strip()
    text = re.sub(r'^(?:想請問|請問|我想知道)', '', text)
    for old, new in [('電子郵件信箱', '電子郵件'), ('電子信箱', '電子郵件'),
                     ('聯絡信箱', '聯絡電子郵件'), ('幾學分', '多少學分')]:
        text = text.replace(old, new)
    return re.sub(r'[\s？?，,。！!]', '', text)


def acceptance(query, candidate, score, threshold, *, mode='experimental', second_score=None, min_margin=0.05):
    """Exact intent only; no similarity-based certification or inferred conditions."""
    if not math.isfinite(score) or score < threshold:
        return False, 'below_threshold'
    if not candidate or not str(candidate.get('answer') or '').strip():
        return False, 'empty_answer'
    if mode != 'experimental':
        raise ValueError('FAQ requires experimental acceptance without review states')
    if second_score is None or not math.isfinite(second_score) or score - second_score < min_margin:
        return False, 'insufficient_top2_margin'
    numbers = lambda text: re.findall(r'[0-9]+(?:\.[0-9]+)?', unicodedata.normalize('NFKC', text))
    if numbers(query) != numbers(candidate['question']):
        return False, 'numeric_or_year_condition_mismatch'
    if lexical_intent(query) != lexical_intent(candidate['question']):
        return False, 'intent_or_conditions_unverified'
    return True, None


class FAQFallback:
    def __init__(self, root, config, adapter):
        from .faq_data import load_csv_candidates
        from experiments.shared.model_resources import inference_identity
        from experiments.baselines.faq_semantic_matching.matcher import FAQMatcher
        from experiments.shared.vector_store import FAISSVectorStore
        import torch
        self.adapter = adapter
        self.pipeline = adapter.pipeline
        options = config['faq']
        validate_faq_config(options)
        from experiments.baselines.rag_baseline.prompts import validate_prompt_version
        validate_prompt_version(self.pipeline.prompt_version)
        identity = inference_identity(adapter.paths['bge'], sentence_model=True)
        if identity['content_sha256'] != options['bge_content_sha256']:
            raise ValueError('FAQ must use the pinned original A1 BGE-M3 identity')
        model = adapter.embedder.model
        if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError('shared BGE must already be float32; no silent casting')
        if model.max_seq_length != 512 or not adapter.embedder.normalize_embeddings:
            raise ValueError('shared BGE requires length 512 and L2 normalization')
        model.eval()
        faq_path = Path(options['faq_file'])
        faq_path = faq_path if faq_path.is_absolute() else Path(root) / faq_path
        rows, faq_hash = load_csv_candidates(faq_path)
        hashes = {str(faq_path): faq_hash}
        canonical = [dict(r, faq_id=r['intent_group_id']) for r in rows]
        matcher = FAQMatcher.__new__(FAQMatcher)
        matcher.faqs, matcher.embedder = canonical, adapter.embedder
        matcher.faq_documents = matcher._faqs_to_documents(canonical)
        store = FAISSVectorStore.__new__(FAISSVectorStore)
        store.embedding_dim, store.use_hnsw = adapter.embedder.embedding_dim, False
        store.build(adapter.embedder.encode([r['question'] for r in canonical]), matcher.faq_documents)
        matcher.vector_store, matcher.index_built = store, True
        self.matcher, self.threshold = matcher, options['threshold']
        self.acceptance_mode = options.get('acceptance_mode', 'experimental')
        self.min_margin = options.get('min_margin', 0.05)
        self.identity = dict(adapter.identity, faq={
            'candidate_count': 138, 'input_sha256': hashes, 'model': identity,
            'declared_revision': options['bge_revision'], 'threshold': self.threshold,
            'threshold_status': 'uncalibrated', 'policy': 'csv_bounded_lexical_aliases_v1',
            'review_required': False,
            'acceptance_mode': self.acceptance_mode, 'min_margin': self.min_margin,
            'experimental_policy': 'bounded_lexical_aliases_v1_not_semantic_certification',
            'policy_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'index': 'in_memory_IndexFlatIP_L2', 'shared_embedding': True,
        })

    def match_only(self, query, *, query_vector=None):
        from experiments.shared.a0_adapter import measured
        self.adapter.validate_query(query)
        # match() returns the raw top-1 FAQ even when legacy global threshold rejects it.
        # Ignore its answer/acceptance; our independent policy owns the decision.
        kwargs = {'return_faq': True}
        if query_vector is not None:
            kwargs['query_vector'] = query_vector
        (_, score, candidate, top2, _), seconds = measured(self.matcher.match, query, **kwargs)
        second_score = top2[1]['similarity_score'] if len(top2) > 1 else None
        mode = getattr(self, 'acceptance_mode', 'experimental')
        accepted, reason = acceptance(query, candidate, score, self.threshold, mode=mode,
                                      second_score=second_score, min_margin=getattr(self, 'min_margin', 0.05))
        route = {'answer_source': 'faq' if accepted else 'rag', 'faq_accepted': accepted,
                 'faq_top1_score': score, 'faq_top2_score': top2[1]['similarity_score'] if len(top2) > 1 else None,
                 'faq_rejection_reason': reason, 'faq_question_id': candidate['question_id'] if candidate else None}
        route.update(faq_acceptance_mode=mode,
                     faq_acceptance_reason='experimental_lexical_match_unverified' if accepted else None,
                     faq_human_verified=False,
                     faq_score_margin=score-second_score if second_score is not None else None)
        return (candidate['answer'] if accepted else None), dict(route, faq_seconds=seconds,
                             faq_sources=candidate['sources'] if accepted else None)

    def answer(self, query):
        answer, route = self.match_only(query)
        if route['faq_accepted']:
            detail = {'retrieval_seconds': None, 'generation_seconds': None, 'input_tokens': None}
        else:
            answer, detail = self.adapter.answer(query)
        return answer, dict(detail, **route)
