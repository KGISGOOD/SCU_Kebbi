"""Conservative FAQ routing around the existing A0; no research runner changes."""
import math
from pathlib import Path


def validate_faq_config(config):
    if not isinstance(config, dict) or set(config) != {'training_bundle', 'threshold', 'threshold_status', 'bge_content_sha256', 'bge_revision'}:
        raise ValueError('invalid Baseline 2 FAQ configuration')
    value = config['threshold']
    if type(value) not in (int, float) or not math.isfinite(value) or not -1 <= value <= 1:
        raise ValueError('FAQ threshold must be explicit and finite, never null')
    if config['threshold_status'] != 'uncalibrated':
        raise ValueError('this candidate configuration must remain uncalibrated')
    for key in ('training_bundle', 'bge_content_sha256', 'bge_revision'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError('missing FAQ ' + key)


def acceptance(query, candidate, score, threshold):
    """Exact intent only; no similarity-based certification or inferred conditions."""
    if not math.isfinite(score) or score < threshold:
        return False, 'below_threshold'
    if not candidate or not str(candidate.get('answer') or '').strip():
        return False, 'empty_answer'
    if query.strip() != candidate['question'].strip():
        return False, 'intent_or_conditions_unverified'
    if candidate.get('review_status') != 'approved':
        return False, 'answer_not_human_approved'
    if not candidate.get('sources'):
        return False, 'missing_evidence'
    return True, None


class FAQFallback:
    def __init__(self, root, config, adapter):
        from experiments.shared.faq_data import load_bundle, read_rows
        from experiments.shared.model_resources import inference_identity
        from experiments.baselines.faq_semantic_matching.matcher import FAQMatcher
        from experiments.shared.vector_store import FAISSVectorStore
        import torch
        self.adapter = adapter
        self.pipeline = adapter.pipeline
        options = config['faq']
        validate_faq_config(options)
        if self.pipeline.prompt_version != 'baseline_senior_v2':
            raise ValueError('Baseline 2 requires baseline_senior_v2')
        identity = inference_identity(adapter.paths['bge'], sentence_model=True)
        if identity['content_sha256'] != options['bge_content_sha256']:
            raise ValueError('FAQ must use the pinned original A1 BGE-M3 identity')
        model = adapter.embedder.model
        if any(p.dtype != torch.float32 for p in model.parameters() if p.is_floating_point()):
            raise ValueError('shared BGE must already be float32; no silent casting')
        if model.max_seq_length != 512 or not adapter.embedder.normalize_embeddings:
            raise ValueError('shared BGE requires length 512 and L2 normalization')
        model.eval()
        bundle = Path(options['training_bundle'])
        _, manifest, hashes = load_bundle(root, bundle if bundle.is_absolute() else Path(root) / bundle)
        master = read_rows(Path(root) / manifest['frozen_dir'] / 'qa_master.jsonl')
        rows = [r for r in master if r['canonical_question_id'] == r['question_id']]
        if len(master) != 336 or len(rows) != 138 or len({r['intent_group_id'] for r in rows}) != 138:
            raise ValueError('expected immutable V2 336 rows and 138 unique canonical intents')
        canonical = [dict(r, faq_id=r['intent_group_id']) for r in rows]
        matcher = FAQMatcher.__new__(FAQMatcher)
        matcher.faqs, matcher.embedder = canonical, adapter.embedder
        matcher.faq_documents = matcher._faqs_to_documents(canonical)
        store = FAISSVectorStore.__new__(FAISSVectorStore)
        store.embedding_dim, store.use_hnsw = adapter.embedder.embedding_dim, False
        store.build(adapter.embedder.encode([r['question'] for r in canonical]), matcher.faq_documents)
        matcher.vector_store, matcher.index_built = store, True
        self.matcher, self.threshold = matcher, options['threshold']
        self.identity = dict(adapter.identity, faq={
            'candidate_count': 138, 'input_sha256': hashes, 'model': identity,
            'declared_revision': options['bge_revision'], 'threshold': self.threshold,
            'threshold_status': 'uncalibrated', 'policy': 'exact_question_and_approved_answer_v1',
            'human_approved_candidates': sum(r.get('review_status') == 'approved' for r in rows),
            'index': 'in_memory_IndexFlatIP_L2', 'shared_embedding': True,
        })

    def answer(self, query):
        from experiments.shared.a0_adapter import measured
        self.adapter.validate_query(query)
        # match() returns the raw top-1 FAQ even when legacy global threshold rejects it.
        # Ignore its answer/acceptance; our independent policy owns the decision.
        (_, score, candidate, top2, _), seconds = measured(self.matcher.match, query, return_faq=True)
        accepted, reason = acceptance(query, candidate, score, self.threshold)
        route = {'answer_source': 'faq' if accepted else 'rag', 'faq_accepted': accepted,
                 'faq_top1_score': score, 'faq_top2_score': top2[1]['similarity_score'] if len(top2) > 1 else None,
                 'faq_rejection_reason': reason, 'faq_question_id': candidate['question_id'] if candidate else None}
        if accepted:
            answer, detail = candidate['answer'], {'retrieval_seconds': None, 'generation_seconds': None,
                                                 'input_tokens': None, 'faq_sources': candidate['sources']}
        else:
            answer, detail = self.adapter.answer(query)
        return answer, dict(detail, faq_seconds=seconds, **route)
