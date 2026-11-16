"""Reviewed-only cache routing; reuse AnswerCache and A0 without auto-approval."""
from datetime import datetime, timezone
import hashlib
import math
import time

from experiments.algorithms.semantic_cache.cache import AnswerCache


def validate_options(options):
    if not isinstance(options, dict) or set(options) != {'threshold', 'threshold_status', 'reuse_mode', 'max_entries', 'persistent', 'ttl_seconds', 'verify_sources_each_request'}:
        raise ValueError('invalid Baseline 3 cache options')
    threshold = options['threshold']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('explicit finite cache threshold required')
    if options['threshold_status'] != 'uncalibrated' or options['reuse_mode'] != 'reviewed_exact':
        raise ValueError('only uncalibrated / reviewed_exact supported')
    if options['persistent'] is not False or options['verify_sources_each_request'] is not True:
        raise ValueError('memory-only cache requires immutable resource verification')
    for key in ('max_entries', 'ttl_seconds'):
        if type(options[key]) is not int or options[key] <= 0:
            raise ValueError('positive integer required: ' + key)


class ReviewedCacheA0:
    def __init__(self, adapter, options):
        validate_options(options)
        if adapter.pipeline.prompt_version != 'baseline_senior_v2':
            raise ValueError('Baseline 3 requires baseline_senior_v2')
        self.adapter, self.options = adapter, dict(options)
        self.identity = {'adapter': adapter.identity, 'policy': self.options}
        self.cache = AnswerCache('semantic', self.identity, adapter.embedder.embedding_dim, options['threshold'])
        self.reviews, self.pending = {}, None

    def verify(self):
        self.adapter.verify_sources()
        self.cache.check_identity({'adapter': self.adapter.identity, 'policy': self.options})

    def answer(self, query):
        self.pending = None
        self.verify()
        vector = self.adapter.embed(query)
        entry, score, candidate = self.cache.lookup(query, vector)
        reason = 'empty_cache' if candidate is None else 'below_threshold'
        if entry is not None:
            review = self.reviews.get(entry.cache_id)
            if not review:
                reason = 'not_reviewed'
            elif time.monotonic() - review['monotonic'] > self.options['ttl_seconds']:
                reason = 'expired'
            elif query.strip() != entry.query.strip():
                # Do not guess equivalence of years, credits, eligibility or negation.
                reason = 'intent_or_conditions_unverified'
            else:
                return entry.answer, {'answer_source': 'cache', 'cache_hit': True, 'similarity': score,
                                      'reason': 'reviewed_exact_question', 'cache_id': entry.cache_id,
                                      'cache_size': len(self.cache.entries), 'review': review,
                                      'saved': False, 'rag_detail': None}
        answer, detail = self.adapter.answer(query)
        if isinstance(answer, str) and answer.strip() and detail.get('documents'):
            self.pending = {'query': query, 'answer': answer, 'vector': vector, 'detail': detail}
        return answer, {'answer_source': 'rag', 'cache_hit': False, 'similarity': score, 'reason': reason,
                        'candidate_query': candidate.query if candidate else None,
                        'cache_size': len(self.cache.entries), 'saved': False,
                        'save_status': 'awaiting_explicit_human_review' if self.pending else 'ineligible',
                        'rag_detail': detail}

    def approve_last(self, reviewer, evidence, *, confirmed=False):
        if not confirmed or not reviewer.strip() or not evidence.strip() or self.pending is None:
            raise ValueError('explicit source/conditions/completeness confirmation, reviewer and evidence required')
        self.verify()
        pending = self.pending
        if (len(self.cache.entries) >= self.options['max_entries']
                and self.cache.key(pending['query']) not in self.cache.exact):
            raise ValueError('cache capacity reached; no automatic eviction')
        key = self.cache.insert(pending['query'], pending['answer'], pending['vector'])
        self.reviews[key] = {'reviewer': reviewer.strip(), 'evidence': evidence.strip(),
                             'reviewed_at': datetime.now(timezone.utc).isoformat(),
                             'monotonic': time.monotonic(),
                             'answer_sha256': hashlib.sha256(pending['answer'].encode('utf-8')).hexdigest()}
        self.pending = None
        return key
