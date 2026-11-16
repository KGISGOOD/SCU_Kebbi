"""Strict reviewed and explicit experimental cache routing; no auto-approval."""
from datetime import datetime, timezone
import hashlib
import math
import time
import os
import re
import unicodedata
import atexit
from pathlib import Path

from experiments.algorithms.semantic_cache.cache import AnswerCache


def validate_options(options):
    if not isinstance(options, dict) or set(options) - {'acceptance_mode', 'auto_write'} != {'threshold', 'threshold_status', 'reuse_mode', 'max_entries', 'persistent', 'ttl_seconds', 'verify_sources_each_request'}:
        raise ValueError('invalid Baseline 3 cache options')
    threshold = options['threshold']
    if type(threshold) not in (int, float) or not math.isfinite(threshold) or not -1 <= threshold <= 1:
        raise ValueError('explicit finite cache threshold required')
    expected_reuse = 'bounded_semantic' if options.get('acceptance_mode', 'strict') == 'experimental' else 'reviewed_exact'
    if options['threshold_status'] != 'uncalibrated' or options['reuse_mode'] != expected_reuse:
        raise ValueError('cache reuse_mode must match the explicit acceptance_mode')
    if options.get('acceptance_mode', 'strict') not in ('strict', 'experimental'):
        raise ValueError('unknown cache acceptance_mode')
    if type(options.get('auto_write', False)) is not bool or (options.get('acceptance_mode', 'strict') == 'strict' and options.get('auto_write', False)):
        raise ValueError('Strict cannot auto-write')
    if options['persistent'] is not False or type(options['verify_sources_each_request']) is not bool:
        raise ValueError('memory-only cache requires immutable resource verification')
    for key in ('max_entries', 'ttl_seconds'):
        if type(options[key]) is not int or options[key] <= 0:
            raise ValueError('positive integer required: ' + key)


def equivalent(left, right):
    """Bounded lexical aliases, not a general semantic equivalence checker."""
    def normalized(text):
        text = unicodedata.normalize('NFKC', text).strip()
        text = re.sub(r'^(?:想請問|請問|我想知道)', '', text)
        for old, new in [('電子郵件信箱', '電子郵件'), ('電子信箱', '電子郵件'),
                         ('幾學分', '多少學分')]:
            text = text.replace(old, new)
        return re.sub(r'[\s？?，,。！!]', '', text)
    numbers = lambda t: re.findall(r'[0-9]+(?:\.[0-9]+)?', unicodedata.normalize('NFKC', t))
    return numbers(left) == numbers(right) and normalized(left) == normalized(right)


class ResourceGuard:
    """Full startup verification under Windows deny-write/delete handles.

    Metadata/inventory checks detect changes; they are NOT content hashes.
    On other systems use full hashing rather than claim an immutable snapshot.
    """
    def __init__(self, adapter):
        self.adapter, self.handles = adapter, []
        self.locked = os.name == 'nt'
        if self.locked:
            import ctypes
            from ctypes import wintypes
            self.api = ctypes.WinDLL('kernel32', use_last_error=True)
            self.api.CreateFileW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD,
                                            wintypes.LPVOID, wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE]
            self.api.CreateFileW.restype = wintypes.HANDLE
            self.api.CloseHandle.argtypes = [wintypes.HANDLE]
            self.api.CloseHandle.restype = wintypes.BOOL
            try:
                before = self.snapshot()
                for path in sorted(before):
                    handle = self.api.CreateFileW(path, 0x80000000, 1, None, 3, 128, None)
                    if handle == ctypes.c_void_p(-1).value:
                        raise OSError(ctypes.get_last_error(), 'cannot protect resource from writes: ' + path)
                    self.handles.append(handle)
                adapter.verify_sources()
                self.initial = self.snapshot()
                if before != self.initial:
                    raise ValueError('resource changed while establishing snapshot')
            except Exception:
                self.close()
                raise
            atexit.register(self.close)
        else:
            adapter.verify_sources()

    def snapshot(self):
        files = set()
        for directory in self.adapter.paths.values():
            files.update(p.resolve() for p in Path(directory).rglob('*') if p.is_file())
        for folder in ('experiments/algorithms', 'experiments/baselines/rag_baseline', 'experiments/shared'):
            files.update(p.resolve() for p in (self.adapter.root / folder).rglob('*.py'))
        files.add(Path(__file__).resolve())
        files.add(Path(__file__).with_name('cache_chat.py').resolve())
        return {str(p): (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ino) for p in files}

    def verify(self):
        if not self.locked:
            self.adapter.verify_sources()
        elif not self.handles or self.snapshot() != self.initial:
            raise ValueError('protected resource inventory changed; restart required')

    def close(self):
        for handle in self.handles:
            self.api.CloseHandle(handle)
        self.handles.clear()


class ReviewedCacheA0:
    def __init__(self, adapter, options, *, resource_guard=None):
        validate_options(options)
        if adapter.pipeline.prompt_version != 'baseline_senior_v2':
            raise ValueError('Baseline 3 requires baseline_senior_v2')
        self.adapter, self.options = adapter, dict(options)
        self.mode = options.get('acceptance_mode', 'strict')
        self.identity = {'adapter': adapter.identity, 'policy': self.options}
        self.cache = AnswerCache('semantic', self.identity, adapter.embedder.embedding_dim, options['threshold'])
        self.reviews, self.pending = {}, None
        self.guard = resource_guard
        if self.guard is None and not options['verify_sources_each_request']:
            self.guard = ResourceGuard(adapter)

    def verify(self):
        if self.guard is not None:
            self.guard.verify()
        else:
            self.adapter.verify_sources()
        self.cache.check_identity({'adapter': self.adapter.identity, 'policy': self.options})

    def close(self):
        if self.guard is not None and hasattr(self.guard, 'close'):
            self.guard.close()

    def eligible(self, answer, detail):
        docs = detail.get('documents') or []
        return (isinstance(answer, str) and bool(answer.strip()) and bool(docs)
                and detail.get('truncated') is not True
                and detail.get('stop_reason') != 'max_new_tokens'
                and all(isinstance(d, dict) and isinstance(d.get('content'), str)
                        and d['content'].strip() and d.get('metadata') for d in docs))

    def save(self, pending, status, reviewer='', evidence=''):
        key = self.cache.key(pending['query'])
        existing = self.cache.entries.get(self.cache.exact.get(key))
        if existing and existing.answer != pending['answer']:
            return None, 'append_only_conflict'
        if existing is None and len(self.cache.entries) >= self.options['max_entries']:
            return None, 'capacity_reached'
        key = self.cache.insert(pending['query'], pending['answer'], pending['vector'])
        self.reviews[key] = {'status': status, 'reviewer': reviewer, 'evidence': evidence,
                             'reviewed_at': datetime.now(timezone.utc).isoformat() if status == 'approved' else None,
                             'monotonic': time.monotonic(),
                             'answer_sha256': hashlib.sha256(pending['answer'].encode('utf-8')).hexdigest()}
        return key, 'stored_' + status

    def answer(self, query):
        from experiments.shared.a0_adapter import measured
        started = time.perf_counter()
        self.pending = None
        timings = {'resource_validation_seconds': 0., 'embedding_seconds': 0.,
                   'cache_lookup_seconds': 0., 'rag_llm_seconds': 0., 'cache_write_seconds': 0.}
        def stage(name, fn, *args):
            result, seconds = measured(fn, *args)
            timings[name] = seconds
            return result
        valid, error = True, None
        try:
            stage('resource_validation_seconds', self.verify)
        except (ValueError, OSError) as exc:
            valid, error = False, str(exc)
        # Unsafe resource identity is a miss and then a hard stop: the loaded RAG
        # resources may themselves be stale. Never silently generate with them.
        if not valid:
            raise ValueError('Cache Miss: resource_version_unverified; RAG also refused; restart: ' + error)
        vector = stage('embedding_seconds', self.adapter.embed, query)
        entry, score, candidate = stage('cache_lookup_seconds', self.cache.lookup, query, vector)
        reason = 'empty_cache' if candidate is None else 'below_threshold'
        hit, hit_type, review = False, None, None
        if entry is not None:
            review = self.reviews.get(entry.cache_id)
            if not review or (self.mode == 'strict' and review['status'] != 'approved'):
                reason = 'not_reviewed'
            elif time.monotonic() - review['monotonic'] > self.options['ttl_seconds']:
                reason = 'expired'
            elif query.strip() == entry.query.strip():
                hit, hit_type, reason = True, 'Exact', 'same_question'
            elif self.mode == 'experimental' and equivalent(query, entry.query):
                hit, hit_type, reason = True, 'Semantic', 'bounded_lexical_equivalence_unverified'
            else:
                reason = 'intent_or_conditions_unverified'
        saved, save_status, detail = False, 'not_applicable', None
        if hit:
            answer = entry.answer
        else:
            answer, detail = stage('rag_llm_seconds', self.adapter.answer, query)
            save_status = 'ineligible'
            if self.eligible(answer, detail):
                self.pending = {'query': query, 'answer': answer, 'vector': vector, 'detail': detail}
                save_status = 'awaiting_explicit_human_review'
                if self.mode == 'experimental' and self.options.get('auto_write', False):
                    key, save_status = stage('cache_write_seconds', self.save, self.pending, 'unverified')
                    saved = key is not None
        return answer, {'answer_source': 'cache' if hit else 'rag', 'cache_hit': hit,
                        'similarity': score, 'hit_type': hit_type, 'reason': reason,
                        'candidate_query': candidate.query if candidate else None,
                        'cache_size': len(self.cache.entries), 'saved': saved, 'save_status': save_status,
                        'answer_verification': review['status'] if hit else 'unverified',
                        'semantic_safety_verified': False, 'acceptance_mode': self.mode,
                        'timings': timings, 'total_seconds': time.perf_counter()-started, 'rag_detail': detail}

    def approve_last(self, reviewer, evidence, *, confirmed=False):
        if not confirmed or not reviewer.strip() or not evidence.strip() or self.pending is None:
            raise ValueError('explicit source/conditions/completeness confirmation, reviewer and evidence required')
        self.verify()
        key, reason = self.save(self.pending, 'approved', reviewer.strip(), evidence.strip())
        if key is None:
            raise ValueError(reason)
        self.pending = None
        return key
