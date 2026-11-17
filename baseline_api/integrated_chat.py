"""FAQ -> reviewed/experimental cache -> existing A0; V6 shared routing runtime."""
import json
from pathlib import Path
import time

from .faq import FAQFallback, validate_faq_config
from .cache import ReviewedCacheA0, ResourceGuard, validate_options
from experiments.shared.a0_adapter import measured, file_hash

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    path = Path(path).expanduser()
    path = path if path.is_absolute() else ROOT / path
    config = json.loads(path.read_text(encoding='utf-8-sig'))
    validate_faq_config(config.get('faq'))
    validate_options(config.get('cache'))
    if (config.get('prompt_version') != 'baseline_senior_v2'
            or config['faq'].get('acceptance_mode') != 'experimental'
            or config['faq']['threshold'] != 0.9 or config['faq'].get('min_margin') != 0.05
            or config['cache'].get('acceptance_mode') != 'experimental'
            or not config['cache'].get('auto_write') or config['cache']['threshold'] != 0.9):
        raise ValueError('integrated candidate requires senior_v2, experimental FAQ/cache, 0.9 thresholds and FAQ margin 0.05')
    from .speculative import validate_speculative
    validate_speculative(config)
    return config


class IntegratedQA:
    def __init__(self, root, config):
        from .speculative import SpeculativeA0
        self.adapter = SpeculativeA0(root, config)
        self.faq = FAQFallback(root, config, self.adapter)
        faq_identity = self.faq.identity['faq']
        hashes = dict(faq_identity['input_sha256'])
        if faq_identity['review']:
            hashes[faq_identity['review']['path']] = faq_identity['review']['sha256']
        # Pin routing code and FAQ resources as well as the existing A0 resources.
        hashes.update({str(p.resolve()): file_hash(p) for p in Path(__file__).parent.glob('*.py')})
        self.guard = ResourceGuard(self.adapter, extra_hashes=hashes)
        try:
            self.cache = ReviewedCacheA0(self.adapter, config['cache'], resource_guard=self.guard)
        except Exception:
            self.guard.close()
            raise
        self.identity = {'a0': self.adapter.identity, 'faq': faq_identity,
                         'cache': self.cache.identity, 'extra_sha256': hashes,
                         'formal_quality_certified': False}

    def answer(self, question):
        start = time.perf_counter()
        _, validation = measured(self.guard.verify)
        vector, embedding = measured(self.adapter.embed, question)
        (answer, faq), faq_seconds = measured(self.faq.match_only, question, query_vector=vector)
        timing = {'resource_validation_seconds': validation, 'embedding_seconds': embedding,
                  'faq_seconds': faq_seconds, 'cache_lookup_seconds': 0.,
                  'rag_llm_seconds': 0., 'cache_write_seconds': 0.}
        cache_detail = None
        if faq['faq_accepted']:
            source, verification = 'faq', 'unverified'
        else:
            answer, cache_detail = self.cache.answer(question, query_vector=vector)
            source, verification = cache_detail['answer_source'], cache_detail['answer_verification']
            for name, value in cache_detail['timings'].items():
                timing[name] = timing.get(name, 0.) + value
        speculative = {}
        if 'speculative' in self.adapter.config:
            from .speculative import unused_statistics
            speculative['speculative'] = ((cache_detail.get('rag_detail') or {}).get('speculative')
                                          if source == 'rag' else unused_statistics())
        return answer, {**speculative, 'answer_source': source, 'answer_verification': verification,
                        'layers': ['faq'] if source == 'faq' else ['faq', 'cache'] if source == 'cache'
                                  else ['faq', 'cache', 'rag', 'llm'],
                        'faq': faq, 'cache': cache_detail,
                        'cache_size': len(self.cache.cache.entries),
                        'timings': timing, 'total_seconds': time.perf_counter()-start}

    def close(self):
        self.guard.close()
        if hasattr(self.adapter, 'close'):
            self.adapter.close()
